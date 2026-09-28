"""Artifact-only protocol boundary for one R/G focus measurement."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from fast_ofm_core.artifacts import ArtifactError, load_json_artifact, resolve_artifact

from .rg_focus_control import RGFocusControlSettings, decide_correction
from .rg_focus_estimator import RGFocusInputs, estimate_rg_shift
from .rg_focus_field import FrameReference, TissueFieldError, prepare_tissue_field
from .rg_focus_model import RGFocusMeasurementPolicy, RGFocusModelProfile

FRAME_MEDIA_TYPE = "application/vnd.fast-ofm.rg-frame+npz"
CALIBRATION_MEDIA_TYPE = "application/vnd.fast-ofm.rg-calibration+json"
MAXIMUM_FRAME_BUNDLE_BYTES = 512 * 1024 * 1024
MAXIMUM_FRAME_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAXIMUM_CALIBRATION_BYTES = 4 * 1024 * 1024
FRAME_ARRAYS = {
    "white_frame",
    "red_plane",
    "green_plane",
    "red_source_jpeg8",
    "green_source_jpeg8",
    "red_valid",
    "green_valid",
    "metadata_json",
}


class RGServiceError(ValueError):
    """A stable request or artifact-content failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _checked_npz(path: Path) -> dict[str, np.ndarray]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if (
                any(member.flag_bits & 0x1 for member in members)
                or sum(member.file_size for member in members) > MAXIMUM_FRAME_UNCOMPRESSED_BYTES
            ):
                raise RGServiceError("INVALID_FRAME_BUNDLE", "Frame archive is unsafe or too large")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != FRAME_ARRAYS:
                raise RGServiceError(
                    "INVALID_FRAME_BUNDLE", "Frame bundle has missing or unknown arrays"
                )
            return {name: np.array(archive[name], copy=True) for name in FRAME_ARRAYS}
    except RGServiceError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise RGServiceError("INVALID_FRAME_BUNDLE", "Frame bundle is not a valid NPZ") from error


def _metadata(value: np.ndarray) -> dict[str, Any]:
    if value.shape != () or value.dtype.kind not in {"U", "S"}:
        raise RGServiceError("INVALID_FRAME_BUNDLE", "metadata_json must be a scalar string")
    try:
        payload = json.loads(str(value.item()))
    except json.JSONDecodeError as error:
        raise RGServiceError("INVALID_FRAME_BUNDLE", "metadata_json is not valid JSON") from error
    if not isinstance(payload, dict):
        raise RGServiceError("INVALID_FRAME_BUNDLE", "Frame metadata must be an object")
    return payload


def _calibration(
    value: Mapping[str, Any],
) -> tuple[
    str,
    Mapping[str, object],
    RGFocusMeasurementPolicy,
    RGFocusModelProfile | None,
    RGFocusControlSettings,
    int,
    float,
]:
    allowed = {
        "schema_version",
        "calibration_id",
        "geometry",
        "measurement_policy",
        "profile",
        "control_settings",
        "iteration",
        "total_correction_um",
    }
    if set(value) - allowed or value.get("schema_version") != "1.0":
        raise RGServiceError("INVALID_CALIBRATION", "Calibration bundle schema is unsupported")
    calibration_id = value.get("calibration_id")
    geometry = value.get("geometry")
    if (
        not isinstance(calibration_id, str)
        or not calibration_id
        or not isinstance(geometry, Mapping)
    ):
        raise RGServiceError("INVALID_CALIBRATION", "Calibration ID and geometry are required")
    try:
        policy = RGFocusMeasurementPolicy.model_validate(value.get("measurement_policy"))
        policy.require_jpeg()
        profile_value = value.get("profile")
        profile = (
            None if profile_value is None else RGFocusModelProfile.model_validate(profile_value)
        )
        control = RGFocusControlSettings.model_validate(value.get("control_settings", {}))
    except (ValidationError, ValueError) as error:
        raise RGServiceError("INVALID_CALIBRATION", str(error)) from error
    iteration = value.get("iteration", 0)
    total = value.get("total_correction_um", 0.0)
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise RGServiceError("INVALID_CALIBRATION", "iteration must be a non-negative integer")
    if isinstance(total, bool) or not isinstance(total, (int, float)) or not np.isfinite(total):
        raise RGServiceError("INVALID_CALIBRATION", "total_correction_um must be finite")
    return calibration_id, geometry, policy, profile, control, iteration, float(total)


def measure_payload(
    payload: Mapping[str, object], *, artifact_roots: Sequence[Path]
) -> dict[str, object]:
    """Measure one verified frame bundle with one immutable calibration bundle."""
    allowed = {
        "frame",
        "calibration",
        "position_um",
        "maximum_abs_correction_um",
        "diagnostics",
    }
    if set(payload) - allowed or not {"frame", "calibration", "position_um"} <= set(payload):
        raise RGServiceError("INVALID_PAYLOAD", "R/G payload fields are invalid")
    position = payload.get("position_um")
    if not isinstance(position, Mapping) or set(position) != {"x_um", "y_um", "z_um"}:
        raise RGServiceError("INVALID_PAYLOAD", "position_um must contain X, Y and Z")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
        for value in position.values()
    ):
        raise RGServiceError("INVALID_PAYLOAD", "position_um values must be finite numbers")
    try:
        frame_path = resolve_artifact(
            payload.get("frame"),
            roots=artifact_roots,
            media_type=FRAME_MEDIA_TYPE,
            maximum_size_bytes=MAXIMUM_FRAME_BUNDLE_BYTES,
        )
        calibration_path = resolve_artifact(
            payload.get("calibration"),
            roots=artifact_roots,
            media_type=CALIBRATION_MEDIA_TYPE,
            maximum_size_bytes=MAXIMUM_CALIBRATION_BYTES,
        )
        calibration = _calibration(load_json_artifact(calibration_path))
        arrays = _checked_npz(frame_path)
        metadata = _metadata(arrays.pop("metadata_json"))
        if set(metadata) != {"white_reference", "red_reference", "green_reference"}:
            raise RGServiceError("INVALID_FRAME_BUNDLE", "Frame metadata fields are invalid")
        white_reference = FrameReference.model_validate(metadata["white_reference"])
        red_reference = FrameReference.model_validate(metadata["red_reference"])
        green_reference = FrameReference.model_validate(metadata["green_reference"])
        calibration_id, geometry, policy, profile, control, iteration, total = calibration
        field = prepare_tissue_field(
            arrays["white_frame"],
            white_reference,
            geometry,
            policy.tissue_field,
            policy.core,
        )
        measurement = estimate_rg_shift(
            RGFocusInputs(
                red_plane=arrays["red_plane"],
                green_plane=arrays["green_plane"],
                red_source_jpeg8=arrays["red_source_jpeg8"],
                green_source_jpeg8=arrays["green_source_jpeg8"],
                red_valid=arrays["red_valid"],
                green_valid=arrays["green_valid"],
                field=field,
                red_reference=red_reference,
                green_reference=green_reference,
                geometry_value=geometry,
            ),
            policy.core,
            policy.estimator,
        )
    except ArtifactError as error:
        raise RGServiceError(error.code, str(error)) from error
    except (KeyError, ValidationError, TissueFieldError, ValueError) as error:
        code = error.code if isinstance(error, TissueFieldError) else "INVALID_FRAME_BUNDLE"
        raise RGServiceError(code.upper(), str(error)) from error

    result: dict[str, object] = {
        "measurement_status": "accepted" if measurement.status == "ready" else "refused",
        "confidence": measurement.confidence,
        "calibration_id": calibration_id,
        "selected_patch_count": measurement.inlier_patch_count,
        "correction_status": "not_requested",
        "qc": measurement.model_dump(mode="json"),
    }
    if measurement.status != "ready" or measurement.dx is None or measurement.dy is None:
        result["refusal_code"] = measurement.status.upper()
        return result
    result["shift_px"] = {"dx_px": measurement.dx, "dy_px": measurement.dy}
    if profile is None:
        return result
    maximum = payload.get("maximum_abs_correction_um")
    if maximum is not None:
        if isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or maximum <= 0:
            raise RGServiceError("INVALID_PAYLOAD", "maximum_abs_correction_um must be positive")
        control = control.model_copy(
            update={
                "maximum_single_correction_um": min(control.maximum_single_correction_um, maximum)
            }
        )
    decision = decide_correction(
        profile,
        measurement,
        control,
        iteration=iteration,
        total_correction_um=total,
    )
    result["correction_status"] = decision.status
    result["qc"]["decision"] = decision.model_dump(mode="json")
    if decision.correction_um is not None:
        result["proposed_z_correction_um"] = decision.correction_um
        requested = None if decision.inferred_z_error_um is None else -decision.inferred_z_error_um
        result["correction_was_clamped"] = requested != decision.correction_um
    return result
