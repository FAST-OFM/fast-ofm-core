"""Artifact and JSON boundary for simultaneous R/G numerical operations."""

from __future__ import annotations

import math
import os
import tempfile
import time
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from fast_ofm_core.artifacts import ArtifactError, describe_artifact, resolve_artifact

from .rg_focus_core import PatchBox
from .rg_simultaneous import (
    SimultaneousCalibrationSettings,
    SimultaneousFocusCurve,
    SimultaneousFocusObservation,
    SimultaneousFocusSearchSettings,
    SimultaneousFocusSettings,
    SimultaneousShiftMeasurement,
    crop_spectral_maps,
    fit_simultaneous_focus_curve,
    fit_spectral_flat_field,
    infer_simultaneous_defocus,
    inspect_simultaneous_focus_field,
    measure_simultaneous_shift,
    select_simultaneous_focus_boxes,
    simultaneous_focus_capture_order,
    unmix_components,
    validate_peripheral_focus,
    validate_spectral_flat_field,
)

INPUT_MEDIA_TYPE = "application/vnd.fast-ofm.rg-simultaneous-input+npz"
RESULT_MEDIA_TYPE = "application/vnd.fast-ofm.rg-simultaneous-result+npz"
MAXIMUM_BUNDLE_BYTES = 2 * 1024 * 1024 * 1024
MAXIMUM_UNCOMPRESSED_BYTES = 4 * 1024 * 1024 * 1024


class SimultaneousServiceError(ValueError):
    """A stable simultaneous R/G request or artifact failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _arrays(path: Path, expected: set[str]) -> dict[str, np.ndarray]:
    """Load one bounded non-pickle NPZ with an exact member set."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if (
                any(member.flag_bits & 0x1 for member in members)
                or sum(member.file_size for member in members) > MAXIMUM_UNCOMPRESSED_BYTES
            ):
                raise SimultaneousServiceError(
                    "INVALID_SIMULTANEOUS_ARTIFACT", "Simultaneous archive is unsafe"
                )
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise SimultaneousServiceError(
                    "INVALID_SIMULTANEOUS_ARTIFACT",
                    "Simultaneous archive has missing or unknown arrays",
                )
            return {name: np.array(archive[name], copy=True) for name in expected}
    except SimultaneousServiceError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise SimultaneousServiceError(
            "INVALID_SIMULTANEOUS_ARTIFACT",
            "Simultaneous archive is not a valid NPZ",
        ) from error


def _input(
    payload: Mapping[str, object],
    *,
    artifact_roots: Sequence[Path],
    expected: set[str],
) -> tuple[Path, dict[str, np.ndarray]]:
    try:
        path = resolve_artifact(
            payload.get("input"),
            roots=artifact_roots,
            media_type=INPUT_MEDIA_TYPE,
            maximum_size_bytes=MAXIMUM_BUNDLE_BYTES,
        )
    except ArtifactError as error:
        raise SimultaneousServiceError(error.code, str(error)) from error
    return path, _arrays(path, expected)


def _write_result(path: Path, **arrays: np.ndarray) -> dict[str, object]:
    """Atomically publish core-produced arrays beside the verified request."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary, **{name: np.asarray(value) for name, value in arrays.items()}
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return describe_artifact(path, RESULT_MEDIA_TYPE, role="rg-simultaneous-result")


def _offsets(value: object) -> list[list[int]]:
    """Validate the exact four-channel Bayer offset matrix."""
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(
            not isinstance(row, list) or len(row) != 2 or any(type(item) is not int for item in row)
            for row in value
        )
    ):
        raise SimultaneousServiceError(
            "INVALID_PAYLOAD", "channel_offsets_xy must contain four integer pairs"
        )
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SimultaneousServiceError("INVALID_PAYLOAD", f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SimultaneousServiceError("INVALID_PAYLOAD", f"{name} must be finite")
    return result


def calibrate_payload(
    payload: Mapping[str, object], *, artifact_roots: Sequence[Path]
) -> dict[str, object]:
    """Fit and validate spectral maps from independent source holdouts."""
    if set(payload) != {"input", "settings", "channel_offsets_xy", "white_level"}:
        raise SimultaneousServiceError("INVALID_PAYLOAD", "Simultaneous calibration fields changed")
    path, arrays = _input(
        payload,
        artifact_roots=artifact_roots,
        expected={
            "dark",
            "red_fit",
            "green_fit",
            "red_holdout",
            "green_holdout",
            "mixed_holdout",
        },
    )
    try:
        settings = SimultaneousCalibrationSettings.model_validate(payload.get("settings"))
        offsets = _offsets(payload.get("channel_offsets_xy"))
        white_level = _number(payload.get("white_level"), "white_level")
        maps, fit_report = fit_spectral_flat_field(
            arrays["dark"],
            arrays["red_fit"],
            arrays["green_fit"],
            offsets_xy=offsets,
            white_level=white_level,
            settings=settings,
        )
        validation = validate_spectral_flat_field(
            arrays["red_holdout"],
            arrays["green_holdout"],
            arrays["mixed_holdout"],
            offsets_xy=offsets,
            white_level=white_level,
            maps=maps,
            settings=settings,
        )
        components, _residual, valid = unmix_components(arrays["mixed_holdout"], offsets, maps)
    except SimultaneousServiceError:
        raise
    except (ValidationError, ValueError) as error:
        raise SimultaneousServiceError("SIMULTANEOUS_REFUSED", str(error)) from error
    descriptor = _write_result(
        path.parent / "simultaneous-calibration-result.npz",
        dark=maps["dark"],
        red_response=maps["red_response"],
        green_response=maps["green_response"],
        valid=maps["valid"],
        components=components,
        component_valid=valid,
    )
    return {
        "arrays": descriptor,
        "fit_report": fit_report,
        "validation": validation,
    }


def _refused(reason: str, candidates: int) -> SimultaneousShiftMeasurement:
    return SimultaneousShiftMeasurement(
        status="refused",
        reason=reason,
        confidence=0,
        candidate_patch_count=candidates,
        accepted_patch_count=0,
        inlier_patch_count=0,
        tissue_coverage=0,
        dx_mad=0,
        dy_mad=0,
        median_response=0,
    )


def focus_sample_payload(  # noqa: C901, PLR0912, PLR0915
    payload: Mapping[str, object], *, artifact_roots: Sequence[Path]
) -> dict[str, object]:
    """Unmix and measure one central or calibrated-area mixed RAW exposure."""
    expected_fields = {
        "input",
        "settings",
        "channel_offsets_xy",
        "white_level",
        "boxes",
        "adaptive_window_selection",
        "inspect_periphery",
    }
    if set(payload) != expected_fields:
        raise SimultaneousServiceError(
            "INVALID_PAYLOAD", "Simultaneous focus-sample fields changed"
        )
    _path, arrays = _input(
        payload,
        artifact_roots=artifact_roots,
        expected={"planes", "dark", "red_response", "green_response", "valid"},
    )
    try:
        settings = SimultaneousFocusSettings.model_validate(payload.get("settings"))
        offsets = _offsets(payload.get("channel_offsets_xy"))
        white_level = _number(payload.get("white_level"), "white_level")
        adaptive = payload.get("adaptive_window_selection")
        inspect = payload.get("inspect_periphery")
        if type(adaptive) is not bool or type(inspect) is not bool:
            raise SimultaneousServiceError("INVALID_PAYLOAD", "Focus-sample flags must be booleans")
        boxes_value = payload.get("boxes")
        if boxes_value is not None and not isinstance(boxes_value, list):
            raise SimultaneousServiceError("INVALID_PAYLOAD", "boxes must be an array or null")
        boxes = (
            None
            if boxes_value is None
            else [PatchBox.model_validate(value) for value in boxes_value]
        )
        maps = {name: arrays[name] for name in ("dark", "red_response", "green_response", "valid")}
        full_shape = maps["valid"].shape
        settings.checked_focus_roi((full_shape[1], full_shape[0]))
        x, y, width, height = settings.focus_plane_roi
        central_maps = crop_spectral_maps(maps, settings.focus_plane_roi)
        planes = arrays["planes"]
        if inspect:
            if planes.shape[1:] != full_shape or boxes is not None or not adaptive:
                raise ValueError("Peripheral focus requires one full-area adaptive exposure")
            central_planes = planes[:, y : y + height, x : x + width]
        else:
            if planes.shape[1:] != (height, width):
                raise ValueError("Central focus RAW geometry changed")
            central_planes = planes
        components, residual, valid = unmix_components(central_planes, offsets, central_maps)
        residual_values = residual[valid & np.isfinite(residual)]
        residual_p95 = float(np.percentile(residual_values, 95)) if residual_values.size else None
        focus_valid = valid & np.isfinite(residual)
        focus_valid &= residual <= settings.maximum_frame_residual_fraction
        saturation = float(np.max(np.mean(central_planes >= white_level, axis=(1, 2))))
        if inspect and not np.any(focus_valid):
            selected: list[PatchBox] = []
        else:
            selected = (
                select_simultaneous_focus_boxes(components, focus_valid, settings)
                if boxes is None
                else list(boxes)
            )
        if (
            saturation > settings.maximum_frame_saturation_fraction
            or residual_p95 is None
            or residual_p95 > settings.maximum_frame_residual_fraction
        ):
            measurement = _refused(
                (
                    "Mixed RAW is saturated"
                    if saturation > settings.maximum_frame_saturation_fraction
                    else "Mixed RAW spectral reconstruction residual is too high"
                ),
                len(selected),
            )
        else:
            measurement = measure_simultaneous_shift(
                components,
                focus_valid,
                settings,
                selected if boxes is not None or not adaptive else None,
                adaptive_window_selection=adaptive,
            )
            if boxes is None and measurement.candidate_patch_count > len(selected):
                selected = select_simultaneous_focus_boxes(
                    components, focus_valid, settings, include_reserve=True
                )[: measurement.candidate_patch_count]
        peripheral = None
        if inspect and measurement.status != "ready":
            full_components, full_residual, full_valid = unmix_components(planes, offsets, maps)
            search = SimultaneousFocusSearchSettings()
            peripheral = inspect_simultaneous_focus_field(
                full_components,
                full_valid,
                full_residual,
                np.any(planes >= white_level, axis=0),
                settings,
                search,
                deadline=time.monotonic() + search.timeout_s,
            )
        elif inspect:
            peripheral = {
                "status": "not_needed",
                "attempted": False,
                "used": False,
                "evaluated_patch_count": 0,
                "peripheral_patch_count": 0,
                "central_anchor_count": 0,
                "elapsed_s": 0.0,
                "reason": "Central crop passed unchanged QC",
                "diagnostic_only": True,
                "autofocus_authorized": False,
            }
    except SimultaneousServiceError:
        raise
    except (KeyError, TypeError, ValidationError, ValueError) as error:
        raise SimultaneousServiceError("SIMULTANEOUS_REFUSED", str(error)) from error
    return {
        "measurement": measurement.model_dump(mode="json"),
        "boxes": [box.model_dump(mode="json") for box in selected],
        "residual_p95": residual_p95,
        "valid_fraction": float(np.mean(focus_valid)),
        "peripheral_search": peripheral,
    }


def focus_plan_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Return the deterministic signed calibration capture order."""
    if set(payload) != {"settings"}:
        raise SimultaneousServiceError("INVALID_PAYLOAD", "Focus-plan fields changed")
    try:
        settings = SimultaneousFocusSettings.model_validate(payload.get("settings"))
    except ValidationError as error:
        raise SimultaneousServiceError("INVALID_PAYLOAD", str(error)) from error
    return {"capture_order_um": list(simultaneous_focus_capture_order(settings))}


def focus_fit_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Fit one signed curve from the complete declared observation grid."""
    if set(payload) != {"settings", "observations"} or not isinstance(
        payload.get("observations"), list
    ):
        raise SimultaneousServiceError("INVALID_PAYLOAD", "Focus-fit fields changed")
    try:
        settings = SimultaneousFocusSettings.model_validate(payload.get("settings"))
        observations = [
            SimultaneousFocusObservation.model_validate(value) for value in payload["observations"]
        ]
        curve = fit_simultaneous_focus_curve(observations, settings)
    except (ValidationError, ValueError) as error:
        raise SimultaneousServiceError("SIMULTANEOUS_REFUSED", str(error)) from error
    return {"curve": curve.model_dump(mode="json")}


def focus_evaluate_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Optionally authorize peripheral evidence and project one ready shift."""
    if set(payload) != {"curve", "measurement", "settings", "peripheral_search"}:
        raise SimultaneousServiceError("INVALID_PAYLOAD", "Focus-evaluate fields changed")
    try:
        curve = SimultaneousFocusCurve.model_validate(payload.get("curve"))
        measurement = SimultaneousShiftMeasurement.model_validate(payload.get("measurement"))
        settings = SimultaneousFocusSettings.model_validate(payload.get("settings"))
        search = payload.get("peripheral_search")
        validation: dict[str, object] | None = None
        if measurement.status != "ready" and search is not None:
            if not isinstance(search, dict):
                raise ValueError("peripheral_search must be an object or null")
            recovered, validation = validate_peripheral_focus(curve, search, settings)
            if recovered is not None:
                measurement = recovered
        result: dict[str, object] = {
            "measurement": measurement.model_dump(mode="json"),
            "peripheral_validation": validation,
        }
        if measurement.status == "ready":
            defocus, cross_track = infer_simultaneous_defocus(curve, measurement)
            result.update(defocus_um=defocus, cross_track_px=cross_track)
        return result
    except (ValidationError, ValueError) as error:
        raise SimultaneousServiceError("SIMULTANEOUS_REFUSED", str(error)) from error
