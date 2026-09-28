"""Artifact-only protocol boundary for signed R/G calibration fitting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import ValidationError

from fast_ofm_core.artifacts import ArtifactError, load_json_artifact, resolve_artifact

from .rg_focus_model import (
    RGFocusCalibrationPoint,
    RGFocusMeasurementPolicy,
    RGFocusModelProfile,
    RGFocusModelSettings,
    RGFocusStationaryObservation,
    fit_rg_focus_model,
)

SERIES_MEDIA_TYPE = "application/vnd.fast-ofm.rg-calibration-series+json"
MAXIMUM_SERIES_BYTES = 8 * 1024 * 1024


class CalibrationServiceError(ValueError):
    """A stable calibration request or content failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def validate_profile_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Revalidate one persisted profile and its optional live focus budget."""
    if set(payload) - {"profile", "focus_tolerance_um"} or "profile" not in payload:
        raise CalibrationServiceError(
            "INVALID_PAYLOAD", "Profile validation requires profile and optional tolerance"
        )
    try:
        profile = RGFocusModelProfile.model_validate(payload["profile"])
        tolerance = payload.get("focus_tolerance_um")
        effective = None
        if tolerance is not None:
            if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
                raise ValueError("focus_tolerance_um must be numeric")
            effective = profile.effective_focus_tolerance_um(float(tolerance))
    except (TypeError, ValueError, ValidationError) as error:
        raise CalibrationServiceError("INVALID_PROFILE", str(error)) from error
    return {
        "profile": profile.model_dump(mode="json"),
        "profile_id": profile.id,
        "profile_status": profile.status,
        "effective_residual_tolerance_um": effective,
    }


def fit_payload(
    payload: Mapping[str, object], *, artifact_roots: Sequence[Path]
) -> dict[str, object]:
    """Fit and validate one immutable observation-series artifact."""
    if set(payload) != {"series"}:
        raise CalibrationServiceError("INVALID_PAYLOAD", "Calibration payload requires only series")
    try:
        path = resolve_artifact(
            payload["series"],
            roots=artifact_roots,
            media_type=SERIES_MEDIA_TYPE,
            maximum_size_bytes=MAXIMUM_SERIES_BYTES,
        )
        value = load_json_artifact(path)
    except ArtifactError as error:
        raise CalibrationServiceError(error.code, str(error)) from error

    required = {
        "schema_version",
        "profile_id",
        "source_series_id",
        "points",
        "settings",
        "reference_white_score",
        "stationary_observations",
        "compatibility",
        "measurement_policy",
    }
    allowed = required | {"source_evidence"}
    if set(value) - allowed or not required <= set(value) or value["schema_version"] != "1.0":
        raise CalibrationServiceError(
            "INVALID_CALIBRATION_SERIES", "Calibration series schema is unsupported"
        )
    points_value = value["points"]
    stationary_value = value["stationary_observations"]
    if not isinstance(points_value, list) or not isinstance(stationary_value, list):
        raise CalibrationServiceError(
            "INVALID_CALIBRATION_SERIES", "Calibration observations must be arrays"
        )
    try:
        points = [RGFocusCalibrationPoint.model_validate(item) for item in points_value]
        stationary = tuple(
            RGFocusStationaryObservation.model_validate(item) for item in stationary_value
        )
        if len(stationary) != 2:
            raise ValueError("Exactly two stationary observations are required")
        settings = RGFocusModelSettings.model_validate(value["settings"])
        policy = RGFocusMeasurementPolicy.model_validate(value["measurement_policy"])
        policy.require_jpeg()
        profile_id = value["profile_id"]
        source_series_id = value["source_series_id"]
        compatibility = value["compatibility"]
        source_evidence = value.get("source_evidence", {})
        if (
            not isinstance(profile_id, str)
            or not profile_id
            or not isinstance(source_series_id, str)
            or not source_series_id
            or not isinstance(compatibility, dict)
            or not isinstance(source_evidence, dict)
        ):
            raise ValueError("Calibration identity and provenance fields are invalid")
        profile = fit_rg_focus_model(
            points,
            settings,
            reference_white_score=float(value["reference_white_score"]),
            stationary_observations=(stationary[0], stationary[1]),
            profile_id=profile_id,
            source_series_id=source_series_id,
            compatibility=compatibility,
            source_evidence=source_evidence,
            measurement_policy=policy,
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise CalibrationServiceError("INVALID_CALIBRATION_SERIES", str(error)) from error
    return {
        "calibration_id": profile.id,
        "calibration_status": profile.status,
        "reason": profile.reason,
        "profile": profile.model_dump(mode="json"),
    }
