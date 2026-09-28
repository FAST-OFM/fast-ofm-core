"""Protocol-facing sparse focus-surface validation and prediction."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from fast_ofm_core.focus.focus_surface import (
    FocusObservation as ExactFocusObservation,
)
from fast_ofm_core.focus.focus_surface import (
    FocusPredictionRequest as ExactFocusPredictionRequest,
)
from fast_ofm_core.focus.focus_surface import (
    predict_focus_surface as predict_exact_focus_surface,
)
from fast_ofm_core.focus.sparse_focus import (
    FocusAnchor,
    SparseFocusSettings,
    estimate_focus_height,
)


class FocusSurfacePayloadError(ValueError):
    """A neutral focus-surface payload is malformed."""


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FocusSurfacePayloadError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise FocusSurfacePayloadError(f"{name} must be a finite number")
    return result


def _anchors(value: object) -> tuple[FocusAnchor, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FocusSurfacePayloadError("anchors must be an array")
    anchors: list[FocusAnchor] = []
    seen: set[str] = set()
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise FocusSurfacePayloadError(f"anchors[{index}] must be an object")
        anchor_id = row.get("anchor_id")
        if not isinstance(anchor_id, str) or not anchor_id or anchor_id in seen:
            raise FocusSurfacePayloadError("anchor_id values must be non-empty and unique")
        seen.add(anchor_id)
        position = row.get("position_um")
        if not isinstance(position, Mapping):
            raise FocusSurfacePayloadError(f"anchors[{index}].position_um must be an object")
        anchors.append(
            FocusAnchor(
                _number(position.get("x_um"), f"anchors[{index}].position_um.x_um"),
                _number(position.get("y_um"), f"anchors[{index}].position_um.y_um"),
                _number(position.get("z_um"), f"anchors[{index}].position_um.z_um"),
            )
        )
    return tuple(anchors)


def _settings(value: object) -> SparseFocusSettings:
    if not isinstance(value, Mapping):
        raise FocusSurfacePayloadError("limits must be an object")
    radius = _number(value.get("maximum_support_distance_um"), "maximum_support_distance_um")
    slope = _number(value.get("maximum_abs_slope"), "maximum_abs_slope")
    residual = _number(value.get("maximum_residual_um"), "maximum_residual_um")
    extrapolation = _number(
        value.get("maximum_extrapolation_um", radius / 2), "maximum_extrapolation_um"
    )
    delta = _number(value.get("maximum_prediction_delta_um", 40.0), "maximum_prediction_delta_um")
    minimum_points = value.get("minimum_plane_points", 4)
    maximum_neighbors = value.get("maximum_neighbors", 12)
    if isinstance(minimum_points, bool) or not isinstance(minimum_points, int):
        raise FocusSurfacePayloadError("minimum_plane_points must be an integer")
    if isinstance(maximum_neighbors, bool) or not isinstance(maximum_neighbors, int):
        raise FocusSurfacePayloadError("maximum_neighbors must be an integer")
    condition = _number(
        value.get("maximum_condition_number", 1000.0),
        "maximum_condition_number",
    )
    return SparseFocusSettings(
        enabled=True,
        minimum_plane_points=minimum_points,
        maximum_neighbors=maximum_neighbors,
        neighbor_radius_um=radius,
        maximum_extrapolation_um=extrapolation,
        maximum_prediction_delta_um=delta,
        maximum_fit_residual_um=residual,
        maximum_plane_slope_um_per_um=slope,
        maximum_condition_number=condition,
    )


def fit_surface_payload(payload: Mapping[str, object]) -> dict[str, Any]:
    """Validate and identify immutable measured support without fitting globally."""
    anchors = _anchors(payload.get("anchors"))
    settings = _settings(payload.get("limits"))
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "surface_id": f"surface-{hashlib.sha256(canonical).hexdigest()[:16]}",
        "anchor_count": len(anchors),
        "settings": asdict(settings),
        "method": "bounded_local_on_demand",
    }


def predict_surface_payload(payload: Mapping[str, object]) -> dict[str, Any]:
    """Predict one bounded target from measured anchors only."""
    if set(payload) == {"observations", "request"}:
        observations_value = payload["observations"]
        if not isinstance(observations_value, Sequence) or isinstance(
            observations_value, (str, bytes)
        ):
            raise FocusSurfacePayloadError("observations must be an array")
        try:
            observations = tuple(
                ExactFocusObservation.model_validate(value) for value in observations_value
            )
            request = ExactFocusPredictionRequest.model_validate(payload["request"])
            return predict_exact_focus_surface(observations, request).model_dump(mode="json")
        except ValueError as error:
            raise FocusSurfacePayloadError(str(error)) from error
    anchors = _anchors(payload.get("anchors"))
    settings = _settings(payload.get("limits"))
    target = payload.get("target_um")
    if not isinstance(target, Mapping):
        raise FocusSurfacePayloadError("target_um must be an object")
    x = _number(target.get("x_um"), "target_um.x_um")
    y = _number(target.get("y_um"), "target_um.y_um")
    current_z = _number(target.get("z_um"), "target_um.z_um")
    estimate = estimate_focus_height(
        anchors,
        target_x_um=x,
        target_y_um=y,
        current_z_um=current_z,
        settings=settings,
    )
    result: dict[str, Any] = {
        "position_um": {"x_um": x, "y_um": y},
        "status": "predicted" if estimate.usable else "refused",
        "support_count": estimate.support_count,
        "reason": estimate.reason,
    }
    for name in (
        "fit_residual_um",
        "slope_um_per_um",
        "extrapolation_um",
        "validation_error_um",
    ):
        value = getattr(estimate, name)
        if value is not None and math.isfinite(value):
            result[name] = value
    if estimate.source in {"nearest", "row", "plane"}:
        result["model"] = estimate.source
    if estimate.usable:
        result["predicted_z_um"] = estimate.target_z_um
        result["uncertainty_um"] = estimate.fit_residual_um or 0.0
        result["support_distance_um"] = estimate.extrapolation_um or 0.0
    else:
        result["refusal_code"] = "FOCUS_SURFACE_UNAVAILABLE"
    return result
