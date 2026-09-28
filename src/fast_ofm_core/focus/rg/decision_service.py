"""JSON protocol boundary for a bounded R/G correction decision."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import ValidationError

from .rg_focus_control import RGFocusControlSettings, decide_correction
from .rg_focus_estimator import RGFocusMeasurement
from .rg_focus_model import RGFocusModelProfile


class DecisionServiceError(ValueError):
    """A stable correction-decision payload failure."""


def decide_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Validate generic JSON values and return a non-hardware decision."""
    required = {"profile", "measurement", "control", "iteration", "total_correction_um"}
    if set(payload) != required:
        raise DecisionServiceError("R/G decision payload fields are invalid")
    iteration = payload["iteration"]
    total = payload["total_correction_um"]
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise DecisionServiceError("iteration must be a non-negative integer")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        raise DecisionServiceError("total_correction_um must be a number")
    try:
        profile = RGFocusModelProfile.model_validate(payload["profile"])
        measurement = RGFocusMeasurement.model_validate(payload["measurement"])
        control = RGFocusControlSettings.model_validate(payload["control"])
        decision = decide_correction(
            profile,
            measurement,
            control,
            iteration=iteration,
            total_correction_um=float(total),
        )
    except (ValidationError, ValueError) as error:
        raise DecisionServiceError(str(error)) from error
    return decision.model_dump(mode="json")
