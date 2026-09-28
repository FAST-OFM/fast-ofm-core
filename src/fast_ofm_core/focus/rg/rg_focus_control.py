"""Pure fail-closed correction decisions for bounded tissue-aware LED autofocus."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import Field

from fast_ofm_core.contracts import StrictModel

from .rg_focus_estimator import RGFocusMeasurement
from .rg_focus_model import RGFocusModelProfile, estimate_z_um


class RGFocusControlSettings(StrictModel):
    """Bound measurement count, elapsed time and every correction envelope."""

    maximum_iterations: int = Field(default=3, ge=1, le=5)
    timeout_s: float = Field(default=120.0, gt=0, le=300)
    minimum_capture_budget_s: float = Field(default=30.0, gt=0, le=60)
    focus_tolerance_um: float = Field(default=1.5, gt=0, le=4)
    cross_track_envelope_multiplier: float = Field(
        default=1.0,
        ge=1.0,
        le=10.0,
        description=(
            "Scale the calibrated cross-track envelope for tissue-dependent "
            "demo operation; axial range and correction limits remain independent."
        ),
    )
    cross_track_measurement_mad_multiplier: float = Field(default=1.0, ge=0.0, le=3.0)
    maximum_single_correction_um: float = Field(default=32.0, gt=0, le=32)
    maximum_total_correction_um: float = Field(default=32.0, gt=0, le=32)
    maximum_absolute_z_excursion_um: float = Field(
        default=42.0,
        gt=0,
        le=48,
        description=(
            "Full standalone autofocus Z path, including preload, relative to its "
            "starting position. Scan paths use their separate field envelope and budget."
        ),
    )


class RGFocusDecision(StrictModel):
    """One non-hardware decision made from a strict measurement and saved model."""

    status: Literal["focused", "move", "refused"]
    reason: str
    inferred_z_position_um: float | None = None
    inferred_z_error_um: float | None = None
    correction_um: float | None = None
    empirical_prediction_error_um: float | None = None
    effective_residual_tolerance_um: float | None = None
    cross_track_residual_px: float | None = None
    cross_track_empirical_limit_px: float | None = None
    cross_track_measurement_mad_px: float | None = None
    cross_track_limit_px: float | None = None
    cross_track_roundoff_px: float | None = None


def decide_correction(
    profile: RGFocusModelProfile,
    measurement: RGFocusMeasurement,
    settings: RGFocusControlSettings,
    *,
    iteration: int,
    total_correction_um: float,
) -> RGFocusDecision:
    """Return a bounded correction or refusal without fallback or hardware access."""
    if profile.status != "valid" or profile.focus_z_um is None:
        return RGFocusDecision(
            status="refused",
            reason=f"R/G focus model is not active: {profile.reason}",
        )
    try:
        effective_tolerance = profile.effective_focus_tolerance_um(settings.focus_tolerance_um)
    except ValueError as exc:
        return RGFocusDecision(
            status="refused",
            reason=str(exc),
            empirical_prediction_error_um=profile.empirical_prediction_error_um,
        )
    empirical_cross_track_limit = profile.empirical_cross_track_limit_px
    slope_x, slope_y = profile.slope_px_per_um
    slope_norm = math.hypot(slope_x, slope_y)
    cross_track_measurement_mad = (
        math.hypot(
            (-slope_y / slope_norm) * measurement.dx_mad,
            (slope_x / slope_norm) * measurement.dy_mad,
        )
        if slope_norm > 0
        else None
    )
    effective_cross_track_limit = (
        empirical_cross_track_limit * settings.cross_track_envelope_multiplier
        + cross_track_measurement_mad * settings.cross_track_measurement_mad_multiplier
        if empirical_cross_track_limit is not None
        and math.isfinite(empirical_cross_track_limit)
        and cross_track_measurement_mad is not None
        else None
    )
    diagnostics = {
        "empirical_prediction_error_um": profile.empirical_prediction_error_um,
        "effective_residual_tolerance_um": effective_tolerance,
        "cross_track_empirical_limit_px": empirical_cross_track_limit,
        "cross_track_measurement_mad_px": cross_track_measurement_mad,
        "cross_track_limit_px": effective_cross_track_limit,
    }
    if measurement.status != "ready" or measurement.dx is None or measurement.dy is None:
        return RGFocusDecision(
            status="refused",
            reason=f"R/G measurement rejected: {measurement.status}: {measurement.reason}",
            **diagnostics,
        )
    residual, roundoff = profile.cross_track_residual_px(measurement.dx, measurement.dy)
    diagnostics.update(cross_track_residual_px=residual, cross_track_roundoff_px=roundoff)
    if (
        effective_cross_track_limit is None
        or abs(residual) > effective_cross_track_limit + roundoff
    ):
        return RGFocusDecision(
            status="refused",
            reason="R/G shift is outside the observed cross-track calibration envelope",
            **diagnostics,
        )
    position_um = estimate_z_um(profile, measurement.dx, measurement.dy)
    error_um = position_um - profile.focus_z_um
    low, high = profile.applicable_z_range_um
    if not low <= position_um <= high:
        return RGFocusDecision(
            status="refused",
            reason="Inferred Z position is outside the independently validated model range",
            inferred_z_position_um=position_um,
            inferred_z_error_um=error_um,
            **diagnostics,
        )
    if abs(error_um) <= effective_tolerance:
        return RGFocusDecision(
            status="focused",
            reason="Inferred Z error is within the configured focus tolerance",
            inferred_z_position_um=position_um,
            inferred_z_error_um=error_um,
            correction_um=0,
            **diagnostics,
        )
    requested_correction_um = -error_um
    if iteration >= settings.maximum_iterations:
        return RGFocusDecision(
            status="refused",
            reason="Iteration budget leaves no independent post-move verification",
            inferred_z_position_um=position_um,
            inferred_z_error_um=error_um,
            **diagnostics,
        )
    if total_correction_um + abs(requested_correction_um) > settings.maximum_total_correction_um:
        return RGFocusDecision(
            status="refused",
            reason="Required Z correction exceeds the total correction envelope",
            inferred_z_position_um=position_um,
            inferred_z_error_um=error_um,
            **diagnostics,
        )
    correction_um = max(
        -settings.maximum_single_correction_um,
        min(settings.maximum_single_correction_um, requested_correction_um),
    )
    coarse = correction_um != requested_correction_um
    return RGFocusDecision(
        status="move",
        reason=(
            "Measurement passed QC and requests one bounded coarse correction"
            if coarse
            else "Measurement passed QC and requests one bounded signed correction"
        ),
        inferred_z_position_um=position_um,
        inferred_z_error_um=error_um,
        correction_um=correction_um,
        **diagnostics,
    )
