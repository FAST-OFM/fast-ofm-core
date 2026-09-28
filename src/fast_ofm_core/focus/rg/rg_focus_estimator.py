"""Strict field-bound quality decision for tissue-aware R/G registration.

This OF-024 layer consumes corrected common R/G planes and a fresh OF-032
``TissueField``.  It owns operational rejection and robust aggregation only; it
does not capture frames, switch lights, move Z, fit a focus model, or fall back
to a whole-frame or sharpness-difference measurement.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from pydantic import Field, model_validator

from fast_ofm_core.contracts import CalibrationStage, StrictModel

from .rg_focus_core import (
    PatchEvaluation,
    PatchMeasurement,
    RGFocusCoreSettings,
    ShiftSummary,
    evaluate_patches,
    summarise_shift,
)
from .rg_focus_field import (
    FrameReference,
    TissueField,
    TissueFieldError,
    require_tissue_field_for_pair,
)

MeasurementStatus = Literal[
    "ready",
    "no_tissue",
    "insufficient_tissue",
    "low_signal",
    "saturated",
    "spectral_mismatch",
    "correlation_failed",
    "inconsistent_shifts",
]


class RGFocusEstimatorSettings(StrictModel):
    """Predeclared operational tolerances for robust shift aggregation."""

    inlier_aggregation: Literal["arithmetic_mean"] = "arithmetic_mean"
    minimum_inlier_fraction: float = Field(default=0.60, gt=0, le=1)
    outlier_mad_multiplier: float = Field(default=3.5, ge=1, le=10)
    minimum_outlier_threshold_px: float = Field(default=0.75, gt=0)
    maximum_dx_mad_px: float = Field(default=2.0, gt=0)
    maximum_dy_mad_px: float = Field(default=2.0, gt=0)
    maximum_radial_p90_px: float = Field(default=3.0, gt=0)
    minimum_confidence: float = Field(default=0.15, ge=0, le=1)

    @model_validator(mode="after")
    def consistent_spread_limits(self) -> Self:
        """Keep the joint residual limit at least as large as each MAD gate."""
        if self.maximum_radial_p90_px < max(self.maximum_dx_mad_px, self.maximum_dy_mad_px):
            raise ValueError("Radial residual limit is smaller than an axis MAD limit")
        return self


class RGFocusMeasurement(StrictModel):
    """Serializable measurement or refusal with complete fixed-window evidence."""

    status: MeasurementStatus
    reason: str
    dx: float | None = None
    dy: float | None = None
    confidence: float = Field(ge=0, le=1)
    median_response: float = Field(ge=0)
    median_spectral_correlation: float = Field(ge=-1, le=1)
    dx_mad: float = Field(ge=0)
    dy_mad: float = Field(ge=0)
    radial_p90: float = Field(ge=0)
    candidate_patch_count: int = Field(ge=0)
    accepted_patch_count: int = Field(ge=0)
    inlier_patch_count: int = Field(ge=0)
    inlier_fraction: float = Field(ge=0, le=1)
    support_fraction: float = Field(ge=0, le=1)
    rejection_counts: dict[str, int]
    windows: tuple[PatchEvaluation, ...] = Field(strict=False)


@dataclass(frozen=True)
class RGFocusInputs:
    """One already captured, corrected pair and its immutable field provenance."""

    red_plane: np.ndarray
    green_plane: np.ndarray
    red_source_jpeg8: np.ndarray
    green_source_jpeg8: np.ndarray
    red_valid: np.ndarray
    green_valid: np.ndarray
    field: TissueField | None
    red_reference: FrameReference
    green_reference: FrameReference
    geometry_value: Mapping[str, object]


def _empty_result(
    status: MeasurementStatus,
    reason: str,
    support_fraction: float,
    windows: tuple[PatchEvaluation, ...] = (),
) -> RGFocusMeasurement:
    """Build a finite fail-closed result for a pre-correlation refusal."""
    counts = Counter(row.status for row in windows if row.status != "accepted")
    return RGFocusMeasurement(
        status=status,
        reason=reason,
        confidence=0,
        median_response=0,
        median_spectral_correlation=0,
        dx_mad=0,
        dy_mad=0,
        radial_p90=0,
        candidate_patch_count=len(windows),
        accepted_patch_count=sum(row.status == "accepted" for row in windows),
        inlier_patch_count=0,
        inlier_fraction=0,
        support_fraction=support_fraction,
        rejection_counts=dict(sorted(counts.items())),
        windows=windows,
    )


def _measurement(row: PatchEvaluation) -> PatchMeasurement:
    """Convert one already accepted detailed row to robust-summary input."""

    def required(value: float | None, name: str) -> float:
        if value is None:
            raise RuntimeError(f"Accepted patch is missing {name}")
        return value

    return PatchMeasurement(
        patch_id=row.patch_id,
        x=row.x,
        y=row.y,
        w=row.w,
        h=row.h,
        dx=required(row.dx, "dx"),
        dy=required(row.dy, "dy"),
        response=required(row.response, "response"),
        peak_margin=required(row.peak_margin, "peak_margin"),
        coverage=row.coverage,
        median_r=required(row.median_r, "median_r"),
        median_g=required(row.median_g, "median_g"),
        std_r=required(row.std_r, "std_r"),
        std_g=required(row.std_g, "std_g"),
        saturation_r=required(row.saturation_r, "saturation_r"),
        saturation_g=required(row.saturation_g, "saturation_g"),
        spectral_correlation=required(row.spectral_correlation, "spectral_correlation"),
    )


def _summarise_inliers(inliers: list[PatchMeasurement]) -> ShiftSummary:
    """Average integer patch shifts after the robust rejection boundary."""
    summary = summarise_shift(inliers)
    return summary.model_copy(
        update={
            "dx": float(np.mean([row.dx for row in inliers])),
            "dy": float(np.mean([row.dy for row in inliers])),
        }
    )


def _dominant_failure(windows: tuple[PatchEvaluation, ...]) -> MeasurementStatus:
    """Choose a stable specific reason when too few windows reach aggregation."""
    counts = Counter(row.status for row in windows if row.status != "accepted")
    for status in ("saturated", "low_signal"):
        if counts[status] > 0 and counts[status] == max(counts.values(), default=0):
            return status  # type: ignore[return-value]
    if counts["spectral_mismatch"] > 0:
        return "spectral_mismatch"
    return "correlation_failed"


def _aggregate_result(
    status: MeasurementStatus, reason: str, values: dict[str, object]
) -> RGFocusMeasurement:
    """Validate a common successful-aggregation payload with its final decision."""
    return RGFocusMeasurement.model_validate({**values, "status": status, "reason": reason})


def estimate_rg_shift(
    inputs: RGFocusInputs,
    core_settings: RGFocusCoreSettings,
    estimator_settings: RGFocusEstimatorSettings,
) -> RGFocusMeasurement:
    """Return one strict robust R/G displacement or a non-motion refusal."""
    field = inputs.field
    if field is None:
        raise TissueFieldError("mask_invalid", "R/G measurement has no WHITE mask")
    if field.metrics.status != "ready":
        status: MeasurementStatus
        if field.metrics.status == "no_tissue":
            status = "no_tissue"
        elif field.metrics.status == "saturated":
            status = "saturated"
        elif field.metrics.status == "low_signal":
            status = "low_signal"
        else:
            status = "insufficient_tissue"
        return _empty_result(status, field.metrics.reason, field.metrics.candidate_coverage)
    require_tissue_field_for_pair(
        field,
        inputs.red_reference,
        inputs.green_reference,
        inputs.geometry_value,
    )

    red = np.asarray(inputs.red_plane)
    green = np.asarray(inputs.green_plane)
    source_r = np.asarray(inputs.red_source_jpeg8)
    source_g = np.asarray(inputs.green_source_jpeg8)
    valid_r = np.asarray(inputs.red_valid)
    valid_g = np.asarray(inputs.green_valid)
    expected_shape = (field.geometry.plane_size[1], field.geometry.plane_size[0])
    if (
        red.shape != expected_shape
        or green.shape != expected_shape
        or source_r.shape != expected_shape
        or source_g.shape != expected_shape
        or source_r.dtype != np.uint8
        or source_g.dtype != np.uint8
        or valid_r.shape != expected_shape
        or valid_g.shape != expected_shape
        or valid_r.dtype != np.bool_
        or valid_g.dtype != np.bool_
    ):
        raise TissueFieldError(
            "mask_invalid",
            "Corrected/source R/G planes or valid masks changed JPEG geometry",
        )
    support = field.mask & valid_r & valid_g
    support_fraction = float(np.mean(support))
    windows = tuple(
        evaluate_patches(
            red,
            green,
            support,
            core_settings,
            boxes=field.boxes,
            red_source_jpeg8=source_r,
            green_source_jpeg8=source_g,
            source_support=field.mask,
        )
    )
    accepted_rows = [row for row in windows if row.status == "accepted"]
    if len(accepted_rows) < core_settings.minimum_patch_count:
        status = _dominant_failure(windows)
        return _empty_result(
            status,
            f"Only {len(accepted_rows)} fixed windows passed per-window QC; "
            f"{core_settings.minimum_patch_count} required",
            support_fraction,
            windows,
        )

    accepted = [_measurement(row) for row in accepted_rows]
    preliminary = summarise_shift(accepted)
    dx_limit = max(
        estimator_settings.minimum_outlier_threshold_px,
        estimator_settings.outlier_mad_multiplier * 1.4826 * preliminary.dx_mad,
    )
    dy_limit = max(
        estimator_settings.minimum_outlier_threshold_px,
        estimator_settings.outlier_mad_multiplier * 1.4826 * preliminary.dy_mad,
    )
    inliers = [
        row
        for row in accepted
        if abs(row.dx - preliminary.dx) <= dx_limit and abs(row.dy - preliminary.dy) <= dy_limit
    ]
    inlier_fraction = len(inliers) / len(accepted)
    if (
        len(inliers) < core_settings.minimum_patch_count
        or inlier_fraction < estimator_settings.minimum_inlier_fraction
    ):
        return _empty_result(
            "inconsistent_shifts",
            "Too few mutually consistent R/G windows after robust outlier rejection",
            support_fraction,
            windows,
        ).model_copy(
            update={
                "candidate_patch_count": len(windows),
                "accepted_patch_count": len(accepted),
                "inlier_patch_count": len(inliers),
                "inlier_fraction": inlier_fraction,
            }
        )

    # Registration is intentionally integer-valued per patch.  Once the robust
    # MAD gate has removed spatial outliers, the arithmetic mean recovers
    # sub-pixel resolution from the independent windows.  A second median here
    # quantises the field result in 0.5/1 px jumps and is measurably too coarse
    # for the 2 um holdout contract at the calibrated ~0.4 px/um slope.
    summary = _summarise_inliers(inliers)
    residuals = np.hypot(
        np.asarray([row.dx - summary.dx for row in inliers]),
        np.asarray([row.dy - summary.dy for row in inliers]),
    )
    radial_p90 = float(np.percentile(residuals, 90))
    spectral = float(np.median([row.spectral_correlation for row in inliers]))
    confidence = float(np.clip(summary.response * inlier_fraction, 0, 1))
    counts = Counter(row.status for row in windows if row.status != "accepted")
    common = {
        "dx": summary.dx,
        "dy": summary.dy,
        "confidence": confidence,
        "median_response": summary.response,
        "median_spectral_correlation": spectral,
        "dx_mad": summary.dx_mad,
        "dy_mad": summary.dy_mad,
        "radial_p90": radial_p90,
        "candidate_patch_count": len(windows),
        "accepted_patch_count": len(accepted),
        "inlier_patch_count": len(inliers),
        "inlier_fraction": inlier_fraction,
        "support_fraction": support_fraction,
        "rejection_counts": dict(sorted(counts.items())),
        "windows": windows,
    }
    if (
        summary.dx_mad > estimator_settings.maximum_dx_mad_px
        or summary.dy_mad > estimator_settings.maximum_dy_mad_px
        or radial_p90 > estimator_settings.maximum_radial_p90_px
    ):
        return _aggregate_result(
            "inconsistent_shifts",
            "R/G window spread exceeds the predeclared consistency tolerance",
            common,
        )
    if confidence < estimator_settings.minimum_confidence:
        return _aggregate_result(
            "correlation_failed",
            "Aggregate R/G confidence is below the configured minimum",
            common,
        )
    return _aggregate_result(
        "ready",
        "Tissue-bound R/G mutual-information inlier-mean shift passed signal and consistency QC",
        common,
    )


def rg_shift_manifest_stage() -> CalibrationStage:
    """Describe the real OF-024 computation for the LED calibration manifest."""
    return CalibrationStage(
        id="rg_shift",
        name="Tissue-aware R/G shift",
        description="Measure mutual-information displacement only on fixed fresh-WHITE tissue windows.",
        inputs=[
            "corrected_red_plane",
            "corrected_green_plane",
            "valid_masks",
            "tissue_field",
            "frame_references",
            "parameters",
        ],
        outputs=["shift", "confidence", "spread", "window_diagnostics"],
        action="estimate_rg_shift",
        success_criterion="Ready shift with sufficient mutually consistent tissue windows",
        timeout_setting="measurement_timeout_s",
        cancellation="Discard the pair; never substitute whole-frame or sharpness-difference focus.",
        hardware_required=False,
    )
