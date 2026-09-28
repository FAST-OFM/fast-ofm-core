"""Parameters and WHITE focus-curve checks for standalone Z calibration."""

from __future__ import annotations

import math
from typing import Literal, Self

import numpy as np
from pydantic import Field, field_validator, model_validator

from fast_ofm_core.contracts import StrictModel

from .z_backlash import ZBacklashEstimate, ZBacklashPolicy, ZFocusObservation


class FocusROI(StrictModel):
    """Operator-selected tissue rectangle, in fractions of the captured image."""

    x: float = Field(default=0.25, ge=0, lt=1)
    y: float = Field(default=0.25, ge=0, lt=1)
    width: float = Field(default=0.5, gt=0, le=1)
    height: float = Field(default=0.5, gt=0, le=1)

    @model_validator(mode="after")
    def inside_frame(self) -> Self:
        """Keep the complete rectangle inside the frame."""
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("Tissue ROI must be inside the frame")
        return self


class ZCalibrationSettings(StrictModel):
    """Explicit measurement parameters; motion dynamics come from the stage profile."""

    span_um: float = Field(default=20.0, gt=0)
    step_um: float = Field(default=2.0, gt=0)
    preload_um: float = Field(default=10.0, gt=0)
    cycles: int = Field(default=2, ge=2, le=10)
    preferred_final_approach_sign: Literal[-1, 1] = 1
    validation_half_range_um: float = Field(default=4.0, gt=0)
    maximum_uncertainty_um: float = Field(default=2.5, gt=0)
    preload_margin_um: float = Field(default=1.0, gt=0)
    maximum_residual_um: float = Field(default=2.5, gt=0)
    maximum_drift_um: float = Field(default=2.0, gt=0)
    minimum_prominence: float = Field(default=0.1, gt=0, lt=1)
    minimum_texture: float = Field(default=0.0001, gt=0)
    minimum_signal: float = Field(default=0.02, gt=0, lt=1)
    maximum_saturation_fraction: float = Field(default=0.02, ge=0, lt=1)
    frame_timeout_s: float = Field(default=5.0, gt=0, le=30)
    timeout_s: float = Field(default=900.0, gt=0, le=3600)
    maximum_frames: int = Field(default=1000, ge=10, le=5000)
    roi: FocusROI = Field(default_factory=FocusROI)
    mechanics_id: str = Field(default="current", min_length=1, max_length=100)
    optics_id: str = Field(default="current", min_length=1, max_length=100)

    @field_validator("preferred_final_approach_sign", mode="before")
    @classmethod
    def signed_integer(cls, value: object) -> object:
        """Require an actual signed integer, not True or a coerced float."""
        return ZBacklashEstimate.check_direction(value)

    @model_validator(mode="after")
    def validate_sampling(self) -> Self:
        """Reject ambiguous grids and excessive capture counts before movement."""
        intervals = self.span_um / self.step_um
        validation_steps = self.validation_half_range_um / self.step_um
        if not math.isfinite(intervals) or not math.isfinite(validation_steps):
            raise ValueError("Non-finite Z sampling grid")
        if intervals < 4 or not intervals.is_integer() or int(intervals) % 2:
            raise ValueError("Span must contain an even number of at least four steps")
        if validation_steps < 2 or not validation_steps.is_integer():
            raise ValueError("Validation half-range must contain at least two whole steps")
        if self.preload_um < self.step_um:
            raise ValueError("Preload must be at least one sampling step")
        if self.validation_half_range_um > self.span_um / 2:
            raise ValueError("Validation range exceeds the measured focus range")
        frames = 1 + self.cycles * 4 * (int(intervals) + 1)
        frames += 2 * (2 * int(validation_steps) + 1)
        if frames > self.maximum_frames:
            raise ValueError("Z calibration exceeds the capture budget")
        return self

    def policy(self) -> ZBacklashPolicy:
        """Use the same accepted parameters in the pure directional estimator."""
        return ZBacklashPolicy(
            minimum_cycles=self.cycles,
            minimum_quality_score=self.minimum_prominence,
            maximum_uncertainty_um=self.maximum_uncertainty_um,
            preload_margin_um=self.preload_margin_um,
        )


class FocusSample(StrictModel):
    """One stopped, fresh WHITE exposure and its ROI sharpness."""

    z_um: float
    time_s: float
    score: float = Field(ge=0)
    frame: str = Field(min_length=1)


class FocusPeak(StrictModel):
    """An interior peak with a conservative sampling-resolution bound."""

    z_um: float
    time_s: float
    uncertainty_um: float = Field(gt=0)
    quality_score: float = Field(gt=0, le=1)


def focus_peak(samples: list[FocusSample], settings: ZCalibrationSettings) -> FocusPeak:
    """Fit only an interior three-point maximum; reject flat or clipped focus curves."""
    ordered = sorted(samples, key=lambda sample: sample.z_um)
    if len(ordered) < 5:
        raise ValueError("Not enough points for a WHITE focus curve")
    z = np.array([sample.z_um for sample in ordered])
    values = np.array([sample.score for sample in ordered])
    steps = np.diff(z)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=0, atol=1e-7):
        raise ValueError("Focus curve must have distinct uniformly spaced Z positions")
    peak = int(np.argmax(values))
    if values[peak] < settings.minimum_texture:
        raise ValueError("Insufficient texture in WHITE focus curve")
    if peak in (0, len(ordered) - 1):
        raise ValueError("WHITE focus peak is outside the sampled range")
    quality = float((values[peak] - max(values[0], values[-1])) / values[peak])
    if quality < settings.minimum_prominence:
        raise ValueError("WHITE focus curve has insufficient prominence")
    competing = np.concatenate((values[: max(0, peak - 1)], values[peak + 2 :]))
    if (
        len(competing)
        and (values[peak] - max(competing)) / values[peak] < settings.minimum_prominence
    ):
        raise ValueError("WHITE focus curve has ambiguous competing peaks")
    left, centre, right = values[peak - 1 : peak + 2]
    curvature = left - 2 * centre + right
    if curvature >= 0:
        raise ValueError("WHITE focus peak is not resolved")
    fraction = float(0.5 * (left - right) / curvature)
    if not math.isfinite(fraction) or abs(fraction) > 0.5:
        raise ValueError("WHITE focus interpolation is inconsistent")
    return FocusPeak(
        z_um=float(z[peak] + fraction * steps[0]),
        time_s=ordered[peak].time_s,
        uncertainty_um=float(steps[0] / 2),
        quality_score=quality,
    )


def abba_observation(
    cycle_id: str, peaks: list[FocusPeak], settings: ZCalibrationSettings
) -> ZFocusObservation:
    """Compare +/−/−/+ curves with linear drift interpolation, not an R/G reference."""
    if len(peaks) != 4:
        raise ValueError("An ABBA cycle requires four focus curves")
    a0, b1, b2, a3 = peaks
    if not a0.time_s < b1.time_s < b2.time_s < a3.time_s:
        raise ValueError("Invalid ABBA exposure order")
    drift = a3.z_um - a0.z_um
    if abs(drift) > settings.maximum_drift_um:
        raise ValueError("WHITE focus drift exceeds the configured limit")
    expected_b_drift = drift * (b2.time_s - b1.time_s) / (a3.time_s - a0.time_s)
    if abs(b2.z_um - b1.z_um - expected_b_drift) > settings.maximum_drift_um:
        raise ValueError("WHITE repeated negative approach is inconsistent")
    b_time = (b1.time_s + b2.time_s) / 2
    positive = a0.z_um + drift * (b_time - a0.time_s) / (a3.time_s - a0.time_s)
    negative = (b1.z_um + b2.z_um) / 2
    return ZFocusObservation(
        cycle_id=cycle_id,
        positive_focus_um=positive,
        negative_focus_um=negative,
        focus_uncertainty_um=(
            max(a0.uncertainty_um, a3.uncertainty_um) + max(b1.uncertainty_um, b2.uncertainty_um)
        ),
        quality_score=min(peak.quality_score for peak in peaks),
    )
