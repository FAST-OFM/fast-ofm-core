"""Validated Fast OFM hardware limits and declarative calibration contracts."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

AxisName = Literal["x", "y", "z"]


class StrictModel(BaseModel):
    """Reject unknown fields, non-finite numbers and in-place model changes."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True, strict=True)


class AxisSettings(StrictModel):
    """Limits in millimetres relative to a fresh operator-confirmed reference."""

    enabled: bool = True
    units_per_mm: float = Field(default=1000, gt=0)
    direction_sign: Literal[-1, 1] = 1
    min_mm: float | None = -5
    max_mm: float | None = 5
    max_move_mm: float = Field(default=0.25, gt=0)
    ui_step_mm: float = Field(default=0.05, gt=0)
    speed_mm_s: float = Field(default=0.25, gt=0)
    accel_mm_s2: float = Field(default=2, gt=0)
    settle_ms: float = Field(default=750, ge=0)
    move_timeout_s: float = Field(default=15, gt=0)

    @model_validator(mode="after")
    def valid_range(self) -> Self:
        """Require a bounded zero-containing range for every enabled axis."""
        if self.enabled:
            if self.ui_step_mm > self.max_move_mm:
                raise ValueError("UI step exceeds the maximum single move")
            if self.min_mm is None or self.max_mm is None:
                raise ValueError("An enabled axis needs both bounds")
            if not self.min_mm <= 0 <= self.max_mm or self.min_mm == self.max_mm:
                raise ValueError("Axis range must contain zero and have positive width")
            if self.max_move_mm > self.max_mm - self.min_mm:
                raise ValueError("Maximum move exceeds the axis range")
        return self

    def require_subset_of(self, hardware: AxisSettings) -> None:
        """Reject an experiment profile that widens the hardware envelope."""
        if (
            self.units_per_mm != hardware.units_per_mm
            or self.direction_sign != hardware.direction_sign
        ):
            raise ValueError("An experiment cannot change units or axis direction")
        if not self.enabled:
            return
        if not hardware.enabled:
            raise ValueError("An experiment cannot enable a disabled hardware axis")
        if self.min_mm is None or self.max_mm is None:
            raise ValueError("Missing experiment bounds")
        if hardware.min_mm is None or hardware.max_mm is None:
            raise ValueError("Missing hardware bounds")
        if self.min_mm < hardware.min_mm or self.max_mm > hardware.max_mm:
            raise ValueError("Experiment range exceeds hardware range")
        for field in ("max_move_mm", "speed_mm_s", "accel_mm_s2"):
            if getattr(self, field) > getattr(hardware, field):
                raise ValueError(f"Experiment {field} exceeds hardware limit")
        if self.settle_ms < hardware.settle_ms:
            raise ValueError("Experiment cannot shorten hardware settling time")


def default_axes() -> dict[AxisName, AxisSettings]:
    """Return the initial XY envelope with Z explicitly disabled."""
    return {
        "x": AxisSettings(),
        "y": AxisSettings(),
        "z": AxisSettings(enabled=False, min_mm=None, max_mm=None),
    }


class MotionSettings(StrictModel):
    """Deployment hardware profile; motion is disabled until explicitly commissioned."""

    allow_motion: bool = False
    allow_operator_controls: bool = False
    ui_repeat_delay_ms: float = Field(default=100, ge=0, le=2000)
    manual_settle_ms: float = Field(default=0, ge=0)
    axes: dict[AxisName, AxisSettings] = Field(default_factory=default_axes)
    request_timeout_s: float = Field(default=3, gt=0)
    position_tolerance_mm: float = Field(default=0.005, gt=0)
    status_max_age_s: float = Field(default=1.0, gt=0, le=5)
    post_command_push_wait_s: float = Field(default=0.3, ge=0, le=1)
    status_reconnect_interval_s: float = Field(default=1.0, gt=0, le=30)

    @model_validator(mode="after")
    def all_axes_present(self) -> Self:
        """Require explicit settings for X, Y and Z."""
        if set(self.axes) != {"x", "y", "z"}:
            raise ValueError("Exactly X, Y and Z settings are required")
        return self


class MappingMotionChecks(StrictModel):
    """Checks made from the existing mapping frames, not a separate calibration."""

    unverified_step_mm: float = Field(default=0.05, gt=0)
    max_excursion_mm: float = Field(default=2.0, gt=0)
    min_shift_px: float = Field(default=3.0, gt=0)
    scale_tolerance: float = Field(default=0.35, gt=0, lt=1)
    max_step_frame_fraction: float = Field(default=0.25, gt=0, le=0.25)


class CalibrationStage(StrictModel):
    """One described stage; references point to implementation inputs and outputs."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    inputs: list[str]
    outputs: list[str]
    action: str = Field(min_length=1)
    success_criterion: str = Field(min_length=1)
    timeout_setting: str = Field(min_length=1)
    cancellation: str = Field(min_length=1)
    hardware_required: bool


class CalibrationManifest(StrictModel):
    """Metadata only, not a second workflow engine or duplicated parameter defaults."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    prerequisites: list[str]
    parameter_refs: list[str]
    stages: list[CalibrationStage] = Field(min_length=1)
    results: list[str]
    acceptance_criteria: list[str] = Field(min_length=1)
    invalidated_by: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_stages(self) -> Self:
        """Reject ambiguous duplicate stage identifiers."""
        ids = [stage.id for stage in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError("Calibration stage IDs must be unique")
        return self
