"""Pure, fail-closed planning for mapped LED focus preparation.

The planner works in physical micrometres and emits logical stage units.  Axis
direction is intentionally carried only as frozen identity: the stage adapter
applies it once when logical units become controller coordinates.  Planning is
not evidence that a preload was executed; only the stage can issue a
``PreparedApproach`` after verified motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Literal, Self, cast

from pydantic import Field, field_validator, model_validator

from fast_ofm_core.contracts import AxisName, StrictModel

from .focus_surface import (
    FocusBinding,
    FocusPrediction,
    WhiteSearchSettings,
    check_focus_binding,
)

XYZUm = tuple[float, float, float]
XYZStageUnits = tuple[int, int, int]
PlanStatus = Literal["planned", "ordinary", "refused", "focused", "resolution_limited"]
PlanPurpose = Literal[
    "preparation",
    "white_peak_preparation",
    "initial_correction",
    "direct_correction",
    "reapproach_correction",
]
SegmentPhase = Literal[
    "target_transit",
    "white_precentre",
    "pre_measurement",
    "measurement",
    "direct_correction",
    "correction_preload",
    "correction_measurement",
]


class AxisEnvelope(StrictModel):
    """One ordered physical interval in the local reference frame."""

    minimum_um: float
    maximum_um: float

    @model_validator(mode="after")
    def ordered(self) -> Self:
        """Reject empty or inverted envelopes rather than clamping a target."""
        if self.minimum_um >= self.maximum_um:
            raise ValueError("Axis envelope must have positive width")
        return self

    def contains(self, value_um: float) -> bool:
        """Return whether one already-quantized coordinate is in the interval."""
        return self.minimum_um <= value_um <= self.maximum_um


class AxisMotionProfile(StrictModel):
    """Frozen conversion, envelopes, dynamics and one-command bound for an axis."""

    axis: AxisName
    units_per_mm: float = Field(gt=0)
    direction_sign: Literal[-1, 1]
    travel: AxisEnvelope
    controller: AxisEnvelope
    experiment: AxisEnvelope
    maximum_single_move_um: float = Field(gt=0)
    velocity_um_s: float = Field(gt=0)
    acceleration_um_s2: float = Field(gt=0)
    move_timeout_s: float = Field(gt=0)
    settle_s: float = Field(ge=0)

    @field_validator("direction_sign", mode="before")
    @classmethod
    def direction_is_integer_sign(cls, value: object) -> object:
        """Reject boolean, float and string aliases for an axis sign."""
        if type(value) is not int or value not in (-1, 1):
            raise ValueError("Axis direction must be integer -1 or +1")
        return value

    @model_validator(mode="after")
    def nested_envelopes(self) -> Self:
        """Keep an experiment from widening either physical envelope."""
        for value in (self.experiment.minimum_um, self.experiment.maximum_um):
            if not self.travel.contains(value) or not self.controller.contains(value):
                raise ValueError("Experiment envelope exceeds stage/controller limits")
        if self.maximum_single_move_um * self.units_per_mm / 1000 < 1:
            raise ValueError("Single-command limit is below one stage unit")
        return self


class StageMotionSnapshot(StrictModel):
    """Commanded readback and immutable stage-owned identity at one boundary."""

    reference_id: Annotated[str, Field(min_length=1)]
    motion_revision: int = Field(ge=0)
    reference_origin_commanded_mm: tuple[float, float, float]
    current_commanded_xyz_um: XYZUm
    axes: tuple[AxisMotionProfile, AxisMotionProfile, AxisMotionProfile]
    controller_max_velocity_um_s: float = Field(gt=0)
    controller_max_acceleration_um_s2: float = Field(gt=0)
    request_timeout_s: float = Field(gt=0)

    @model_validator(mode="after")
    def exactly_xyz(self) -> Self:
        """Require one canonical scale and envelope per logical axis."""
        if tuple(axis.axis for axis in self.axes) != ("x", "y", "z"):
            raise ValueError("Motion profiles must contain X, Y and Z in order")
        for coordinate, axis in zip(self.current_quantized_xyz_um, self.axes, strict=True):
            if not (
                axis.travel.contains(coordinate)
                and axis.controller.contains(coordinate)
                and axis.experiment.contains(coordinate)
            ):
                raise ValueError("Current readback is outside a frozen envelope")
        return self

    @property
    def current_stage_units(self) -> XYZStageUnits:
        """Quantize physical readback exactly once into logical stage units."""
        return tuple(
            round(value * axis.units_per_mm / 1000)
            for value, axis in zip(self.current_commanded_xyz_um, self.axes, strict=True)
        )  # type: ignore[return-value]

    @property
    def current_quantized_xyz_um(self) -> XYZUm:
        """Return the physical coordinates that the emitted integer units represent."""
        return _units_to_um(self.current_stage_units, self.axes)


class FocusFieldBudget(StrictModel):
    """One field-scoped origin and cumulative budget that retries cannot reset."""

    budget_id: Annotated[str, Field(min_length=1)]
    origin_xyz_um: XYZUm
    deadline_monotonic_s: float = Field(gt=0)
    maximum_elapsed_s: float = Field(gt=0)
    elapsed_s: float = Field(ge=0)
    maximum_z_travel_um: float = Field(gt=0)
    z_travel_um: float = Field(ge=0)
    post_move_verification_reserve_s: float = Field(gt=0)

    @model_validator(mode="after")
    def usage_within_limit(self) -> Self:
        """Reject a reset-looking or already-exhausted budget snapshot."""
        if self.elapsed_s > self.maximum_elapsed_s:
            raise ValueError("Elapsed field budget is already exceeded")
        if self.z_travel_um > self.maximum_z_travel_um:
            raise ValueError("Field Z-travel budget is already exceeded")
        return self

    def same_frozen_limit_as(self, previous: FocusFieldBudget) -> bool:
        """Compare immutable identity, origin and maxima while allowing usage to grow."""
        return (
            self.budget_id == previous.budget_id
            and self.origin_xyz_um == previous.origin_xyz_um
            and self.deadline_monotonic_s == previous.deadline_monotonic_s
            and self.maximum_elapsed_s == previous.maximum_elapsed_s
            and self.maximum_z_travel_um == previous.maximum_z_travel_um
            and self.post_move_verification_reserve_s == previous.post_move_verification_reserve_s
            and self.elapsed_s >= previous.elapsed_s
            and self.z_travel_um >= previous.z_travel_um
        )

    def elapsed_at(self, monotonic_s: float) -> float:
        """Include delay against the frozen field deadline without resetting usage."""
        field_started_s = self.deadline_monotonic_s - self.maximum_elapsed_s
        return max(self.elapsed_s, monotonic_s - field_started_s)


class FocusApproachPolicy(StrictModel):
    """Frozen prediction, approach, R/G range and correction policy."""

    approach_profile_id: Annotated[str, Field(min_length=1)]
    # Signed physical offset, shared with FocusSurfaceSettings.  Directional
    # compatibility is checked with the frozen approach sign by the planner.
    measurement_offset_um: float
    expected_prediction_error_range_um: tuple[float, float]
    rg_applicable_error_range_um: tuple[float, float]
    focus_tolerance_um: float = Field(gt=0)
    maximum_single_correction_um: float = Field(gt=0)
    maximum_total_correction_um: float = Field(gt=0)
    maximum_iterations: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered_ranges(self) -> Self:
        """Require explicit two-sided prediction and calibrated R/G ranges."""
        for name, limits in (
            ("prediction error", self.expected_prediction_error_range_um),
            ("R/G applicable", self.rg_applicable_error_range_um),
        ):
            low, high = limits
            if not low <= 0 <= high or low == high:
                raise ValueError(f"{name} range must contain zero with positive width")
        if self.focus_tolerance_um > self.maximum_single_correction_um:
            raise ValueError("Focus tolerance exceeds the single correction limit")
        if self.maximum_single_correction_um > self.maximum_total_correction_um:
            raise ValueError("Single correction limit exceeds total correction limit")
        return self


class CorrectionProgress(StrictModel):
    """Cumulative R/G iteration and correction use for the current field."""

    measurement_iteration: int = Field(ge=1)
    total_correction_um: float = Field(ge=0)


class MotionSegment(StrictModel):
    """One already-quantized command target within every single-command bound."""

    phase: SegmentPhase
    target_stage_units: XYZStageUnits
    target_xyz_um: XYZUm
    estimated_duration_s: float = Field(gt=0)


class PreparedApproach(StrictModel):
    """Stage-issued evidence of an executed and verified final approach.

    This is not accepted from a public ``prepared=True`` flag.  Any later stage
    motion changes ``motion_revision`` and invalidates the evidence, even if the
    stage returns to the same XYZ.
    """

    source_plan_id: Annotated[str, Field(min_length=1)]
    reference_id: Annotated[str, Field(min_length=1)]
    motion_revision: int = Field(ge=0)
    xyz_stage_units: XYZStageUnits
    binding: FocusBinding
    policy: FocusApproachPolicy
    budget_after: FocusFieldBudget
    progress: CorrectionProgress
    approach_sign: Literal[-1, 1]


class WhitePeakTransfer(StrictModel):
    """One ordered LED hand-off for an extreme possible native WHITE peak."""

    peak_z_stage_units: int
    peak_z_um: float
    approach_segments: tuple[MotionSegment, ...]
    correction_reserve_z_um: tuple[float, ...]
    worst_case_motion_time_s: float = Field(gt=0)
    worst_case_z_travel_um: float = Field(gt=0)


class WhiteSweepPlan(StrictModel):
    """Pure bounded native WHITE sweep and worst-case hand-off calculation."""

    plan_id: Annotated[str, Field(min_length=1)]
    status: Literal["planned", "ordinary", "refused"]
    reason: Annotated[str, Field(min_length=1)]
    reason_code: Annotated[str, Field(min_length=1)]
    target_transit_segments: tuple[MotionSegment, ...] = ()
    precentre_segments: tuple[MotionSegment, ...] = ()
    native_start: Literal["base"] = "base"
    native_total_dz_stage_units: int | None = Field(default=None, gt=0)
    native_base_z_stage_units: int | None = None
    native_sweep_high_z_stage_units: int | None = None
    native_return_base_z_stage_units: int | None = None
    possible_peak_z_stage_units: tuple[int, int] | None = None
    possible_peak_z_um: tuple[float, float] | None = None
    peak_transfers: tuple[WhitePeakTransfer, ...] = ()
    transfer_envelope_z_um: tuple[float, ...] = ()
    worst_case_motion_time_s: float = Field(default=0, ge=0)
    worst_case_z_travel_um: float = Field(default=0, ge=0)
    budget_after_precentre_and_search: FocusFieldBudget | None = None
    leaves_preparation_unproved: Literal[True] = True
    # The accepted actual-peak hand-off retains the complete original planning
    # context.  Refusals keep these empty because they can never authorize motion.
    stage_snapshot: StageMotionSnapshot | None = None
    binding: FocusBinding | None = None
    policy: FocusApproachPolicy | None = None
    budget_before: FocusFieldBudget | None = None
    settings: WhiteSearchSettings | None = None
    target_xy_um: tuple[float, float] | None = None
    centre_z_um: float | None = None
    total_focus_deadline_monotonic_s: float | None = None

    @model_validator(mode="after")
    def complete_planned_context(self) -> Self:
        """Require every frozen input needed by a planned sweep hand-off."""
        if self.status != "planned":
            return self
        required = (
            self.stage_snapshot,
            self.binding,
            self.policy,
            self.budget_before,
            self.settings,
            self.target_xy_um,
            self.centre_z_um,
            self.total_focus_deadline_monotonic_s,
            self.possible_peak_z_stage_units,
            self.budget_after_precentre_and_search,
        )
        if any(value is None for value in required):
            raise ValueError("Planned WHITE sweep omitted its frozen hand-off context")
        return self


class FocusMotionPlan(StrictModel):
    """Complete pure decision for preparation or one post-R/G correction."""

    plan_id: Annotated[str, Field(min_length=1)]
    status: PlanStatus
    purpose: PlanPurpose
    reason: Annotated[str, Field(min_length=1)]
    reason_code: Annotated[str, Field(min_length=1)]
    stage_snapshot: StageMotionSnapshot
    binding: FocusBinding
    policy: FocusApproachPolicy
    progress: CorrectionProgress
    budget_before: FocusFieldBudget
    budget_after: FocusFieldBudget
    segments: tuple[MotionSegment, ...] = ()
    target_xy_um: tuple[float, float] | None = None
    expected_focus_z_um: float | None = None
    measurement_z_um: float | None = None
    correction_error_um: float | None = None
    required_preparation: PreparedApproach | None = None
    white_sweep_plan: WhiteSweepPlan | None = None
    reserved_correction_endpoints_um: tuple[float, ...] = ()
    estimated_motion_time_s: float = Field(default=0, ge=0)
    reserved_motion_time_s: float = Field(default=0, ge=0)
    z_travel_um: float = Field(default=0, ge=0)
    reserved_z_travel_um: float = Field(default=0, ge=0)
    prepares_on_verified_execution: bool = False
    requires_rg_verification: bool = False

    @model_validator(mode="after")
    def status_payload(self) -> Self:
        """Only executable plans may contain commands or establish preparation."""
        if self.status == "planned":
            if not self.segments or not self.prepares_on_verified_execution:
                raise ValueError("Executable plan needs motion and final preparation")
        elif self.segments or self.prepares_on_verified_execution:
            raise ValueError("Non-executable decision cannot contain stage commands")
        has_white_context = self.white_sweep_plan is not None
        if has_white_context != (self.purpose == "white_peak_preparation"):
            raise ValueError("WHITE peak preparation requires its original sweep plan")
        return self

    @property
    def final_stage_units(self) -> XYZStageUnits:
        """Return the final commanded target, or the unchanged start."""
        if not self.segments:
            return self.stage_snapshot.current_stage_units
        return self.segments[-1].target_stage_units


class FocusApproachRequest(StrictModel):
    """All live and frozen inputs for one predicted-focus preparation."""

    plan_id: Annotated[str, Field(min_length=1)]
    prediction: FocusPrediction
    target_xy_um: tuple[float, float]
    stage_snapshot: StageMotionSnapshot
    frozen_binding: FocusBinding
    current_binding: FocusBinding
    policy: FocusApproachPolicy
    budget: FocusFieldBudget
    progress: CorrectionProgress


class FocusCorrectionRequest(StrictModel):
    """One measured R/G error plus live evidence required to correct it."""

    plan_id: Annotated[str, Field(min_length=1)]
    inferred_error_um: float
    stage_snapshot: StageMotionSnapshot
    current_binding: FocusBinding
    prepared: PreparedApproach
    budget: FocusFieldBudget
    progress: CorrectionProgress


class WhiteSweepRequest(StrictModel):
    """Frozen unavailable prediction and bounded WHITE-search inputs."""

    plan_id: Annotated[str, Field(min_length=1)]
    prediction: FocusPrediction
    target_xy_um: tuple[float, float]
    centre_z_um: float
    stage_snapshot: StageMotionSnapshot
    frozen_binding: FocusBinding
    current_binding: FocusBinding
    policy: FocusApproachPolicy
    budget: FocusFieldBudget
    settings: WhiteSearchSettings


class WhitePeakApproachRequest(StrictModel):
    """Actual native WHITE peak plus the complete plan that bounded its discovery."""

    plan_id: Annotated[str, Field(min_length=1)]
    sweep_plan: WhiteSweepPlan
    actual_peak_z_stage_units: int
    stage_snapshot: StageMotionSnapshot
    frozen_binding: FocusBinding
    current_binding: FocusBinding
    policy: FocusApproachPolicy
    budget: FocusFieldBudget
    progress: CorrectionProgress
    current_monotonic_s: float = Field(gt=0)


class FocusInitialCorrectionRequest(StrictModel):
    """First measured R/G correction when no stage preparation exists yet."""

    plan_id: Annotated[str, Field(min_length=1)]
    inferred_error_um: float
    stage_snapshot: StageMotionSnapshot
    frozen_binding: FocusBinding
    current_binding: FocusBinding
    policy: FocusApproachPolicy
    budget: FocusFieldBudget
    progress: CorrectionProgress


class _PlanRefusalError(ValueError):
    """Internal expected safety refusal with a stable narrow code."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(reason)
        self.code = code


@dataclass(frozen=True)
class _Path:
    """Quantized segments and conservative motion costs for one path."""

    segments: tuple[MotionSegment, ...]
    duration_s: float
    z_travel_um: float


@dataclass(frozen=True)
class _CanonicalFocusExecution:
    """Values independently recomputed from semantic plan inputs."""

    path: _Path
    target_xy_um: tuple[float, float]
    expected_focus_z_um: float | None
    measurement_z_um: float
    reserved_correction_endpoints_um: tuple[float, ...]
    reserved_motion_time_s: float
    reserved_z_travel_um: float
    budget_after: FocusFieldBudget


def _units_to_um(
    values: XYZStageUnits,
    axes: tuple[AxisMotionProfile, AxisMotionProfile, AxisMotionProfile],
) -> XYZUm:
    return tuple(value * 1000 / axis.units_per_mm for value, axis in zip(values, axes, strict=True))  # type: ignore[return-value]


def _um_to_units(
    values: XYZUm,
    axes: tuple[AxisMotionProfile, AxisMotionProfile, AxisMotionProfile],
) -> XYZStageUnits:
    return tuple(
        round(value * axis.units_per_mm / 1000) for value, axis in zip(values, axes, strict=True)
    )  # type: ignore[return-value]


def _validate_point(point: XYZUm, snapshot: StageMotionSnapshot) -> None:
    for value, axis in zip(point, snapshot.axes, strict=True):
        if not axis.travel.contains(value):
            raise _PlanRefusalError("travel_envelope", f"Axis {axis.axis} exceeds travel")
        if not axis.controller.contains(value):
            raise _PlanRefusalError(
                "controller_envelope", f"Axis {axis.axis} exceeds controller travel"
            )
        if not axis.experiment.contains(value):
            raise _PlanRefusalError(
                "experiment_envelope", f"Axis {axis.axis} exceeds experiment travel"
            )


def _segment_duration_s(start_um: XYZUm, target_um: XYZUm, snapshot: StageMotionSnapshot) -> float:
    components_mm = tuple(abs(b - a) / 1000 for a, b in zip(start_um, target_um, strict=True))
    active = tuple(index for index, value in enumerate(components_mm) if value > 0)
    if not active:
        return 0
    distance_mm = math.hypot(*(components_mm[index] for index in active))
    speed_mm_s = min(
        snapshot.controller_max_velocity_um_s / 1000,
        *(
            snapshot.axes[index].velocity_um_s / 1000 * distance_mm / components_mm[index]
            for index in active
        ),
    )
    acceleration_mm_s2 = min(
        snapshot.controller_max_acceleration_um_s2 / 1000,
        *(
            snapshot.axes[index].acceleration_um_s2 / 1000 * distance_mm / components_mm[index]
            for index in active
        ),
    )
    speed_mm_s = math.floor(speed_mm_s * 1_000_000) / 1_000_000
    acceleration_mm_s2 = math.floor(acceleration_mm_s2 * 1_000_000) / 1_000_000
    if speed_mm_s <= 0 or acceleration_mm_s2 <= 0:
        raise _PlanRefusalError("dynamics_resolution", "Dynamics are below G-code precision")
    required = (
        distance_mm / speed_mm_s + 2 * speed_mm_s / acceleration_mm_s2 + snapshot.request_timeout_s
    )
    timeout = min(snapshot.axes[index].move_timeout_s for index in active)
    if timeout <= required:
        raise _PlanRefusalError("move_timeout", "A segment exceeds its axis timeout")
    return required + max(snapshot.axes[index].settle_s for index in active)


def _build_path(
    snapshot: StageMotionSnapshot,
    start_units: XYZStageUnits,
    legs: tuple[tuple[SegmentPhase, XYZUm], ...],
) -> _Path:
    axes = snapshot.axes
    current_units = start_units
    current_um = _units_to_um(current_units, axes)
    _validate_point(current_um, snapshot)
    segments: list[MotionSegment] = []
    total_duration = 0.0
    z_travel = 0.0
    for phase, target_um in legs:
        final_units = _um_to_units(target_um, axes)
        maximum_segments = 1
        for index, axis in enumerate(axes):
            delta_units = final_units[index] - current_units[index]
            step_units = math.floor(axis.maximum_single_move_um * axis.units_per_mm / 1000)
            maximum_segments = max(maximum_segments, math.ceil(abs(delta_units) / step_units))
        leg_start = current_units
        for segment_index in range(1, maximum_segments + 1):
            next_units = cast(
                XYZStageUnits,
                tuple(
                    leg_start[index]
                    + round(
                        (final_units[index] - leg_start[index]) * segment_index / maximum_segments
                    )
                    for index in range(3)
                ),
            )
            if next_units == current_units:
                continue
            next_um = _units_to_um(next_units, axes)
            _validate_point(next_um, snapshot)
            for index, axis in enumerate(axes):
                if abs(next_um[index] - current_um[index]) > (axis.maximum_single_move_um + 1e-9):
                    raise _PlanRefusalError(
                        "single_command_limit",
                        f"Axis {axis.axis} exceeds its single-command limit",
                    )
            duration = _segment_duration_s(current_um, next_um, snapshot)
            segments.append(
                MotionSegment(
                    phase=phase,
                    target_stage_units=next_units,
                    target_xyz_um=next_um,
                    estimated_duration_s=duration,
                )
            )
            total_duration += duration
            z_travel += abs(next_um[2] - current_um[2])
            current_units = next_units
            current_um = next_um
    return _Path(tuple(segments), total_duration, z_travel)


def _validate_common(
    *,
    snapshot: StageMotionSnapshot,
    frozen_binding: FocusBinding,
    current_binding: FocusBinding,
    policy: FocusApproachPolicy,
) -> None:
    check = check_focus_binding(frozen_binding, current_binding)
    if not check.compatible or not check.usable:
        raise _PlanRefusalError("binding_changed", check.reason)
    if snapshot.reference_id != frozen_binding.reference_id:
        raise _PlanRefusalError("reference_changed", "Stage reference identity changed")
    if policy.approach_profile_id != frozen_binding.approach_profile_id:
        raise _PlanRefusalError("policy_changed", "Approach profile identity changed")
    approach_sign = frozen_binding.approach_parameters.approach_sign
    if policy.measurement_offset_um * approach_sign >= 0:
        raise _PlanRefusalError(
            "measurement_offset_direction",
            "Signed measurement offset is not before focus in the approach direction",
        )
    for scale, axis in zip(frozen_binding.axis_scales, snapshot.axes, strict=True):
        if (
            scale.axis != axis.axis
            or scale.units_per_mm != axis.units_per_mm
            or scale.direction_sign != axis.direction_sign
        ):
            raise _PlanRefusalError("axis_binding_changed", "Axis scale or sign changed")


def _validate_prediction_context(
    prediction: FocusPrediction,
    target_xy_um: tuple[float, float],
    frozen_binding: FocusBinding,
) -> None:
    """Keep a target or binding fault from becoming optical unavailable."""
    if prediction.target_x_um != target_xy_um[0] or (prediction.target_y_um != target_xy_um[1]):
        raise _PlanRefusalError("target_changed", "Prediction does not match actual XY")
    if (
        prediction.binding is None
        or not check_focus_binding(prediction.binding, frozen_binding).usable
    ):
        raise _PlanRefusalError("prediction_binding", "Prediction binding changed")


def _measurement_points(
    expected_focus_z_um: float,
    snapshot: StageMotionSnapshot,
    binding: FocusBinding,
    policy: FocusApproachPolicy,
) -> tuple[float, float]:
    """Quantize measurement and round preload away from it to preserve loading."""
    sign = binding.approach_parameters.approach_sign
    z_axis = snapshot.axes[2]
    measurement_units = round(
        (expected_focus_z_um + policy.measurement_offset_um) * z_axis.units_per_mm / 1000
    )
    measurement = measurement_units * 1000 / z_axis.units_per_mm
    if (measurement - expected_focus_z_um) * sign >= 0:
        raise _PlanRefusalError(
            "measurement_offset_resolution",
            "Measurement offset is not representable on the selected side",
        )
    preload_units = math.ceil(binding.approach_parameters.preload_um * z_axis.units_per_mm / 1000)
    pre_measurement = (measurement_units - sign * preload_units) * 1000 / z_axis.units_per_mm
    return pre_measurement, measurement


def _expected_measurement_error_range(
    policy: FocusApproachPolicy,
    measurement_defocus_um: float,
) -> tuple[float, float]:
    """Return the only first-pair errors authorised by the prediction contract."""
    prediction_low, prediction_high = policy.expected_prediction_error_range_um
    return (
        measurement_defocus_um - prediction_high,
        measurement_defocus_um - prediction_low,
    )


def _validate_rg_scope(policy: FocusApproachPolicy, expected_measurement_defocus_um: float) -> None:
    defocus_low, defocus_high = _expected_measurement_error_range(
        policy, expected_measurement_defocus_um
    )
    applicable_low, applicable_high = policy.rg_applicable_error_range_um
    if defocus_low < applicable_low or defocus_high > applicable_high:
        raise _PlanRefusalError(
            "rg_range",
            "Prediction error and measurement offset exceed the asymmetric R/G range",
        )


def _correction_legs(
    start_xyz_um: XYZUm,
    error_um: float,
    snapshot: StageMotionSnapshot,
    binding: FocusBinding,
) -> tuple[tuple[SegmentPhase, XYZUm], ...]:
    z_axis = snapshot.axes[2]
    corrected_units = round((start_xyz_um[2] - error_um) * z_axis.units_per_mm / 1000)
    corrected = (
        start_xyz_um[0],
        start_xyz_um[1],
        corrected_units * 1000 / z_axis.units_per_mm,
    )
    correction = -error_um
    sign = binding.approach_parameters.approach_sign
    if correction == 0 or math.copysign(1, correction) == sign:
        return (("direct_correction", corrected),)
    preload_units = math.ceil(binding.approach_parameters.preload_um * z_axis.units_per_mm / 1000)
    pre = (
        corrected[0],
        corrected[1],
        (corrected_units - sign * preload_units) * 1000 / z_axis.units_per_mm,
    )
    return (("correction_preload", pre), ("correction_measurement", corrected))


def _correction_reserve(
    snapshot: StageMotionSnapshot,
    start_units: XYZStageUnits,
    binding: FocusBinding,
    policy: FocusApproachPolicy,
    progress: CorrectionProgress,
    *,
    measurement_defocus_um: float,
) -> tuple[float, float, tuple[float, ...]]:
    worst_time = 0.0
    worst_z = 0.0
    endpoints: list[float] = []
    start_um = _units_to_um(start_units, snapshot.axes)
    for error in _expected_measurement_error_range(policy, measurement_defocus_um):
        if abs(error) <= policy.focus_tolerance_um:
            continue
        if progress.measurement_iteration >= policy.maximum_iterations:
            raise _PlanRefusalError(
                "iteration_reserve",
                "No iteration remains for reserved correction verification",
            )
        if abs(error) > policy.maximum_single_correction_um or (
            progress.total_correction_um + abs(error) > policy.maximum_total_correction_um
        ):
            raise _PlanRefusalError(
                "correction_reserve",
                "Asymmetric R/G range exceeds the remaining correction budget",
            )
        path = _build_path(
            snapshot,
            start_units,
            _correction_legs(start_um, error, snapshot, binding),
        )
        actual_correction = abs(path.segments[-1].target_xyz_um[2] - start_um[2])
        if actual_correction > policy.maximum_single_correction_um or (
            progress.total_correction_um + actual_correction > policy.maximum_total_correction_um
        ):
            raise _PlanRefusalError(
                "correction_quantization",
                "Quantized correction exceeds the remaining correction budget",
            )
        worst_time = max(worst_time, path.duration_s)
        worst_z = max(worst_z, path.z_travel_um)
        endpoints.extend(segment.target_xyz_um[2] for segment in path.segments)
    return worst_time, worst_z, tuple(endpoints)


def _apply_budget(
    budget: FocusFieldBudget,
    *,
    actual_time_s: float,
    actual_z_um: float,
    reserve_time_s: float,
    reserve_z_um: float,
) -> FocusFieldBudget:
    if budget.elapsed_s + actual_time_s + reserve_time_s > budget.maximum_elapsed_s:
        raise _PlanRefusalError("time_budget", "Field time budget cannot cover the full path")
    if budget.z_travel_um + actual_z_um + reserve_z_um > (budget.maximum_z_travel_um):
        raise _PlanRefusalError(
            "z_travel_budget", "Field Z-travel budget cannot cover the full path"
        )
    return budget.model_copy(
        update={
            "elapsed_s": budget.elapsed_s + actual_time_s,
            "z_travel_um": budget.z_travel_um + actual_z_um,
        }
    )


def _calculate_preparation(
    *,
    snapshot: StageMotionSnapshot,
    binding: FocusBinding,
    policy: FocusApproachPolicy,
    budget: FocusFieldBudget,
    progress: CorrectionProgress,
    target_xy_um: tuple[float, float],
    expected_focus_z_um: float,
) -> _CanonicalFocusExecution:
    """Build the one canonical preload/measurement path for a known focus Z."""
    pre_z, measurement_z = _measurement_points(expected_focus_z_um, snapshot, binding, policy)
    path = _build_path(
        snapshot,
        snapshot.current_stage_units,
        (
            ("pre_measurement", (target_xy_um[0], target_xy_um[1], pre_z)),
            ("measurement", (target_xy_um[0], target_xy_um[1], measurement_z)),
        ),
    )
    _validate_loaded_final_approach(
        path,
        snapshot=snapshot,
        start_units=snapshot.current_stage_units,
        binding=binding,
        pre_target_um=(target_xy_um[0], target_xy_um[1], pre_z),
        preload_phase="pre_measurement",
        final_phase="measurement",
    )
    _validate_rg_scope(policy, path.segments[-1].target_xyz_um[2] - expected_focus_z_um)
    reserve_time, reserve_z, endpoints = _correction_reserve(
        snapshot,
        path.segments[-1].target_stage_units,
        binding,
        policy,
        progress,
        measurement_defocus_um=measurement_z - expected_focus_z_um,
    )
    reserve_time += budget.post_move_verification_reserve_s
    budget_after = _apply_budget(
        budget,
        actual_time_s=path.duration_s,
        actual_z_um=path.z_travel_um,
        reserve_time_s=reserve_time,
        reserve_z_um=reserve_z,
    )
    return _CanonicalFocusExecution(
        path=path,
        target_xy_um=target_xy_um,
        expected_focus_z_um=expected_focus_z_um,
        measurement_z_um=path.segments[-1].target_xyz_um[2],
        reserved_correction_endpoints_um=endpoints,
        reserved_motion_time_s=reserve_time,
        reserved_z_travel_um=reserve_z,
        budget_after=budget_after,
    )


def _empty_motion_plan(
    request: FocusApproachRequest,
    status: PlanStatus,
    code: str,
    reason: str,
) -> FocusMotionPlan:
    return FocusMotionPlan(
        plan_id=request.plan_id,
        status=status,
        purpose="preparation",
        reason=reason,
        reason_code=code,
        stage_snapshot=request.stage_snapshot,
        binding=request.frozen_binding,
        policy=request.policy,
        progress=request.progress,
        budget_before=request.budget,
        budget_after=request.budget,
    )


def _validate_loaded_final_approach(
    path: _Path,
    *,
    snapshot: StageMotionSnapshot,
    start_units: XYZStageUnits,
    binding: FocusBinding,
    pre_target_um: XYZUm,
    preload_phase: SegmentPhase,
    final_phase: SegmentPhase,
) -> None:
    """Prove a complete, Z-only final approach after reaching its preload point."""
    final_indices = [
        index for index, segment in enumerate(path.segments) if segment.phase == final_phase
    ]
    if not final_indices:
        raise _PlanRefusalError("preload_resolution_limited", "Final preload is not representable")
    first_final = final_indices[0]
    if any(segment.phase != preload_phase for segment in path.segments[:first_final]) or any(
        segment.phase != final_phase for segment in path.segments[first_final:]
    ):
        raise _PlanRefusalError(
            "preload_sequence", "Final approach phases are missing or out of order"
        )
    pre_units = _um_to_units(pre_target_um, snapshot.axes)
    reached_pre_units = (
        start_units if first_final == 0 else path.segments[first_final - 1].target_stage_units
    )
    if reached_pre_units != pre_units:
        raise _PlanRefusalError(
            "preload_endpoint", "Final approach did not start at the preload point"
        )
    sign = binding.approach_parameters.approach_sign
    previous = pre_units
    for segment in path.segments[first_final:]:
        target = segment.target_stage_units
        if target[:2] != previous[:2] or (target[2] - previous[2]) * sign <= 0:
            raise _PlanRefusalError(
                "preload_direction",
                "Final approach must be Z-only in the selected direction",
            )
        previous = target
    preload_units = math.ceil(
        binding.approach_parameters.preload_um * snapshot.axes[2].units_per_mm / 1000
    )
    if (previous[2] - pre_units[2]) * sign < preload_units:
        raise _PlanRefusalError(
            "preload_distance", "Final approach is shorter than the frozen preload"
        )


def plan_focus_approach(request: FocusApproachRequest) -> FocusMotionPlan:
    """Plan joint XYZ preload and a complete final Z approach before R/G."""
    prediction = request.prediction
    if prediction.status == "disabled":
        return _empty_motion_plan(
            request,
            "ordinary",
            "prediction_disabled",
            "No usable mapped focus prediction; keep the ordinary selected path",
        )
    if prediction.status == "unavailable":
        try:
            _validate_common(
                snapshot=request.stage_snapshot,
                frozen_binding=request.frozen_binding,
                current_binding=request.current_binding,
                policy=request.policy,
            )
            _validate_prediction_context(prediction, request.target_xy_um, request.frozen_binding)
        except _PlanRefusalError as exc:
            return _empty_motion_plan(request, "refused", exc.code, str(exc))
        return _empty_motion_plan(
            request,
            "ordinary",
            "prediction_unavailable",
            "No usable mapped focus prediction; keep the ordinary selected path",
        )
    if prediction.status != "usable" or prediction.target_z_um is None:
        return _empty_motion_plan(
            request,
            "refused",
            "prediction_fault",
            "A prediction fault cannot authorise an optimized motion",
        )
    try:
        _validate_common(
            snapshot=request.stage_snapshot,
            frozen_binding=request.frozen_binding,
            current_binding=request.current_binding,
            policy=request.policy,
        )
        _validate_prediction_context(prediction, request.target_xy_um, request.frozen_binding)
        canonical = _calculate_preparation(
            snapshot=request.stage_snapshot,
            binding=request.frozen_binding,
            policy=request.policy,
            budget=request.budget,
            progress=request.progress,
            target_xy_um=request.target_xy_um,
            expected_focus_z_um=prediction.target_z_um,
        )
    except _PlanRefusalError as exc:
        return _empty_motion_plan(request, "refused", exc.code, str(exc))
    return FocusMotionPlan(
        plan_id=request.plan_id,
        status="planned",
        purpose="preparation",
        reason="Full XYZ preparation and R/G correction reserve are safe",
        reason_code="prepared_path",
        stage_snapshot=request.stage_snapshot,
        binding=request.frozen_binding,
        policy=request.policy,
        progress=request.progress,
        budget_before=request.budget,
        segments=canonical.path.segments,
        target_xy_um=request.target_xy_um,
        expected_focus_z_um=prediction.target_z_um,
        measurement_z_um=canonical.measurement_z_um,
        reserved_correction_endpoints_um=(canonical.reserved_correction_endpoints_um),
        estimated_motion_time_s=canonical.path.duration_s,
        reserved_motion_time_s=canonical.reserved_motion_time_s,
        z_travel_um=canonical.path.z_travel_um,
        reserved_z_travel_um=canonical.reserved_z_travel_um,
        budget_after=canonical.budget_after,
        prepares_on_verified_execution=True,
    )


def _initial_correction_refusal(
    request: FocusInitialCorrectionRequest, code: str, reason: str
) -> FocusMotionPlan:
    return FocusMotionPlan(
        plan_id=request.plan_id,
        status="refused",
        purpose="initial_correction",
        reason=reason,
        reason_code=code,
        stage_snapshot=request.stage_snapshot,
        binding=request.frozen_binding,
        policy=request.policy,
        progress=request.progress,
        budget_before=request.budget,
        budget_after=request.budget,
    )


def _calculate_initial_correction(
    request: FocusInitialCorrectionRequest,
) -> _CanonicalFocusExecution:
    """Build a complete preload to the first measured correction endpoint."""
    _validate_common(
        snapshot=request.stage_snapshot,
        frozen_binding=request.frozen_binding,
        current_binding=request.current_binding,
        policy=request.policy,
    )
    low, high = request.policy.rg_applicable_error_range_um
    error = request.inferred_error_um
    if not low <= error <= high:
        raise _PlanRefusalError("rg_range", "Measured error is outside R/G range")
    if abs(error) <= request.policy.focus_tolerance_um:
        raise _PlanRefusalError(
            "in_tolerance", "In-tolerance measurement needs no initial correction"
        )
    if request.progress.measurement_iteration >= request.policy.maximum_iterations:
        raise _PlanRefusalError(
            "iteration_budget", "No iteration remains for independent verification"
        )
    start_units = request.stage_snapshot.current_stage_units
    start_um = _units_to_um(start_units, request.stage_snapshot.axes)
    corrected_units = round(
        (start_um[2] - error) * request.stage_snapshot.axes[2].units_per_mm / 1000
    )
    if corrected_units == start_units[2]:
        raise _PlanRefusalError(
            "resolution_limited",
            "Out-of-tolerance correction rounds to zero stage units",
        )
    correction_um = abs(
        (corrected_units - start_units[2]) * 1000 / request.stage_snapshot.axes[2].units_per_mm
    )
    if correction_um > request.policy.maximum_single_correction_um or (
        request.progress.total_correction_um + correction_um
        > request.policy.maximum_total_correction_um
    ):
        raise _PlanRefusalError("correction_budget", "Correction exceeds frozen limits")
    sign = request.frozen_binding.approach_parameters.approach_sign
    preload_units = math.ceil(
        request.frozen_binding.approach_parameters.preload_um
        * request.stage_snapshot.axes[2].units_per_mm
        / 1000
    )
    target_z_um = corrected_units * 1000 / request.stage_snapshot.axes[2].units_per_mm
    preload_z_um = (
        (corrected_units - sign * preload_units)
        * 1000
        / request.stage_snapshot.axes[2].units_per_mm
    )
    target_xy = (start_um[0], start_um[1])
    path = _build_path(
        request.stage_snapshot,
        start_units,
        (
            ("correction_preload", (target_xy[0], target_xy[1], preload_z_um)),
            ("correction_measurement", (target_xy[0], target_xy[1], target_z_um)),
        ),
    )
    _validate_loaded_final_approach(
        path,
        snapshot=request.stage_snapshot,
        start_units=start_units,
        binding=request.frozen_binding,
        pre_target_um=(target_xy[0], target_xy[1], preload_z_um),
        preload_phase="correction_preload",
        final_phase="correction_measurement",
    )
    reserve_time = request.budget.post_move_verification_reserve_s
    budget_after = _apply_budget(
        request.budget,
        actual_time_s=path.duration_s,
        actual_z_um=path.z_travel_um,
        reserve_time_s=reserve_time,
        reserve_z_um=0,
    )
    return _CanonicalFocusExecution(
        path=path,
        target_xy_um=target_xy,
        expected_focus_z_um=None,
        measurement_z_um=target_z_um,
        reserved_correction_endpoints_um=(),
        reserved_motion_time_s=reserve_time,
        reserved_z_travel_um=0,
        budget_after=budget_after,
    )


def plan_initial_focus_correction(
    request: FocusInitialCorrectionRequest,
) -> FocusMotionPlan:
    """Plan the first enabled selected-method correction as one full approach."""
    try:
        canonical = _calculate_initial_correction(request)
        correction_um = abs(
            canonical.measurement_z_um - request.stage_snapshot.current_quantized_xyz_um[2]
        )
        progress = CorrectionProgress(
            measurement_iteration=request.progress.measurement_iteration + 1,
            total_correction_um=request.progress.total_correction_um + correction_um,
        )
    except _PlanRefusalError as exc:
        return _initial_correction_refusal(request, exc.code, str(exc))
    return FocusMotionPlan(
        plan_id=request.plan_id,
        status="planned",
        purpose="initial_correction",
        reason="First measured correction uses a complete verified final approach",
        reason_code="initial_correction",
        stage_snapshot=request.stage_snapshot,
        binding=request.frozen_binding,
        policy=request.policy,
        progress=progress,
        budget_before=request.budget,
        budget_after=canonical.budget_after,
        segments=canonical.path.segments,
        target_xy_um=canonical.target_xy_um,
        measurement_z_um=canonical.measurement_z_um,
        correction_error_um=request.inferred_error_um,
        estimated_motion_time_s=canonical.path.duration_s,
        reserved_motion_time_s=canonical.reserved_motion_time_s,
        z_travel_um=canonical.path.z_travel_um,
        prepares_on_verified_execution=True,
        requires_rg_verification=True,
    )


def _correction_refusal(
    request: FocusCorrectionRequest,
    status: PlanStatus,
    code: str,
    reason: str,
) -> FocusMotionPlan:
    prepared = request.prepared
    return FocusMotionPlan(
        plan_id=request.plan_id,
        status=status,
        purpose="direct_correction",
        reason=reason,
        reason_code=code,
        stage_snapshot=request.stage_snapshot,
        binding=prepared.binding,
        policy=prepared.policy,
        progress=request.progress,
        budget_before=request.budget,
        budget_after=request.budget,
    )


def _validate_prepared(request: FocusCorrectionRequest) -> None:
    prepared = request.prepared
    _validate_common(
        snapshot=request.stage_snapshot,
        frozen_binding=prepared.binding,
        current_binding=request.current_binding,
        policy=prepared.policy,
    )
    if request.stage_snapshot.reference_id != prepared.reference_id:
        raise _PlanRefusalError("prepared_reference", "Prepared reference is no longer live")
    if request.stage_snapshot.motion_revision != prepared.motion_revision:
        raise _PlanRefusalError("prepared_history", "Stage moved after preparation")
    if request.stage_snapshot.current_stage_units != prepared.xyz_stage_units:
        raise _PlanRefusalError("prepared_position", "Stage is not at prepared XYZ")
    if request.progress != prepared.progress:
        raise _PlanRefusalError("progress_changed", "Correction progress was reset or changed")
    if not request.budget.same_frozen_limit_as(prepared.budget_after):
        raise _PlanRefusalError("budget_changed", "Field budget was reset or changed")


def plan_focus_correction(request: FocusCorrectionRequest) -> FocusMotionPlan:
    """Plan a direct same-load correction or a complete reapproach and recheck."""
    try:
        _validate_prepared(request)
        policy = request.prepared.policy
        low, high = policy.rg_applicable_error_range_um
        error = request.inferred_error_um
        if not low <= error <= high:
            raise _PlanRefusalError("rg_range", "Measured error is outside R/G range")
        if abs(error) <= policy.focus_tolerance_um:
            return _correction_refusal(
                request,
                "focused",
                "in_tolerance",
                "Measured error is in tolerance; no movement is needed",
            )
        if request.progress.measurement_iteration >= policy.maximum_iterations:
            raise _PlanRefusalError(
                "iteration_budget", "No iteration remains for independent verification"
            )
        correction = -error
        if abs(correction) > policy.maximum_single_correction_um or (
            request.progress.total_correction_um + abs(correction)
            > policy.maximum_total_correction_um
        ):
            raise _PlanRefusalError("correction_budget", "Correction exceeds frozen limits")
        start_units = request.stage_snapshot.current_stage_units
        start_um = _units_to_um(start_units, request.stage_snapshot.axes)
        corrected_units = _um_to_units(
            (start_um[0], start_um[1], start_um[2] + correction),
            request.stage_snapshot.axes,
        )
        if corrected_units[2] == start_units[2]:
            return _correction_refusal(
                request,
                "resolution_limited",
                "resolution_limited",
                "Out-of-tolerance correction rounds to zero stage units",
            )
        quantized_correction_um = (
            abs(corrected_units[2] - start_units[2])
            * 1000
            / request.stage_snapshot.axes[2].units_per_mm
        )
        if quantized_correction_um > policy.maximum_single_correction_um or (
            request.progress.total_correction_um + quantized_correction_um
            > policy.maximum_total_correction_um
        ):
            raise _PlanRefusalError(
                "correction_quantization", "Quantized correction exceeds frozen limits"
            )
        direction = 1 if corrected_units[2] > start_units[2] else -1
        approach_sign = request.prepared.approach_sign
        purpose: PlanPurpose = (
            "direct_correction" if direction == approach_sign else "reapproach_correction"
        )
        correction_legs = _correction_legs(
            start_um, error, request.stage_snapshot, request.prepared.binding
        )
        path = _build_path(
            request.stage_snapshot,
            start_units,
            correction_legs,
        )
        if purpose == "reapproach_correction":
            _validate_loaded_final_approach(
                path,
                snapshot=request.stage_snapshot,
                start_units=start_units,
                binding=request.prepared.binding,
                pre_target_um=correction_legs[0][1],
                preload_phase="correction_preload",
                final_phase="correction_measurement",
            )
        budget_after = _apply_budget(
            request.budget,
            actual_time_s=path.duration_s,
            actual_z_um=path.z_travel_um,
            reserve_time_s=request.budget.post_move_verification_reserve_s,
            reserve_z_um=0,
        )
        next_progress = CorrectionProgress(
            measurement_iteration=request.progress.measurement_iteration + 1,
            total_correction_um=(request.progress.total_correction_um + quantized_correction_um),
        )
    except _PlanRefusalError as exc:
        return _correction_refusal(request, "refused", exc.code, str(exc))
    return FocusMotionPlan(
        plan_id=request.plan_id,
        status="planned",
        purpose=purpose,
        reason=(
            "Correction preserves the verified load direction"
            if purpose == "direct_correction"
            else "Reverse correction uses a complete preload and final approach"
        ),
        reason_code=purpose,
        stage_snapshot=request.stage_snapshot,
        binding=request.prepared.binding,
        policy=policy,
        progress=next_progress,
        budget_before=request.budget,
        segments=path.segments,
        target_xy_um=(start_um[0], start_um[1]),
        measurement_z_um=path.segments[-1].target_xyz_um[2],
        correction_error_um=error,
        required_preparation=request.prepared,
        estimated_motion_time_s=path.duration_s,
        reserved_motion_time_s=request.budget.post_move_verification_reserve_s,
        z_travel_um=path.z_travel_um,
        budget_after=budget_after,
        prepares_on_verified_execution=True,
        requires_rg_verification=True,
    )


def _require_plan_value(condition: bool, reason: str) -> None:
    if not condition:
        raise _PlanRefusalError("semantic_path", reason)


def _canonical_preparation(plan: FocusMotionPlan) -> _CanonicalFocusExecution:
    _require_plan_value(
        plan.target_xy_um is not None and plan.expected_focus_z_um is not None,
        "Preparation target or expected focus is missing",
    )
    _require_plan_value(
        plan.correction_error_um is None and plan.required_preparation is None,
        "Preparation contains correction-only evidence",
    )
    target_xy = cast(tuple[float, float], plan.target_xy_um)
    expected_focus = cast(float, plan.expected_focus_z_um)
    if plan.purpose == "white_peak_preparation":
        _validate_white_peak_plan_context(plan, expected_focus)
    canonical = _calculate_preparation(
        snapshot=plan.stage_snapshot,
        binding=plan.binding,
        policy=plan.policy,
        budget=plan.budget_before,
        progress=plan.progress,
        target_xy_um=target_xy,
        expected_focus_z_um=expected_focus,
    )
    _require_plan_value(not plan.requires_rg_verification, "Wrong preparation flag")
    return canonical


def _canonical_initial_correction(plan: FocusMotionPlan) -> _CanonicalFocusExecution:
    error = plan.correction_error_um
    _require_plan_value(error is not None, "Initial correction lacks measured error")
    error = cast(float, error)
    start_z = plan.stage_snapshot.current_quantized_xyz_um[2]
    z_scale = plan.stage_snapshot.axes[2].units_per_mm / 1000
    target_units = round((start_z - error) * z_scale)
    actual_correction = abs(
        target_units / z_scale - plan.stage_snapshot.current_stage_units[2] / z_scale
    )
    previous = CorrectionProgress(
        measurement_iteration=plan.progress.measurement_iteration - 1,
        total_correction_um=plan.progress.total_correction_um - actual_correction,
    )
    request = FocusInitialCorrectionRequest(
        plan_id=plan.plan_id,
        inferred_error_um=error,
        stage_snapshot=plan.stage_snapshot,
        frozen_binding=plan.binding,
        current_binding=plan.binding,
        policy=plan.policy,
        budget=plan.budget_before,
        progress=previous,
    )
    canonical = _calculate_initial_correction(request)
    expected_progress = CorrectionProgress(
        measurement_iteration=previous.measurement_iteration + 1,
        total_correction_um=previous.total_correction_um + actual_correction,
    )
    _require_plan_value(plan.progress == expected_progress, "Initial progress changed")
    _require_plan_value(
        plan.required_preparation is None,
        "Initial correction cannot claim earlier preparation",
    )
    _require_plan_value(plan.requires_rg_verification, "R/G recheck was disabled")
    return canonical


def _canonical_correction(plan: FocusMotionPlan) -> _CanonicalFocusExecution:
    prepared = plan.required_preparation
    error = plan.correction_error_um
    _require_plan_value(
        prepared is not None and error is not None,
        "Correction lacks the preparation or measured error it depends on",
    )
    prepared = cast(PreparedApproach, prepared)
    error = cast(float, error)
    _require_plan_value(
        prepared.binding == plan.binding
        and prepared.policy == plan.policy
        and prepared.reference_id == plan.stage_snapshot.reference_id
        and prepared.motion_revision == plan.stage_snapshot.motion_revision
        and prepared.xyz_stage_units == plan.stage_snapshot.current_stage_units
        and prepared.approach_sign == plan.binding.approach_parameters.approach_sign,
        "Correction preparation does not match its start boundary",
    )
    _require_plan_value(
        plan.budget_before.same_frozen_limit_as(prepared.budget_after),
        "Correction field budget was reset or replaced",
    )
    low, high = plan.policy.rg_applicable_error_range_um
    _require_plan_value(low <= error <= high, "Correction is outside R/G range")
    _require_plan_value(
        abs(error) > plan.policy.focus_tolerance_um,
        "In-tolerance result cannot contain movement",
    )
    _require_plan_value(
        prepared.progress.measurement_iteration < plan.policy.maximum_iterations,
        "No verification iteration remains",
    )
    _require_plan_value(
        abs(error) <= plan.policy.maximum_single_correction_um
        and prepared.progress.total_correction_um + abs(error)
        <= plan.policy.maximum_total_correction_um,
        "Correction exceeds its frozen limit",
    )
    start_units = plan.stage_snapshot.current_stage_units
    start_um = _units_to_um(start_units, plan.stage_snapshot.axes)
    legs = _correction_legs(start_um, error, plan.stage_snapshot, plan.binding)
    path = _build_path(plan.stage_snapshot, start_units, legs)
    _require_plan_value(bool(path.segments), "Correction rounds to no movement")
    quantized_correction = abs(path.segments[-1].target_xyz_um[2] - start_um[2])
    _require_plan_value(
        quantized_correction <= plan.policy.maximum_single_correction_um
        and prepared.progress.total_correction_um + quantized_correction
        <= plan.policy.maximum_total_correction_um,
        "Quantized correction exceeds its frozen limit",
    )
    direction = 1 if path.segments[-1].target_stage_units[2] > start_units[2] else -1
    purpose: PlanPurpose = (
        "direct_correction" if direction == prepared.approach_sign else "reapproach_correction"
    )
    _require_plan_value(plan.purpose == purpose, "Correction purpose was replaced")
    if purpose == "reapproach_correction":
        _validate_loaded_final_approach(
            path,
            snapshot=plan.stage_snapshot,
            start_units=start_units,
            binding=plan.binding,
            pre_target_um=legs[0][1],
            preload_phase="correction_preload",
            final_phase="correction_measurement",
        )
    next_progress = CorrectionProgress(
        measurement_iteration=prepared.progress.measurement_iteration + 1,
        total_correction_um=prepared.progress.total_correction_um + quantized_correction,
    )
    _require_plan_value(plan.progress == next_progress, "Correction progress changed")
    _require_plan_value(plan.requires_rg_verification, "R/G recheck was disabled")
    reserve_time = plan.budget_before.post_move_verification_reserve_s
    budget_after = _apply_budget(
        plan.budget_before,
        actual_time_s=path.duration_s,
        actual_z_um=path.z_travel_um,
        reserve_time_s=reserve_time,
        reserve_z_um=0,
    )
    return _CanonicalFocusExecution(
        path,
        (start_um[0], start_um[1]),
        None,
        path.segments[-1].target_xyz_um[2],
        (),
        reserve_time,
        0,
        budget_after,
    )


def validate_executable_focus_plan(plan: FocusMotionPlan) -> None:
    """Recompute an executable plan's path at the stage boundary.

    Schema validation deliberately is not execution evidence.  The stage calls
    this function before its first command so serialized phase, endpoint or cost
    substitutions cannot manufacture a prepared approach.
    """
    _require_plan_value(plan.status == "planned", "Plan is not executable")
    _validate_common(
        snapshot=plan.stage_snapshot,
        frozen_binding=plan.binding,
        current_binding=plan.binding,
        policy=plan.policy,
    )
    if plan.purpose in ("preparation", "white_peak_preparation"):
        canonical = _canonical_preparation(plan)
    elif plan.purpose == "initial_correction":
        canonical = _canonical_initial_correction(plan)
    else:
        canonical = _canonical_correction(plan)
    _require_plan_value(plan.segments == canonical.path.segments, "Motion segments were replaced")
    _require_plan_value(plan.target_xy_um == canonical.target_xy_um, "Target XY was replaced")
    _require_plan_value(
        plan.expected_focus_z_um == canonical.expected_focus_z_um,
        "Expected focus was replaced",
    )
    _require_plan_value(
        plan.measurement_z_um == canonical.measurement_z_um,
        "Measurement endpoint was replaced",
    )
    _require_plan_value(
        plan.reserved_correction_endpoints_um == canonical.reserved_correction_endpoints_um,
        "Correction reserve endpoints were replaced",
    )
    _require_plan_value(
        plan.estimated_motion_time_s == canonical.path.duration_s
        and plan.reserved_motion_time_s == canonical.reserved_motion_time_s
        and plan.z_travel_um == canonical.path.z_travel_um
        and plan.reserved_z_travel_um == canonical.reserved_z_travel_um,
        "Motion cost or reserve was replaced",
    )
    _require_plan_value(
        plan.budget_after == canonical.budget_after,
        "Estimated field budget was replaced",
    )


def _native_z_move_duration(
    snapshot: StageMotionSnapshot, start_units: int, target_units: int
) -> float:
    z_axis = snapshot.axes[2]
    delta_um = abs(target_units - start_units) * 1000 / z_axis.units_per_mm
    if delta_um > z_axis.maximum_single_move_um + 1e-9:
        raise _PlanRefusalError(
            "white_single_command",
            "Native WHITE total dz exceeds the Z single-command limit",
        )
    start = list(snapshot.current_quantized_xyz_um)
    target = list(start)
    start[2] = start_units * 1000 / z_axis.units_per_mm
    target[2] = target_units * 1000 / z_axis.units_per_mm
    _validate_point(tuple(start), snapshot)  # type: ignore[arg-type]
    _validate_point(tuple(target), snapshot)  # type: ignore[arg-type]
    return _segment_duration_s(cast(XYZUm, tuple(start)), cast(XYZUm, tuple(target)), snapshot)


def _white_refusal(
    request: WhiteSweepRequest, status: Literal["ordinary", "refused"], code: str
) -> WhiteSweepPlan:
    return WhiteSweepPlan(
        plan_id=request.plan_id,
        status=status,
        reason=(
            "WHITE search is not applicable to this prediction"
            if status == "ordinary"
            else code.replace("_", " ")
        ),
        reason_code=code,
    )


def _white_transfer_reserve(
    request: WhiteSweepRequest,
    target_units: XYZStageUnits,
    base_units: int,
    high_units: int,
    white_policy: FocusApproachPolicy,
) -> tuple[WhitePeakTransfer, WhitePeakTransfer]:
    """Check transfer/preload/correction for both extrema of an unknown peak."""
    axes = request.stage_snapshot.axes
    transfers: list[WhitePeakTransfer] = []
    for peak_units in (base_units, high_units):
        peak_um = peak_units * 1000 / axes[2].units_per_mm
        pre_z, measurement_z = _measurement_points(
            peak_um,
            request.stage_snapshot,
            request.frozen_binding,
            white_policy,
        )
        peak_xyz = (request.target_xy_um[0], request.target_xy_um[1], peak_um)
        transfer = _build_path(
            request.stage_snapshot,
            (target_units[0], target_units[1], peak_units),
            (
                ("pre_measurement", (peak_xyz[0], peak_xyz[1], pre_z)),
                ("measurement", (peak_xyz[0], peak_xyz[1], measurement_z)),
            ),
        )
        _validate_rg_scope(white_policy, transfer.segments[-1].target_xyz_um[2] - peak_um)
        correction_time, correction_z, correction_points = _correction_reserve(
            request.stage_snapshot,
            transfer.segments[-1].target_stage_units,
            request.frozen_binding,
            white_policy,
            CorrectionProgress(measurement_iteration=1, total_correction_um=0.0),
            measurement_defocus_um=measurement_z - peak_um,
        )
        transfers.append(
            WhitePeakTransfer(
                peak_z_stage_units=peak_units,
                peak_z_um=peak_um,
                approach_segments=transfer.segments,
                correction_reserve_z_um=correction_points,
                worst_case_motion_time_s=transfer.duration_s + correction_time,
                worst_case_z_travel_um=transfer.z_travel_um + correction_z,
            )
        )
    return transfers[0], transfers[1]


def _white_sweep_bounds(
    snapshot: StageMotionSnapshot, settings: WhiteSearchSettings, centre_z_um: float
) -> tuple[int, int]:
    """Quantize the declared WHITE interval inward using its frozen axis scale."""
    z_scale = snapshot.axes[2].units_per_mm / 1000
    centre_units = round(centre_z_um * z_scale)
    low_offset, high_offset = settings.search_z_range_um
    base_units = math.ceil((centre_z_um + low_offset) * z_scale)
    high_units = math.floor((centre_z_um + high_offset) * z_scale)
    if not base_units < centre_units < high_units:
        raise _PlanRefusalError("white_range_resolution", "Quantized WHITE range is not two-sided")
    return base_units, high_units


def plan_white_sweep(request: WhiteSweepRequest) -> WhiteSweepPlan:
    """Calculate a full asymmetric native fast-autofocus and LED hand-off envelope."""
    if request.prediction.status != "unavailable":
        status: Literal["ordinary", "refused"] = (
            "refused" if request.prediction.status == "fault" else "ordinary"
        )
        return _white_refusal(request, status, "prediction_not_unavailable")
    try:
        _validate_common(
            snapshot=request.stage_snapshot,
            frozen_binding=request.frozen_binding,
            current_binding=request.current_binding,
            policy=request.policy,
        )
        _validate_prediction_context(
            request.prediction, request.target_xy_um, request.frozen_binding
        )
        axes = request.stage_snapshot.axes
        # Quantize inward: a bounded range may narrow by one stage quantum,
        # but it must never expand outside either frozen WHITE endpoint.
        base_units, high_units = _white_sweep_bounds(
            request.stage_snapshot, request.settings, request.centre_z_um
        )
        total_dz = high_units - base_units
        current = request.stage_snapshot.current_quantized_xyz_um
        target_path = _build_path(
            request.stage_snapshot,
            request.stage_snapshot.current_stage_units,
            (
                (
                    "target_transit",
                    (request.target_xy_um[0], request.target_xy_um[1], current[2]),
                ),
            ),
        )
        target_units = (
            target_path.segments[-1].target_stage_units
            if target_path.segments
            else request.stage_snapshot.current_stage_units
        )
        precentre = _build_path(
            request.stage_snapshot,
            target_units,
            (
                (
                    "white_precentre",
                    (
                        request.target_xy_um[0],
                        request.target_xy_um[1],
                        base_units * 1000 / axes[2].units_per_mm,
                    ),
                ),
            ),
        )
        native_up = _native_z_move_duration(request.stage_snapshot, base_units, high_units)
        native_down = _native_z_move_duration(request.stage_snapshot, high_units, base_units)
        native_peak = _native_z_move_duration(request.stage_snapshot, base_units, high_units)
        native_time = precentre.duration_s + native_up + native_down + native_peak
        if native_time > request.settings.white_search_timeout_s:
            raise _PlanRefusalError("white_timeout", "Full native WHITE path exceeds timeout")
        white_policy = request.policy.model_copy(
            update={
                "expected_prediction_error_range_um": (
                    -request.settings.maximum_white_led_disagreement_um,
                    request.settings.maximum_white_led_disagreement_um,
                )
            }
        )
        peak_transfers = _white_transfer_reserve(
            request, target_units, base_units, high_units, white_policy
        )
        worst_transfer_time = max(transfer.worst_case_motion_time_s for transfer in peak_transfers)
        worst_transfer_z = max(transfer.worst_case_z_travel_um for transfer in peak_transfers)
        transfer_endpoints = tuple(
            z
            for transfer in peak_transfers
            for z in (
                *(segment.target_xyz_um[2] for segment in transfer.approach_segments),
                *transfer.correction_reserve_z_um,
            )
        )
        transfer_reserve = worst_transfer_time + request.budget.post_move_verification_reserve_s
        if transfer_reserve > request.settings.approach_and_rg_reserve_s:
            raise _PlanRefusalError(
                "white_transfer_reserve", "WHITE hand-off exceeds its frozen reserve"
            )
        total_time = target_path.duration_s + native_time + transfer_reserve
        if request.budget.elapsed_s + total_time > request.settings.total_focus_budget_s:
            raise _PlanRefusalError("white_total_budget", "Full WHITE-to-R/G path exceeds budget")
        native_z_travel = 3 * (high_units - base_units) * 1000 / axes[2].units_per_mm
        total_z = target_path.z_travel_um + precentre.z_travel_um + native_z_travel
        budget_after = _apply_budget(
            request.budget,
            actual_time_s=target_path.duration_s + native_time,
            actual_z_um=total_z,
            reserve_time_s=transfer_reserve,
            reserve_z_um=worst_transfer_z,
        )
    except _PlanRefusalError as exc:
        return WhiteSweepPlan(
            plan_id=request.plan_id,
            status="refused",
            reason=str(exc),
            reason_code=exc.code,
        )
    return WhiteSweepPlan(
        plan_id=request.plan_id,
        status="planned",
        reason="Full native WHITE range and every bounded LED hand-off are safe",
        reason_code="white_path",
        target_transit_segments=target_path.segments,
        precentre_segments=precentre.segments,
        native_total_dz_stage_units=total_dz,
        native_base_z_stage_units=base_units,
        native_sweep_high_z_stage_units=high_units,
        native_return_base_z_stage_units=base_units,
        possible_peak_z_stage_units=(base_units, high_units),
        possible_peak_z_um=(
            base_units * 1000 / axes[2].units_per_mm,
            high_units * 1000 / axes[2].units_per_mm,
        ),
        peak_transfers=peak_transfers,
        transfer_envelope_z_um=tuple(sorted(set(transfer_endpoints))),
        worst_case_motion_time_s=total_time,
        worst_case_z_travel_um=total_z + worst_transfer_z,
        budget_after_precentre_and_search=budget_after,
        stage_snapshot=request.stage_snapshot,
        binding=request.frozen_binding,
        policy=request.policy,
        budget_before=request.budget,
        settings=request.settings,
        target_xy_um=request.target_xy_um,
        centre_z_um=request.centre_z_um,
        total_focus_deadline_monotonic_s=(
            request.budget.deadline_monotonic_s
            - request.budget.maximum_elapsed_s
            + request.settings.total_focus_budget_s
        ),
    )


def _validate_white_sweep_context(sweep: WhiteSweepPlan) -> None:
    """Keep serialized native bounds and deadline tied to their frozen inputs."""
    original = cast(StageMotionSnapshot, sweep.stage_snapshot)
    settings = cast(WhiteSearchSettings, sweep.settings)
    before = cast(FocusFieldBudget, sweep.budget_before)
    base, high = _white_sweep_bounds(original, settings, cast(float, sweep.centre_z_um))
    scale = original.axes[2].units_per_mm
    _require_plan_value(
        sweep.possible_peak_z_stage_units == (base, high)
        and sweep.possible_peak_z_um == (base * 1000 / scale, high * 1000 / scale)
        and sweep.native_base_z_stage_units == base
        and sweep.native_sweep_high_z_stage_units == high
        and sweep.native_return_base_z_stage_units == base
        and sweep.native_total_dz_stage_units == high - base,
        "WHITE native range differs from its frozen search interval",
    )
    _require_plan_value(
        sweep.total_focus_deadline_monotonic_s
        == before.deadline_monotonic_s - before.maximum_elapsed_s + settings.total_focus_budget_s,
        "WHITE total deadline differs from its frozen field budget",
    )


def _validate_white_peak_request(request: WhitePeakApproachRequest) -> None:
    """Bind one actual peak to the immutable sweep that was executed to find it."""
    sweep = request.sweep_plan
    if sweep.status != "planned":
        raise _PlanRefusalError(
            "white_plan_status", "Only a planned WHITE sweep can hand off a peak"
        )
    if (
        sweep.stage_snapshot is None
        or sweep.binding is None
        or sweep.policy is None
        or sweep.budget_before is None
        or sweep.settings is None
        or sweep.target_xy_um is None
        or sweep.possible_peak_z_stage_units is None
        or sweep.total_focus_deadline_monotonic_s is None
    ):
        raise _PlanRefusalError(
            "white_plan_context", "WHITE sweep omitted its frozen planning context"
        )
    _validate_white_sweep_context(sweep)
    _validate_common(
        snapshot=request.stage_snapshot,
        frozen_binding=request.frozen_binding,
        current_binding=request.current_binding,
        policy=request.policy,
    )
    if (
        sweep.binding != request.frozen_binding
        or sweep.policy != request.policy
        or _um_to_units(
            (sweep.target_xy_um[0], sweep.target_xy_um[1], 0),
            request.stage_snapshot.axes,
        )[:2]
        != request.stage_snapshot.current_stage_units[:2]
    ):
        raise _PlanRefusalError(
            "white_binding_changed", "WHITE sweep context changed before hand-off"
        )
    original = sweep.stage_snapshot
    current = request.stage_snapshot
    if (
        original.reference_id != current.reference_id
        or original.reference_origin_commanded_mm != current.reference_origin_commanded_mm
        or original.axes != current.axes
        or original.controller_max_velocity_um_s != current.controller_max_velocity_um_s
        or original.controller_max_acceleration_um_s2 != current.controller_max_acceleration_um_s2
        or original.request_timeout_s != current.request_timeout_s
        or current.motion_revision < original.motion_revision
    ):
        raise _PlanRefusalError(
            "white_stage_changed", "Stage identity or frozen motion profile changed"
        )
    low, high = sweep.possible_peak_z_stage_units
    peak = request.actual_peak_z_stage_units
    if not low <= peak <= high or current.current_stage_units[2] != peak:
        raise _PlanRefusalError(
            "white_peak_range", "Actual WHITE peak is outside its declared sweep"
        )
    if not request.budget.same_frozen_limit_as(sweep.budget_before):
        raise _PlanRefusalError("white_budget_changed", "WHITE field budget was reset or replaced")


def _validate_white_peak_plan_context(plan: FocusMotionPlan, expected_focus_z_um: float) -> None:
    """Recheck serialized WHITE provenance at the native stage boundary."""
    sweep = plan.white_sweep_plan
    _require_plan_value(sweep is not None, "WHITE sweep plan is missing")
    sweep = cast(WhiteSweepPlan, sweep)
    settings = sweep.settings
    peaks = sweep.possible_peak_z_stage_units
    original = sweep.stage_snapshot
    _require_plan_value(
        sweep.status == "planned"
        and sweep.binding == plan.binding
        and sweep.policy is not None
        and settings is not None
        and peaks is not None
        and original is not None
        and sweep.target_xy_um == plan.target_xy_um,
        "WHITE sweep context changed",
    )
    original = cast(StageMotionSnapshot, original)
    _validate_white_sweep_context(sweep)
    _require_plan_value(
        plan.budget_before.same_frozen_limit_as(cast(FocusFieldBudget, sweep.budget_before)),
        "WHITE field budget was reset or replaced",
    )
    _require_plan_value(
        original.reference_id == plan.stage_snapshot.reference_id
        and original.reference_origin_commanded_mm
        == plan.stage_snapshot.reference_origin_commanded_mm
        and original.axes == plan.stage_snapshot.axes
        and original.controller_max_velocity_um_s
        == plan.stage_snapshot.controller_max_velocity_um_s
        and original.controller_max_acceleration_um_s2
        == plan.stage_snapshot.controller_max_acceleration_um_s2
        and original.request_timeout_s == plan.stage_snapshot.request_timeout_s
        and plan.stage_snapshot.motion_revision >= original.motion_revision,
        "WHITE stage identity or motion profile changed",
    )
    peak_units = round(expected_focus_z_um * plan.stage_snapshot.axes[2].units_per_mm / 1000)
    _require_plan_value(
        cast(tuple[int, int], peaks)[0] <= peak_units <= cast(tuple[int, int], peaks)[1]
        and peak_units == plan.stage_snapshot.current_stage_units[2],
        "WHITE peak is outside the declared sweep",
    )
    disagreement = cast(WhiteSearchSettings, settings).maximum_white_led_disagreement_um
    _require_plan_value(
        plan.policy
        == cast(FocusApproachPolicy, sweep.policy).model_copy(
            update={"expected_prediction_error_range_um": (-disagreement, disagreement)}
        ),
        "WHITE hand-off policy differs from its frozen sweep",
    )


def plan_white_peak_approach(
    request: WhitePeakApproachRequest,
) -> FocusMotionPlan:
    """Plan the actual WHITE peak hand-off without inventing a map prediction."""
    white_policy = request.policy
    try:
        _validate_white_peak_request(request)
        settings = request.sweep_plan.settings
        deadline = request.sweep_plan.total_focus_deadline_monotonic_s
        if settings is None or deadline is None:
            raise _PlanRefusalError("white_plan_context", "WHITE sweep omitted its runtime limits")
        white_policy = request.policy.model_copy(
            update={
                "expected_prediction_error_range_um": (
                    -settings.maximum_white_led_disagreement_um,
                    settings.maximum_white_led_disagreement_um,
                )
            }
        )
        peak_z_um = (
            request.actual_peak_z_stage_units * 1000 / request.stage_snapshot.axes[2].units_per_mm
        )
        target_xy = cast(tuple[float, float], request.sweep_plan.target_xy_um)
        canonical = _calculate_preparation(
            snapshot=request.stage_snapshot,
            binding=request.frozen_binding,
            policy=white_policy,
            budget=request.budget,
            progress=request.progress,
            target_xy_um=target_xy,
            expected_focus_z_um=peak_z_um,
        )
        if (
            request.current_monotonic_s
            + canonical.path.duration_s
            + canonical.reserved_motion_time_s
            > deadline
        ):
            raise _PlanRefusalError(
                "white_total_budget",
                "Actual WHITE-to-R/G deadline cannot cover the hand-off",
            )
    except _PlanRefusalError as exc:
        return FocusMotionPlan(
            plan_id=request.plan_id,
            status="refused",
            purpose="white_peak_preparation",
            reason=str(exc),
            reason_code=exc.code,
            stage_snapshot=request.stage_snapshot,
            binding=request.frozen_binding,
            policy=white_policy,
            progress=request.progress,
            budget_before=request.budget,
            budget_after=request.budget,
            white_sweep_plan=request.sweep_plan,
        )
    return FocusMotionPlan(
        plan_id=request.plan_id,
        status="planned",
        purpose="white_peak_preparation",
        reason="Actual bounded WHITE peak has a complete verified LED hand-off",
        reason_code="white_peak_handoff",
        stage_snapshot=request.stage_snapshot,
        binding=request.frozen_binding,
        policy=white_policy,
        progress=request.progress,
        budget_before=request.budget,
        budget_after=canonical.budget_after,
        segments=canonical.path.segments,
        target_xy_um=canonical.target_xy_um,
        expected_focus_z_um=canonical.expected_focus_z_um,
        measurement_z_um=canonical.measurement_z_um,
        white_sweep_plan=request.sweep_plan,
        reserved_correction_endpoints_um=(canonical.reserved_correction_endpoints_um),
        estimated_motion_time_s=canonical.path.duration_s,
        reserved_motion_time_s=canonical.reserved_motion_time_s,
        z_travel_um=canonical.path.z_travel_um,
        reserved_z_travel_um=canonical.reserved_z_travel_um,
        prepares_on_verified_execution=True,
    )
