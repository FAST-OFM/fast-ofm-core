"""Strict, hardware-free contracts and prediction for an LED focus surface.

This module contains immutable data models and a bounded local numerical
predictor.  It does not plan a route, move a stage, capture a frame, or choose an
autofocus implementation at runtime.

All coordinates are physical micrometres.  Stage conversion is represented by
an explicit frozen binding, and every readback is labelled as commanded state,
not encoder feedback.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Self, cast

import numpy as np
from pydantic import Field, field_validator, model_validator

from fast_ofm_core.contracts import AxisName, StrictModel

from .rg.rg_focus_calibration import RGFocusApproachSettings
from .rg.rg_focus_estimator import MeasurementStatus

Identifier = Annotated[str, Field(min_length=1, pattern=r"\S")]
AutofocusMethod = Literal["none", "openflexure", "led", "simultaneous_rg"]
FocusStrategy = Literal["single_autofocus", "smart_stack"]
UnpredictedFocusMode = Literal["selected_method", "white_then_led"]
NoTissueMode = Literal["keep_z", "skip_tile", "pause"]

FocusObservationSource = Literal[
    "seed",
    "led_autofocus",
    "white_autofocus",
    "surface_prediction",
    "preload",
    "none",
]
FocusObservationStatus = Literal[
    "confirmed",
    "candidate",
    "intermediate",
    "failed",
    "cancelled",
    "no_tissue",
    "imaged",
]
FocusPredictionStatus = Literal["usable", "unavailable", "disabled", "fault"]
FocusPredictionSource = Literal["seed", "nearest_neighbor", "local_plane"]
FocusUnavailableReason = Literal[
    "no_observations",
    "insufficient_support",
    "stale_support",
    "poor_local_fit",
    "outside_support",
    "support_discontinuity",
    "prediction_delta_exceeded",
]
FocusObservationFaultCode = Literal[
    "foreign_scan",
    "binding_mismatch",
    "binding_unusable",
    "future_timestamp",
    "observation_id_conflict",
    "ambiguous_xy_timestamp",
    "candidate_limit_exceeded",
]


class FocusObservationContractError(ValueError):
    """A non-optical contract fault that must not become unavailable support."""

    def __init__(
        self,
        code: FocusObservationFaultCode,
        observation_id: str | None,
        detail: str,
    ) -> None:
        """Store a narrow machine-readable code and affected observation identity."""
        super().__init__(detail)
        self.code = code
        self.observation_id = observation_id


def _json_tuple(value: object) -> object:
    """Accept a decoded JSON array as a tuple without coercing its members."""
    if isinstance(value, list):
        return tuple(value)
    return value


def _strict_sign(value: object) -> object:
    """Reject booleans/floats that Pydantic's integer Literal accepts as equal."""
    if type(value) is not int or value not in (-1, 1):
        raise ValueError("Axis or approach sign must be integer -1 or +1")
    return value


class FocusAxisScale(StrictModel):
    """One axis conversion frozen at the stage boundary, not a motion profile."""

    axis: AxisName
    units_per_mm: float = Field(gt=0)
    direction_sign: Literal[-1, 1]

    @field_validator("direction_sign", mode="before")
    @classmethod
    def direction_is_an_integer_sign(cls, value: object) -> object:
        """Keep bool/float/string values out of the signed axis binding."""
        return _strict_sign(value)


class FocusBinding(StrictModel):
    """Explicit identities and parameters that make surface data compatible.

    The reference has no age or TTL.  A changed/lost reference is represented by
    its identity/status and must be checked against the live frozen snapshot.
    """

    stage_controller_id: Identifier
    reference_id: Identifier
    reference_status: Literal["valid", "lost"]
    axis_scales: tuple[FocusAxisScale, FocusAxisScale, FocusAxisScale]
    camera_stage_mapping_id: Identifier
    camera_stage_mapping_status: Literal["valid", "candidate"]
    geometry_id: Identifier
    rg_focus_model_id: Identifier
    rg_focus_model_status: Literal["valid", "candidate"]
    red_flat_field_profile_id: Identifier
    red_flat_field_status: Literal["valid", "validation_failed"]
    green_flat_field_profile_id: Identifier
    green_flat_field_status: Literal["valid", "validation_failed"]
    approach_profile_id: Identifier
    approach_status: Literal["valid", "candidate"]
    approach_parameters: RGFocusApproachSettings

    @field_validator("axis_scales", mode="before")
    @classmethod
    def axes_accept_json_array(cls, value: object) -> object:
        """Convert only the JSON container; nested numeric values stay strict."""
        return _json_tuple(value)

    @field_validator("approach_parameters", mode="before")
    @classmethod
    def approach_sign_is_strict(cls, value: object) -> object:
        """Protect the reused approach model from bool/float sign coercion."""
        if isinstance(value, Mapping) and "approach_sign" in value:
            _strict_sign(value["approach_sign"])
        return value

    @model_validator(mode="after")
    def exactly_xyz_scales(self) -> Self:
        """Require one canonical, unambiguous scale for each physical axis."""
        axes = tuple(scale.axis for scale in self.axis_scales)
        if axes != ("x", "y", "z"):
            raise ValueError("Axis scales must contain X, Y and Z exactly once in order")
        return self


class FocusBindingCheck(StrictModel):
    """Pure compatibility/readiness result with explicit mismatched fields."""

    compatible: bool
    usable: bool
    reason: Identifier
    mismatched_fields: tuple[str, ...] = ()

    @field_validator("mismatched_fields", mode="before")
    @classmethod
    def fields_accept_json_array(cls, value: object) -> object:
        """Accept persisted JSON while retaining an immutable tuple."""
        return _json_tuple(value)


_BINDING_COMPATIBILITY_FIELDS = (
    "stage_controller_id",
    "reference_id",
    "axis_scales",
    "camera_stage_mapping_id",
    "geometry_id",
    "rg_focus_model_id",
    "red_flat_field_profile_id",
    "green_flat_field_profile_id",
    "approach_profile_id",
    "approach_parameters",
)


def _binding_readiness_problems(binding: FocusBinding) -> tuple[str, ...]:
    """Return every explicit status that prevents LED surface use."""
    expected_statuses = {
        "reference_status": "valid",
        "camera_stage_mapping_status": "valid",
        "rg_focus_model_status": "valid",
        "red_flat_field_status": "valid",
        "green_flat_field_status": "valid",
        "approach_status": "valid",
    }
    return tuple(
        name for name, expected in expected_statuses.items() if getattr(binding, name) != expected
    )


def check_focus_binding(expected: FocusBinding, current: FocusBinding) -> FocusBindingCheck:
    """Compare known fields and require readiness of both frozen and current sides."""
    mismatches = tuple(
        name
        for name in _BINDING_COMPATIBILITY_FIELDS
        if getattr(expected, name) != getattr(current, name)
    )
    expected_problems = tuple(f"expected.{name}" for name in _binding_readiness_problems(expected))
    current_problems = tuple(f"current.{name}" for name in _binding_readiness_problems(current))
    if mismatches:
        return FocusBindingCheck(
            compatible=False,
            usable=False,
            reason="Focus binding identities or frozen parameters changed",
            mismatched_fields=mismatches + expected_problems + current_problems,
        )
    if expected_problems or current_problems:
        if expected_problems and current_problems:
            reason = "Frozen and current focus bindings are not valid for LED surface use"
        elif expected_problems:
            reason = "Frozen expected focus binding is not valid for LED surface use"
        else:
            reason = "Current focus binding is not valid for LED surface use"
        return FocusBindingCheck(
            compatible=True,
            usable=False,
            reason=reason,
            mismatched_fields=expected_problems + current_problems,
        )
    return FocusBindingCheck(
        compatible=True,
        usable=True,
        reason="Focus binding identities, parameters and statuses are compatible",
    )


class FocusSurfaceSettings(StrictModel):
    """Run-frozen local-surface gates; disabled settings authorise nothing.

    Operational distance, fit and age limits intentionally have no guessed
    defaults.  They must all be supplied before ``enabled`` can be true.
    Observation age limits surface support only; they are not a reference TTL.
    """

    enabled: bool = False
    minimum_plane_points: int = Field(default=4, ge=4, le=64)
    maximum_neighbors: int = Field(default=12, ge=4, le=64)
    maximum_candidate_observations: int = Field(default=256, ge=4, le=4096)
    neighbor_radius_um: float | None = Field(default=None, gt=0)
    allow_extrapolation: bool = False
    maximum_extrapolation_um: float | None = Field(default=None, ge=0)
    maximum_prediction_delta_um: float | None = Field(default=None, gt=0)
    measurement_offset_um: float | None = None
    maximum_fit_residual_um: float | None = Field(default=None, gt=0)
    minimum_fit_inlier_fraction: float | None = Field(default=None, gt=0, le=1)
    maximum_fit_condition_number: float | None = Field(default=None, gt=1)
    maximum_plane_slope_um_per_um: float | None = Field(default=None, gt=0)
    maximum_observation_age_s: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def complete_operational_gates(self) -> Self:
        """Require a coherent, fully explicit set of limits when enabled."""
        if self.maximum_neighbors < self.minimum_plane_points:
            raise ValueError("Maximum neighbors cannot be below minimum plane points")
        if self.maximum_candidate_observations < self.maximum_neighbors:
            raise ValueError("Maximum candidate observations cannot be below maximum neighbors")
        required = (
            "neighbor_radius_um",
            "maximum_extrapolation_um",
            "maximum_prediction_delta_um",
            "measurement_offset_um",
            "maximum_fit_residual_um",
            "minimum_fit_inlier_fraction",
            "maximum_fit_condition_number",
            "maximum_plane_slope_um_per_um",
            "maximum_observation_age_s",
        )
        if self.enabled and any(getattr(self, name) is None for name in required):
            raise ValueError("Enabled focus surface requires every operational gate")
        if not self.enabled:
            return self
        if self.maximum_extrapolation_um is None or self.neighbor_radius_um is None:
            raise ValueError("Enabled focus surface requires distance limits")
        if self.allow_extrapolation != (self.maximum_extrapolation_um > 0):
            raise ValueError("Extrapolation flag and maximum extrapolation distance disagree")
        if self.maximum_extrapolation_um > self.neighbor_radius_um:
            raise ValueError("Extrapolation distance cannot exceed neighbor radius")
        if (
            self.maximum_fit_residual_um is None
            or self.maximum_prediction_delta_um is None
            or self.maximum_fit_residual_um > self.maximum_prediction_delta_um
        ):
            raise ValueError("Fit residual gate cannot exceed prediction delta gate")
        if (
            self.measurement_offset_um is None
            or abs(self.measurement_offset_um) > self.maximum_prediction_delta_um
        ):
            raise ValueError("Measurement offset exceeds the prediction delta gate")
        return self


class WhiteSearchSettings(StrictModel):
    """Frozen bounds and time reserve for one WHITE-to-LED field transfer.

    These are algorithm budgets only.  They never widen the stage profile; a
    later approach/path layer must validate every endpoint and overshoot.
    """

    search_z_range_um: tuple[float, float]
    white_search_timeout_s: float = Field(gt=0)
    total_focus_budget_s: float = Field(gt=0)
    approach_and_rg_reserve_s: float = Field(gt=0)
    maximum_white_led_disagreement_um: float = Field(gt=0)
    maximum_white_searches_per_field: Literal[1] = 1

    @field_validator("search_z_range_um", mode="before")
    @classmethod
    def range_accepts_json_array(cls, value: object) -> object:
        """Accept a JSON array but retain strict numeric item validation."""
        return _json_tuple(value)

    @field_validator("maximum_white_searches_per_field", mode="before")
    @classmethod
    def exactly_one_integer_search(cls, value: object) -> object:
        """Make the one-WHITE-per-field invariant resistant to bool coercion."""
        if type(value) is not int or value != 1:
            raise ValueError("Exactly one WHITE search is allowed per field attempt")
        return value

    @model_validator(mode="after")
    def bounded_transfer_budget(self) -> Self:
        """Require a two-sided finite range and preserve the post-WHITE reserve."""
        low, high = self.search_z_range_um
        if not low < 0 < high:
            raise ValueError("WHITE search range must extend to both sides of zero")
        if self.white_search_timeout_s + self.approach_and_rg_reserve_s > (
            self.total_focus_budget_s
        ):
            raise ValueError("WHITE timeout and approach/RG reserve exceed total budget")
        if self.maximum_white_led_disagreement_um > high - low:
            raise ValueError("WHITE/LED disagreement gate exceeds the search range")
        return self


class FocusRunSettings(StrictModel):
    """Frozen scan choices that can activate the otherwise inert surface contract."""

    autofocus_method: AutofocusMethod = "openflexure"
    focus_strategy: FocusStrategy = "smart_stack"
    surface: FocusSurfaceSettings = Field(default_factory=FocusSurfaceSettings)
    unpredicted_focus_mode: UnpredictedFocusMode = "selected_method"
    white_search: WhiteSearchSettings | None = None
    no_tissue_mode: NoTissueMode = "keep_z"
    on_focus_failure: Literal["pause"] = "pause"
    binding: FocusBinding | None = None

    @model_validator(mode="after")
    def supported_active_combination(self) -> Self:
        """Fail before motion for every unsupported active mixed-focus selection."""
        if not self.surface.enabled:
            if self.unpredicted_focus_mode == "white_then_led":
                raise ValueError("white_then_led requires an enabled focus surface")
            if self.white_search is not None:
                raise ValueError("WHITE transfer settings require white_then_led")
            return self
        if self.autofocus_method != "led" or self.focus_strategy != "single_autofocus":
            raise ValueError("Focus surface supports only LED single_autofocus")
        if self.binding is None:
            raise ValueError("Enabled focus surface requires a frozen binding")
        binding_check = check_focus_binding(self.binding, self.binding)
        if not binding_check.usable:
            raise ValueError("Enabled focus surface requires valid compatible profiles")
        if self.unpredicted_focus_mode == "white_then_led":
            if self.white_search is None:
                raise ValueError("white_then_led requires bounded WHITE transfer settings")
        elif self.white_search is not None:
            raise ValueError("WHITE transfer settings require white_then_led")
        return self


class FocusObservationQuality(StrictModel):
    """Saved LED measurement QC; confidence is diagnostic, not confirmation alone."""

    measurement_status: MeasurementStatus
    confidence: float = Field(ge=0, le=1)
    accepted_patch_count: int = Field(ge=0)
    inlier_fraction: float = Field(ge=0, le=1)
    final_error_um: float | None = None
    focus_tolerance_um: float = Field(gt=0)
    independent_post_move_verification: bool

    @property
    def confirms_led_focus(self) -> bool:
        """Return whether QC, tolerance and independent verification all passed."""
        return (
            self.measurement_status == "ready"
            and self.accepted_patch_count > 0
            and self.final_error_um is not None
            and abs(self.final_error_um) <= self.focus_tolerance_um
            and self.independent_post_move_verification
        )


class FocusObservationProvenance(StrictModel):
    """Stable result/report/frame identities; paths are references, not file contents."""

    result_id: Identifier
    report_ref: Identifier
    rg_measurement_id: Identifier | None = None
    capture_ids: tuple[str, ...] = ()

    @field_validator("capture_ids", mode="before")
    @classmethod
    def captures_accept_json_array(cls, value: object) -> object:
        """Accept saved JSON arrays and keep the snapshot immutable."""
        return _json_tuple(value)

    @model_validator(mode="after")
    def unique_capture_ids(self) -> Self:
        """Prevent a repeated frame identity from inflating provenance."""
        if len(self.capture_ids) != len(set(self.capture_ids)):
            raise ValueError("Capture IDs must be unique")
        if any(not value or value.isspace() for value in self.capture_ids):
            raise ValueError("Capture IDs must be non-empty")
        return self


class FocusPredictionEvidence(StrictModel):
    """Prediction saved before an observation, including its later signed error."""

    prediction_id: Identifier
    predicted_z_um: float
    prediction_error_um: float
    support_age_s: float = Field(ge=0)


class FocusObservation(StrictModel):
    """One immutable field result at a settled commanded XYZ readback."""

    scan_id: Identifier
    field_id: Identifier
    attempt_id: Identifier
    observation_id: Identifier
    x_um: float
    y_um: float
    z_um: float
    readback_kind: Literal["commanded_not_encoder"] = "commanded_not_encoder"
    final_readback_settled: bool
    final_readback_monotonic_s: float = Field(ge=0)
    recorded_monotonic_s: float = Field(ge=0)
    source: FocusObservationSource
    status: FocusObservationStatus
    binding: FocusBinding
    provenance: FocusObservationProvenance
    quality: FocusObservationQuality | None = None
    pre_update_prediction: FocusPredictionEvidence | None = None

    @model_validator(mode="after")
    def internally_consistent_result(self) -> Self:
        """Keep numeric positions from upgrading non-LED or unverified results."""
        if self.final_readback_monotonic_s > self.recorded_monotonic_s:
            raise ValueError("Final readback cannot occur after observation recording")
        if self.pre_update_prediction is not None and not math.isclose(
            self.pre_update_prediction.prediction_error_um,
            self.z_um - self.pre_update_prediction.predicted_z_um,
            rel_tol=0,
            abs_tol=1e-9,
        ):
            raise ValueError("Saved pre-update prediction error is inconsistent")
        if self.status != "confirmed":
            return self
        if self.source != "led_autofocus":
            raise ValueError("Only LED autofocus can create a confirmed observation")
        if not self.final_readback_settled:
            raise ValueError("Confirmed observation requires a settled final readback")
        if self.quality is None or not self.quality.confirms_led_focus:
            raise ValueError("Confirmed observation requires independently verified LED QC")
        if self.provenance.rg_measurement_id is None or len(self.provenance.capture_ids) < 2:
            raise ValueError("Confirmed observation requires R/G measurement provenance")
        if not check_focus_binding(self.binding, self.binding).usable:
            raise ValueError("Confirmed observation requires a valid LED binding")
        return self


class FocusObservationSuitability(StrictModel):
    """Pure decision about whether one observation may support the surface."""

    usable: bool
    reason: Identifier
    age_s: float | None = Field(default=None, ge=0)


def _expected_session_time(expected_scan_id: str, now_monotonic_s: float) -> float:
    """Validate the caller-supplied session identity and monotonic clock value."""
    if not isinstance(expected_scan_id, str) or not expected_scan_id.strip():
        raise ValueError("Expected scan ID must be a non-empty string")
    if isinstance(now_monotonic_s, bool) or not isinstance(now_monotonic_s, (int, float)):
        raise TypeError("Current monotonic time must be a number")
    now = float(now_monotonic_s)
    if not math.isfinite(now) or now < 0:
        raise ValueError("Current monotonic time must be finite and non-negative")
    return now


def _observation_age_in_expected_context(
    observation: FocusObservation,
    *,
    expected_scan_id: str,
    expected_binding: FocusBinding,
    now_monotonic_s: float,
) -> float:
    """Return comparable in-session age or raise a non-optical contract error."""
    if observation.scan_id != expected_scan_id:
        raise FocusObservationContractError(
            "foreign_scan",
            observation.observation_id,
            "Observation belongs to a different scan session",
        )
    binding_check = check_focus_binding(observation.binding, expected_binding)
    if not binding_check.compatible:
        raise FocusObservationContractError(
            "binding_mismatch",
            observation.observation_id,
            binding_check.reason,
        )
    if not binding_check.usable:
        raise FocusObservationContractError(
            "binding_unusable",
            observation.observation_id,
            binding_check.reason,
        )
    age = now_monotonic_s - observation.recorded_monotonic_s
    if age < 0:
        raise FocusObservationContractError(
            "future_timestamp",
            observation.observation_id,
            "Observation time is ahead of the expected scan session time",
        )
    return age


def assess_focus_observation(
    observation: FocusObservation,
    *,
    expected_scan_id: str,
    expected_binding: FocusBinding,
    settings: FocusSurfaceSettings,
    now_monotonic_s: float,
) -> FocusObservationSuitability:
    """Assess optical suitability or raise a narrow non-optical contract fault."""
    now = _expected_session_time(expected_scan_id, now_monotonic_s)
    if not settings.enabled:
        return FocusObservationSuitability(usable=False, reason="Focus surface is disabled")
    age = _observation_age_in_expected_context(
        observation,
        expected_scan_id=expected_scan_id,
        expected_binding=expected_binding,
        now_monotonic_s=now,
    )
    if observation.source != "led_autofocus" or observation.status != "confirmed":
        return FocusObservationSuitability(
            usable=False,
            reason="Observation is not a confirmed LED focus result",
            age_s=age,
        )
    if observation.quality is None or not observation.quality.confirms_led_focus:
        return FocusObservationSuitability(
            usable=False,
            reason="Observation LED quality is not independently verified",
            age_s=age,
        )
    maximum_age = settings.maximum_observation_age_s
    if maximum_age is None:
        return FocusObservationSuitability(
            usable=False, reason="Observation age gate is not configured", age_s=age
        )
    if age > maximum_age:
        return FocusObservationSuitability(
            usable=False, reason="Observation support is stale", age_s=age
        )
    return FocusObservationSuitability(
        usable=True, reason="Observation is valid surface support", age_s=age
    )


@dataclass(frozen=True)
class _ObservationSelection:
    """Deterministic selection plus counts needed for unavailable diagnostics."""

    observations: tuple[FocusObservation, ...]
    unique_observation_count: int
    stale_observation_count: int


def _select_observation_support(
    observations: Sequence[FocusObservation],
    *,
    expected_scan_id: str,
    expected_binding: FocusBinding,
    settings: FocusSurfaceSettings,
    now_monotonic_s: float,
) -> _ObservationSelection:
    """Deduplicate retries and choose the latest accepted result at each exact XY.

    Later failures remain in the caller's history but cannot erase an earlier good
    result.  Conflicting content under one observation ID fails closed.
    """
    _expected_session_time(expected_scan_id, now_monotonic_s)
    if settings.enabled:
        binding_check = check_focus_binding(expected_binding, expected_binding)
        if not binding_check.usable:
            raise FocusObservationContractError("binding_unusable", None, binding_check.reason)
    unique: dict[str, FocusObservation] = {}
    for observation in observations:
        previous = unique.get(observation.observation_id)
        if previous is not None and previous != observation:
            raise FocusObservationContractError(
                "observation_id_conflict",
                observation.observation_id,
                "Conflicting observations share one observation ID",
            )
        unique[observation.observation_id] = observation

    accepted_by_xy: dict[tuple[float, float], list[FocusObservation]] = {}
    stale_count = 0
    for observation in unique.values():
        suitability = assess_focus_observation(
            observation,
            expected_scan_id=expected_scan_id,
            expected_binding=expected_binding,
            settings=settings,
            now_monotonic_s=now_monotonic_s,
        )
        if not suitability.usable:
            if suitability.reason == "Observation support is stale":
                stale_count += 1
            continue
        xy = (observation.x_um, observation.y_um)
        accepted_by_xy.setdefault(xy, []).append(observation)

    latest: list[FocusObservation] = []
    for same_xy in accepted_by_xy.values():
        latest_time = max(item.recorded_monotonic_s for item in same_xy)
        latest_at_time = tuple(item for item in same_xy if item.recorded_monotonic_s == latest_time)
        if len(latest_at_time) != 1:
            raise FocusObservationContractError(
                "ambiguous_xy_timestamp",
                None,
                "Two accepted observations at one XY have the same time",
            )
        latest.append(latest_at_time[0])
    selected = tuple(
        sorted(latest, key=lambda item: (item.recorded_monotonic_s, item.observation_id))
    )
    return _ObservationSelection(
        observations=selected,
        unique_observation_count=len(unique),
        stale_observation_count=stale_count,
    )


def select_usable_observations(
    observations: Sequence[FocusObservation],
    *,
    expected_scan_id: str,
    expected_binding: FocusBinding,
    settings: FocusSurfaceSettings,
    now_monotonic_s: float,
) -> tuple[FocusObservation, ...]:
    """Return deterministic accepted support while preserving narrow faults."""
    return _select_observation_support(
        observations,
        expected_scan_id=expected_scan_id,
        expected_binding=expected_binding,
        settings=settings,
        now_monotonic_s=now_monotonic_s,
    ).observations


class FocusPredictionDiagnostics(StrictModel):
    """Fit/support diagnostics, explicitly not a confidence interval."""

    interpretation: Literal["diagnostic_not_confidence_interval"] = (
        "diagnostic_not_confidence_interval"
    )
    candidate_observation_count: int = Field(ge=0)
    local_observation_count: int = Field(default=0, ge=0)
    selected_support_count: int = Field(default=0, ge=0)
    nearest_distance_um: float | None = Field(default=None, ge=0)
    support_age_s: float | None = Field(default=None, ge=0)
    fit_rmse_um: float | None = Field(default=None, ge=0)
    maximum_absolute_residual_um: float | None = Field(default=None, ge=0)
    fit_rank: int | None = Field(default=None, ge=0, le=3)
    fit_condition_number: float | None = Field(default=None, ge=1)
    fit_inlier_fraction: float | None = Field(default=None, ge=0, le=1)
    plane_slope_x_um_per_um: float | None = None
    plane_slope_y_um_per_um: float | None = None
    plane_slope_magnitude_um_per_um: float | None = Field(default=None, ge=0)
    excluded_observation_ids: tuple[str, ...] = ()
    extrapolation_distance_um: float | None = Field(default=None, ge=0)
    prediction_delta_um: float | None = None

    @field_validator("excluded_observation_ids", mode="before")
    @classmethod
    def excluded_ids_accept_json_array(cls, value: object) -> object:
        """Accept persisted JSON arrays while retaining immutable identities."""
        return _json_tuple(value)

    @model_validator(mode="after")
    def diagnostic_counts_and_ids_are_coherent(self) -> Self:
        """Keep reported bounded selection counts internally consistent."""
        if self.local_observation_count > self.candidate_observation_count:
            raise ValueError("Local observation count exceeds candidate count")
        if self.selected_support_count > self.local_observation_count:
            raise ValueError("Selected support count exceeds local count")
        if len(self.excluded_observation_ids) != len(set(self.excluded_observation_ids)):
            raise ValueError("Excluded observation IDs must be unique")
        return self


class FocusPredictionRequest(StrictModel):
    """One pure query in physical micrometres against a frozen scan context."""

    expected_scan_id: Identifier
    field_id: Identifier
    attempt_id: Identifier
    prediction_id: Identifier
    target_x_um: float
    target_y_um: float
    current_z_um: float
    predicted_monotonic_s: float = Field(ge=0)
    expected_binding: FocusBinding | None = None
    settings: FocusSurfaceSettings


class FocusPrediction(StrictModel):
    """A target-Z decision whose four statuses have disjoint payloads."""

    scan_id: Identifier
    field_id: Identifier
    attempt_id: Identifier
    prediction_id: Identifier
    target_x_um: float
    target_y_um: float
    target_z_um: float | None = None
    status: FocusPredictionStatus
    source: FocusPredictionSource | None = None
    reason: Identifier
    unavailable_reason: FocusUnavailableReason | None = None
    binding: FocusBinding | None = None
    support_observation_ids: tuple[str, ...] = ()
    support_field_ids: tuple[str, ...] = ()
    support_distances_um: tuple[Annotated[float, Field(ge=0)], ...] = ()
    diagnostics: FocusPredictionDiagnostics | None = None
    extrapolated: bool = False
    fault_code: FocusObservationFaultCode | None = None
    predicted_monotonic_s: float = Field(ge=0)

    @field_validator(
        "support_observation_ids",
        "support_field_ids",
        "support_distances_um",
        mode="before",
    )
    @classmethod
    def supports_accept_json_array(cls, value: object) -> object:
        """Accept JSON arrays while retaining immutable strict string tuples."""
        return _json_tuple(value)

    def _validate_support_ids(self) -> None:
        """Validate the paired immutable support identity collections."""
        support_ids = self.support_observation_ids
        field_ids = self.support_field_ids
        if len(support_ids) != len(set(support_ids)) or len(field_ids) != len(set(field_ids)):
            raise ValueError("Prediction support IDs must be unique")
        if len(support_ids) != len(field_ids):
            raise ValueError("Prediction support observation and field IDs must align")
        if len(support_ids) != len(self.support_distances_um):
            raise ValueError("Prediction support IDs and distances must align")
        if any(not value or value.isspace() for value in (*support_ids, *field_ids)):
            raise ValueError("Prediction support IDs must be non-empty")

    def _validate_usable_payload(self) -> None:
        """Require complete optical evidence for one usable target Z."""
        if self.target_z_um is None:
            raise ValueError("Usable prediction requires target Z, including Z=0")
        if self.source not in ("nearest_neighbor", "local_plane"):
            raise ValueError("Usable prediction requires an optical support source")
        if not self.support_observation_ids or self.binding is None or self.diagnostics is None:
            raise ValueError("Usable prediction requires binding, support and diagnostics")
        if not check_focus_binding(self.binding, self.binding).usable:
            raise ValueError("Usable prediction requires a valid LED binding")
        if self.unavailable_reason is not None:
            raise ValueError("Usable prediction cannot carry an unavailable reason")
        if self.fault_code is not None:
            raise ValueError("Usable prediction cannot carry a fault code")
        if self.diagnostics.support_age_s is None:
            raise ValueError("Usable prediction requires support age")
        if self.source == "local_plane" and len(self.support_observation_ids) < 4:
            raise ValueError("Local plane prediction requires at least four supports")

    def _validate_unavailable_payload(self) -> None:
        """Allow only expected optical-support failures to be unavailable."""
        if self.unavailable_reason is None:
            raise ValueError("Unavailable prediction requires an optical reason")
        if self.binding is None or not check_focus_binding(self.binding, self.binding).usable:
            raise ValueError("Profile/reference faults cannot be labelled optical unavailable")

    def _validate_nonusable_payload(self) -> None:
        """Keep target/support data out of unavailable, disabled, and fault states."""
        if (
            self.target_z_um is not None
            or self.support_observation_ids
            or self.support_field_ids
            or self.support_distances_um
        ):
            raise ValueError("Only usable prediction may carry target Z or support IDs")
        if self.extrapolated:
            raise ValueError("Only usable prediction may claim bounded extrapolation")
        if self.status == "unavailable":
            if self.fault_code is not None:
                raise ValueError("Unavailable prediction cannot carry a fault code")
            self._validate_unavailable_payload()
            return
        if self.unavailable_reason is not None:
            raise ValueError("Disabled/fault prediction cannot carry unavailable reason")
        if self.source is not None:
            raise ValueError("Disabled/fault prediction cannot claim an optical source")
        if self.diagnostics is not None:
            raise ValueError("Disabled/fault prediction cannot carry fit diagnostics")
        if self.status == "fault" and self.fault_code is None:
            raise ValueError("Fault prediction requires a narrow fault code")
        if self.status == "disabled" and self.fault_code is not None:
            raise ValueError("Disabled prediction cannot carry a fault code")

    @model_validator(mode="after")
    def status_payload_is_disjoint(self) -> Self:
        """Keep unavailable/disabled/fault from masquerading as target Z values."""
        self._validate_support_ids()
        if self.status == "usable":
            self._validate_usable_payload()
        else:
            self._validate_nonusable_payload()
        return self


@dataclass(frozen=True)
class _PlaneCalculation:
    """One centred and isotropically scaled least-squares calculation."""

    observations: tuple[FocusObservation, ...]
    predicted_z_um: float | None
    fit_rmse_um: float | None
    maximum_absolute_residual_um: float | None
    rank: int
    condition_number: float | None
    slope_x_um_per_um: float | None
    slope_y_um_per_um: float | None
    slope_magnitude_um_per_um: float | None


@dataclass(frozen=True)
class _PlaneOutcome:
    """Accepted robust plane or an explicit optical rejection."""

    calculation: _PlaneCalculation
    unavailable_reason: Literal["poor_local_fit", "support_discontinuity"] | None
    excluded_observation_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _LocalSupport:
    """Bounded nearest candidates around one physical target."""

    selection: _ObservationSelection
    observations: tuple[FocusObservation, ...]
    distances_um: tuple[float, ...]
    local_observation_count: int
    nearest_distance_um: float | None


def _calculate_plane(
    observations: tuple[FocusObservation, ...],
    target_x_um: float,
    target_y_um: float,
) -> _PlaneCalculation:
    """Fit ``z = a + bx + cy`` after centring and one physical XY scale."""
    xy = np.asarray([(item.x_um, item.y_um) for item in observations], dtype=float)
    z = np.asarray([item.z_um for item in observations], dtype=float)
    centre = np.mean(xy, axis=0)
    centred = xy - centre
    coordinate_scale_um = float(np.sqrt(np.mean(np.sum(centred * centred, axis=1))))
    if coordinate_scale_um == 0:
        return _PlaneCalculation(
            observations=observations,
            predicted_z_um=None,
            fit_rmse_um=None,
            maximum_absolute_residual_um=None,
            rank=1,
            condition_number=None,
            slope_x_um_per_um=None,
            slope_y_um_per_um=None,
            slope_magnitude_um_per_um=None,
        )
    design = np.column_stack((np.ones(len(observations)), centred / coordinate_scale_um))
    singular_values = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    condition_number = None
    if singular_values[-1] > 0:
        candidate_condition = float(singular_values[0] / singular_values[-1])
        if math.isfinite(candidate_condition):
            condition_number = candidate_condition
    if rank < 3:
        return _PlaneCalculation(
            observations=observations,
            predicted_z_um=None,
            fit_rmse_um=None,
            maximum_absolute_residual_um=None,
            rank=rank,
            condition_number=condition_number,
            slope_x_um_per_um=None,
            slope_y_um_per_um=None,
            slope_magnitude_um_per_um=None,
        )
    coefficients = np.linalg.lstsq(design, z, rcond=None)[0]
    fitted = design @ coefficients
    residuals = z - fitted
    target_scaled = (np.asarray((target_x_um, target_y_um)) - centre) / (coordinate_scale_um)
    predicted_z_um = float(
        coefficients[0] + coefficients[1] * target_scaled[0] + coefficients[2] * target_scaled[1]
    )
    slope_x = float(coefficients[1] / coordinate_scale_um)
    slope_y = float(coefficients[2] / coordinate_scale_um)
    return _PlaneCalculation(
        observations=observations,
        predicted_z_um=predicted_z_um,
        fit_rmse_um=float(np.sqrt(np.mean(residuals * residuals))),
        maximum_absolute_residual_um=float(np.max(np.abs(residuals))),
        rank=rank,
        condition_number=condition_number,
        slope_x_um_per_um=slope_x,
        slope_y_um_per_um=slope_y,
        slope_magnitude_um_per_um=math.hypot(slope_x, slope_y),
    )


def _plane_geometry_is_usable(
    calculation: _PlaneCalculation, settings: FocusSurfaceSettings
) -> bool:
    """Apply full-rank and condition gates without hiding singular geometry."""
    maximum_condition = cast(float, settings.maximum_fit_condition_number)
    return (
        calculation.rank == 3
        and calculation.condition_number is not None
        and calculation.condition_number <= maximum_condition
    )


def _plane_residual_is_usable(
    calculation: _PlaneCalculation, settings: FocusSurfaceSettings
) -> bool:
    """Apply the configured maximum absolute residual gate."""
    maximum_residual = cast(float, settings.maximum_fit_residual_um)
    return (
        calculation.maximum_absolute_residual_um is not None
        and calculation.maximum_absolute_residual_um <= maximum_residual
    )


def _plane_slope_is_usable(calculation: _PlaneCalculation, settings: FocusSurfaceSettings) -> bool:
    """Apply the physical gradient-magnitude gate."""
    maximum_slope = cast(float, settings.maximum_plane_slope_um_per_um)
    return (
        calculation.slope_magnitude_um_per_um is not None
        and calculation.slope_magnitude_um_per_um <= maximum_slope
    )


def _excluded_residual_um(
    calculation: _PlaneCalculation,
    excluded: FocusObservation,
    target_x_um: float,
    target_y_um: float,
) -> float:
    """Measure an excluded point against the candidate plane."""
    predicted_at_excluded = (
        cast(float, calculation.predicted_z_um)
        + cast(float, calculation.slope_x_um_per_um) * (excluded.x_um - target_x_um)
        + cast(float, calculation.slope_y_um_per_um) * (excluded.y_um - target_y_um)
    )
    return abs(excluded.z_um - predicted_at_excluded)


def _fit_bounded_plane(
    observations: tuple[FocusObservation, ...],
    request: FocusPredictionRequest,
) -> _PlaneOutcome:
    """Accept a clean plane or one uniquely identifiable independent outlier."""
    full = _calculate_plane(observations, request.target_x_um, request.target_y_um)
    if not _plane_geometry_is_usable(full, request.settings):
        return _PlaneOutcome(full, "poor_local_fit")
    if _plane_residual_is_usable(full, request.settings):
        reason: Literal["poor_local_fit"] | None = (
            None if _plane_slope_is_usable(full, request.settings) else "poor_local_fit"
        )
        return _PlaneOutcome(full, reason)

    inlier_count = len(observations) - 1
    minimum_fraction = cast(float, request.settings.minimum_fit_inlier_fraction)
    if (
        inlier_count < request.settings.minimum_plane_points
        or inlier_count / len(observations) < minimum_fraction
    ):
        return _PlaneOutcome(full, "support_discontinuity")

    robust_candidates: list[tuple[_PlaneCalculation, FocusObservation]] = []
    maximum_residual = cast(float, request.settings.maximum_fit_residual_um)
    for index, excluded in enumerate(observations):
        retained = observations[:index] + observations[index + 1 :]
        calculation = _calculate_plane(retained, request.target_x_um, request.target_y_um)
        if not (
            _plane_geometry_is_usable(calculation, request.settings)
            and _plane_residual_is_usable(calculation, request.settings)
            and _plane_slope_is_usable(calculation, request.settings)
        ):
            continue
        if (
            _excluded_residual_um(
                calculation,
                excluded,
                request.target_x_um,
                request.target_y_um,
            )
            <= maximum_residual
        ):
            continue
        robust_candidates.append((calculation, excluded))
    if len(robust_candidates) != 1:
        return _PlaneOutcome(full, "support_discontinuity")
    calculation, excluded = robust_candidates[0]
    return _PlaneOutcome(
        calculation=calculation,
        unavailable_reason=None,
        excluded_observation_ids=(excluded.observation_id,),
    )


def _cross(
    origin: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    """Return the signed 2-D cross product around ``origin``."""
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
        second[0] - origin[0]
    )


def _convex_hull(
    points: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Return a deterministic counter-clockwise convex hull without SciPy state."""
    unique = sorted(set(points))
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def _distance_from_origin_to_segment(start: tuple[float, float], end: tuple[float, float]) -> float:
    """Return Euclidean distance from the translated target to a hull edge."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(*start)
    fraction = -(start[0] * dx + start[1] * dy) / length_squared
    fraction = min(1.0, max(0.0, fraction))
    return math.hypot(start[0] + fraction * dx, start[1] + fraction * dy)


def _extrapolation_distance_um(
    observations: tuple[FocusObservation, ...],
    target_x_um: float,
    target_y_um: float,
) -> float:
    """Return zero inside the support hull, else distance to its boundary."""
    translated = tuple((item.x_um - target_x_um, item.y_um - target_y_um) for item in observations)
    hull = _convex_hull(translated)
    coordinate_scale = max(1.0, *(max(abs(x), abs(y)) for x, y in hull))
    tolerance = np.finfo(float).eps * coordinate_scale * coordinate_scale * 64
    if all(
        _cross(start, end, (0.0, 0.0)) >= -tolerance
        for start, end in zip(hull, hull[1:] + hull[:1], strict=True)
    ):
        return 0.0
    return min(
        _distance_from_origin_to_segment(start, end)
        for start, end in zip(hull, hull[1:] + hull[:1], strict=True)
    )


def _local_support(
    selection: _ObservationSelection, request: FocusPredictionRequest
) -> _LocalSupport:
    """Sort by physical distance and retain at most the frozen neighbor count."""
    distance_rows = sorted(
        (
            (
                math.hypot(
                    item.x_um - request.target_x_um,
                    item.y_um - request.target_y_um,
                ),
                item,
            )
            for item in selection.observations
        ),
        key=lambda row: (row[0], row[1].observation_id),
    )
    nearest_distance = distance_rows[0][0] if distance_rows else None
    radius = cast(float, request.settings.neighbor_radius_um)
    local_rows = [row for row in distance_rows if row[0] <= radius]
    bounded_rows = local_rows[: request.settings.maximum_neighbors]
    return _LocalSupport(
        selection=selection,
        observations=tuple(row[1] for row in bounded_rows),
        distances_um=tuple(row[0] for row in bounded_rows),
        local_observation_count=len(local_rows),
        nearest_distance_um=nearest_distance,
    )


def _prediction_diagnostics(
    support: _LocalSupport,
    request: FocusPredictionRequest,
    *,
    calculation: _PlaneCalculation | None = None,
    used_observations: tuple[FocusObservation, ...] | None = None,
    excluded_observation_ids: tuple[str, ...] = (),
    extrapolation_distance_um: float | None = None,
    predicted_z_um: float | None = None,
) -> FocusPredictionDiagnostics:
    """Build bounded support/fit diagnostics without claiming uncertainty."""
    used = calculation.observations if calculation is not None else used_observations or ()
    support_age_s = None
    if used:
        support_age_s = max(
            request.predicted_monotonic_s - item.recorded_monotonic_s for item in used
        )
    return FocusPredictionDiagnostics(
        candidate_observation_count=len(support.selection.observations),
        local_observation_count=support.local_observation_count,
        selected_support_count=len(used),
        nearest_distance_um=support.nearest_distance_um,
        support_age_s=support_age_s,
        fit_rmse_um=None if calculation is None else calculation.fit_rmse_um,
        maximum_absolute_residual_um=(
            None if calculation is None else calculation.maximum_absolute_residual_um
        ),
        fit_rank=None if calculation is None else calculation.rank,
        fit_condition_number=(None if calculation is None else calculation.condition_number),
        fit_inlier_fraction=(
            None
            if calculation is None
            else len(calculation.observations) / len(support.observations)
        ),
        plane_slope_x_um_per_um=(None if calculation is None else calculation.slope_x_um_per_um),
        plane_slope_y_um_per_um=(None if calculation is None else calculation.slope_y_um_per_um),
        plane_slope_magnitude_um_per_um=(
            None if calculation is None else calculation.slope_magnitude_um_per_um
        ),
        excluded_observation_ids=excluded_observation_ids,
        extrapolation_distance_um=extrapolation_distance_um,
        prediction_delta_um=(
            None if predicted_z_um is None else predicted_z_um - request.current_z_um
        ),
    )


def _unavailable_prediction(
    request: FocusPredictionRequest,
    unavailable_reason: FocusUnavailableReason,
    reason: str,
    diagnostics: FocusPredictionDiagnostics,
) -> FocusPrediction:
    """Return an optical support failure that may permit configured orchestration."""
    return FocusPrediction(
        scan_id=request.expected_scan_id,
        field_id=request.field_id,
        attempt_id=request.attempt_id,
        prediction_id=request.prediction_id,
        target_x_um=request.target_x_um,
        target_y_um=request.target_y_um,
        status="unavailable",
        reason=reason,
        unavailable_reason=unavailable_reason,
        binding=cast(FocusBinding, request.expected_binding),
        diagnostics=diagnostics,
        predicted_monotonic_s=request.predicted_monotonic_s,
    )


def _fault_prediction(
    request: FocusPredictionRequest, error: FocusObservationContractError
) -> FocusPrediction:
    """Preserve a narrow contract error as fault, never optical unavailable."""
    observation_detail = "" if error.observation_id is None else f" [{error.observation_id}]"
    return FocusPrediction(
        scan_id=request.expected_scan_id,
        field_id=request.field_id,
        attempt_id=request.attempt_id,
        prediction_id=request.prediction_id,
        target_x_um=request.target_x_um,
        target_y_um=request.target_y_um,
        status="fault",
        reason=f"{error.code}{observation_detail}: {error}",
        binding=request.expected_binding,
        fault_code=error.code,
        predicted_monotonic_s=request.predicted_monotonic_s,
    )


def _usable_prediction(
    request: FocusPredictionRequest,
    source: Literal["nearest_neighbor", "local_plane"],
    *,
    observations: tuple[FocusObservation, ...],
    distances_um: tuple[float, ...],
    predicted_z_um: float,
    diagnostics: FocusPredictionDiagnostics,
    extrapolated: bool = False,
) -> FocusPrediction:
    """Return one usable prediction with aligned support identities/distances."""
    return FocusPrediction(
        scan_id=request.expected_scan_id,
        field_id=request.field_id,
        attempt_id=request.attempt_id,
        prediction_id=request.prediction_id,
        target_x_um=request.target_x_um,
        target_y_um=request.target_y_um,
        target_z_um=predicted_z_um,
        status="usable",
        source=source,
        reason=f"Bounded {source.replace('_', ' ')} support passed all gates",
        binding=cast(FocusBinding, request.expected_binding),
        support_observation_ids=tuple(item.observation_id for item in observations),
        support_field_ids=tuple(item.field_id for item in observations),
        support_distances_um=distances_um,
        diagnostics=diagnostics,
        extrapolated=extrapolated,
        predicted_monotonic_s=request.predicted_monotonic_s,
    )


def _prediction_delta_is_usable(predicted_z_um: float, request: FocusPredictionRequest) -> bool:
    """Check prediction against the caller's explicit current physical Z."""
    maximum_delta = cast(float, request.settings.maximum_prediction_delta_um)
    return abs(predicted_z_um - request.current_z_um) <= maximum_delta


def _predict_from_nearest(
    support: _LocalSupport, request: FocusPredictionRequest
) -> FocusPrediction:
    """Use one bounded nearest observation without inventing transverse slope."""
    maximum_slope = cast(float, request.settings.maximum_plane_slope_um_per_um)
    maximum_residual = cast(float, request.settings.maximum_fit_residual_um)
    for index, left in enumerate(support.observations):
        for right in support.observations[index + 1 :]:
            separation_um = math.hypot(left.x_um - right.x_um, left.y_um - right.y_um)
            # Any admissible plane permits this slope plus one residual per point.
            # Sparse support cannot identify a plane, but can contradict its bounds.
            if abs(left.z_um - right.z_um) > (maximum_slope * separation_um + 2 * maximum_residual):
                return _unavailable_prediction(
                    request,
                    "support_discontinuity",
                    "Sparse local heights contradict configured slope/residual bounds",
                    _prediction_diagnostics(support, request),
                )
    nearest = support.observations[0]
    distance = support.distances_um[0]
    diagnostics = _prediction_diagnostics(
        support,
        request,
        used_observations=(nearest,),
        predicted_z_um=nearest.z_um,
    )
    if not _prediction_delta_is_usable(nearest.z_um, request):
        return _unavailable_prediction(
            request,
            "prediction_delta_exceeded",
            "Nearest focus exceeds the configured current-Z delta",
            diagnostics,
        )
    return _usable_prediction(
        request,
        "nearest_neighbor",
        observations=(nearest,),
        distances_um=(distance,),
        predicted_z_um=nearest.z_um,
        diagnostics=diagnostics,
    )


def _predict_from_plane(support: _LocalSupport, request: FocusPredictionRequest) -> FocusPrediction:
    """Fit, robustly gate, and geometrically bound a local plane."""
    outcome = _fit_bounded_plane(support.observations, request)
    calculation = outcome.calculation
    if outcome.unavailable_reason is not None:
        diagnostics = _prediction_diagnostics(support, request, calculation=calculation)
        reason = (
            "Local XY geometry or slope failed configured plane gates"
            if outcome.unavailable_reason == "poor_local_fit"
            else "Local heights do not identify one unambiguous plane"
        )
        return _unavailable_prediction(request, outcome.unavailable_reason, reason, diagnostics)

    extrapolation_distance = _extrapolation_distance_um(
        calculation.observations, request.target_x_um, request.target_y_um
    )
    predicted_z_um = cast(float, calculation.predicted_z_um)
    diagnostics = _prediction_diagnostics(
        support,
        request,
        calculation=calculation,
        excluded_observation_ids=outcome.excluded_observation_ids,
        extrapolation_distance_um=extrapolation_distance,
        predicted_z_um=predicted_z_um,
    )
    maximum_extrapolation = cast(float, request.settings.maximum_extrapolation_um)
    if extrapolation_distance > 0 and (
        not request.settings.allow_extrapolation or extrapolation_distance > maximum_extrapolation
    ):
        return _unavailable_prediction(
            request,
            "outside_support",
            "Target lies outside the configured local support hull",
            diagnostics,
        )
    if not _prediction_delta_is_usable(predicted_z_um, request):
        return _unavailable_prediction(
            request,
            "prediction_delta_exceeded",
            "Plane focus exceeds the configured current-Z delta",
            diagnostics,
        )
    support_distances = tuple(
        math.hypot(
            item.x_um - request.target_x_um,
            item.y_um - request.target_y_um,
        )
        for item in calculation.observations
    )
    return _usable_prediction(
        request,
        "local_plane",
        observations=calculation.observations,
        distances_um=support_distances,
        predicted_z_um=predicted_z_um,
        diagnostics=diagnostics,
        extrapolated=extrapolation_distance > 0,
    )


def _predict_with_active_surface(
    selection: _ObservationSelection, request: FocusPredictionRequest
) -> FocusPrediction:
    """Choose optical unavailable, nearest, or plane for an active valid context."""
    support = _local_support(selection, request)
    if not selection.observations:
        if selection.unique_observation_count == 0:
            unavailable_reason: FocusUnavailableReason = "no_observations"
            reason = "No observations have been recorded in this scan"
        elif selection.stale_observation_count:
            unavailable_reason = "stale_support"
            reason = "Accepted LED support is older than the frozen age limit"
        else:
            unavailable_reason = "insufficient_support"
            reason = "No recorded result is accepted verified LED support"
        return _unavailable_prediction(
            request,
            unavailable_reason,
            reason,
            _prediction_diagnostics(support, request),
        )
    if not support.observations:
        return _unavailable_prediction(
            request,
            "outside_support",
            "No accepted observation is within the frozen neighbor radius",
            _prediction_diagnostics(support, request),
        )
    if len(support.observations) < request.settings.minimum_plane_points:
        return _predict_from_nearest(support, request)
    return _predict_from_plane(support, request)


def predict_focus_surface(
    observations: Sequence[FocusObservation], request: FocusPredictionRequest
) -> FocusPrediction:
    """Return a deterministic bounded prediction without hardware or runtime access.

    The caller supplies a bounded candidate window, the actual physical target XY,
    and current physical Z.  Only ``FocusObservationContractError`` is converted to
    a fault result; unrelated programmer errors are never relabelled unavailable.
    """
    if not request.settings.enabled:
        return FocusPrediction(
            scan_id=request.expected_scan_id,
            field_id=request.field_id,
            attempt_id=request.attempt_id,
            prediction_id=request.prediction_id,
            target_x_um=request.target_x_um,
            target_y_um=request.target_y_um,
            status="disabled",
            reason="Focus surface is disabled in frozen scan settings",
            predicted_monotonic_s=request.predicted_monotonic_s,
        )
    try:
        if request.expected_binding is None:
            raise FocusObservationContractError(
                "binding_unusable",
                None,
                "Enabled focus prediction requires a frozen binding",
            )
        if len(observations) > request.settings.maximum_candidate_observations:
            raise FocusObservationContractError(
                "candidate_limit_exceeded",
                None,
                "Observation candidate window exceeds its frozen computational bound",
            )
        selection = _select_observation_support(
            observations,
            expected_scan_id=request.expected_scan_id,
            expected_binding=request.expected_binding,
            settings=request.settings,
            now_monotonic_s=request.predicted_monotonic_s,
        )
    except FocusObservationContractError as error:
        return _fault_prediction(request, error)
    return _predict_with_active_surface(selection, request)


class FocusFieldProgress(StrictModel):
    """Field-scoped WHITE-to-LED checkpoint data, not autofocus orchestration."""

    scan_id: Identifier
    field_id: Identifier
    attempt_id: Identifier
    prediction_id: Identifier
    stage: Literal["planned", "white_completed", "rg_verified"]
    white_search_id: Identifier
    white_search_count: Literal[1] = 1
    white_status: Literal["planned", "succeeded", "failed", "unknown"]
    white_result_id: Identifier | None = None
    rg_status: Literal["not_started", "succeeded", "failed", "unknown"]
    rg_result_id: Identifier | None = None
    automatic_white_retry_allowed: Literal[False] = False
    recorded_monotonic_s: float = Field(ge=0)

    @field_validator("white_search_count", mode="before")
    @classmethod
    def one_white_search_only(cls, value: object) -> object:
        """Reject bool/coerced/repeated WHITE search counts."""
        if type(value) is not int or value != 1:
            raise ValueError("Exactly one WHITE search is recorded per field attempt")
        return value

    def _validate_result_ids(self) -> None:
        """Require result identities exactly when an operation completed."""
        white_has_result = self.white_status in ("succeeded", "failed")
        if white_has_result != (self.white_result_id is not None):
            raise ValueError("WHITE completion status and result ID disagree")
        rg_has_result = self.rg_status in ("succeeded", "failed")
        if rg_has_result != (self.rg_result_id is not None):
            raise ValueError("R/G completion status and result ID disagree")

    @model_validator(mode="after")
    def sequential_checkpoint(self) -> Self:
        """Reject impossible stage/outcome combinations and hidden retries."""
        self._validate_result_ids()
        if self.stage == "planned":
            if self.white_status == "succeeded" or self.rg_status != "not_started":
                raise ValueError("Planned stage cannot claim completed WHITE or R/G")
        elif self.stage == "white_completed":
            if self.white_status != "succeeded" or self.rg_status == "succeeded":
                raise ValueError("white_completed requires WHITE success before R/G")
        elif self.white_status != "succeeded" or self.rg_status != "succeeded":
            raise ValueError("rg_verified requires successful WHITE and R/G results")
        return self
