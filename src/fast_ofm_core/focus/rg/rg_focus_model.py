"""Fit and apply the bounded signed 2D R/G shift-to-Z calibration model."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Iterable
from typing import Literal, Self, cast

import numpy as np
from pydantic import AliasChoices, Field, model_validator

from fast_ofm_core.contracts import StrictModel

from .rg_focus_core import RGFocusCoreSettings
from .rg_focus_estimator import MeasurementStatus, RGFocusEstimatorSettings
from .rg_focus_field import TissueFieldSettings

CalibrationRole = Literal["fit", "holdout", "approach_above", "approach_below"]
ApproachRole = Literal["approach_above", "approach_below"]

_SUPPORTED_GRID_ROLE_SETS: tuple[dict[float, str], ...] = (
    {
        -8.0: "fit",
        -6.0: "holdout",
        -4.0: "fit",
        -2.0: "holdout",
        0.0: "fit",
        2.0: "holdout",
        4.0: "fit",
        6.0: "holdout",
        8.0: "fit",
    },
    {
        -9.0: "fit",
        -6.0: "holdout",
        -3.0: "fit",
        0.0: "fit",
        3.0: "fit",
        6.0: "holdout",
        9.0: "fit",
    },
    {
        -12.0: "fit",
        -9.0: "holdout",
        -6.0: "fit",
        -3.0: "holdout",
        0.0: "fit",
        3.0: "holdout",
        6.0: "fit",
        9.0: "holdout",
        12.0: "fit",
    },
    {
        -16.0: "fit",
        -12.0: "holdout",
        -8.0: "fit",
        -4.0: "holdout",
        0.0: "fit",
        4.0: "holdout",
        8.0: "fit",
        12.0: "holdout",
        16.0: "fit",
    },
    {
        -32.0: "fit",
        -28.0: "holdout",
        -24.0: "fit",
        -20.0: "holdout",
        -16.0: "fit",
        -12.0: "holdout",
        -8.0: "fit",
        -4.0: "holdout",
        0.0: "fit",
        4.0: "holdout",
        8.0: "fit",
        12.0: "holdout",
        16.0: "fit",
        20.0: "holdout",
        24.0: "fit",
        28.0: "holdout",
        32.0: "fit",
    },
)


def _supported_grid_roles(
    rows: Iterable[RGFocusCalibrationPoint | RGFocusModelObservation],
) -> bool:
    """Accept only one complete legacy or widened fit/holdout layout."""
    grid = list(rows)
    observed = {(float(row.z_um), row.role) for row in grid}
    return any(
        len(grid) == len(expected) and observed == set(expected.items())
        for expected in _SUPPORTED_GRID_ROLE_SETS
    )


def _required_half_range(
    rows: Iterable[RGFocusCalibrationPoint | RGFocusModelObservation],
) -> float:
    """Return the validated interpolation range for one supported grid."""
    maximum = max(abs(float(row.z_um)) for row in rows)
    return 6.0 if maximum <= 8.0 else maximum


JPEG_DRAFT_POLICY_SHA256 = "82ceeb734f77c07260dcf72fccb5ed03eee4a876721386d9f452298c65a415fb"
JPEG_DRAFT_POLICY_ID = (
    f"ofm-rg-focus-jpeg8-imx477-4x-roi810x760-draft-v1+sha256-{JPEG_DRAFT_POLICY_SHA256}"
)


class RGFocusCalibrationPoint(StrictModel):
    """One preassigned calibration observation, including rejected measurements."""

    capture_id: str = Field(min_length=1)
    role: CalibrationRole
    z_um: float
    measurement_status: MeasurementStatus
    dx: float | None = None
    dy: float | None = None
    confidence: float = Field(ge=0, le=1)
    white_score: float = Field(gt=0)

    @model_validator(mode="after")
    def ready_has_shift(self) -> Self:
        """Require a complete finite shift vector for every ready observation."""
        if self.measurement_status == "ready" and (self.dx is None or self.dy is None):
            raise ValueError("A ready calibration point is missing its R/G shift")
        return self


class RGFocusModelSettings(StrictModel):
    """Predeclared validation and approach-repeatability requirements."""

    minimum_valid_fit_points: int = Field(default=5, ge=3)
    minimum_valid_holdout_points: int = Field(default=2, ge=2)
    maximum_holdout_error_um: float = Field(default=4.0, gt=0)
    minimum_slope_norm_px_per_um: float = Field(default=0.05, gt=0)
    maximum_approach_shift_difference_px: float = Field(default=1.0, gt=0)
    maximum_return_error_um: float = Field(default=2.0, gt=0)
    maximum_return_prediction_difference_um: float = Field(default=2.0, gt=0)
    maximum_stationary_prediction_difference_um: float = Field(default=2.0, gt=0)
    minimum_white_score_fraction: float = Field(default=0.85, gt=0, le=1)
    maximum_reference_focus_offset_um: float = Field(default=12.0, gt=0)
    maximum_approach_white_score_fraction: float = Field(default=0.15, gt=0, le=1)


class RGFocusMeasurementPolicy(StrictModel):
    """Freeze every tissue and shift QC parameter used with a model profile."""

    measurement_domain: Literal["legacy-raw", "processed-jpeg-rgb8"] = "legacy-raw"
    legacy_raw_saturation_level: float | None = Field(default=None, gt=0)
    core: RGFocusCoreSettings = Field(default_factory=RGFocusCoreSettings)
    estimator: RGFocusEstimatorSettings = Field(default_factory=RGFocusEstimatorSettings)
    tissue_field: TissueFieldSettings = Field(default_factory=TissueFieldSettings)

    @model_validator(mode="before")
    @classmethod
    def preserve_compatible_schema(cls, value: object) -> object:
        """Load historical evidence while removing only retired derived fields."""
        if not isinstance(value, dict):
            return value
        result = copy.deepcopy(value)
        core = result.get("core")
        if isinstance(core, dict):
            core.pop("minimum_measurement_response", None)
        if isinstance(core, dict) and "saturation_level" in core:
            if result.get("measurement_domain", "legacy-raw") != "legacy-raw":
                raise ValueError("RAW saturation cannot be used in a JPEG policy")
            result["measurement_domain"] = "legacy-raw"
            result["legacy_raw_saturation_level"] = core.pop("saturation_level")
        return result

    def require_jpeg(self) -> None:
        """Refuse historical policies instead of guessing JPEG8 DN/pixel settings."""
        if (
            self.measurement_domain != "processed-jpeg-rgb8"
            or self.legacy_raw_saturation_level is not None
        ):
            raise ValueError("Save an explicit processed-JPEG RGB8 focus policy")


def canonical_measurement_policy_bytes(policy: RGFocusMeasurementPolicy) -> bytes:
    """Return the policy-only canonical bytes used by the reviewed draft artifact."""
    payload = json.dumps(
        policy.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"{payload}\n".encode()


def measurement_policy_sha256(policy: RGFocusMeasurementPolicy) -> str:
    """Identify a complete normalized measurement policy, including its final LF."""
    return hashlib.sha256(canonical_measurement_policy_bytes(policy)).hexdigest()


def jpeg_measurement_draft_policy() -> RGFocusMeasurementPolicy:
    """Build the complete source-owned, uncommissioned JPEG draft policy."""
    policy = RGFocusMeasurementPolicy.model_validate(
        {
            "measurement_domain": "processed-jpeg-rgb8",
            "legacy_raw_saturation_level": None,
            "core": {
                "registration_highpass_sigma_px": 4.0,
                "mutual_information_bins": 32,
                "texture_background_sigma_px": 6.0,
                "texture_score_sigma_px": 2.0,
                "texture_gradient_weight": 0.25,
                "texture_percentile": 55.0,
                "texture_morphology_kernel_px": 3,
                "patch_size_px": 128,
                "patch_stride_px": 64,
                "minimum_tissue_coverage": 0.35,
                "minimum_patch_signal": 8.0,
                "jpeg8_saturation_level": 250.0,
                "maximum_patch_saturation_fraction": 0.01,
                "minimum_patch_std": 4.0,
                "minimum_patch_response": 0.05,
                "minimum_patch_peak_margin": 0.03,
                "minimum_patch_spectral_correlation": 0.05,
                "maximum_absolute_shift_px": 24.0,
                "maximum_absolute_orthogonal_shift_px": 4.0,
                "masked_inpaint_radius_px": 1.5,
                "minimum_shifted_support_fraction": 0.5,
                "maximum_patch_count": 24,
                "minimum_patch_count": 9,
            },
            "estimator": {
                "inlier_aggregation": "arithmetic_mean",
                "minimum_inlier_fraction": 0.6,
                "outlier_mad_multiplier": 2.5,
                "minimum_outlier_threshold_px": 1.0,
                "maximum_dx_mad_px": 1.0,
                "maximum_dy_mad_px": 1.0,
                "maximum_radial_p90_px": 3.25,
                "minimum_confidence": 0.05,
            },
            "tissue_field": {
                "edge_margin_px": 128,
                "minimum_white_median": 16.0,
                "white_saturation_level": 250.0,
                "maximum_white_saturation_fraction": 0.02,
                "coherent_texture_blur_sigma_px": 0.75,
                "coherent_texture_background_sigma_px": 4.0,
                "minimum_coherent_texture_fraction": 0.01,
                "minimum_component_pixels": 512,
                "minimum_tissue_fraction": 0.03,
            },
        }
    )
    if measurement_policy_sha256(policy) != JPEG_DRAFT_POLICY_SHA256:
        raise RuntimeError("Source JPEG draft policy differs from its immutable identity")
    return policy


class RGFocusStationaryObservation(StrictModel):
    """One no-motion commissioning result bound to its frozen field identity."""

    capture_id: str = Field(min_length=1)
    measurement_status: MeasurementStatus
    dx: float | None = None
    dy: float | None = None
    white_score: float = Field(gt=0)
    accepted_patch_count: int = Field(ge=0)
    inlier_fraction: float = Field(ge=0, le=1)
    field_id: str = Field(min_length=1)
    geometry_id: str = Field(min_length=1)
    binding_sha256: str = Field(min_length=64, max_length=64)
    mask_sha256: str = Field(min_length=64, max_length=64)
    window_ids: tuple[str, ...] = Field(strict=False)

    @model_validator(mode="after")
    def ready_has_shift(self) -> Self:
        """Require a finite 2D shift for every ready stationary result."""
        if self.measurement_status == "ready" and (
            self.dx is None
            or self.dy is None
            or not math.isfinite(self.dx)
            or not math.isfinite(self.dy)
        ):
            raise ValueError("A ready stationary observation is missing its R/G shift")
        return self


class RGFocusApproachCheck(StrictModel):
    """Projected-Z and WHITE evidence from both final-approach returns."""

    status: Literal["passed", "failed"]
    reason: str
    projected_z_um: dict[ApproachRole, float]
    absolute_target_errors_um: dict[ApproachRole, float]
    mutual_prediction_difference_um: float | None = Field(default=None, ge=0)
    white_scores: dict[ApproachRole, float]
    white_to_reference_fractions: dict[ApproachRole, float]
    white_mutual_fraction: float | None = Field(default=None, ge=0)
    target_z_um: float = 0.0
    maximum_return_error_um: float = Field(gt=0)
    maximum_mutual_prediction_difference_um: float = Field(gt=0)
    minimum_white_score_fraction: float = Field(gt=0, le=1)
    maximum_white_mutual_fraction: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def passed_is_exact(self) -> Self:
        """Reject a serialized pass that loosens or omits a frozen return gate."""
        if self.status == "passed" and (
            self.target_z_um != 0
            or self.maximum_return_error_um != 2.0
            or self.maximum_mutual_prediction_difference_um != 2.0
            or self.minimum_white_score_fraction != 0.85
            or self.maximum_white_mutual_fraction != 0.15
            or set(self.projected_z_um) != {"approach_above", "approach_below"}
            or set(self.absolute_target_errors_um) != {"approach_above", "approach_below"}
            or any(value > 2.0 for value in self.absolute_target_errors_um.values())
            or self.mutual_prediction_difference_um is None
            or self.mutual_prediction_difference_um > 2.0
            or set(self.white_to_reference_fractions) != {"approach_above", "approach_below"}
            or any(value < 0.85 for value in self.white_to_reference_fractions.values())
            or self.white_mutual_fraction is None
            or self.white_mutual_fraction > 0.15
        ):
            raise ValueError("Passed return evidence exceeds a frozen JPEG gate")
        return self


class RGFocusActivationEvidence(StrictModel):
    """Complete prospective gate ledger required by every active JPEG model."""

    contract_id: Literal["jpeg-demo-calibration-v1"] = "jpeg-demo-calibration-v1"
    status: Literal["passed", "failed"]
    failures: tuple[str, ...] = Field(strict=False)
    exact_grid_and_returns: bool
    all_grid_and_returns_ready: bool
    design_rank: int = Field(ge=0)
    slope_finite_nonzero: bool
    applicable_range_covers_contract: bool = Field(
        validation_alias=AliasChoices(
            "applicable_range_covers_contract",
            "applicable_range_covers_minus6_plus6",
        )
    )
    reference_white_score: float = Field(gt=0)
    maximum_grid_white_score: float = Field(gt=0)
    maximum_grid_white_z_um: float
    grid_white_maximum_z_positions_um: tuple[float, ...] = Field(strict=False)
    maximum_reference_focus_offset_um: float = Field(gt=0)
    reference_white_fraction: float = Field(ge=0)
    grid_white_maximum_internal: bool
    grid_white_maximum_has_measured_neighbours: bool
    approach: RGFocusApproachCheck
    stationary_observations: tuple[RGFocusStationaryObservation, RGFocusStationaryObservation] = (
        Field(strict=False)
    )
    stationary_identity_match: bool
    stationary_projected_z_um: tuple[float, float] | None = Field(default=None, strict=False)
    stationary_prediction_difference_um: float | None = Field(default=None, ge=0)
    maximum_stationary_prediction_difference_um: float = Field(gt=0)

    @model_validator(mode="after")
    def passed_is_complete(self) -> Self:
        """Reject a serialized valid decision that omits or relabels a gate."""
        reference_is_strictly_sharper = self.reference_white_score > self.maximum_grid_white_score
        if self.status == "passed" and (
            self.failures
            or not self.exact_grid_and_returns
            or not self.all_grid_and_returns_ready
            or self.design_rank != 2
            or not self.slope_finite_nonzero
            or not self.applicable_range_covers_contract
            or (
                not reference_is_strictly_sharper
                and (
                    not self.grid_white_maximum_internal
                    or not self.grid_white_maximum_has_measured_neighbours
                    or len(self.grid_white_maximum_z_positions_um) != 1
                    or self.maximum_grid_white_z_um != self.grid_white_maximum_z_positions_um[0]
                    or abs(self.maximum_grid_white_z_um) > self.maximum_reference_focus_offset_um
                )
            )
            or self.maximum_reference_focus_offset_um != 12.0
            or self.approach.status != "passed"
            or not self.stationary_identity_match
            or self.stationary_prediction_difference_um is None
            or self.stationary_prediction_difference_um
            > self.maximum_stationary_prediction_difference_um
            or self.maximum_stationary_prediction_difference_um != 2.0
            or any(
                observation.measurement_status != "ready"
                or observation.accepted_patch_count < 9
                or observation.inlier_fraction < 0.6
                for observation in self.stationary_observations
            )
            or self.stationary_observations[0].capture_id
            == self.stationary_observations[1].capture_id
            or abs(
                self.stationary_observations[0].white_score
                - self.stationary_observations[1].white_score
            )
            / max(
                self.stationary_observations[0].white_score,
                self.stationary_observations[1].white_score,
            )
            > 0.15
        ):
            raise ValueError("Passed JPEG activation evidence contains a failed gate")
        return self


class RGFocusModelObservation(StrictModel):
    """One accepted source observation for the frozen empirical model envelopes."""

    capture_id: str = Field(min_length=1)
    role: Literal["fit", "holdout", "approach_above", "approach_below", "stationary"]
    z_um: float
    dx: float
    dy: float
    signed_cross_track_residual_px: float


def _cross_track(
    offset: tuple[float, float], slope: tuple[float, float], dx: float, dy: float
) -> tuple[float, float]:
    """Return signed residual and only a scale-aware floating-point error allowance."""
    norm = math.hypot(*slope)
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Cross-track residual requires a finite nonzero slope")
    normal = (-slope[1] / norm, slope[0] / norm)
    residual = (dx - offset[0]) * normal[0] + (dy - offset[1]) * normal[1]
    roundoff = 32 * np.finfo(float).eps * (abs(dx) + abs(dy) + abs(offset[0]) + abs(offset[1]))
    return residual, float(roundoff)


class RGFocusModelProfile(StrictModel):
    """A serialisable signed model with validation scope and activation decision."""

    id: str = Field(min_length=1)
    status: Literal["candidate", "valid"]
    reason: str
    source_series_id: str = Field(min_length=1)
    method: Literal["linear_2d_shift_projection"] = "linear_2d_shift_projection"
    offset_px: tuple[float, float] = Field(strict=False)
    slope_px_per_um: tuple[float, float] = Field(strict=False)
    slope_norm_px_per_um: float = Field(ge=0)
    focus_z_um: float | None
    empirical_observations: tuple[RGFocusModelObservation, ...] = Field(strict=False)
    empirical_cross_track_limit_px: float | None = Field(ge=0)
    empirical_prediction_error_um: float | None = Field(ge=0)
    fit_capture_ids: tuple[str, ...] = Field(strict=False)
    holdout_capture_ids: tuple[str, ...] = Field(strict=False)
    rejected_capture_reasons: dict[str, str]
    fit_z_range_um: tuple[float, float] = Field(strict=False)
    holdout_z_range_um: tuple[float, float] = Field(strict=False)
    applicable_z_range_um: tuple[float, float] = Field(strict=False)
    fit_rmse_um: float | None = Field(default=None, ge=0)
    holdout_rmse_um: float | None = Field(default=None, ge=0)
    holdout_max_absolute_error_um: float | None = Field(default=None, ge=0)
    holdout_absolute_error_p95_um: float | None = Field(default=None, ge=0)
    holdout_predictions_um: dict[str, float]
    holdout_errors_um: dict[str, float]
    approach_check: RGFocusApproachCheck
    activation_evidence: RGFocusActivationEvidence
    compatibility: dict[str, object]
    source_evidence: dict[str, str]
    measurement_policy: RGFocusMeasurementPolicy
    settings: RGFocusModelSettings

    @model_validator(mode="after")
    def ordered_ranges(self) -> Self:
        """Prevent a profile from publishing an empty or inverted calibration scope."""
        for low, high in (
            self.fit_z_range_um,
            self.holdout_z_range_um,
            self.applicable_z_range_um,
        ):
            if low >= high:
                raise ValueError("R/G model ranges must have positive width")
        if self.status == "valid":
            grid_rows = [
                row for row in self.empirical_observations if row.role in ("fit", "holdout")
            ]
            required_half_range = _required_half_range(grid_rows)
            if self.activation_evidence.status != "passed":
                raise ValueError("Valid profile requires passed JPEG activation evidence")
            if self.slope_norm_px_per_um <= 0:
                raise ValueError("Valid profile requires a finite non-zero signed slope")
            expected_focus_z_um = (
                0.0
                if self.activation_evidence.reference_white_score
                > self.activation_evidence.maximum_grid_white_score
                else self.activation_evidence.maximum_grid_white_z_um
            )
            if (
                self.holdout_max_absolute_error_um is None
                or self.holdout_max_absolute_error_um > 2.0
                or self.applicable_z_range_um[0] > -required_half_range
                or self.applicable_z_range_um[1] < required_half_range
                or self.approach_check != self.activation_evidence.approach
                or self.focus_z_um is None
                or self.focus_z_um != expected_focus_z_um
            ):
                raise ValueError("Valid profile exceeds a frozen JPEG model gate")
            self._validate_empirical_evidence()
        return self

    def _validate_empirical_evidence(self) -> None:
        """Recompute both immutable envelopes from the complete supported source rows."""
        rows = self.empirical_observations
        by_id = {row.capture_id: row for row in rows}
        role_ids = {
            role: {row.capture_id for row in rows if row.role == role}
            for role in (
                "fit",
                "holdout",
                "approach_above",
                "approach_below",
                "stationary",
            )
        }
        stationary = self.activation_evidence.stationary_observations
        grid_rows = [row for row in rows if row.role in ("fit", "holdout")]
        if (
            len(rows) != len(grid_rows) + 4
            or len(by_id) != len(rows)
            or role_ids["fit"] != set(self.fit_capture_ids)
            or role_ids["holdout"] != set(self.holdout_capture_ids)
            or not _supported_grid_roles(grid_rows)
            or set(self.holdout_errors_um) != role_ids["holdout"]
            or len(role_ids["approach_above"]) != 1
            or len(role_ids["approach_below"]) != 1
            or role_ids["stationary"] != {row.capture_id for row in stationary}
            or self.empirical_cross_track_limit_px is None
            or self.empirical_prediction_error_um is None
        ):
            raise ValueError("Valid model requires complete supported empirical observations")
        errors: list[float] = []
        residuals: list[float] = []
        allowances: list[float] = []
        for row in rows:
            residual, allowance = _cross_track(self.offset_px, self.slope_px_per_um, row.dx, row.dy)
            if abs(row.signed_cross_track_residual_px - residual) > allowance:
                raise ValueError("Saved cross-track residual does not match its observation")
            residuals.append(abs(residual))
            allowances.append(allowance)
            predicted = estimate_z_um(self, row.dx, row.dy)
            if row.role in ("holdout", "approach_above", "approach_below"):
                error = predicted - row.z_um
                if row.role == "holdout":
                    saved_error = self.holdout_errors_um[row.capture_id]
                else:
                    if row.z_um != 0:
                        raise ValueError("Return empirical observation must use commanded zero")
                    saved_error = self.approach_check.projected_z_um[row.role]
                if not math.isclose(
                    error,
                    saved_error,
                    rel_tol=32 * np.finfo(float).eps,
                    abs_tol=32 * np.finfo(float).eps,
                ):
                    raise ValueError("Saved prediction error does not match its observation")
                errors.append(abs(error))
        for observation in stationary:
            row = by_id[observation.capture_id]
            if row.z_um != 0 or (row.dx, row.dy) != (observation.dx, observation.dy):
                raise ValueError("Stationary empirical observation changed")
        if abs(self.empirical_cross_track_limit_px - max(residuals)) > max(allowances):
            raise ValueError("Saved cross-track envelope is not the observed maximum")
        if not math.isclose(
            self.empirical_prediction_error_um,
            max(errors),
            rel_tol=32 * np.finfo(float).eps,
            abs_tol=32 * np.finfo(float).eps,
        ):
            raise ValueError("Saved empirical prediction error is not the independent maximum")

    def effective_focus_tolerance_um(self, total_tolerance_um: float) -> float:
        """Reserve the observed model error from one total empirical focus budget."""
        error = self.empirical_prediction_error_um
        if error is None or not math.isfinite(error) or error >= total_tolerance_um:
            raise ValueError("Empirical prediction error exhausts the total focus budget")
        return total_tolerance_um - error

    def cross_track_residual_px(self, dx: float, dy: float) -> tuple[float, float]:
        """Return the live signed residual and machine-roundoff allowance, not noise."""
        return _cross_track(self.offset_px, self.slope_px_per_um, dx, dy)

    @property
    def applicable_error_range_um(self) -> tuple[float, float]:
        """Translate the validated commanded-Z interval around the WHITE target."""
        if self.focus_z_um is None:
            raise ValueError("R/G model has no unambiguous WHITE focus target")
        return (
            self.applicable_z_range_um[0] - self.focus_z_um,
            self.applicable_z_range_um[1] - self.focus_z_um,
        )


def _usable(point: RGFocusCalibrationPoint) -> bool:
    """Return whether a calibration point has a QC-approved finite shift."""
    return (
        point.measurement_status == "ready"
        and point.dx is not None
        and point.dy is not None
        and math.isfinite(point.dx)
        and math.isfinite(point.dy)
    )


def estimate_z_um(profile: RGFocusModelProfile, dx: float, dy: float) -> float:
    """Infer commanded calibration Z; subtract focus_z_um for the focus error."""
    shift = np.asarray([dx, dy], dtype=np.float64)
    offset = np.asarray(profile.offset_px, dtype=np.float64)
    slope = np.asarray(profile.slope_px_per_um, dtype=np.float64)
    if not np.isfinite(shift).all():
        raise ValueError("R/G shift must be finite")
    if not np.isfinite(slope).all() or float(np.dot(slope, slope)) <= 0:
        raise ValueError("R/G model does not have a finite non-zero signed slope")
    return float(np.dot(shift - offset, slope) / np.dot(slope, slope))


def _approach_check(
    points: list[RGFocusCalibrationPoint],
    settings: RGFocusModelSettings,
    *,
    offset_px: tuple[float, float],
    slope_px_per_um: tuple[float, float],
    reference_white_score: float,
) -> RGFocusApproachCheck:
    """Compare both returns to fitted target Z and frozen WHITE gates."""
    above_points = [point for point in points if point.role == "approach_above"]
    below_points = [point for point in points if point.role == "approach_below"]
    rows: dict[ApproachRole, RGFocusCalibrationPoint] = {}
    if len(above_points) == 1:
        rows["approach_above"] = above_points[0]
    if len(below_points) == 1:
        rows["approach_below"] = below_points[0]
    independent_ready = (
        set(rows) == {"approach_above", "approach_below"}
        and rows["approach_above"].capture_id != rows["approach_below"].capture_id
        and all(_usable(point) and point.z_um == 0 for point in rows.values())
    )
    slope = np.asarray(slope_px_per_um, dtype=np.float64)
    offset = np.asarray(offset_px, dtype=np.float64)
    denominator = float(np.dot(slope, slope))
    projection_available = (
        independent_ready
        and np.isfinite(slope).all()
        and np.isfinite(offset).all()
        and denominator > 0
    )
    projected: dict[ApproachRole, float] = {}
    if projection_available:
        projected = {
            role: float(
                np.dot(
                    np.asarray([point.dx, point.dy], dtype=np.float64) - offset,
                    slope,
                )
                / denominator
            )
            for role, point in rows.items()
        }
    absolute_errors = {role: abs(value) for role, value in projected.items()}
    mutual_prediction = (
        abs(projected["approach_above"] - projected["approach_below"])
        if len(projected) == 2
        else None
    )
    white_scores: dict[ApproachRole, float] = {
        role: point.white_score for role, point in rows.items()
    }
    white_fractions = {role: value / reference_white_score for role, value in white_scores.items()}
    white_mutual = (
        abs(white_scores["approach_above"] - white_scores["approach_below"])
        / max(white_scores.values())
        if len(white_scores) == 2
        else None
    )
    passed = bool(
        independent_ready
        and len(projected) == 2
        and all(value <= settings.maximum_return_error_um for value in absolute_errors.values())
        and mutual_prediction is not None
        and mutual_prediction <= settings.maximum_return_prediction_difference_um
        and all(
            value >= settings.minimum_white_score_fraction for value in white_fractions.values()
        )
        and white_mutual is not None
        and white_mutual <= settings.maximum_approach_white_score_fraction
    )
    return RGFocusApproachCheck(
        status="passed" if passed else "failed",
        reason=(
            "Both returns pass projected-target and WHITE gates"
            if passed
            else "Two-sided return projected-target or WHITE gate failed"
        ),
        projected_z_um=projected,
        absolute_target_errors_um=absolute_errors,
        mutual_prediction_difference_um=mutual_prediction,
        white_scores=white_scores,
        white_to_reference_fractions=white_fractions,
        white_mutual_fraction=white_mutual,
        maximum_return_error_um=settings.maximum_return_error_um,
        maximum_mutual_prediction_difference_um=(settings.maximum_return_prediction_difference_um),
        minimum_white_score_fraction=settings.minimum_white_score_fraction,
        maximum_white_mutual_fraction=(settings.maximum_approach_white_score_fraction),
    )


def fit_rg_focus_model(
    points: list[RGFocusCalibrationPoint],
    settings: RGFocusModelSettings,
    *,
    reference_white_score: float,
    stationary_observations: tuple[RGFocusStationaryObservation, RGFocusStationaryObservation],
    profile_id: str,
    source_series_id: str,
    compatibility: dict[str, object],
    source_evidence: dict[str, str] | None = None,
    measurement_policy: RGFocusMeasurementPolicy | None = None,
) -> RGFocusModelProfile:
    """Fit the fixed JPEG split and retain every prospective activation gate."""
    if not math.isfinite(reference_white_score) or reference_white_score <= 0:
        raise ValueError("Reference WHITE score must be finite and positive")
    grid = [point for point in points if point.role in ("fit", "holdout")]
    returns = [point for point in points if point.role in ("approach_above", "approach_below")]
    exact_grid_and_returns = (
        len(returns) == 2
        and len({point.capture_id for point in grid + returns}) == len(grid) + 2
        and _supported_grid_roles(grid)
        and {point.role for point in returns} == {"approach_above", "approach_below"}
        and all(point.z_um == 0 for point in returns)
    )
    all_grid_and_returns_ready = exact_grid_and_returns and all(
        _usable(point) for point in grid + returns
    )
    fit = [point for point in points if point.role == "fit" and _usable(point)]
    holdout = [point for point in points if point.role == "holdout" and _usable(point)]
    if len(fit) < 3:
        raise ValueError("At least three QC-approved fit points are required")
    if len(holdout) < 2:
        raise ValueError("At least two QC-approved holdout points are required")
    if not (min(point.z_um for point in fit) < 0 < max(point.z_um for point in fit)):
        raise ValueError("Fit observations must span both sides of focus")
    if not (min(point.z_um for point in holdout) < 0 < max(point.z_um for point in holdout)):
        raise ValueError("Holdout observations must span both sides of focus")

    fit_z = np.asarray([point.z_um for point in fit], dtype=np.float64)
    fit_shift = np.asarray([[point.dx, point.dy] for point in fit], dtype=np.float64)
    design = np.column_stack([np.ones(len(fit_z)), fit_z])
    coefficients, _, rank, _ = np.linalg.lstsq(design, fit_shift, rcond=None)
    offset = coefficients[0]
    slope = coefficients[1]
    slope_norm = float(np.linalg.norm(slope))
    numerical_zero = (
        np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(offset))))
        * max(1.0, float(np.max(np.abs(fit_z))))
        * len(fit_z)
    )
    slope_finite_nonzero = bool(
        rank == 2
        and np.isfinite(offset).all()
        and np.isfinite(slope).all()
        and math.isfinite(slope_norm)
        and slope_norm > numerical_zero
    )

    def infer(point: RGFocusCalibrationPoint) -> float:
        shift = np.asarray([point.dx, point.dy], dtype=np.float64)
        return float(np.dot(shift - offset, slope) / np.dot(slope, slope))

    fit_errors = (
        np.asarray([infer(point) - point.z_um for point in fit]) if slope_finite_nonzero else None
    )
    predictions = (
        {point.capture_id: infer(point) for point in holdout} if slope_finite_nonzero else {}
    )
    errors = (
        {point.capture_id: predictions[point.capture_id] - point.z_um for point in holdout}
        if slope_finite_nonzero
        else {}
    )
    holdout_abs = np.abs(np.asarray(list(errors.values()))) if errors else np.asarray([])
    approach = _approach_check(
        points,
        settings,
        offset_px=(float(offset[0]), float(offset[1])),
        slope_px_per_um=(float(slope[0]), float(slope[1])),
        reference_white_score=reference_white_score,
    )
    rejected: dict[str, str] = {
        point.capture_id: point.measurement_status
        for point in points
        if point.role in ("fit", "holdout") and not _usable(point)
    }
    fit_range = (min(point.z_um for point in fit), max(point.z_um for point in fit))
    holdout_range = (
        min(point.z_um for point in holdout),
        max(point.z_um for point in holdout),
    )
    required_half_range = _required_half_range(grid)
    # Widened fit points bound interpolation at both edges; independent holdouts
    # test the same line close to those edges without reusing fit data.
    applicable_range = (
        fit_range
        if required_half_range > 6.0
        else (
            max(fit_range[0], holdout_range[0]),
            min(fit_range[1], holdout_range[1]),
        )
    )
    applicable_range_covers = (
        applicable_range[0] <= -required_half_range and applicable_range[1] >= required_half_range
    )
    maximum_grid_white = max(point.white_score for point in grid)
    maximum_grid_points = [point for point in grid if point.white_score == maximum_grid_white]
    white_maximum_positions = tuple(point.z_um for point in maximum_grid_points)
    grid_by_z = {point.z_um: point for point in grid}
    minimum_grid_z = min(grid_by_z)
    maximum_grid_z = max(grid_by_z)
    internal_maximum = bool(
        maximum_grid_points
        and all(minimum_grid_z < point.z_um < maximum_grid_z for point in maximum_grid_points)
    )
    measured_neighbours = bool(maximum_grid_points)
    for point in maximum_grid_points:
        lower = [z for z in grid_by_z if z < point.z_um]
        upper = [z for z in grid_by_z if z > point.z_um]
        measured_neighbours = bool(
            measured_neighbours
            and lower
            and upper
            and _usable(grid_by_z[max(lower)])
            and _usable(grid_by_z[min(upper)])
        )
    reference_white_fraction = reference_white_score / maximum_grid_white
    reference_is_strictly_sharper = reference_white_score > maximum_grid_white
    focus_z_um = (
        0.0
        if reference_is_strictly_sharper
        else (white_maximum_positions[0] if len(white_maximum_positions) == 1 else None)
    )

    stationary_identity_match = stationary_observations[0].capture_id != stationary_observations[
        1
    ].capture_id and all(
        getattr(stationary_observations[0], name) == getattr(stationary_observations[1], name)
        for name in (
            "field_id",
            "geometry_id",
            "binding_sha256",
            "mask_sha256",
            "window_ids",
        )
    )
    stationary_ready = all(
        observation.measurement_status == "ready"
        and observation.dx is not None
        and observation.dy is not None
        for observation in stationary_observations
    )
    stationary_white_mutual = abs(
        stationary_observations[0].white_score - stationary_observations[1].white_score
    ) / max(
        stationary_observations[0].white_score,
        stationary_observations[1].white_score,
    )
    stationary_projected = cast(
        tuple[float, float] | None,
        tuple(
            float(
                np.dot(
                    np.asarray([observation.dx, observation.dy], dtype=np.float64) - offset,
                    slope,
                )
                / np.dot(slope, slope)
            )
            for observation in stationary_observations
        )
        if stationary_ready and slope_finite_nonzero
        else None,
    )
    stationary_prediction_difference = (
        abs(stationary_projected[0] - stationary_projected[1])
        if stationary_projected is not None
        else None
    )
    failures: list[str] = []
    if (
        settings.maximum_holdout_error_um != 2.0
        or settings.maximum_return_error_um != 2.0
        or settings.maximum_return_prediction_difference_um != 2.0
        or settings.maximum_stationary_prediction_difference_um != 2.0
        or settings.minimum_white_score_fraction != 0.85
        or settings.maximum_approach_white_score_fraction != 0.15
        or settings.maximum_reference_focus_offset_um != 12.0
    ):
        failures.append("model_gate_contract")
    if not exact_grid_and_returns:
        failures.append("exact_grid_and_returns")
    if not all_grid_and_returns_ready:
        failures.append("all_grid_and_returns_ready")
    if rank != 2:
        failures.append("rank_2")
    if not slope_finite_nonzero:
        failures.append("slope_finite_nonzero")
    if not applicable_range_covers:
        failures.append(
            f"applicable_range_minus{required_half_range:g}_plus{required_half_range:g}"
        )
    if not errors or float(np.max(holdout_abs)) > 2.0:
        failures.append("holdout_maximum_error")
    if not reference_is_strictly_sharper:
        if not internal_maximum:
            failures.append("grid_white_maximum_internal")
        if not measured_neighbours:
            failures.append("grid_white_maximum_measured_neighbours")
        if focus_z_um is None:
            failures.append("grid_white_maximum_ambiguous")
        elif abs(focus_z_um) > settings.maximum_reference_focus_offset_um:
            failures.append("white_focus_reference_offset")
    if approach.status != "passed":
        if (
            any(
                value > settings.maximum_return_error_um
                for value in approach.absolute_target_errors_um.values()
            )
            or len(approach.absolute_target_errors_um) != 2
        ):
            failures.append("return_projected_target_error")
        if (
            approach.mutual_prediction_difference_um is None
            or approach.mutual_prediction_difference_um
            > settings.maximum_return_prediction_difference_um
        ):
            failures.append("return_mutual_prediction_difference")
        if (
            any(
                value < settings.minimum_white_score_fraction
                for value in approach.white_to_reference_fractions.values()
            )
            or len(approach.white_to_reference_fractions) != 2
        ):
            failures.append("return_white_fraction")
        if (
            approach.white_mutual_fraction is None
            or approach.white_mutual_fraction > settings.maximum_approach_white_score_fraction
        ):
            failures.append("return_white_mutual_fraction")
    if not stationary_ready:
        failures.append("stationary_ready")
    if not stationary_identity_match:
        failures.append("stationary_identity")
    if any(observation.accepted_patch_count < 9 for observation in stationary_observations):
        failures.append("stationary_patch_count")
    if any(observation.inlier_fraction < 0.6 for observation in stationary_observations):
        failures.append("stationary_inlier_fraction")
    if stationary_white_mutual > settings.maximum_approach_white_score_fraction:
        failures.append("stationary_white_mutual_fraction")
    if (
        stationary_prediction_difference is None
        or stationary_prediction_difference > settings.maximum_stationary_prediction_difference_um
    ):
        failures.append("stationary_projected_difference")
    status: Literal["candidate", "valid"] = "candidate" if failures else "valid"
    reason = (
        "; ".join(failures)
        if failures
        else "All frozen JPEG calibration and stationary activation gates passed"
    )
    activation = RGFocusActivationEvidence(
        status="failed" if failures else "passed",
        failures=tuple(failures),
        exact_grid_and_returns=exact_grid_and_returns,
        all_grid_and_returns_ready=all_grid_and_returns_ready,
        design_rank=int(rank),
        slope_finite_nonzero=slope_finite_nonzero,
        applicable_range_covers_contract=applicable_range_covers,
        reference_white_score=reference_white_score,
        maximum_grid_white_score=maximum_grid_white,
        maximum_grid_white_z_um=maximum_grid_points[0].z_um,
        grid_white_maximum_z_positions_um=white_maximum_positions,
        maximum_reference_focus_offset_um=settings.maximum_reference_focus_offset_um,
        reference_white_fraction=reference_white_fraction,
        grid_white_maximum_internal=internal_maximum,
        grid_white_maximum_has_measured_neighbours=measured_neighbours,
        approach=approach,
        stationary_observations=stationary_observations,
        stationary_identity_match=stationary_identity_match,
        stationary_projected_z_um=stationary_projected,
        stationary_prediction_difference_um=stationary_prediction_difference,
        maximum_stationary_prediction_difference_um=(
            settings.maximum_stationary_prediction_difference_um
        ),
    )
    empirical_rows: list[RGFocusModelObservation] = []
    if slope_finite_nonzero:
        for point in [*grid, *returns]:
            if _usable(point):
                dx, dy = cast(float, point.dx), cast(float, point.dy)
                empirical_rows.append(
                    RGFocusModelObservation(
                        capture_id=point.capture_id,
                        role=point.role,
                        z_um=point.z_um,
                        dx=dx,
                        dy=dy,
                        signed_cross_track_residual_px=_cross_track(
                            tuple(offset), tuple(slope), dx, dy
                        )[0],
                    )
                )
        for observation in stationary_observations:
            if observation.measurement_status == "ready":
                dx, dy = cast(float, observation.dx), cast(float, observation.dy)
                empirical_rows.append(
                    RGFocusModelObservation(
                        capture_id=observation.capture_id,
                        role="stationary",
                        z_um=0,
                        dx=dx,
                        dy=dy,
                        signed_cross_track_residual_px=_cross_track(
                            tuple(offset), tuple(slope), dx, dy
                        )[0],
                    )
                )
    independent_errors = [abs(value) for value in errors.values()]
    independent_errors.extend(approach.absolute_target_errors_um.values())
    return RGFocusModelProfile(
        id=profile_id,
        status=status,
        reason=reason,
        source_series_id=source_series_id,
        offset_px=(float(offset[0]), float(offset[1])),
        slope_px_per_um=(float(slope[0]), float(slope[1])),
        slope_norm_px_per_um=slope_norm,
        focus_z_um=focus_z_um,
        empirical_observations=tuple(empirical_rows),
        empirical_cross_track_limit_px=(
            max(abs(row.signed_cross_track_residual_px) for row in empirical_rows)
            if empirical_rows
            else None
        ),
        empirical_prediction_error_um=(max(independent_errors) if independent_errors else None),
        fit_capture_ids=tuple(point.capture_id for point in fit),
        holdout_capture_ids=tuple(point.capture_id for point in holdout),
        rejected_capture_reasons=rejected,
        fit_z_range_um=fit_range,
        holdout_z_range_um=holdout_range,
        applicable_z_range_um=applicable_range,
        fit_rmse_um=(
            float(np.sqrt(np.mean(fit_errors * fit_errors))) if fit_errors is not None else None
        ),
        holdout_rmse_um=(
            float(np.sqrt(np.mean(holdout_abs * holdout_abs))) if holdout_abs.size else None
        ),
        holdout_max_absolute_error_um=(float(np.max(holdout_abs)) if holdout_abs.size else None),
        holdout_absolute_error_p95_um=(
            float(np.percentile(holdout_abs, 95)) if holdout_abs.size else None
        ),
        holdout_predictions_um=predictions,
        holdout_errors_um=errors,
        approach_check=approach,
        activation_evidence=activation,
        compatibility=compatibility,
        source_evidence=source_evidence or {},
        measurement_policy=measurement_policy or RGFocusMeasurementPolicy(),
        settings=settings,
    )
