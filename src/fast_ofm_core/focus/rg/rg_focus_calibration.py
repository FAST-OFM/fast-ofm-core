"""Small declarative contract for the bounded LED Z calibration series."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from fast_ofm_core.contracts import CalibrationManifest, CalibrationStage, StrictModel

from .rg_focus_estimator import rg_shift_manifest_stage
from .rg_focus_field import tissue_field_manifest_stage

StackRole = Literal["reference", "fit", "holdout", "approach_above", "approach_below"]

DEMO_Z_OFFSETS_UM = tuple(range(-32, 33, 4))
DEMO_FIT_OFFSETS_UM = (-32, -24, -16, -8, 0, 8, 16, 24, 32)
DEMO_HOLDOUT_OFFSETS_UM = (-28, -20, -12, -4, 4, 12, 20, 28)
DEMO_CALIBRATION_ENVELOPE_UM = 42
DEMO_PAIR_BUDGET = 20


class RGFocusApproachSettings(StrictModel):
    """One explicit stage-owned final approach shared by calibration and focus."""

    preload_um: int = Field(default=8, ge=1, le=16)
    approach_sign: Literal[-1, 1] = 1

    def path(self, target_um: int) -> tuple[int, int]:
        """Return both endpoints; the caller must check its whole allowed envelope."""
        return target_um - self.approach_sign * self.preload_um, target_um


class RGFocusStackSettings(StrictModel):
    """Predeclared Z grid, data split, budgets, and acceptance tolerances."""

    z_offsets_um: tuple[int, ...] = Field(default=DEMO_Z_OFFSETS_UM, strict=False)
    fit_offsets_um: tuple[int, ...] = Field(default=DEMO_FIT_OFFSETS_UM, strict=False)
    holdout_offsets_um: tuple[int, ...] = Field(default=DEMO_HOLDOUT_OFFSETS_UM, strict=False)
    approach_start_um: int = Field(default=8, ge=4, le=24)
    focus_target_um: int = 0
    maximum_absolute_z_um: int = Field(default=DEMO_CALIBRATION_ENVELOPE_UM, ge=4, le=48)
    step_um: int = Field(default=4, ge=1, le=4)
    action_timeout_s: float = Field(default=60, gt=0, le=120)
    timeout_s: float = Field(default=900, gt=0, le=1800)
    deadline_guard_s: float = Field(default=120, ge=60, le=600)
    maximum_pairs: int = Field(default=DEMO_PAIR_BUDGET, ge=1, le=20)
    minimum_valid_fit_points: int = Field(default=5, ge=3)
    minimum_valid_holdout_points: int = Field(default=2, ge=2)
    maximum_holdout_error_um: float = Field(default=4.0, gt=0, le=8)
    maximum_approach_shift_difference_px: float = Field(default=1.0, gt=0, le=4)
    maximum_approach_white_score_fraction: float = Field(default=0.15, gt=0, le=0.5)

    @model_validator(mode="after")
    def safe_complete_grid(self) -> Self:
        """Reject widened, reordered, post-hoc, or underpowered experiments."""
        if not self.z_offsets_um:
            raise ValueError("The Z grid cannot be empty")
        expected = tuple(range(self.z_offsets_um[0], self.z_offsets_um[-1] + 1, self.step_um))
        if self.z_offsets_um != expected or any(
            abs(z) > self.maximum_absolute_z_um for z in self.z_offsets_um
        ):
            raise ValueError("The Z grid must be uniform, increasing and inside its envelope")
        if self.focus_target_um != 0 or self.approach_start_um > self.maximum_absolute_z_um:
            raise ValueError("The operator focus is Z0 and both starts must fit the envelope")
        if set(self.fit_offsets_um) & set(self.holdout_offsets_um):
            raise ValueError("Fit and holdout Z points must be disjoint")
        if set(self.fit_offsets_um) | set(self.holdout_offsets_um) != set(self.z_offsets_um):
            raise ValueError("Fit and holdout must partition the declared Z grid")
        if self.focus_target_um not in self.fit_offsets_um:
            raise ValueError("The commanded calibration origin must be a fit point")
        if (
            len(self.fit_offsets_um) < self.minimum_valid_fit_points
            or len(self.holdout_offsets_um) < self.minimum_valid_holdout_points
        ):
            raise ValueError("The declared data split cannot meet its minimum counts")
        if self.maximum_pairs < len(self.z_offsets_um) + 3:
            raise ValueError("Pair budget omits reference or two-sided approach checks")
        return self


def calibration_path(
    settings: RGFocusStackSettings, approach: RGFocusApproachSettings
) -> tuple[int, ...]:
    """Return every endpoint in the exact acquisition order before any action."""
    path: list[int] = []
    for capture in capture_plan(settings)[1:]:
        if capture.role == "approach_above":
            path.append(settings.approach_start_um)
            path.extend(approach.path(0))
        elif capture.role == "approach_below":
            path.append(-settings.approach_start_um)
            path.extend(approach.path(0))
        else:
            path.extend(approach.path(capture.z_um))
    if any(abs(z) > settings.maximum_absolute_z_um for z in path):
        raise ValueError("Z grid plus final-approach preload exceeds the calibration envelope")
    return tuple(path)


class RGFocusCapture(StrictModel):
    """One capture role at a commanded Z offset; XY is intentionally absent."""

    capture_id: str
    role: StackRole
    z_um: int


def capture_plan(settings: RGFocusStackSettings) -> tuple[RGFocusCapture, ...]:
    """Return the immutable assignment in its deterministic acquisition order.

    The accepted demo grid is ordered so a linear time drift is orthogonal to Z
    separately for fit and holdout points. Both groups and the approach checks
    also have the same mean acquisition time. Other declarative grids retain
    their increasing order; they are not accepted by the fixed demo contract.
    """
    captures = [RGFocusCapture(capture_id="reference", role="reference", z_um=0)]
    for index, z_um in enumerate(settings.z_offsets_um):
        role: StackRole = "fit" if z_um in settings.fit_offsets_um else "holdout"
        captures.append(RGFocusCapture(capture_id=f"stack-{index:02d}", role=role, z_um=z_um))
    captures.extend(
        [
            RGFocusCapture(capture_id="approach-above", role="approach_above", z_um=0),
            RGFocusCapture(capture_id="approach-below", role="approach_below", z_um=0),
        ]
    )
    if (
        settings.z_offsets_um == tuple(range(-9, 10, 3))
        and settings.fit_offsets_um == (-9, -3, 0, 3, 9)
        and settings.holdout_offsets_um == (-6, 6)
    ):
        by_z = {capture.z_um: capture for capture in captures if capture.role in ("fit", "holdout")}
        by_role = {capture.role: capture for capture in captures}
        captures = [
            captures[0],
            by_role["approach_above"],
            by_z[-9],
            by_z[9],
            by_z[-6],
            by_z[3],
            by_z[6],
            by_z[0],
            by_z[-3],
            by_role["approach_below"],
        ]
    elif (
        settings.z_offsets_um == tuple(range(-16, 17, 4))
        and settings.fit_offsets_um == (-16, -8, 0, 8, 16)
        and settings.holdout_offsets_um == (-12, -4, 4, 12)
    ):
        by_z = {capture.z_um: capture for capture in captures if capture.role in ("fit", "holdout")}
        by_role = {capture.role: capture for capture in captures}
        captures = [
            captures[0],
            by_z[4],
            by_z[-16],
            by_z[16],
            by_z[-4],
            by_role["approach_above"],
            by_z[8],
            by_role["approach_below"],
            by_z[-8],
            by_z[-12],
            by_z[12],
            by_z[0],
        ]
    elif (
        settings.z_offsets_um == DEMO_Z_OFFSETS_UM
        and settings.fit_offsets_um == DEMO_FIT_OFFSETS_UM
        and settings.holdout_offsets_um == DEMO_HOLDOUT_OFFSETS_UM
    ):
        by_z = {capture.z_um: capture for capture in captures if capture.role in ("fit", "holdout")}
        by_role = {capture.role: capture for capture in captures}
        captures = [
            captures[0],
            by_z[-8],
            by_z[12],
            by_z[32],
            by_z[4],
            by_z[-20],
            by_z[-32],
            by_z[-16],
            by_z[-4],
            by_role["approach_above"],
            by_z[8],
            by_role["approach_below"],
            by_z[-12],
            by_z[16],
            by_z[24],
            by_z[28],
            by_z[20],
            by_z[-24],
            by_z[-28],
            by_z[0],
        ]
    if len(captures) > settings.maximum_pairs:
        raise ValueError("Capture plan exceeds the configured pair budget")
    return tuple(captures)


def led_calibration_manifest(settings: RGFocusStackSettings) -> CalibrationManifest:
    """Describe the existing-API acquisition and offline fit without a new scheduler."""
    field_stage = tissue_field_manifest_stage().model_copy(update={"hardware_required": False})
    return CalibrationManifest(
        id="led_focus_calibration",
        name="Tissue-aware R/G LED focus calibration",
        description="Discrete WHITE Z stack plus tissue-window R/G mutual-information model and holdouts.",
        prerequisites=[
            "operator_prepared_dense_single_layer_tissue_and_white_focus_within_12_um",
            "valid_stage_reference_and_exclusive_control",
            "compatible_checked_red_green_flat_fields",
            "fixed_manual_camera_configuration",
            "confirmed_white_illumination",
        ],
        parameter_refs=list(RGFocusStackSettings.model_fields)
        + [
            "approach_parameters.preload_um",
            "approach_parameters.approach_sign",
        ],
        stages=[
            CalibrationStage(
                id="preflight",
                name="Readiness and bounded path",
                description="Freeze camera/profile/controller identity and validate every Z target.",
                inputs=[
                    "parameters",
                    "controller_state",
                    "camera_binding",
                    "flat_fields",
                ],
                outputs=["frozen_binding", "capture_plan"],
                action="existing OFM read-only properties",
                success_criterion="XY zero, Z zero, WHITE, compatible profiles and the complete declared bounded path",
                timeout_setting="action_timeout_s",
                cancellation="Issue no motion or light command.",
                hardware_required=True,
            ),
            CalibrationStage(
                id="capture",
                name="Discrete WHITE and common final-approach R/G stack",
                description="Score a fresh WHITE frame at every Z point, then capture the same-owner R/G pair after the stage-owned preload path.",
                inputs=["frozen_binding", "capture_plan"],
                outputs=["WHITE_RED_GREEN_processed_JPEG_RGB8_metadata"],
                action="capture_plan",
                success_criterion="Fresh WHITE/R/G triplets at every declared fit/holdout point and one interior discrete WHITE target with measured neighbours",
                timeout_setting="action_timeout_s",
                cancellation="Stop before a new action; read back uncertain motion; restore WHITE and Z only when confirmed safe.",
                hardware_required=True,
            ),
            field_stage,
            rg_shift_manifest_stage(),
            CalibrationStage(
                id="approach_check",
                name="Two-sided final approach check",
                description="Compare independent target captures after starts above and below focus.",
                inputs=["approach_above", "approach_below", "parameters"],
                outputs=["repeatability_result"],
                action="offline approach comparison",
                success_criterion="R/G shift and WHITE score agree within predefined tolerances",
                timeout_setting="action_timeout_s",
                cancellation="Keep model candidate-only; do not activate LED autofocus.",
                hardware_required=False,
            ),
            CalibrationStage(
                id="fit_holdout",
                name="Signed model and holdout",
                description="Fit the signed 2D mutual-information inlier-mean model, select the stopped-frame WHITE target, and test untouched Z points.",
                inputs=[
                    "fit measurements",
                    "holdout measurements",
                    "repeatability_result",
                ],
                outputs=["candidate_or_valid_profile", "fit_report"],
                action="fit_led_focus_model",
                success_criterion="Holdout error and uncertainty pass predefined tolerances",
                timeout_setting="action_timeout_s",
                cancellation="Preserve evidence and keep LED unavailable.",
                hardware_required=False,
            ),
        ],
        results=[
            "manifest",
            "processed JPEG RGB8 frames and metadata",
            "fixed tissue windows and measurements",
            "independent WHITE scores",
            "fit/holdout report",
            "candidate_or_valid_profile",
        ],
        acceptance_criteria=[
            f"At least {settings.minimum_valid_fit_points} fit and {settings.minimum_valid_holdout_points} holdout points",
            f"Holdout Z error <= {settings.maximum_holdout_error_um:g} um",
            "Unique stopped-frame WHITE target is internal and no more than 12 um from the operator reference",
            "Two-sided target approach passes R/G and WHITE repeatability",
            "No command exceeds ±26 um and final WHITE/Z0 are confirmed",
        ],
        invalidated_by=[
            "camera/exposure/ROI/tuning",
            "R/G flat-field profile",
            "optics/illumination",
            "stage Z profile or approach policy",
        ],
    )
