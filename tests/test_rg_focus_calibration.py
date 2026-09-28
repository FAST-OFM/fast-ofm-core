"""Bounded manifest and predeclared split tests for OF-025."""

import pytest
from pydantic import ValidationError

from fast_ofm_core.focus.rg.rg_focus_calibration import (
    DEMO_FIT_OFFSETS_UM,
    DEMO_HOLDOUT_OFFSETS_UM,
    DEMO_Z_OFFSETS_UM,
    RGFocusApproachSettings,
    RGFocusStackSettings,
    calibration_path,
    capture_plan,
    led_calibration_manifest,
)


def test_default_plan_is_bounded_and_split_before_capture():
    """The first capture grid and fit/holdout roles are immutable and bounded."""
    settings = RGFocusStackSettings()
    plan = capture_plan(settings)
    stack = [item for item in plan if item.role in ("fit", "holdout")]
    assert sorted(item.z_um for item in stack) == list(DEMO_Z_OFFSETS_UM)
    assert {item.z_um for item in stack if item.role == "fit"} == set(settings.fit_offsets_um)
    assert {item.z_um for item in stack if item.role == "holdout"} == set(
        settings.holdout_offsets_um
    )
    assert len(plan) == 20
    assert all(abs(item.z_um) <= 42 for item in plan)


def test_demo_plan_is_time_balanced_for_fit_holdout_and_approach_checks():
    """Linear drift cannot alias into Z slope or separate fit from holdout."""
    settings = RGFocusStackSettings(
        z_offsets_um=DEMO_Z_OFFSETS_UM,
        fit_offsets_um=DEMO_FIT_OFFSETS_UM,
        holdout_offsets_um=DEMO_HOLDOUT_OFFSETS_UM,
        step_um=4,
        maximum_pairs=20,
        minimum_valid_holdout_points=2,
    )
    plan = capture_plan(settings)
    timed = list(enumerate(plan[1:], start=1))
    fit = [(time, item) for time, item in timed if item.role == "fit"]
    holdout = [(time, item) for time, item in timed if item.role == "holdout"]
    approaches = [
        (time, item) for time, item in timed if item.role in ("approach_above", "approach_below")
    ]

    assert [item.z_um for _, item in timed] == [
        -8,
        12,
        32,
        4,
        -20,
        -32,
        -16,
        -4,
        0,
        8,
        0,
        -12,
        16,
        24,
        28,
        20,
        -24,
        -28,
        0,
    ]
    assert sum(time for time, _ in fit) / len(fit) == 10
    assert sum(time for time, _ in holdout) / len(holdout) == 10
    assert sum(time for time, _ in approaches) / len(approaches) == 10
    assert sum(time * item.z_um for time, item in fit) == 0
    assert sum(time * item.z_um for time, item in holdout) == 0


@pytest.mark.parametrize(
    "change",
    [
        {"maximum_absolute_z_um": 49},
        {"step_um": 3},
        {"holdout_offsets_um": (-24, -12, -4, 4, 12, 20)},
        {"maximum_pairs": 19},
    ],
)
def test_widened_changed_or_overlapping_plan_is_rejected(change):
    """Unsafe range, step, data reuse, and insufficient budgets fail validation."""
    with pytest.raises(ValidationError):
        RGFocusStackSettings(**change)


def test_manifest_reuses_real_field_and_shift_actions_without_xy_or_fallback():
    """The manifest points at implemented math and has no XY or fallback controls."""
    manifest = led_calibration_manifest(RGFocusStackSettings())
    stages = {stage.id: stage for stage in manifest.stages}
    assert stages["tissue_field"].action == "prepare_tissue_field"
    assert stages["rg_shift"].action == "estimate_rg_shift"
    assert "whole-frame" in stages["tissue_field"].cancellation
    assert "sharpness-difference" in stages["rg_shift"].cancellation
    assert "x" not in RGFocusStackSettings.model_fields
    assert "y" not in RGFocusStackSettings.model_fields


@pytest.mark.parametrize("sign", [-1, 1])
def test_entire_preload_path_is_bounded_for_both_signs(sign):
    """Both shared policies reserve room inside the ±42 um envelope."""
    approach = RGFocusApproachSettings(preload_um=10, approach_sign=sign)
    path = calibration_path(RGFocusStackSettings(), approach)
    assert min(path) >= -42
    assert max(path) <= 42
    assert approach.path(0) == (-sign * 10, 0)


def test_old_edge_to_edge_grid_requires_explicit_narrowing_before_preload():
    """Historical settings still load but cannot send an unsafe overshoot."""
    settings = RGFocusStackSettings(
        z_offsets_um=tuple(range(-24, 25, 4)),
        fit_offsets_um=(-24, -16, -8, 0, 8, 16, 24),
        holdout_offsets_um=(-20, -12, -4, 4, 12, 20),
        step_um=4,
        maximum_absolute_z_um=24,
        maximum_pairs=16,
    )
    with pytest.raises(ValueError, match="preload exceeds"):
        calibration_path(settings, RGFocusApproachSettings())
