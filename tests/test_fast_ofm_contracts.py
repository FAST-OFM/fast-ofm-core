"""Checks for the configurable Fast OFM envelope and calibration descriptions."""

import pytest
from pydantic import ValidationError

from fast_ofm_core.contracts import (
    AxisSettings,
    CalibrationManifest,
    MotionSettings,
)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["min_mm", "max_mm", "speed_mm_s", "units_per_mm"])
def test_nonfinite_rejected(field, bad):
    """Non-finite limits must not pass validation."""
    with pytest.raises(ValidationError):
        AxisSettings(**{field: bad})


def test_z_disabled_and_incomplete_range_rejected():
    """There is no guessed Z envelope or enabled unbounded axis."""
    settings = MotionSettings()
    assert not settings.allow_motion
    assert not settings.axes["z"].enabled
    assert settings.axes["z"].min_mm is None
    with pytest.raises(ValidationError):
        AxisSettings(min_mm=None)
    with pytest.raises(ValidationError):
        MotionSettings(axes={"x": AxisSettings()})


@pytest.mark.parametrize(
    "change",
    [
        {"max_mm": 6},
        {"min_mm": -6},
        {"max_move_mm": 0.3},
        {"speed_mm_s": 1},
        {"accel_mm_s2": 10},
        {"settle_ms": 100},
        {"direction_sign": -1},
        {"units_per_mm": 1},
    ],
)
def test_experiment_cannot_expand_hardware(change):
    """An experiment may not weaken the hardware profile."""
    with pytest.raises(ValueError, match="hardware|units|direction"):
        AxisSettings(**change).require_subset_of(AxisSettings())


def test_narrower_profile_and_disabled_axis():
    """A smaller experiment is allowed; enabling an uncommissioned axis is not."""
    AxisSettings(min_mm=-1, max_mm=1, max_move_mm=0.1).require_subset_of(AxisSettings())
    with pytest.raises(ValueError, match="disabled hardware axis"):
        AxisSettings().require_subset_of(AxisSettings(enabled=False))


def test_reference_has_no_expiry_setting():
    """The obsolete age limit cannot silently be re-enabled by deployment settings."""
    assert "reference_max_age_s" not in MotionSettings.model_json_schema()["properties"]
    assert "reference_max_age_s" not in MotionSettings().model_dump()
    with pytest.raises(ValidationError):
        MotionSettings(reference_max_age_s=600)


def test_no_unrecognised_units_or_missing_manifest_fields():
    """Unknown fields and empty calibration descriptions must be rejected."""
    with pytest.raises(ValidationError):
        AxisSettings(speed_inches_s=1)
    with pytest.raises(ValidationError):
        CalibrationManifest(id="backlash")


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_manual_timing_requires_finite_nonnegative_values(value):
    """Short manual delays are permitted, invalid timing is not."""
    assert MotionSettings(manual_settle_ms=0.0, ui_repeat_delay_ms=20.0)
    with pytest.raises(ValidationError):
        MotionSettings(manual_settle_ms=value)
    with pytest.raises(ValidationError):
        MotionSettings(ui_repeat_delay_ms=value)
