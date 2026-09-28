"""Fail-closed finite correction policy tests for OF-027."""

import pytest
from pydantic import ValidationError
from test_rg_focus_model import (
    empirical_profile,
    shifted_white_profile,
    stationary_observations,
)

from fast_ofm_core.focus.rg.rg_focus_control import (
    RGFocusControlSettings,
    decide_correction,
)
from fast_ofm_core.focus.rg.rg_focus_estimator import RGFocusMeasurement
from fast_ofm_core.focus.rg.rg_focus_model import (
    RGFocusCalibrationPoint,
    RGFocusModelSettings,
    fit_rg_focus_model,
)


def calibration_point(capture_id, role, z, *, white=None):
    """Create exact signed data with a non-zero shift at focus."""
    return RGFocusCalibrationPoint(
        capture_id=capture_id,
        role=role,
        z_um=float(z),
        measurement_status="ready",
        dx=3.0 - z,
        dy=-0.5 + 0.1 * z,
        confidence=0.8,
        white_score=float(100 - abs(z) if white is None else white),
    )


@pytest.fixture
def profile():
    """Return a valid model over the exact prospective JPEG grid."""
    points = [calibration_point(f"fit-{z}", "fit", z) for z in (-8, -4, 0, 4, 8)]
    points += [calibration_point(f"hold-{z}", "holdout", z) for z in (-6, -2, 2, 6)]
    points += [
        calibration_point("above", "approach_above", 0),
        calibration_point("below", "approach_below", 0),
    ]
    return fit_rg_focus_model(
        points,
        RGFocusModelSettings(maximum_holdout_error_um=2.0),
        reference_white_score=100.0,
        stationary_observations=stationary_observations(),
        profile_id="profile",
        source_series_id="series",
        compatibility={},
    )


def measurement(dx=3.0, dy=-0.5, status="ready"):
    """Build a compact aggregate; per-window checks are tested by OF-024."""
    return RGFocusMeasurement(
        status=status,
        reason="test",
        dx=dx if status == "ready" else None,
        dy=dy if status == "ready" else None,
        confidence=0.8 if status == "ready" else 0,
        median_response=0.8 if status == "ready" else 0,
        median_spectral_correlation=0.8 if status == "ready" else 0,
        dx_mad=0.1 if status == "ready" else 0,
        dy_mad=0.1 if status == "ready" else 0,
        radial_p90=0.2 if status == "ready" else 0,
        candidate_patch_count=20,
        accepted_patch_count=20 if status == "ready" else 0,
        inlier_patch_count=20 if status == "ready" else 0,
        inlier_fraction=1 if status == "ready" else 0,
        support_fraction=0.4,
        rejection_counts={},
        windows=(),
    )


def test_demo_cross_track_multiplier_has_a_bounded_extended_range():
    """Commissioned demo evidence may widen cross-track without removing its gate."""
    assert (
        RGFocusControlSettings(cross_track_envelope_multiplier=8).cross_track_envelope_multiplier
        == 8
    )
    with pytest.raises(ValidationError):
        RGFocusControlSettings(cross_track_envelope_multiplier=10.01)


def test_cross_track_rejection_precedes_focused_even_with_high_window_confidence():
    """Consistent windows cannot authorise a shift perpendicular to the model."""
    profile = empirical_profile(prediction_error=0.0, cross_track=0.5)
    value = measurement(dx=5.25, dy=6.5).model_copy(update={"confidence": 1.0})
    decision = decide_correction(
        profile,
        value,
        RGFocusControlSettings(focus_tolerance_um=2),
        iteration=1,
        total_correction_um=0,
    )
    assert decision.status == "refused"
    assert "cross-track" in decision.reason
    assert abs(decision.cross_track_residual_px) > decision.cross_track_limit_px
    assert decision.cross_track_measurement_mad_px == pytest.approx(0.1)
    assert decision.cross_track_limit_px == pytest.approx(
        decision.cross_track_empirical_limit_px + 0.1
    )
    assert decision.cross_track_roundoff_px < 1e-10
    assert decision.correction_um is None


def test_cross_track_observed_boundary_is_inclusive_with_declared_live_margin():
    """The exact calibration envelope is preserved while live repeats get margin."""
    profile = empirical_profile(prediction_error=0.0, cross_track=0.5)
    row = max(
        profile.empirical_observations,
        key=lambda row: abs(row.signed_cross_track_residual_px),
    )
    decision = decide_correction(
        profile,
        measurement(dx=row.dx, dy=row.dy),
        RGFocusControlSettings(focus_tolerance_um=2),
        iteration=1,
        total_correction_um=0,
    )
    assert "cross-track" not in decision.reason
    assert decision.cross_track_empirical_limit_px == profile.empirical_cross_track_limit_px
    assert decision.cross_track_measurement_mad_px == pytest.approx(0.1)
    assert decision.cross_track_limit_px == pytest.approx(
        profile.empirical_cross_track_limit_px + 0.1
    )


def test_cross_track_live_repeat_inside_empirical_plus_measured_mad_is_accepted():
    """A new aggregate may vary modestly without disabling an otherwise valid Z estimate."""
    profile = empirical_profile(prediction_error=0.0, cross_track=0.5)
    sx, sy = profile.slope_px_per_um
    norm = (sx * sx + sy * sy) ** 0.5
    residual = 1.5 * profile.empirical_cross_track_limit_px
    value = measurement(
        dx=profile.offset_px[0] - sy / norm * residual,
        dy=profile.offset_px[1] + sx / norm * residual,
    ).model_copy(update={"dx_mad": 0.5, "dy_mad": 0.5})

    decision = decide_correction(
        profile,
        value,
        RGFocusControlSettings(focus_tolerance_um=2),
        iteration=1,
        total_correction_um=0,
    )

    assert decision.status == "focused"
    assert decision.cross_track_residual_px == pytest.approx(residual)
    assert decision.cross_track_limit_px == pytest.approx(
        profile.empirical_cross_track_limit_px + 0.5
    )


@pytest.mark.parametrize(("error", "status"), [(0.5, "focused"), (1.0, "move")])
def test_one_empirical_budget_includes_model_error_and_live_residual(error, status):
    """A one-micron residual needs correction when 1.2 microns are already reserved."""
    profile = empirical_profile()
    decision = decide_correction(
        profile,
        measurement(dx=4.25 - 0.8 * error, dy=-1.5 + 0.1 * error),
        RGFocusControlSettings(focus_tolerance_um=2),
        iteration=1,
        total_correction_um=0,
    )
    assert decision.status == status
    assert decision.empirical_prediction_error_um == pytest.approx(1.2)
    assert decision.effective_residual_tolerance_um == pytest.approx(0.8)
    assert decision.correction_um == pytest.approx(-error if status == "move" else 0)


def test_empirical_error_exhausting_budget_refuses_even_a_zero_residual():
    """There is no residual budget when all tolerance is consumed by model error."""
    profile = empirical_profile()
    decision = decide_correction(
        profile,
        measurement(dx=4.25, dy=-1.5),
        RGFocusControlSettings(focus_tolerance_um=profile.empirical_prediction_error_um),
        iteration=1,
        total_correction_um=0,
    )
    assert decision.status == "refused"
    assert "exhausts" in decision.reason
    assert decision.correction_um is None


def test_signed_error_requests_opposite_bounded_correction(profile):
    """A positive inferred Z error requests a negative move, not an unsigned search."""
    decision = decide_correction(
        profile,
        measurement(dx=-1.0, dy=-0.1),
        RGFocusControlSettings(),
        iteration=1,
        total_correction_um=0,
    )
    assert decision.status == "move"
    assert decision.inferred_z_error_um == pytest.approx(4)
    assert decision.correction_um == pytest.approx(-4)


@pytest.mark.parametrize(
    ("position", "status", "correction"),
    [(0, "move", 2), (2, "focused", 0), (5, "move", -3)],
)
def test_focus_error_targets_white_peak_in_commanded_coordinates(position, status, correction):
    """A WHITE peak at +2 is the target without moving the calibration origin."""
    profile = shifted_white_profile(2)
    decision = decide_correction(
        profile,
        measurement(dx=4.25 - 0.8 * position, dy=-1.5 + 0.1 * position),
        RGFocusControlSettings(focus_tolerance_um=0.5),
        iteration=1,
        total_correction_um=0,
    )
    assert decision.status == status
    assert decision.inferred_z_position_um == pytest.approx(position)
    assert decision.inferred_z_error_um == pytest.approx(position - 2)
    assert decision.correction_um == pytest.approx(correction)


def test_range_gate_checks_commanded_position_before_white_relative_error():
    """An error inside the old interval cannot authorise unmeasured commanded Z."""
    profile = shifted_white_profile(2)
    decision = decide_correction(
        profile,
        measurement(dx=4.25 - 0.8 * 7, dy=-1.5 + 0.1 * 7),
        RGFocusControlSettings(),
        iteration=1,
        total_correction_um=0,
    )
    assert decision.inferred_z_position_um == pytest.approx(7)
    assert decision.inferred_z_error_um == pytest.approx(5)
    assert decision.status == "refused"
    assert decision.correction_um is None


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (measurement(status="no_tissue"), "measurement rejected"),
        (measurement(dx=-7, dy=0.5), "validated model range"),
    ],
)
def test_bad_qc_and_out_of_range_never_authorise_motion(profile, value, reason):
    """Neither missing tissue nor extrapolation produces a correction."""
    decision = decide_correction(
        profile,
        value,
        RGFocusControlSettings(),
        iteration=1,
        total_correction_um=0,
    )
    assert decision.status == "refused"
    assert decision.correction_um is None
    assert reason in decision.reason


def test_last_iteration_refuses_unverified_move(profile):
    """The loop never moves when no iteration remains for post-move verification."""
    settings = RGFocusControlSettings(maximum_iterations=2)
    decision = decide_correction(
        profile,
        measurement(dx=7, dy=-0.9),
        settings,
        iteration=2,
        total_correction_um=0,
    )
    assert decision.status == "refused"
    assert "verification" in decision.reason


def test_total_envelope_includes_prior_corrections(profile):
    """Accumulated movement is bounded independently of each acceptable estimate."""
    decision = decide_correction(
        profile,
        measurement(dx=8.9, dy=-1.5),
        RGFocusControlSettings(maximum_total_correction_um=12),
        iteration=1,
        total_correction_um=7,
    )
    assert decision.status == "refused"
    assert "total correction" in decision.reason


@pytest.mark.parametrize(("budget", "status"), [(12, "refused"), (16, "move")])
def test_second_correction_reserves_model_error_beyond_validated_input_range(
    profile, budget, status
):
    """A valid coarse estimate may need a small extra verified move after model error."""
    z_um = 4.6
    decision = decide_correction(
        profile,
        measurement(dx=3.0 - z_um, dy=-0.5 + 0.1 * z_um),
        RGFocusControlSettings(maximum_total_correction_um=budget),
        iteration=2,
        total_correction_um=8,
    )

    assert decision.status == status
    assert decision.correction_um == (pytest.approx(-z_um) if status == "move" else None)


def test_large_valid_error_is_split_into_a_bounded_coarse_step(profile):
    """A wide validated model may approach focus without one oversized move."""
    wide_profile = profile.model_copy(update={"applicable_z_range_um": (-12.0, 12.0)})
    decision = decide_correction(
        wide_profile,
        measurement(dx=14.0, dy=-1.6),
        RGFocusControlSettings(
            maximum_iterations=3,
            maximum_single_correction_um=8,
            maximum_total_correction_um=12,
        ),
        iteration=1,
        total_correction_um=0,
    )
    assert decision.status == "move"
    assert decision.inferred_z_error_um == pytest.approx(-11)
    assert decision.correction_um == 8
    assert "coarse" in decision.reason


def test_within_tolerance_is_terminal_without_move(profile):
    """A QC-approved near-focus point needs no artificial correction/retry."""
    decision = decide_correction(
        profile,
        measurement(dx=3.5, dy=-0.55),
        RGFocusControlSettings(),
        iteration=1,
        total_correction_um=0,
    )
    assert decision.status == "focused"
    assert decision.correction_um == 0
