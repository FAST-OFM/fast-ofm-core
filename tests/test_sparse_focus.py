"""Pure tests for measured-anchor sparse focus scheduling."""

import pytest

from fast_ofm_core.focus.sparse_focus import (
    FocusAnchor,
    FocusEstimate,
    SparseFocusScheduler,
    SparseFocusSettings,
    estimate_focus_height,
)


def _plane(x_um: float, y_um: float) -> float:
    return 12.0 + 0.002 * x_um - 0.001 * y_um


def _plane_anchors() -> tuple[FocusAnchor, ...]:
    return tuple(
        FocusAnchor(x, y, _plane(x, y))
        for x, y in ((0.0, 0.0), (1000.0, 0.0), (0.0, 1000.0), (1000.0, 1000.0))
    )


def test_local_plane_predicts_height_and_reports_geometry() -> None:
    """A clean four-point plane predicts the exact interior height."""
    result = estimate_focus_height(
        _plane_anchors(),
        target_x_um=500.0,
        target_y_um=750.0,
        current_z_um=12.0,
        settings=SparseFocusSettings(enabled=True),
    )

    assert result.usable
    assert result.source == "plane"
    assert result.target_z_um == pytest.approx(_plane(500.0, 750.0))
    assert result.support_count == 4
    assert result.fit_residual_um == pytest.approx(0.0, abs=1e-10)
    assert result.extrapolation_um == 0
    assert result.validation_error_um == pytest.approx(0.0, abs=1e-10)


def test_prediction_outside_bounded_support_requires_real_focus() -> None:
    """Excessive extrapolation refuses prediction instead of guessing."""
    result = estimate_focus_height(
        _plane_anchors(),
        target_x_um=5000.0,
        target_y_um=5000.0,
        current_z_um=12.0,
        settings=SparseFocusSettings(
            enabled=True,
            neighbor_radius_um=10000.0,
            maximum_extrapolation_um=500.0,
        ),
    )

    assert not result.usable
    assert result.target_z_um is None
    assert "outside measured support" in result.reason


def test_scheduler_warms_up_then_uses_prediction_without_self_training() -> None:
    """Only real warm-up autofocus results become surface anchors."""
    scheduler = SparseFocusScheduler(
        SparseFocusSettings(enabled=True, anchor_interval=8),
        x_um_per_unit=1.0,
        y_um_per_unit=1.0,
        z_um_per_unit=1.0,
    )

    for x, y in ((0, 0), (1000, 0), (0, 1000), (1000, 1000)):
        decision = scheduler.prepare(x, y, 12, 12)
        assert decision.requires_anchor
        scheduler.complete_anchor(round(_plane(x, y)))

    decision = scheduler.prepare(500, 500, 12, 12)
    assert not decision.requires_anchor
    assert decision.estimate.source == "plane"
    scheduler.complete_prediction()

    assert scheduler.anchor_fields == 4
    assert scheduler.predicted_fields == 1
    assert len(scheduler.anchors) == 4


def test_background_does_not_advance_anchor_cadence() -> None:
    """Skipped background fields do not bring the next anchor closer."""
    scheduler = SparseFocusScheduler(
        SparseFocusSettings(enabled=True, anchor_interval=2),
        x_um_per_unit=1.0,
        y_um_per_unit=1.0,
        z_um_per_unit=1.0,
    )
    scheduler.anchors.extend(_plane_anchors())

    assert not scheduler.prepare(500, 500, 12, 12).requires_anchor
    scheduler.skip_background()
    assert scheduler.fields_since_anchor == 0
    assert scheduler.background_fields == 1

    assert not scheduler.prepare(500, 500, 12, 12).requires_anchor
    scheduler.complete_prediction()
    assert scheduler.prepare(500, 500, 12, 12).requires_anchor


def test_one_off_line_anchor_cannot_validate_cross_row_slope():
    """A zero training residual does not make a single transverse anchor safe."""
    anchors = tuple(FocusAnchor(x, 0, 0) for x in (0, 1000, 2000, 3000))
    anchors += (FocusAnchor(3000, 1000, 20),)
    result = estimate_focus_height(
        anchors,
        target_x_um=2000,
        target_y_um=1000,
        current_z_um=20,
        settings=SparseFocusSettings(enabled=True),
    )
    assert not result.usable
    assert "leave-one-out" in result.reason
    assert result.fit_residual_um == pytest.approx(0, abs=1e-10)


def test_remote_bend_does_not_disable_validated_nearby_plane():
    """Shrink a spatial neighborhood, keeping every closer measured point."""
    anchors = _plane_anchors() + (
        FocusAnchor(4000, 0, 60),
        FocusAnchor(4000, 1000, 60),
    )
    result = estimate_focus_height(
        anchors,
        target_x_um=500,
        target_y_um=500,
        current_z_um=12,
        settings=SparseFocusSettings(enabled=True),
    )
    assert result.usable
    assert result.source == "plane"
    assert result.support_count == 4
    assert result.target_z_um == pytest.approx(_plane(500, 500))


def test_duplicate_off_line_measurements_do_not_fake_independent_geometry():
    """Repeated XY cannot conceal the only off-line point's unit leverage."""
    anchors = tuple(FocusAnchor(x, 0, 0) for x in (0, 1000, 2000, 3000))
    anchors += (FocusAnchor(3000, 1000, 20),) * 3
    result = estimate_focus_height(
        anchors,
        target_x_um=2000,
        target_y_um=1000,
        current_z_um=20,
        settings=SparseFocusSettings(enabled=True),
    )
    assert not result.usable


def test_nearby_bad_anchor_is_not_removed_to_manufacture_a_good_fit():
    """Closest-first prefixes cannot discard a contradictory interior point."""
    result = estimate_focus_height(
        _plane_anchors() + (FocusAnchor(500, 500, 100),),
        target_x_um=500,
        target_y_um=500,
        current_z_um=12,
        settings=SparseFocusSettings(enabled=True),
    )
    assert not result.usable


@pytest.mark.parametrize("direction", [1, -1])
def test_different_row_slopes_use_short_validated_tangent(direction):
    """Both snake directions support a one-gap, three-measurement row forecast."""
    anchors = tuple(FocusAnchor(direction * x, 0, 0) for x in (0, 1000, 2000))
    anchors += tuple(FocusAnchor(direction * x, 1000, x / 100) for x in (0, 1000, 2000))
    result = estimate_focus_height(
        anchors,
        target_x_um=-direction * 1000,
        target_y_um=1000,
        current_z_um=0,
        settings=SparseFocusSettings(enabled=True),
    )
    assert result.usable
    assert result.source == "row"
    assert result.support_count == 3
    assert result.target_z_um == pytest.approx(-10)
    assert result.validation_error_um == pytest.approx(0, abs=1e-10)
    assert result.extrapolation_um == 1000
    farther = estimate_focus_height(
        anchors,
        target_x_um=-direction * 2000,
        target_y_um=1000,
        current_z_um=0,
        settings=SparseFocusSettings(enabled=True),
    )
    assert not farther.usable


@pytest.mark.parametrize(
    "settings",
    [
        SparseFocusSettings(maximum_prediction_delta_um=5),
        SparseFocusSettings(maximum_plane_slope_um_per_um=0.005),
        SparseFocusSettings(maximum_extrapolation_um=500),
    ],
)
def test_row_tangent_never_bypasses_physical_limits(settings):
    """Row prediction retains Z-delta, slope and extrapolation guards."""
    anchors = tuple(FocusAnchor(x, 0, 0) for x in (0, 1000, 2000))
    anchors += tuple(FocusAnchor(x, 1000, x / 100) for x in (0, 1000, 2000))
    result = estimate_focus_height(
        anchors,
        target_x_um=-1000,
        target_y_um=1000,
        current_z_um=0,
        settings=settings,
    )
    assert not result.usable


def test_initial_line_remains_measured_and_nearest_is_not_a_surface():
    """Warm-up and a sparse distant island cannot turn one Z into an accepted map."""
    scheduler = SparseFocusScheduler(
        SparseFocusSettings(enabled=True),
        x_um_per_unit=1,
        y_um_per_unit=1,
        z_um_per_unit=1,
    )
    for x in range(0, 6000, 1000):
        assert scheduler.prepare(x, 0, 0, 0).requires_anchor
        scheduler.complete_anchor(0)
    scheduler.anchors.append(FocusAnchor(20000, 0, 0))
    decision = scheduler.prepare(20000, 1000, 0, 0)
    assert decision.estimate.source == "nearest"
    assert decision.requires_anchor


def test_long_clean_snake_predicts_after_first_line_and_keeps_eight_field_cadence():
    """Only measured anchors enter a 10x6 scan's model, including across row turns."""
    scheduler = SparseFocusScheduler(
        SparseFocusSettings(enabled=True, maximum_recovery_anchors=3),
        x_um_per_unit=1,
        y_um_per_unit=1,
        z_um_per_unit=1,
    )
    current_z, predicted_run = 12, 0
    first_row_anchors = 0
    for row in range(6):
        columns = range(10) if row % 2 == 0 else reversed(range(10))
        for col in columns:
            x, y = col * 1101, -row * 825
            decision = scheduler.prepare(x, y, current_z, current_z)
            if decision.requires_anchor:
                # Exact integer-valued tilted surface isolates routing/cadence
                # from stage quantization amplified by extrapolation.
                current_z = 12 + 2 * col + row
                scheduler.complete_anchor(current_z)
                first_row_anchors += row == 0
                predicted_run = 0
            else:
                current_z = decision.target_z_units
                assert current_z == 12 + 2 * col + row
                before = tuple(scheduler.anchors)
                scheduler.complete_prediction()
                assert tuple(scheduler.anchors) == before
                predicted_run += 1
                assert predicted_run <= 7
    assert first_row_anchors == 10
    assert scheduler.predicted_fields >= 40
    assert len(scheduler.anchors) == scheduler.anchor_fields


# Saved commanded focus heights from the cancelled 2026-09-12 scan, not an
# independent optical ground truth. Each prediction is compared only afterwards;
# its saved height never enters the simulated model.
_SAVED_ROWS = (
    (-9, -9, -9, -9, -9, -9, -9, -9, -21, -43),
    (-7, -7, -7, -7, -7, -28, -41, -49, -58, -66),
)


@pytest.mark.parametrize(("columns", "expected_anchors", "max_error"), [(10, 16, 0), (9, 14, 4)])
def test_cancelled_scan_replay_uses_only_past_measured_anchors(
    columns, expected_anchors, max_error
):
    """Actual first-line bends no longer poison every later prediction."""
    scheduler = SparseFocusScheduler(
        SparseFocusSettings(enabled=True, maximum_recovery_anchors=3),
        x_um_per_unit=1,
        y_um_per_unit=1,
        z_um_per_unit=1,
    )
    z, errors = 0, []
    for row, heights in enumerate(_SAVED_ROWS):
        cols = range(columns) if row == 0 else reversed(range(columns))
        for col in cols:
            decision = scheduler.prepare(col * 1101, -row * 825, z, z)
            if decision.requires_anchor:
                z = heights[col]
                scheduler.complete_anchor(z)
            else:
                z = decision.target_z_units
                errors.append(abs(z - heights[col]))
                scheduler.complete_prediction()
    assert scheduler.anchor_fields == expected_anchors
    assert scheduler.predicted_fields == 4
    assert max(errors) == max_error


def test_unavailable_prediction_enters_bounded_white_mode(monkeypatch):
    """After three probes use seven WHITE fields, then one RG reprobe, no abort."""
    monkeypatch.setattr(
        "fast_ofm_core.focus.sparse_focus.estimate_focus_height",
        lambda *_args, **_kwargs: FocusEstimate(False, None, "plane", "inconsistent anchors"),
    )
    scheduler = SparseFocusScheduler(
        SparseFocusSettings(enabled=True, maximum_recovery_anchors=3),
        x_um_per_unit=1,
        y_um_per_unit=1,
        z_um_per_unit=1,
    )
    for x in range(10):
        assert scheduler.prepare(x, 0, 0, 0).requires_anchor
        scheduler.complete_anchor(0)
    assert scheduler.prepare(10, 1, 0, 0).requires_anchor
    scheduler.skip_background()
    for x in range(3):
        assert scheduler.prepare(x, 1, 0, 0).requires_anchor
        scheduler.complete_anchor(0)
    anchors = tuple(scheduler.anchors)
    for x in range(3, 10):
        decision = scheduler.prepare(x, 1, 0, 23)
        assert decision.white_only
        assert decision.requires_anchor
        assert decision.target_z_units == 0
        scheduler.complete_white_fallback(rg_attempted=False)
    decision = scheduler.prepare(10, 1, 0, 0)
    assert decision.requires_anchor
    assert not decision.white_only
    scheduler.complete_white_fallback(rg_attempted=True)
    assert scheduler.prepare(11, 1, 0, 0).white_only
    assert tuple(scheduler.anchors) == anchors
    assert scheduler.anchor_fields == 13
    assert scheduler.background_fields == 1
    assert scheduler.white_fallback_fields == 8


def test_none_recovery_limit_does_not_enable_implicit_white_degradation(monkeypatch):
    """An explicitly disabled cap remains disabled beyond three recovery probes."""
    monkeypatch.setattr(
        "fast_ofm_core.focus.sparse_focus.estimate_focus_height",
        lambda *_args, **_kwargs: FocusEstimate(False, None, "plane", "inconsistent anchors"),
    )
    scheduler = SparseFocusScheduler(
        SparseFocusSettings(enabled=True, maximum_recovery_anchors=None),
        x_um_per_unit=1,
        y_um_per_unit=1,
        z_um_per_unit=1,
    )
    for row in range(2):
        for x in range(12):
            decision = scheduler.prepare(x, row, 0, 0)
            assert decision.requires_anchor
            assert not decision.white_only
            scheduler.complete_anchor(0)
    assert scheduler._recovery_anchors == 12
    assert scheduler.white_fallback_fields == 0


def test_failed_rg_warmup_does_not_prevent_white_degradation():
    """Too few valid RG supports must not cause RG autofocus on every later field."""
    scheduler = SparseFocusScheduler(
        SparseFocusSettings(enabled=True, maximum_recovery_anchors=3),
        x_um_per_unit=1,
        y_um_per_unit=1,
        z_um_per_unit=1,
    )
    for x in range(9):
        assert not scheduler.prepare(x, 0, 0, 0).white_only
        scheduler.complete_white_fallback(rg_attempted=True)
    for x in range(3):
        assert not scheduler.prepare(x, 1, 0, 0).white_only
        scheduler.complete_white_fallback(rg_attempted=True)
    assert scheduler.prepare(3, 1, 0, 0).white_only
    scheduler.skip_background()
    assert scheduler.prepare(4, 1, 0, 0).white_only
    assert scheduler.fields_since_rg_probe == 0
    assert not scheduler.anchors
    assert scheduler.anchor_fields == 0


def test_white_does_not_train_and_good_local_surface_resumes_immediately(monkeypatch):
    """A previously invalid target must not poison support or latch degraded mode."""
    scheduler = SparseFocusScheduler(
        SparseFocusSettings(enabled=True, maximum_recovery_anchors=3),
        x_um_per_unit=1,
        y_um_per_unit=1,
        z_um_per_unit=1,
    )
    for x, y in ((0, 0), (1, 0), (0, 1), (1, 1)):
        scheduler.prepare(x, y, 0, 0)
        scheduler.complete_anchor(0)
    anchors = tuple(scheduler.anchors)
    monkeypatch.setattr(
        "fast_ofm_core.focus.sparse_focus.estimate_focus_height",
        lambda *_args, **_kwargs: FocusEstimate(False, None, "plane", "invalid here"),
    )
    assert not scheduler.prepare(0, 2, 0, 0).white_only
    scheduler.complete_white_fallback(rg_attempted=True)
    assert scheduler.prepare(1, 2, 0, 0).white_only
    scheduler.complete_white_fallback(rg_attempted=False)
    monkeypatch.setattr(
        "fast_ofm_core.focus.sparse_focus.estimate_focus_height",
        lambda *_args, **_kwargs: FocusEstimate(True, 0, "plane", "valid here"),
    )
    decision = scheduler.prepare(2, 2, 0, 0)
    assert not decision.requires_anchor
    assert not decision.white_only
    scheduler.complete_prediction()
    assert tuple(scheduler.anchors) == anchors
    assert scheduler._recovery_anchors == 0


@pytest.mark.parametrize("source", ["white_fallback", "prediction", "unknown", "rg"])
def test_non_rg_source_cannot_be_inserted_as_anchor(source):
    """The model API rejects provenance mistakes instead of learning WHITE heights."""
    scheduler = SparseFocusScheduler(
        SparseFocusSettings(enabled=True),
        x_um_per_unit=1,
        y_um_per_unit=1,
        z_um_per_unit=1,
    )
    scheduler.prepare(0, 0, 0, 0)
    with pytest.raises(ValueError, match="Only measured simultaneous RG"):
        scheduler.complete_anchor(-100, focus_method=source)
    assert not scheduler.anchors


def test_white_on_due_anchor_resumes_valid_predictions_without_extra_rg(monkeypatch):
    """A failed scheduled RG shot counts for cadence, but never adds WHITE as support."""
    scheduler = SparseFocusScheduler(
        SparseFocusSettings(enabled=True, anchor_interval=8),
        x_um_per_unit=1,
        y_um_per_unit=1,
        z_um_per_unit=1,
    )
    scheduler.anchors.extend(
        FocusAnchor(x, y, 7) for x, y in ((0, 0), (1000, 0), (0, 1000), (1000, 1000))
    )
    monkeypatch.setattr(
        "fast_ofm_core.focus.sparse_focus.estimate_focus_height",
        lambda *_args, **_kwargs: FocusEstimate(True, 7, "plane", "validated"),
    )
    for x in range(7):
        assert not scheduler.prepare(x, 500, 7, 7).requires_anchor
        scheduler.complete_prediction()
    assert scheduler.prepare(7, 500, 7, 7).requires_anchor
    scheduler.complete_white_fallback(rg_attempted=True)
    for x in range(8, 15):
        decision = scheduler.prepare(x, 500, -6, -6)
        assert not decision.requires_anchor
        assert decision.target_z_units == 7
        scheduler.complete_prediction()
    assert scheduler.prepare(15, 500, 7, 7).requires_anchor
    assert scheduler.anchors == [
        FocusAnchor(x, y, 7) for x, y in ((0, 0), (1000, 0), (0, 1000), (1000, 1000))
    ]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_focus_evidence_is_refused(bad):
    """Malformed observations cannot produce NaN motor targets."""
    result = estimate_focus_height(
        _plane_anchors() + (FocusAnchor(500, 500, bad),),
        target_x_um=500,
        target_y_um=500,
        current_z_um=12,
        settings=SparseFocusSettings(),
    )
    assert not result.usable
    assert result.target_z_um is None


def test_invalid_recovery_limit_is_rejected():
    """A disabled cap is explicit; zero is never an implicit infinite budget."""
    with pytest.raises(ValueError, match="Recovery anchor limit"):
        SparseFocusSettings(maximum_recovery_anchors=0)
