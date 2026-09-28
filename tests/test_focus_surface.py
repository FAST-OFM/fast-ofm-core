"""Contract and pure-predictor checks for the run-scoped focus surface."""

import json
import math

import pytest
from pydantic import ValidationError

from fast_ofm_core.focus.focus_surface import (
    FocusAxisScale,
    FocusBinding,
    FocusFieldProgress,
    FocusObservation,
    FocusObservationContractError,
    FocusObservationProvenance,
    FocusObservationQuality,
    FocusPrediction,
    FocusPredictionDiagnostics,
    FocusPredictionEvidence,
    FocusPredictionRequest,
    FocusRunSettings,
    FocusSurfaceSettings,
    WhiteSearchSettings,
    assess_focus_observation,
    check_focus_binding,
    predict_focus_surface,
    select_usable_observations,
)
from fast_ofm_core.focus.rg.rg_focus_calibration import (
    RGFocusApproachSettings,
)


def binding(**overrides):
    """Return a complete explicit valid binding with no runtime-state hash."""
    values = {
        "stage_controller_id": "moonraker-main",
        "reference_id": "operator-zero-7",
        "reference_status": "valid",
        "axis_scales": (
            FocusAxisScale(axis="x", units_per_mm=1000.0, direction_sign=1),
            FocusAxisScale(axis="y", units_per_mm=800.0, direction_sign=-1),
            FocusAxisScale(axis="z", units_per_mm=1200.0, direction_sign=1),
        ),
        "camera_stage_mapping_id": "csm-3",
        "camera_stage_mapping_status": "valid",
        "geometry_id": "hq-crop-2800-common-plane",
        "rg_focus_model_id": "rg-model-9",
        "rg_focus_model_status": "valid",
        "red_flat_field_profile_id": "red-flat-4",
        "red_flat_field_status": "valid",
        "green_flat_field_profile_id": "green-flat-5",
        "green_flat_field_status": "valid",
        "approach_profile_id": "z-approach-2",
        "approach_status": "valid",
        "approach_parameters": RGFocusApproachSettings(preload_um=8, approach_sign=1),
    }
    values.update(overrides)
    return FocusBinding(**values)


def surface_settings(**overrides):
    """Use explicit test-only gates; the production contract has no active defaults."""
    values = {
        "enabled": True,
        "minimum_plane_points": 4,
        "maximum_neighbors": 12,
        "neighbor_radius_um": 1500.0,
        "allow_extrapolation": False,
        "maximum_extrapolation_um": 0.0,
        "maximum_prediction_delta_um": 12.0,
        "measurement_offset_um": -3.0,
        "maximum_fit_residual_um": 2.0,
        "minimum_fit_inlier_fraction": 0.75,
        "maximum_fit_condition_number": 100.0,
        "maximum_plane_slope_um_per_um": 0.02,
        "maximum_observation_age_s": 60.0,
    }
    values.update(overrides)
    return FocusSurfaceSettings(**values)


def white_settings(**overrides):
    """Return explicit bounded test values, not commissioned hardware numbers."""
    values = {
        "search_z_range_um": (-12.0, 8.0),
        "white_search_timeout_s": 35.0,
        "total_focus_budget_s": 80.0,
        "approach_and_rg_reserve_s": 40.0,
        "maximum_white_led_disagreement_um": 4.0,
    }
    values.update(overrides)
    return WhiteSearchSettings(**values)


def quality(**overrides):
    """Return independently verified final LED measurement quality."""
    values = {
        "measurement_status": "ready",
        "confidence": 0.8,
        "accepted_patch_count": 6,
        "inlier_fraction": 0.75,
        "final_error_um": 0.25,
        "focus_tolerance_um": 1.5,
        "independent_post_move_verification": True,
    }
    values.update(overrides)
    return FocusObservationQuality(**values)


def provenance(**overrides):
    """Return final R/G provenance with distinct fresh frame identities."""
    values = {
        "result_id": "af-result-1",
        "report_ref": "reports/field-1.json",
        "rg_measurement_id": "rg-measurement-1",
        "capture_ids": ("red-1", "green-1"),
    }
    values.update(overrides)
    return FocusObservationProvenance(**values)


def observation(**overrides):
    """Return one confirmed LED observation in physical micrometres."""
    values = {
        "scan_id": "scan-1",
        "field_id": "field-1",
        "attempt_id": "attempt-1",
        "observation_id": "observation-1",
        "x_um": -125.5,
        "y_um": 250.25,
        "z_um": 0.0,
        "final_readback_settled": True,
        "final_readback_monotonic_s": 9.0,
        "recorded_monotonic_s": 10.0,
        "source": "led_autofocus",
        "status": "confirmed",
        "binding": binding(),
        "provenance": provenance(),
        "quality": quality(),
    }
    values.update(overrides)
    return FocusObservation(**values)


def diagnostics(**overrides):
    """Return explicit fit/support diagnostics, not an uncertainty interval."""
    values = {
        "candidate_observation_count": 4,
        "nearest_distance_um": 100.0,
        "support_age_s": 5.0,
        "fit_rmse_um": 0.5,
        "maximum_absolute_residual_um": 0.8,
        "fit_condition_number": 12.0,
        "extrapolation_distance_um": 0.0,
        "prediction_delta_um": -1.0,
    }
    values.update(overrides)
    return FocusPredictionDiagnostics(**values)


def usable_prediction(**overrides):
    """Return a local-plane prediction with four distinct supports."""
    values = {
        "scan_id": "scan-1",
        "field_id": "field-5",
        "attempt_id": "attempt-5",
        "prediction_id": "prediction-5",
        "target_x_um": -500.25,
        "target_y_um": 750.5,
        "target_z_um": 0.0,
        "status": "usable",
        "source": "local_plane",
        "reason": "Local support passed all configured gates",
        "binding": binding(),
        "support_observation_ids": ("o1", "o2", "o3", "o4"),
        "support_field_ids": ("f1", "f2", "f3", "f4"),
        "support_distances_um": (100.0, 200.0, 300.0, 400.0),
        "diagnostics": diagnostics(),
        "predicted_monotonic_s": 20.0,
    }
    values.update(overrides)
    return FocusPrediction(**values)


def surface_observation(index, x_um, y_um, z_um, **overrides):
    """Return a distinct accepted point for deterministic synthetic surfaces."""
    recorded = overrides.pop("recorded_monotonic_s", 90.0 + index * 0.01)
    values = {
        "field_id": f"support-field-{index}",
        "attempt_id": f"support-attempt-{index}",
        "observation_id": f"support-observation-{index:03d}",
        "x_um": x_um,
        "y_um": y_um,
        "z_um": z_um,
        "final_readback_monotonic_s": recorded - 0.005,
        "recorded_monotonic_s": recorded,
        "provenance": provenance(
            result_id=f"result-{index}",
            report_ref=f"reports/support-{index}.json",
            rg_measurement_id=f"rg-{index}",
            capture_ids=(f"red-{index}", f"green-{index}"),
        ),
    }
    values.update(overrides)
    return observation(**values)


def prediction_request(**overrides):
    """Return a bounded, frozen test query in physical micrometres."""
    values = {
        "expected_scan_id": "scan-1",
        "field_id": "target-field",
        "attempt_id": "target-attempt",
        "prediction_id": "target-prediction",
        "target_x_um": 0.0,
        "target_y_um": 0.0,
        "current_z_um": 0.0,
        "predicted_monotonic_s": 100.0,
        "expected_binding": binding(),
        "settings": surface_settings(),
    }
    values.update(overrides)
    return FocusPredictionRequest(**values)


def test_disabled_defaults_preserve_old_policy_and_have_no_reference_ttl():
    """E01/E28: missing new settings keep the old path and never age operator zero."""
    restored = FocusRunSettings.model_validate_json(
        '{"autofocus_method":"none","focus_strategy":"single_autofocus"}'
    )
    assert not restored.surface.enabled
    assert restored.unpredicted_focus_mode == "selected_method"
    assert restored.white_search is None
    assert restored.autofocus_method == "none"
    assert "initial_focus_mode" not in FocusRunSettings.model_fields
    assert "reference_max_age_s" not in FocusSurfaceSettings.model_fields
    assert "reference_max_age_s" not in FocusBinding.model_fields


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("neighbor_radius_um", float("nan")),
        ("maximum_prediction_delta_um", float("inf")),
        ("measurement_offset_um", -float("inf")),
        ("maximum_fit_residual_um", float("nan")),
        ("maximum_plane_slope_um_per_um", float("inf")),
        ("maximum_observation_age_s", float("inf")),
    ],
)
def test_surface_limits_are_finite(field, value):
    """E04/E09: non-finite physical, fit, and age gates fail closed."""
    with pytest.raises(ValidationError):
        surface_settings(**{field: value})


@pytest.mark.parametrize("field", ["x_um", "y_um", "z_um"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_observation_coordinates_are_finite_physical_um(field, value):
    """E04: every saved physical coordinate must be finite."""
    with pytest.raises(ValidationError):
        observation(**{field: value})


def test_enabled_surface_requires_coherent_explicit_operational_gates():
    """Uncommissioned defaults, too few plane points, and inconsistent limits reject."""
    assert not FocusSurfaceSettings().enabled
    with pytest.raises(ValidationError, match="every operational gate"):
        FocusSurfaceSettings(enabled=True)
    with pytest.raises(ValidationError):
        surface_settings(minimum_plane_points=3)
    with pytest.raises(ValidationError, match="Maximum neighbors"):
        surface_settings(minimum_plane_points=8, maximum_neighbors=7)
    with pytest.raises(ValidationError, match="candidate observations"):
        surface_settings(maximum_candidate_observations=8, maximum_neighbors=12)
    with pytest.raises(ValidationError, match="Extrapolation flag"):
        surface_settings(maximum_extrapolation_um=1.0)
    with pytest.raises(ValidationError, match="neighbor radius"):
        surface_settings(allow_extrapolation=True, maximum_extrapolation_um=2000.0)
    with pytest.raises(ValidationError, match="residual gate"):
        surface_settings(maximum_fit_residual_um=13.0)
    with pytest.raises(ValidationError, match="Measurement offset"):
        surface_settings(measurement_offset_um=-13.0)


@pytest.mark.parametrize("sign", [True, False, 1.0, -1.0, "1", 0, 2])
def test_axis_scales_require_positive_values_and_strict_integer_signs(sign):
    """E04: signs are exactly integer ±1, never truthy or coerced values."""
    with pytest.raises(ValidationError):
        FocusAxisScale(axis="x", units_per_mm=1000.0, direction_sign=sign)
    with pytest.raises(ValidationError):
        FocusAxisScale(axis="x", units_per_mm=0.0, direction_sign=1)


def test_binding_requires_exact_unique_xyz_scale_collection():
    """E04: three differently scaled signed axes remain explicit and unambiguous."""
    duplicate = (
        FocusAxisScale(axis="x", units_per_mm=1.0, direction_sign=1),
        FocusAxisScale(axis="x", units_per_mm=2.0, direction_sign=1),
        FocusAxisScale(axis="z", units_per_mm=3.0, direction_sign=1),
    )
    with pytest.raises(ValidationError, match="exactly once"):
        binding(axis_scales=duplicate)


def test_json_arrays_round_trip_as_tuples_without_numeric_string_coercion():
    """The decoded settings boundary accepts arrays but rejects wrong item types."""
    run = FocusRunSettings(
        autofocus_method="led",
        focus_strategy="single_autofocus",
        surface=surface_settings(),
        unpredicted_focus_mode="white_then_led",
        white_search=white_settings(),
        binding=binding(),
    )
    payload = json.loads(run.model_dump_json())
    restored = FocusRunSettings.model_validate(payload)
    assert restored == run
    assert isinstance(restored.binding.axis_scales, tuple)
    assert isinstance(restored.white_search.search_z_range_um, tuple)

    payload["binding"]["axis_scales"][0]["units_per_mm"] = "1000"
    with pytest.raises(ValidationError):
        FocusRunSettings.model_validate(payload)
    payload = json.loads(run.model_dump_json())
    payload["white_search"]["search_z_range_um"][0] = "-12"
    with pytest.raises(ValidationError):
        FocusRunSettings.model_validate(payload)
    payload = json.loads(run.model_dump_json())
    payload["binding"]["approach_parameters"]["approach_sign"] = True
    with pytest.raises(ValidationError):
        FocusRunSettings.model_validate(payload)


def test_models_and_nested_collections_are_frozen_snapshots():
    """E20/E28: later form or nested collection changes cannot mutate a run."""
    run = FocusRunSettings(
        autofocus_method="led",
        focus_strategy="single_autofocus",
        surface=surface_settings(),
        binding=binding(),
    )
    assert isinstance(run.binding.axis_scales, tuple)
    with pytest.raises(ValidationError):
        run.unpredicted_focus_mode = "white_then_led"
    with pytest.raises(ValidationError):
        run.surface.enabled = False
    with pytest.raises(ValidationError):
        run.binding.axis_scales[0].units_per_mm = 999.0
    with pytest.raises(ValidationError):
        run.binding.approach_parameters.preload_um = 9


def test_binding_changes_and_invalid_statuses_are_distinct_fail_closed_results():
    """E19/E20: lost zero, changed geometry/scale, and candidate profiles invalidate use."""
    expected = binding()
    changed_geometry = binding(geometry_id="other-geometry")
    changed = check_focus_binding(expected, changed_geometry)
    assert not changed.compatible
    assert not changed.usable
    assert changed.mismatched_fields == ("geometry_id",)

    changed_scale = binding(
        axis_scales=(
            FocusAxisScale(axis="x", units_per_mm=999.0, direction_sign=1),
            *expected.axis_scales[1:],
        )
    )
    assert "axis_scales" in check_focus_binding(expected, changed_scale).mismatched_fields

    lost = check_focus_binding(expected, binding(reference_status="lost"))
    assert lost.compatible
    assert not lost.usable
    assert "current.reference_status" in lost.mismatched_fields

    candidate = check_focus_binding(expected, binding(rg_focus_model_status="candidate"))
    assert candidate.compatible
    assert not candidate.usable
    assert "current.rg_focus_model_status" in candidate.mismatched_fields


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("reference_status", "lost"),
        ("rg_focus_model_status", "candidate"),
    ],
)
def test_binding_readiness_is_required_on_both_sides(field, invalid_value):
    """E19/E20: a lost/candidate frozen side cannot become valid by status change."""
    valid = binding()
    invalid = binding(**{field: invalid_value})

    frozen_invalid = check_focus_binding(invalid, valid)
    assert frozen_invalid.compatible
    assert not frozen_invalid.usable
    assert f"expected.{field}" in frozen_invalid.mismatched_fields
    assert "Frozen expected" in frozen_invalid.reason

    current_invalid = check_focus_binding(valid, invalid)
    assert current_invalid.compatible
    assert not current_invalid.usable
    assert f"current.{field}" in current_invalid.mismatched_fields
    assert "Current" in current_invalid.reason

    both_invalid = check_focus_binding(invalid, invalid)
    assert both_invalid.compatible
    assert not both_invalid.usable
    assert both_invalid.mismatched_fields == (
        f"expected.{field}",
        f"current.{field}",
    )
    assert "Frozen and current" in both_invalid.reason


def test_zero_is_a_confirmed_value_with_commanded_not_encoder_readback():
    """E01: Z=0 is not null and readback provenance does not claim an encoder."""
    item = observation(z_um=0.0)
    result = assess_focus_observation(
        item,
        expected_scan_id="scan-1",
        expected_binding=binding(),
        settings=surface_settings(),
        now_monotonic_s=20.0,
    )
    assert result.usable
    assert item.z_um == 0.0
    assert item.readback_kind == "commanded_not_encoder"
    assert "encoder" not in FocusObservation.model_fields


@pytest.mark.parametrize(
    ("source", "status"),
    [
        ("seed", "confirmed"),
        ("white_autofocus", "confirmed"),
        ("surface_prediction", "confirmed"),
        ("preload", "confirmed"),
        ("none", "confirmed"),
    ],
)
def test_numeric_non_led_z_cannot_become_confirmed(source, status):
    """E01/E06: seed/WHITE/prediction/preload/none never train by numeric Z alone."""
    with pytest.raises(ValidationError, match="Only LED autofocus"):
        observation(source=source, status=status)


def test_none_white_failed_and_candidate_results_are_not_surface_support():
    """E06: imaged/intermediate/failed/candidate values stay out of the map."""
    settings = surface_settings()
    expected = binding()
    rows = (
        observation(source="none", status="imaged", quality=None),
        observation(source="white_autofocus", status="intermediate", quality=None),
        observation(source="led_autofocus", status="failed", quality=None),
        observation(source="led_autofocus", status="candidate"),
    )
    for row in rows:
        assessed = assess_focus_observation(
            row,
            expected_scan_id="scan-1",
            expected_binding=expected,
            settings=settings,
            now_monotonic_s=20.0,
        )
        assert not assessed.usable

    with pytest.raises(ValidationError, match="valid LED binding"):
        observation(binding=binding(rg_focus_model_status="candidate"))


def test_confirmed_led_requires_settled_readback_qc_and_provenance():
    """A numerical final position does not replace final LED evidence."""
    with pytest.raises(ValidationError, match="settled"):
        observation(final_readback_settled=False)
    with pytest.raises(ValidationError, match="verified LED QC"):
        observation(quality=quality(independent_post_move_verification=False))
    with pytest.raises(ValidationError, match="verified LED QC"):
        observation(quality=quality(final_error_um=2.0))
    with pytest.raises(ValidationError, match="measurement provenance"):
        observation(provenance=provenance(rg_measurement_id=None))


def test_pre_update_prediction_error_and_monotonic_order_are_validated():
    """E09: pre-update error is preserved and timestamps stay monotonic in-session."""
    evidence = FocusPredictionEvidence(
        prediction_id="prediction-1",
        predicted_z_um=-2.0,
        prediction_error_um=2.0,
        support_age_s=5.0,
    )
    assert observation(pre_update_prediction=evidence).pre_update_prediction == evidence
    with pytest.raises(ValidationError, match="prediction error"):
        observation(
            pre_update_prediction=FocusPredictionEvidence(
                prediction_id="prediction-1",
                predicted_z_um=-2.0,
                prediction_error_um=1.0,
                support_age_s=5.0,
            )
        )
    with pytest.raises(ValidationError, match="readback"):
        observation(final_readback_monotonic_s=11.0)


def test_stale_is_unusable_but_future_time_is_fault_without_reference_ttl():
    """E09: support ages out, while invalid session time faults without zero TTL."""
    stale = assess_focus_observation(
        observation(),
        expected_scan_id="scan-1",
        expected_binding=binding(),
        settings=surface_settings(maximum_observation_age_s=5.0),
        now_monotonic_s=20.0,
    )
    assert not stale.usable
    assert stale.age_s == 10.0
    assert "stale" in stale.reason

    with pytest.raises(FocusObservationContractError) as error:
        assess_focus_observation(
            observation(),
            expected_scan_id="scan-1",
            expected_binding=binding(),
            settings=surface_settings(),
            now_monotonic_s=9.0,
        )
    assert error.value.code == "future_timestamp"


@pytest.mark.parametrize("rows", [("foreign",), ("good", "foreign")])
def test_single_foreign_or_mixed_scan_is_a_fault(rows):
    """A shared reference does not make monotonic times comparable across scans."""
    good = observation(observation_id="good", field_id="good", attempt_id="good")
    foreign = observation(
        scan_id="scan-2",
        observation_id="foreign",
        field_id="foreign",
        attempt_id="foreign",
        x_um=10.0,
    )
    lookup = {"good": good, "foreign": foreign}
    with pytest.raises(FocusObservationContractError) as error:
        select_usable_observations(
            tuple(lookup[name] for name in rows),
            expected_scan_id="scan-1",
            expected_binding=binding(),
            settings=surface_settings(),
            now_monotonic_s=20.0,
        )
    assert error.value.code == "foreign_scan"
    assert error.value.observation_id == "foreign"


def test_binding_mismatch_is_not_lost_after_a_good_observation():
    """E19/E25: incompatible frozen/live identities fault instead of unavailable."""
    good = observation(observation_id="good", field_id="good", attempt_id="good")
    mismatched = observation(
        observation_id="mismatch",
        field_id="mismatch",
        attempt_id="mismatch",
        x_um=10.0,
        binding=binding(reference_id="other-reference"),
    )
    with pytest.raises(FocusObservationContractError) as error:
        select_usable_observations(
            (good, mismatched),
            expected_scan_id="scan-1",
            expected_binding=binding(),
            settings=surface_settings(),
            now_monotonic_s=20.0,
        )
    assert error.value.code == "binding_mismatch"
    assert error.value.observation_id == "mismatch"

    with pytest.raises(FocusObservationContractError) as single_error:
        select_usable_observations(
            (good,),
            expected_scan_id="scan-1",
            expected_binding=binding(reference_id="other-reference"),
            settings=surface_settings(),
            now_monotonic_s=20.0,
        )
    assert single_error.value.code == "binding_mismatch"


@pytest.mark.parametrize("field", ["reference_status", "rg_focus_model_status"])
@pytest.mark.parametrize("invalid_side", ["frozen", "current"])
def test_unready_binding_on_either_selection_side_is_a_fault(field, invalid_side):
    """Lost/candidate frozen or live status cannot degrade to empty optical support."""
    invalid_value = "lost" if field == "reference_status" else "candidate"
    invalid = binding(**{field: invalid_value})
    if invalid_side == "frozen":
        row = observation(status="candidate", binding=invalid)
        expected = binding()
    else:
        row = observation()
        expected = invalid
    with pytest.raises(FocusObservationContractError) as error:
        select_usable_observations(
            (row,),
            expected_scan_id="scan-1",
            expected_binding=expected,
            settings=surface_settings(),
            now_monotonic_s=20.0,
        )
    assert error.value.code == "binding_unusable"


def test_future_timestamp_is_not_lost_after_a_good_observation():
    """A future session timestamp faults even when another support is usable."""
    good = observation(observation_id="good", field_id="good", attempt_id="good")
    future = observation(
        observation_id="future",
        field_id="future",
        attempt_id="future",
        x_um=10.0,
        final_readback_monotonic_s=29.0,
        recorded_monotonic_s=30.0,
    )
    with pytest.raises(FocusObservationContractError) as error:
        select_usable_observations(
            (good, future),
            expected_scan_id="scan-1",
            expected_binding=binding(),
            settings=surface_settings(),
            now_monotonic_s=20.0,
        )
    assert error.value.code == "future_timestamp"
    assert error.value.observation_id == "future"


@pytest.mark.parametrize(
    ("overrides", "error_type"),
    [
        ({"expected_scan_id": ""}, ValueError),
        ({"expected_scan_id": " "}, ValueError),
        ({"now_monotonic_s": float("nan")}, ValueError),
        ({"now_monotonic_s": float("inf")}, ValueError),
        ({"now_monotonic_s": -1.0}, ValueError),
        ({"now_monotonic_s": True}, TypeError),
        ({"now_monotonic_s": "20"}, TypeError),
    ],
)
def test_empty_selection_still_validates_session_inputs(overrides, error_type):
    """No observations must not hide a damaged session boundary."""
    kwargs = {
        "expected_scan_id": "scan-1",
        "expected_binding": binding(),
        "settings": surface_settings(),
        "now_monotonic_s": 20.0,
    }
    kwargs.update(overrides)
    with pytest.raises(error_type):
        select_usable_observations([], **kwargs)


@pytest.mark.parametrize(
    "overrides",
    [{"reference_status": "lost"}, {"rg_focus_model_status": "candidate"}],
)
def test_empty_selection_cannot_hide_an_unusable_binding(overrides):
    """Before the first field, a bad binding is a fault, not missing optical support."""
    with pytest.raises(FocusObservationContractError) as error:
        select_usable_observations(
            [],
            expected_scan_id="scan-1",
            expected_binding=binding(**overrides),
            settings=surface_settings(),
            now_monotonic_s=20.0,
        )
    assert error.value.code == "binding_unusable"
    assert error.value.observation_id is None


def test_empty_selection_with_valid_context_is_ordinary_missing_support():
    """A valid new scan still starts with no observations and no fault."""
    assert (
        select_usable_observations(
            [],
            expected_scan_id="scan-1",
            expected_binding=binding(),
            settings=surface_settings(),
            now_monotonic_s=0.0,
        )
        == ()
    )


def test_stale_rejected_and_disabled_observations_remain_nonfault_empty_support():
    """Stale/ordinary rejection/disabled stay optical-unusable, not contract faults."""
    stale = select_usable_observations(
        (observation(),),
        expected_scan_id="scan-1",
        expected_binding=binding(),
        settings=surface_settings(maximum_observation_age_s=5.0),
        now_monotonic_s=20.0,
    )
    assert stale == ()

    rejected = observation(source="led_autofocus", status="failed", quality=None)
    assert (
        select_usable_observations(
            (rejected,),
            expected_scan_id="scan-1",
            expected_binding=binding(),
            settings=surface_settings(),
            now_monotonic_s=20.0,
        )
        == ()
    )

    foreign = observation(scan_id="scan-2")
    assert (
        select_usable_observations(
            (foreign,),
            expected_scan_id="scan-1",
            expected_binding=binding(reference_id="other-reference"),
            settings=FocusSurfaceSettings(),
            now_monotonic_s=1.0,
        )
        == ()
    )


def test_retry_dedup_and_latest_good_at_xy_do_not_let_failure_erase_history():
    """E05: retry-save has one weight; a later failure cannot replace a good point."""
    first = observation()
    failure = observation(
        field_id="field-1",
        attempt_id="attempt-2",
        observation_id="observation-failed",
        source="led_autofocus",
        status="failed",
        quality=None,
        recorded_monotonic_s=20.0,
        final_readback_monotonic_s=19.0,
    )
    selected = select_usable_observations(
        (first, first, failure),
        expected_scan_id="scan-1",
        expected_binding=binding(),
        settings=surface_settings(),
        now_monotonic_s=30.0,
    )
    assert selected == (first,)

    newer = observation(
        field_id="field-1",
        attempt_id="attempt-3",
        observation_id="observation-2",
        z_um=1.25,
        recorded_monotonic_s=25.0,
        final_readback_monotonic_s=24.0,
    )
    selected = select_usable_observations(
        (first, failure, newer),
        expected_scan_id="scan-1",
        expected_binding=binding(),
        settings=surface_settings(),
        now_monotonic_s=30.0,
    )
    assert selected == (newer,)

    conflicting = observation(z_um=1.0)
    with pytest.raises(ValueError, match="Conflicting observations"):
        select_usable_observations(
            (first, conflicting),
            expected_scan_id="scan-1",
            expected_binding=binding(),
            settings=surface_settings(),
            now_monotonic_s=30.0,
        )


def test_only_usable_prediction_carries_target_and_unique_optical_support():
    """E01/E25: usable Z=0 has support; other statuses have no target/support payload."""
    prediction = usable_prediction(target_z_um=0.0)
    assert prediction.target_z_um == 0.0
    assert prediction.diagnostics.interpretation == ("diagnostic_not_confidence_interval")
    assert "confidence_interval" not in FocusPredictionDiagnostics.model_fields

    with pytest.raises(ValidationError, match="unique"):
        usable_prediction(support_observation_ids=("o1", "o1", "o3", "o4"))
    with pytest.raises(ValidationError, match="at least four"):
        usable_prediction(
            support_observation_ids=("o1",),
            support_field_ids=("f1",),
            support_distances_um=(100.0,),
        )
    with pytest.raises(ValidationError, match="Only usable"):
        FocusPrediction(
            scan_id="scan-1",
            field_id="field-1",
            attempt_id="attempt-1",
            prediction_id="p-disabled",
            target_x_um=0.0,
            target_y_um=0.0,
            target_z_um=0.0,
            status="disabled",
            reason="Surface is off",
            predicted_monotonic_s=1.0,
        )


def test_unavailable_is_optical_only_and_distinct_from_disabled_and_fault():
    """E25: expected support gaps are unavailable; reference/profile failures are fault."""
    unavailable = FocusPrediction(
        scan_id="scan-1",
        field_id="field-1",
        attempt_id="attempt-1",
        prediction_id="p-unavailable",
        target_x_um=0.0,
        target_y_um=0.0,
        status="unavailable",
        source="seed",
        reason="No accepted optical support yet",
        unavailable_reason="no_observations",
        binding=binding(),
        diagnostics=diagnostics(candidate_observation_count=0, support_age_s=None),
        predicted_monotonic_s=1.0,
    )
    assert unavailable.target_z_um is None
    assert unavailable.support_observation_ids == ()

    with pytest.raises(ValidationError, match="cannot be labelled"):
        FocusPrediction(
            scan_id="scan-1",
            field_id="field-1",
            attempt_id="attempt-1",
            prediction_id="p-wrong-unavailable",
            target_x_um=0.0,
            target_y_um=0.0,
            status="unavailable",
            reason="Reference was lost",
            unavailable_reason="no_observations",
            binding=binding(reference_status="lost"),
            predicted_monotonic_s=1.0,
        )

    disabled = FocusPrediction(
        scan_id="scan-1",
        field_id="field-1",
        attempt_id="attempt-1",
        prediction_id="p-disabled",
        target_x_um=0.0,
        target_y_um=0.0,
        status="disabled",
        reason="Surface is disabled",
        predicted_monotonic_s=1.0,
    )
    fault = FocusPrediction(
        scan_id="scan-1",
        field_id="field-1",
        attempt_id="attempt-1",
        prediction_id="p-fault",
        target_x_um=0.0,
        target_y_um=0.0,
        status="fault",
        reason="Frozen binding no longer matches",
        binding=binding(reference_status="lost"),
        fault_code="binding_unusable",
        predicted_monotonic_s=1.0,
    )
    assert {unavailable.status, disabled.status, fault.status} == {
        "unavailable",
        "disabled",
        "fault",
    }


def test_white_then_led_only_accepts_enabled_led_single_with_valid_binding():
    """E25/E28: mixed focus is explicit and unsupported active combinations reject."""
    with pytest.raises(ValidationError, match="enabled focus surface"):
        FocusRunSettings(
            autofocus_method="led",
            focus_strategy="single_autofocus",
            unpredicted_focus_mode="white_then_led",
        )
    with pytest.raises(ValidationError, match="only LED single"):
        FocusRunSettings(
            autofocus_method="openflexure",
            focus_strategy="single_autofocus",
            surface=surface_settings(),
            binding=binding(),
        )
    with pytest.raises(ValidationError, match="only LED single"):
        FocusRunSettings(
            autofocus_method="led",
            focus_strategy="smart_stack",
            surface=surface_settings(),
            binding=binding(),
        )
    with pytest.raises(ValidationError, match="valid compatible profiles"):
        FocusRunSettings(
            autofocus_method="led",
            focus_strategy="single_autofocus",
            surface=surface_settings(),
            binding=binding(approach_status="candidate"),
        )
    with pytest.raises(ValidationError, match="bounded WHITE"):
        FocusRunSettings(
            autofocus_method="led",
            focus_strategy="single_autofocus",
            surface=surface_settings(),
            unpredicted_focus_mode="white_then_led",
            binding=binding(),
        )

    run = FocusRunSettings(
        autofocus_method="led",
        focus_strategy="single_autofocus",
        surface=surface_settings(),
        unpredicted_focus_mode="white_then_led",
        white_search=white_settings(),
        binding=binding(),
    )
    assert run.white_search.maximum_white_searches_per_field == 1


def test_white_budget_is_bounded_and_preserves_approach_rg_reserve():
    """WHITE search range, timeout, transfer reserve and disagreement gate are coherent."""
    with pytest.raises(ValidationError, match="both sides"):
        white_settings(search_z_range_um=(0.0, 12.0))
    with pytest.raises(ValidationError, match="exceed total"):
        white_settings(total_focus_budget_s=70.0, approach_and_rg_reserve_s=40.0)
    with pytest.raises(ValidationError, match="disagreement"):
        white_settings(maximum_white_led_disagreement_um=21.0)
    with pytest.raises(ValidationError):
        white_settings(maximum_white_searches_per_field=2)


def test_field_progress_round_trip_records_one_white_without_retry():
    """E27: field/attempt stages survive JSON and cannot represent a second WHITE."""
    planned = FocusFieldProgress(
        scan_id="scan-1",
        field_id="field-1",
        attempt_id="attempt-1",
        prediction_id="prediction-1",
        stage="planned",
        white_search_id="white-search-1",
        white_status="unknown",
        rg_status="not_started",
        recorded_monotonic_s=10.0,
    )
    restored = FocusFieldProgress.model_validate_json(planned.model_dump_json())
    assert restored == planned
    assert not restored.automatic_white_retry_allowed
    assert restored.white_search_count == 1

    completed = FocusFieldProgress(
        scan_id="scan-1",
        field_id="field-1",
        attempt_id="attempt-1",
        prediction_id="prediction-1",
        stage="white_completed",
        white_search_id="white-search-1",
        white_status="succeeded",
        white_result_id="white-result-1",
        rg_status="failed",
        rg_result_id="rg-result-1",
        recorded_monotonic_s=20.0,
    )
    assert completed.stage == "white_completed"
    with pytest.raises(ValidationError):
        FocusFieldProgress(**{**planned.model_dump(), "white_search_count": 2})
    with pytest.raises(ValidationError):
        FocusFieldProgress(**{**planned.model_dump(), "automatic_white_retry_allowed": True})
    with pytest.raises(ValidationError, match="Planned stage"):
        FocusFieldProgress(
            **{
                **planned.model_dump(),
                "white_status": "succeeded",
                "white_result_id": "white-result-1",
            }
        )


def test_rg_verified_requires_both_explicit_results():
    """E27: intermediate WHITE alone never becomes a verified LED field result."""
    verified = FocusFieldProgress(
        scan_id="scan-1",
        field_id="field-1",
        attempt_id="attempt-1",
        prediction_id="prediction-1",
        stage="rg_verified",
        white_search_id="white-search-1",
        white_status="succeeded",
        white_result_id="white-result-1",
        rg_status="succeeded",
        rg_result_id="rg-result-1",
        recorded_monotonic_s=30.0,
    )
    assert verified.stage == "rg_verified"
    with pytest.raises(ValidationError, match="rg_verified"):
        FocusFieldProgress(
            **{
                **verified.model_dump(),
                "rg_status": "unknown",
                "rg_result_id": None,
            }
        )


def test_predictor_separates_disabled_unavailable_and_fault_states():
    """E01/E25: off, optical absence, and invalid binding remain disjoint."""
    disabled = predict_focus_surface(
        (observation(scan_id="foreign-scan"),),
        prediction_request(settings=FocusSurfaceSettings(), expected_binding=None),
    )
    assert disabled.status == "disabled"
    assert disabled.unavailable_reason is None

    unavailable = predict_focus_surface((), prediction_request())
    assert unavailable.status == "unavailable"
    assert unavailable.unavailable_reason == "no_observations"
    assert unavailable.diagnostics.candidate_observation_count == 0

    fault = predict_focus_surface(
        (), prediction_request(expected_binding=binding(reference_status="lost"))
    )
    assert fault.status == "fault"
    assert fault.fault_code == "binding_unusable"
    assert fault.unavailable_reason is None

    missing_binding = predict_focus_surface((), prediction_request(expected_binding=None))
    assert missing_binding.status == "fault"
    assert missing_binding.fault_code == "binding_unusable"

    foreign_scan = predict_focus_surface(
        (observation(scan_id="foreign-scan"),), prediction_request()
    )
    assert foreign_scan.status == "fault"
    assert foreign_scan.fault_code == "foreign_scan"


def test_seed_white_and_failed_results_do_not_train_the_predictor():
    """E01/E06: numerical non-LED and failed rows remain optical non-support."""
    rows = (
        observation(source="seed", status="candidate", quality=None),
        observation(
            observation_id="white",
            source="white_autofocus",
            status="intermediate",
            quality=None,
        ),
        observation(
            observation_id="failed",
            source="led_autofocus",
            status="failed",
            quality=None,
        ),
    )
    result = predict_focus_surface(rows, prediction_request())
    assert result.status == "unavailable"
    assert result.unavailable_reason == "insufficient_support"
    assert result.support_observation_ids == ()


def test_unexpected_programmer_error_is_not_relabelled_optical_unavailable():
    """E25: only the narrow contract fault is converted to a prediction fault."""

    class ExplodingSequence:
        """A deliberately broken caller-owned candidate collection."""

        def __len__(self):
            """Raise an error outside the predictor's narrow fault contract."""
            raise RuntimeError("caller sequence failed")

    with pytest.raises(RuntimeError, match="caller sequence failed"):
        predict_focus_surface(ExplodingSequence(), prediction_request())


def test_zero_and_limited_first_row_use_bounded_nearest_without_a_plane():
    """E01/E02: Z=0 and one to three collinear supports never infer Y slope."""
    rows = tuple(surface_observation(index, index * 100.0, 0.0, float(index)) for index in range(3))
    one = predict_focus_surface((rows[0],), prediction_request(target_x_um=10.0))
    assert one.status == "usable"
    assert one.source == "nearest_neighbor"
    assert one.target_z_um == 0.0
    assert one.support_distances_um == (10.0,)

    three = predict_focus_surface(rows, prediction_request(target_x_um=190.0, target_y_um=50.0))
    assert three.status == "usable"
    assert three.source == "nearest_neighbor"
    assert three.support_observation_ids == (rows[2].observation_id,)


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_sparse_conflicting_heights_do_not_choose_an_arbitrary_nearest(count, reverse):
    """E07: a sparse discontinuity is unavailable before any nearest-Z choice."""
    rows = (
        surface_observation(0, -10.0, 0.0, -8.0),
        surface_observation(1, 10.0, 0.0, 8.0),
        surface_observation(2, 0.0, 10.0, 0.0),
    )[:count]
    if reverse:
        rows = tuple(reversed(rows))
    result = predict_focus_surface(
        rows,
        prediction_request(
            settings=surface_settings(
                maximum_plane_slope_um_per_um=0.1,
                maximum_fit_residual_um=0.25,
                maximum_prediction_delta_um=20.0,
            )
        ),
    )
    assert result.status == "unavailable"
    assert result.unavailable_reason == "support_discontinuity"
    assert result.target_z_um is None
    assert result.support_observation_ids == ()


@pytest.mark.parametrize("count", [2, 3])
@pytest.mark.parametrize(("height", "usable"), [(2.5, True), (2.500001, False)])
def test_sparse_support_checks_pairwise_slope_and_two_residuals(count, height, usable):
    """The exact shared-plane bound is inclusive and uses physical XY distance."""
    rows = (
        surface_observation(0, 0.0, 0.0, 0.0),
        surface_observation(1, 12.0, 16.0, height),
        surface_observation(2, 6.0, 8.0, height / 2),
    )[:count]
    result = predict_focus_surface(
        rows,
        prediction_request(
            settings=surface_settings(
                maximum_plane_slope_um_per_um=0.1,
                maximum_fit_residual_um=0.25,
            )
        ),
    )
    if usable:
        assert result.status == "usable"
        assert result.source == "nearest_neighbor"
        assert result.target_z_um == 0.0
    else:
        assert result.status == "unavailable"
        assert result.unavailable_reason == "support_discontinuity"


@pytest.mark.parametrize("surface_name", ["flat", "tilted"])
def test_local_plane_recovers_predeclared_flat_and_tilted_holdouts(surface_name):
    """E03/E24: target holdouts are predicted before, and absent from, support."""
    points = (
        (-500.0, -500.0),
        (-500.0, 500.0),
        (500.0, -500.0),
        (500.0, 500.0),
    )
    if surface_name == "flat":
        expected = 2.5
        heights = (expected,) * len(points)
    else:
        expected = 1.25
        heights = tuple(1.25 + 0.003 * x - 0.002 * y for x, y in points)
    rows = tuple(
        surface_observation(index, x, y, z)
        for index, ((x, y), z) in enumerate(zip(points, heights, strict=True))
    )
    result = predict_focus_surface(rows, prediction_request(current_z_um=expected))
    assert result.status == "usable"
    assert result.source == "local_plane"
    assert result.target_z_um == pytest.approx(expected, abs=1e-12)
    assert len(result.support_observation_ids) == 4
    assert "target" not in result.support_observation_ids
    assert result.diagnostics.fit_rank == 3
    assert result.diagnostics.interpretation == "diagnostic_not_confidence_interval"


@pytest.mark.parametrize("surface_name", ["flat", "tilted", "smooth_curved"])
def test_predeclared_holdouts_compare_plane_with_same_bounded_nearest_baseline(
    surface_name,
):
    """E24: compare algorithms on untouched targets, without claiming physical accuracy."""
    points = tuple(
        (x, y) for x in (-500.0, 0.0, 500.0) for y in (-500.0, 0.0, 500.0) if (x, y) != (0.0, 0.0)
    )
    if surface_name == "flat":
        heights = (0.0,) * len(points)
    elif surface_name == "tilted":
        heights = tuple(0.002 * x - 0.001 * y for x, y in points)
    else:
        heights = tuple(0.002 * x - 0.001 * y + 0.0000005 * (x * x + y * y) for x, y in points)
    rows = tuple(
        surface_observation(index, x, y, z)
        for index, ((x, y), z) in enumerate(zip(points, heights, strict=True))
    )
    plane = predict_focus_surface(rows, prediction_request())
    nearest = min(
        rows,
        key=lambda item: (
            math.hypot(item.x_um, item.y_um),
            item.observation_id,
        ),
    )
    assert plane.status == "usable"
    assert plane.source == "local_plane"
    plane_error = abs(plane.target_z_um)
    nearest_error = abs(nearest.z_um)
    if surface_name == "flat":
        assert plane_error == nearest_error == 0.0
    else:
        assert plane_error < nearest_error


@pytest.mark.parametrize(
    "ys",
    [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.001, -0.001, 0.002),
    ],
)
def test_degenerate_or_nearly_collinear_plane_does_not_fallback_to_nearest(ys):
    """E02: four unstable points cannot hide behind a convenient nearest result."""
    xs = (-600.0, -200.0, 200.0, 600.0)
    rows = tuple(
        surface_observation(index, x, y, 0.001 * x)
        for index, (x, y) in enumerate(zip(xs, ys, strict=True))
    )
    result = predict_focus_surface(rows, prediction_request())
    assert result.status == "unavailable"
    assert result.unavailable_reason == "poor_local_fit"
    assert result.source is None
    assert result.diagnostics.fit_rank in (2, 3)


def test_one_independent_outlier_is_excluded_but_fold_is_unavailable():
    """E07: one unique outlier is robust; a multi-point step is not one plane."""
    ring = tuple(
        (x, y) for x in (-500.0, 0.0, 500.0) for y in (-500.0, 0.0, 500.0) if (x, y) != (0.0, 0.0)
    )
    rows = []
    for index, (x, y) in enumerate(ring):
        z = 0.002 * x - 0.001 * y
        if index == 0:
            z += 8.0
        rows.append(surface_observation(index, x, y, z))
    robust = predict_focus_surface(
        tuple(rows),
        prediction_request(settings=surface_settings(maximum_fit_residual_um=0.5)),
    )
    assert robust.status == "usable"
    assert robust.target_z_um == pytest.approx(0.0, abs=1e-12)
    assert robust.diagnostics.excluded_observation_ids == (rows[0].observation_id,)
    assert rows[0].observation_id not in robust.support_observation_ids

    fold_rows = tuple(
        surface_observation(
            index,
            x,
            y,
            -4.0 if x < 0 else 4.0,
        )
        for index, (x, y) in enumerate(
            pair for x in (-600.0, -200.0, 200.0, 600.0) for pair in ((x, -200.0), (x, 200.0))
        )
    )
    fold = predict_focus_surface(
        fold_rows,
        prediction_request(settings=surface_settings(maximum_fit_residual_um=0.5)),
    )
    assert fold.status == "unavailable"
    assert fold.unavailable_reason == "support_discontinuity"


def test_plane_slope_gate_rejects_an_otherwise_exact_fit():
    """E07: a low-residual but over-steep plane is not treated as reliable."""
    rows = tuple(
        surface_observation(index, x, y, 0.03 * x)
        for index, (x, y) in enumerate(
            ((-100.0, -100.0), (-100.0, 100.0), (100.0, -100.0), (100.0, 100.0))
        )
    )
    result = predict_focus_surface(rows, prediction_request())
    assert result.status == "unavailable"
    assert result.unavailable_reason == "poor_local_fit"
    assert result.diagnostics.maximum_absolute_residual_um == pytest.approx(0.0)
    assert result.diagnostics.plane_slope_magnitude_um_per_um == pytest.approx(0.03)


def test_extrapolation_is_off_by_default_and_then_strictly_bounded():
    """E08: a target outside the support hull needs an explicit distance allowance."""
    rows = tuple(
        surface_observation(index, x, y, 0.001 * x)
        for index, (x, y) in enumerate(((0.0, 0.0), (0.0, 100.0), (100.0, 0.0), (100.0, 100.0)))
    )
    target = {"target_x_um": 110.0, "target_y_um": 50.0}
    blocked = predict_focus_surface(rows, prediction_request(**target))
    assert blocked.status == "unavailable"
    assert blocked.unavailable_reason == "outside_support"
    assert blocked.diagnostics.extrapolation_distance_um == pytest.approx(10.0)

    allowed = predict_focus_surface(
        rows,
        prediction_request(
            **target,
            settings=surface_settings(allow_extrapolation=True, maximum_extrapolation_um=20.0),
        ),
    )
    assert allowed.status == "usable"
    assert allowed.extrapolated

    too_far = predict_focus_surface(
        rows,
        prediction_request(
            **target,
            settings=surface_settings(allow_extrapolation=True, maximum_extrapolation_um=5.0),
        ),
    )
    assert too_far.status == "unavailable"
    assert too_far.unavailable_reason == "outside_support"


def test_remote_island_cannot_supply_nearest_or_plane_support():
    """E08: old focus on a distant tissue island does not become a prediction."""
    rows = tuple(
        surface_observation(index, x, y, 1.0)
        for index, (x, y) in enumerate(((0.0, 0.0), (0.0, 100.0), (100.0, 0.0), (100.0, 100.0)))
    )
    result = predict_focus_surface(
        rows,
        prediction_request(
            target_x_um=10000.0,
            target_y_um=-10000.0,
            settings=surface_settings(neighbor_radius_um=500.0),
        ),
    )
    assert result.status == "unavailable"
    assert result.unavailable_reason == "outside_support"
    assert result.diagnostics.nearest_distance_um > 13000.0


def test_prediction_delta_uses_explicit_current_physical_z_and_is_diagnostic():
    """E08: current Z, not a seed or frame index, gates the numerical prediction."""
    row = surface_observation(0, 0.0, 0.0, 0.0)
    result = predict_focus_surface((row,), prediction_request(current_z_um=-20.0))
    assert result.status == "unavailable"
    assert result.unavailable_reason == "prediction_delta_exceeded"
    assert result.diagnostics.prediction_delta_um == 20.0


def test_stale_support_is_optical_unavailable_but_future_time_is_fault():
    """E09/E25: expiry may allow policy; invalid session time must stop it."""
    stale = surface_observation(0, 0.0, 0.0, 0.0, recorded_monotonic_s=90.0)
    stale_result = predict_focus_surface(
        (stale,),
        prediction_request(settings=surface_settings(maximum_observation_age_s=5.0)),
    )
    assert stale_result.status == "unavailable"
    assert stale_result.unavailable_reason == "stale_support"

    future = surface_observation(1, 0.0, 0.0, 0.0, recorded_monotonic_s=101.0)
    future_result = predict_focus_surface((future,), prediction_request())
    assert future_result.status == "fault"
    assert future_result.fault_code == "future_timestamp"


def test_duplicate_id_and_ambiguous_latest_xy_are_narrow_faults():
    """E05/E25: corrupt identity/time data cannot become no observations."""
    first = surface_observation(0, 0.0, 0.0, 0.0)
    conflict = first.model_copy(update={"z_um": 1.0})
    conflict_result = predict_focus_surface((first, conflict), prediction_request())
    assert conflict_result.status == "fault"
    assert conflict_result.fault_code == "observation_id_conflict"

    ambiguous = surface_observation(
        1,
        0.0,
        0.0,
        1.0,
        recorded_monotonic_s=first.recorded_monotonic_s,
        final_readback_monotonic_s=first.final_readback_monotonic_s,
    )
    ambiguous_result = predict_focus_surface((ambiguous, first), prediction_request())
    ambiguous_reverse = predict_focus_surface((first, ambiguous), prediction_request())
    assert ambiguous_result == ambiguous_reverse
    assert ambiguous_result.status == "fault"
    assert ambiguous_result.fault_code == "ambiguous_xy_timestamp"


def test_latest_good_at_xy_is_order_independent_and_failure_does_not_erase_it():
    """E03/E05: retry order is physical-time based and failed retry stays history."""
    old = surface_observation(0, 0.0, 0.0, 1.0, recorded_monotonic_s=90.0)
    new = surface_observation(1, 0.0, 0.0, 2.0, recorded_monotonic_s=91.0)
    failed = surface_observation(
        2,
        0.0,
        0.0,
        9.0,
        recorded_monotonic_s=92.0,
        source="led_autofocus",
        status="failed",
        quality=None,
    )
    forward = predict_focus_surface((old, failed, new), prediction_request())
    reverse = predict_focus_surface((new, failed, old), prediction_request())
    assert forward == reverse
    assert forward.status == "usable"
    assert forward.target_z_um == 2.0
    assert forward.support_observation_ids == (new.observation_id,)


def test_large_negative_coordinates_order_and_support_distances_are_deterministic():
    """E03/E04: list order and large signed origins do not change the local fit."""
    origin_x = -1_000_000_000.25
    origin_y = 2_000_000_000.5
    offsets = ((-500.0, -500.0), (-500.0, 500.0), (500.0, -500.0), (500.0, 500.0))
    rows = tuple(
        surface_observation(index, origin_x + dx, origin_y + dy, 0.002 * dx - 0.001 * dy)
        for index, (dx, dy) in enumerate(offsets)
    )
    request = prediction_request(target_x_um=origin_x, target_y_um=origin_y)
    forward = predict_focus_surface(rows, request)
    reverse = predict_focus_surface(tuple(reversed(rows)), request)
    assert forward == reverse
    assert forward.status == "usable"
    assert forward.target_z_um == pytest.approx(0.0, abs=1e-9)
    assert forward.support_distances_um == pytest.approx((math.sqrt(500000.0),) * 4)


def test_neighbor_and_candidate_working_sets_have_explicit_hard_bounds():
    """E23: fitting is max-neighbor bounded and oversized candidate windows fault."""
    near = ((-100.0, -100.0), (-100.0, 100.0), (100.0, -100.0), (100.0, 100.0))
    far = ((-500.0, -500.0), (-500.0, 500.0), (500.0, -500.0), (500.0, 500.0))
    rows = tuple(
        surface_observation(index, x, y, 0.001 * x) for index, (x, y) in enumerate(near + far)
    )
    bounded = predict_focus_surface(
        rows,
        prediction_request(settings=surface_settings(maximum_neighbors=4)),
    )
    assert bounded.status == "usable"
    assert len(bounded.support_observation_ids) == 4
    assert bounded.diagnostics.local_observation_count == 8

    duplicated = predict_focus_surface(
        (rows[0], *rows),
        prediction_request(settings=surface_settings(maximum_neighbors=4)),
    )
    assert duplicated == bounded

    oversized = predict_focus_surface(
        rows[:5],
        prediction_request(
            settings=surface_settings(
                maximum_neighbors=4,
                maximum_candidate_observations=4,
            )
        ),
    )
    assert oversized.status == "fault"
    assert oversized.fault_code == "candidate_limit_exceeded"
