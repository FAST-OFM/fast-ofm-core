"""Signed fit, independent holdout and activation-policy tests for OF-026."""

import pytest
from pydantic import ValidationError

from fast_ofm_core.focus.rg.rg_focus_model import (
    RGFocusCalibrationPoint,
    RGFocusModelProfile,
    RGFocusModelSettings,
    RGFocusStationaryObservation,
    _approach_check,
    estimate_z_um,
    fit_rg_focus_model,
    jpeg_measurement_draft_policy,
)


def point(capture_id, role, z, *, error=0.0, status="ready", white=100.0):
    """Build points on a line with a deliberately non-zero focus offset."""
    return RGFocusCalibrationPoint(
        capture_id=capture_id,
        role=role,
        z_um=float(z),
        measurement_status=status,
        dx=None if status != "ready" else 4.25 - 0.8 * z + error,
        dy=None if status != "ready" else -1.5 + 0.1 * z - error / 2,
        confidence=0.7 if status == "ready" else 0.0,
        white_score=white,
    )


def observations(*, approach_error=0.0, holdout_error=0.0):
    """Return disjoint fit, holdout and two-sided target observations."""
    result = [point(f"fit-{z}", "fit", z, white=100 - abs(z)) for z in (-8, -4, 0, 4, 8)]
    result += [
        point(
            f"holdout-{z}",
            "holdout",
            z,
            error=holdout_error,
            white=100 - abs(z),
        )
        for z in (-6, -2, 2, 6)
    ]
    result += [
        point("above", "approach_above", 0),
        point("below", "approach_below", 0, error=approach_error),
    ]
    return result


def widened_observations(*, approach_error=0.0, holdout_error=0.0):
    """Return the same-size grid with independent evidence through +/-12 um."""
    result = [point(f"fit-{z}", "fit", z, white=100 - abs(z)) for z in (-12, -6, 0, 6, 12)]
    result += [
        point(
            f"holdout-{z}",
            "holdout",
            z,
            error=holdout_error,
            white=100 - abs(z),
        )
        for z in (-9, -3, 3, 9)
    ]
    result += [
        point("above", "approach_above", 0),
        point("below", "approach_below", 0, error=approach_error),
    ]
    return result


def stationary_observations(*, error=0.0, **changes):
    """Return two identity-bound, ready no-motion commissioning observations."""
    common = {
        "measurement_status": "ready",
        "white_score": 100.0,
        "accepted_patch_count": 10,
        "inlier_fraction": 0.6,
        "field_id": "field-a",
        "geometry_id": "geometry-a",
        "binding_sha256": "a" * 64,
        "mask_sha256": "b" * 64,
        "window_ids": tuple(f"patch-{index}" for index in range(10)),
    }
    common.update(changes)
    return (
        RGFocusStationaryObservation(
            capture_id="stationary-0",
            dx=4.25,
            dy=-1.5,
            **common,
        ),
        RGFocusStationaryObservation(
            capture_id="stationary-1",
            dx=4.25 + error,
            dy=-1.5 - error / 2,
            **common,
        ),
    )


def fit(points, *, stationary=None, reference_white=100.0, settings=None):
    """Fit with stable test provenance."""
    return fit_rg_focus_model(
        points,
        settings or RGFocusModelSettings(maximum_holdout_error_um=2.0),
        reference_white_score=reference_white,
        stationary_observations=stationary or stationary_observations(),
        profile_id="profile",
        source_series_id="series",
        compatibility={"geometry_id": "g", "profile_id": "ff"},
    )


def test_signed_2d_model_keeps_nonzero_offset_and_passes_untouched_holdout():
    """Focus is not assumed to be zero shift and both shift axes inform signed Z."""
    profile = fit(observations())
    assert profile.status == "valid"
    assert profile.offset_px == pytest.approx((4.25, -1.5))
    assert profile.slope_px_per_um == pytest.approx((-0.8, 0.1))
    assert profile.holdout_max_absolute_error_um < 1e-10
    assert estimate_z_um(profile, 12.25, -2.5) == pytest.approx(-10)


def test_widened_grid_validates_reloadable_plus_minus_twelve_range():
    """The new grid proves a wider range without adding calibration captures."""
    profile = fit(widened_observations())
    assert profile.status == "valid"
    assert profile.applicable_z_range_um == (-12, 12)
    assert len(profile.empirical_observations) == 13
    assert RGFocusModelProfile.model_validate_json(profile.model_dump_json()) == profile

    payload = profile.model_dump(mode="json")
    payload["empirical_observations"][0]["z_um"] = -15
    with pytest.raises(ValidationError):
        RGFocusModelProfile.model_validate(payload)


def test_legacy_range_gate_field_still_loads_as_generic_contract_evidence():
    """Existing commissioned profiles survive the range-ledger field rename."""
    profile = fit(observations())
    payload = profile.model_dump(mode="json")
    evidence = payload["activation_evidence"]
    evidence["applicable_range_covers_minus6_plus6"] = evidence.pop(
        "applicable_range_covers_contract"
    )

    assert RGFocusModelProfile.model_validate(payload) == profile


def shifted_white_profile(focus_z):
    """Fit unchanged commanded shifts with an independently displaced WHITE peak."""
    rows = [
        row.model_copy(update={"white_score": 100.0 - 0.25 * (row.z_um - focus_z) ** 2})
        for row in observations()
    ]
    reference_white = 100.0 - 0.25 * focus_z**2
    return fit(
        rows,
        stationary=stationary_observations(white_score=reference_white),
        reference_white=reference_white,
    )


def empirical_profile(prediction_error=1.2, cross_track=0.0):
    """Keep the fit fixed while independent rows expose error and cross-track limits."""
    rows = observations()
    for index, row in enumerate(rows):
        if row.role in ("holdout", "approach_above", "approach_below"):
            rows[index] = row.model_copy(
                update={
                    "dx": row.dx - 0.8 * prediction_error + 0.1 * cross_track,
                    "dy": row.dy + 0.1 * prediction_error + 0.8 * cross_track,
                }
            )
    return fit_rg_focus_model(
        rows,
        RGFocusModelSettings(maximum_holdout_error_um=2.0),
        reference_white_score=100.0,
        stationary_observations=stationary_observations(),
        profile_id="profile",
        source_series_id="series",
        compatibility={},
        measurement_policy=jpeg_measurement_draft_policy(),
    )


def test_empirical_envelopes_use_all_thirteen_existing_accepted_rows():
    """Both returns and both stationary captures participate alongside the full grid."""
    result = empirical_profile(cross_track=0.5)
    assert result.status == "valid"
    rows = result.empirical_observations
    assert len(rows) == len({row.capture_id for row in rows}) == 13
    assert {row.capture_id for row in rows} == set(
        result.fit_capture_ids + result.holdout_capture_ids
    ) | {"above", "below", "stationary-0", "stationary-1"}
    residuals = [result.cross_track_residual_px(row.dx, row.dy)[0] for row in rows]
    assert result.empirical_cross_track_limit_px == pytest.approx(
        max(abs(value) for value in residuals)
    )
    assert result.empirical_prediction_error_um == pytest.approx(1.2)
    assert result.effective_focus_tolerance_um(2) == pytest.approx(0.8)
    assert RGFocusModelProfile.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "limit",
        "residual",
        "prediction_error",
        "nonfinite",
        "stationary",
    ],
)
def test_saved_empirical_envelopes_are_required_complete_and_consistent(defect):
    """A serialized active model cannot omit or widen empirical acceptance evidence."""
    payload = empirical_profile(cross_track=0.5).model_dump(mode="json")
    if defect == "missing":
        del payload["empirical_observations"]
    elif defect == "duplicate":
        payload["empirical_observations"][-1] = payload["empirical_observations"][0]
    elif defect == "limit":
        payload["empirical_cross_track_limit_px"] += 0.1
    elif defect == "residual":
        payload["empirical_observations"][0]["signed_cross_track_residual_px"] += 0.1
    elif defect == "prediction_error":
        payload["empirical_prediction_error_um"] = 0.0
    elif defect == "nonfinite":
        payload["empirical_observations"][0]["dx"] = float("nan")
    else:
        payload["empirical_observations"][-1]["z_um"] = 1.0
    with pytest.raises(ValidationError):
        RGFocusModelProfile.model_validate(payload)


@pytest.mark.parametrize("focus_z", [-6, -4, -2, 0, 2, 4, 6])
def test_white_target_keeps_commanded_fit_and_translates_only_error_origin(focus_z):
    """A measured WHITE target does not reinterpret held-out positions or returns."""
    result = shifted_white_profile(focus_z)
    assert result.status == "valid"
    assert result.focus_z_um == focus_z
    assert result.applicable_z_range_um == (-6, 6)
    assert result.applicable_error_range_um == (-6 - focus_z, 6 - focus_z)
    assert result.offset_px == pytest.approx((4.25, -1.5))
    assert result.slope_px_per_um == pytest.approx((-0.8, 0.1))
    assert result.holdout_max_absolute_error_um < 1e-10
    assert result.approach_check.target_z_um == 0
    assert result.activation_evidence.grid_white_maximum_z_positions_um == (focus_z,)
    assert estimate_z_um(result, 4.25 - 0.8 * focus_z, -1.5 + 0.1 * focus_z) == pytest.approx(
        focus_z
    )
    payload = result.model_dump(mode="json")
    assert "applicable_error_range_um" not in payload
    assert RGFocusModelProfile.model_validate_json(result.model_dump_json()) == result


@pytest.mark.parametrize("focus_z", [-8, 8])
def test_endpoint_white_peak_cannot_activate_without_two_neighbours(focus_z):
    """A boundary maximum cannot prove that the discrete stack contains focus."""
    result = shifted_white_profile(focus_z)
    assert result.status == "candidate"
    assert result.focus_z_um == focus_z
    assert "grid_white_maximum_internal" in result.activation_evidence.failures
    assert result.holdout_max_absolute_error_um < 1e-10


def test_strictly_sharper_independent_reference_keeps_commanded_zero():
    """A sharper independent Z0 outranks a noisier grid maximum away from zero."""
    rows = [
        row.model_copy(update={"white_score": 99.0 if row.z_um == 4 else 95.0 - abs(row.z_um)})
        for row in observations()
    ]
    result = fit(
        rows,
        stationary=stationary_observations(white_score=100.0),
        reference_white=100.0,
    )
    assert result.status == "valid"
    assert result.focus_z_um == 0
    assert result.activation_evidence.maximum_grid_white_z_um == 4
    assert result.activation_evidence.reference_white_fraction > 1
    assert "white_focus_reference_offset" not in result.activation_evidence.failures


def test_tied_white_maxima_do_not_guess_a_focus_target():
    """Even a small plateau inside the tolerance has no unique measured target."""
    rows = [
        row.model_copy(update={"white_score": 100.0 if row.z_um in (0, 2) else 90.0})
        for row in observations()
    ]
    result = fit(rows)
    assert result.status == "candidate"
    assert result.focus_z_um is None
    assert "grid_white_maximum_ambiguous" in result.activation_evidence.failures
    with pytest.raises(ValueError, match="unambiguous WHITE"):
        _ = result.applicable_error_range_um


@pytest.mark.parametrize(
    "defect",
    ["missing_target", "missing_evidence", "wrong_target", "ambiguous_evidence"],
)
def test_saved_valid_profile_requires_matching_white_target_evidence(defect):
    """Saved or installed profiles cannot retain the old implicit-zero meaning."""
    payload = shifted_white_profile(2).model_dump(mode="json")
    if defect == "missing_target":
        del payload["focus_z_um"]
    elif defect == "missing_evidence":
        del payload["activation_evidence"]["grid_white_maximum_z_positions_um"]
    elif defect == "wrong_target":
        payload["focus_z_um"] = 0.0
    else:
        payload["activation_evidence"]["grid_white_maximum_z_positions_um"] = [0.0, 2.0]
    with pytest.raises(ValidationError):
        RGFocusModelProfile.model_validate(payload)


def test_json_action_payload_accepts_arrays_for_serialised_tuples():
    """The HTTP action can receive tuple fields through JSON arrays."""
    fitted = fit(observations())
    payload = fitted.model_dump(mode="json")
    restored = RGFocusModelProfile.model_validate(payload)
    assert restored == fitted


def test_holdout_is_not_used_to_fit_coefficients():
    """Changing every holdout leaves fit coefficients fixed and fails validation."""
    baseline = fit(observations())
    changed = fit(observations(holdout_error=8.0))
    assert changed.offset_px == pytest.approx(baseline.offset_px)
    assert changed.slope_px_per_um == pytest.approx(baseline.slope_px_per_um)
    assert changed.status == "candidate"
    assert "holdout_maximum_error" in changed.reason


def test_failed_two_sided_approach_keeps_an_accurate_model_candidate_only():
    """Good fit/holdout cannot activate a model with irreproducible final approach."""
    profile = fit(observations(approach_error=2.0))
    assert profile.holdout_max_absolute_error_um < 1e-10
    assert profile.approach_check.status == "failed"
    assert profile.status == "candidate"
    assert "return_projected_target_error" in profile.reason


def test_qc_rejections_are_retained_and_minimum_counts_gate_activation():
    """Rejected observations never enter the regression and remain visible."""
    points = observations()
    points[0] = point("fit--8", "fit", -8, status="inconsistent_shifts")
    profile = fit(points)
    assert profile.status == "candidate"
    assert profile.rejected_capture_reasons == {"fit--8": "inconsistent_shifts"}
    assert "all_grid_and_returns_ready" in profile.reason


def test_fit_and_holdout_must_each_span_focus():
    """A one-sided dataset cannot identify the declared operational focus range."""
    points = observations()
    points = [item for item in points if item.role != "holdout" or item.z_um < 0]
    with pytest.raises(ValueError, match="Holdout observations must span"):
        fit(points)


def test_equal_bias_of_both_returns_cannot_pass_against_fitted_target():
    """OF-062: agreement between returns alone cannot hide their common drift."""
    baseline = fit(observations())
    rows = observations()
    rows[-2:] = [
        point("above", "approach_above", 0, error=2.0),
        point("below", "approach_below", 0, error=2.0),
    ]
    changed = fit(rows)
    assert changed.approach_check.mutual_prediction_difference_um == 0.0
    assert changed.approach_check.white_mutual_fraction == 0.0
    assert changed.approach_check.status == "failed"
    assert changed.status == "candidate"
    assert changed.offset_px == baseline.offset_px
    assert changed.slope_px_per_um == baseline.slope_px_per_um
    assert changed.holdout_errors_um == baseline.holdout_errors_um
    assert all(value > 2.0 for value in changed.approach_check.absolute_target_errors_um.values())


def test_returns_at_a_nonzero_target_cannot_activate_demo_contract():
    """The frozen two-sided returns must both be measured at A0."""
    rows = observations()
    rows[-2:] = [
        point("above", "approach_above", -4),
        point("below", "approach_below", -4),
    ]
    fitted = fit(rows)
    assert fitted.status == "candidate"
    assert "exact_grid_and_returns" in fitted.reason
    assert fitted.approach_check.status == "failed"


@pytest.mark.parametrize(("bias", "status"), [(2.0, "passed"), (2.000001, "failed")])
def test_target_model_shift_tolerance_has_an_inclusive_exact_boundary(bias, status):
    """Use exact binary coordinates to separate equality from a numerical fit error."""
    rows = [
        RGFocusCalibrationPoint(
            capture_id=role,
            role=role,
            z_um=0.0,
            measurement_status="ready",
            dx=bias,
            dy=0.0,
            confidence=0.8,
            white_score=100.0,
        )
        for role in ("approach_above", "approach_below")
    ]
    check = _approach_check(
        rows,
        RGFocusModelSettings(),
        offset_px=(0.0, 0.0),
        slope_px_per_um=(1.0, 0.0),
        reference_white_score=100.0,
    )
    assert check.status == status


@pytest.mark.parametrize("defect", ["duplicate_role", "shared_capture", "missing_side"])
def test_approach_requires_exactly_two_independent_target_observations(defect):
    """Extra returns cannot silently overwrite a failed side by dictionary order."""
    rows = observations()
    if defect == "duplicate_role":
        rows.append(point("another-above", "approach_above", 0, error=5.0))
    elif defect == "shared_capture":
        rows[-1] = rows[-1].model_copy(update={"capture_id": rows[-2].capture_id})
    else:
        rows.pop()
    assert fit(rows).status == "candidate"
    assert fit(rows).approach_check.status == "failed"


@pytest.mark.parametrize(
    ("defect", "gate"),
    [
        ("grid_not_ready", "all_grid_and_returns_ready"),
        ("holdout", "holdout_maximum_error"),
        ("range", "applicable_range_minus6_plus6"),
        ("endpoint_white", "grid_white_maximum_internal"),
        ("neighbour", "grid_white_maximum_measured_neighbours"),
        ("return_target", "return_projected_target_error"),
        ("return_mutual", "return_mutual_prediction_difference"),
        ("return_white", "return_white_fraction"),
        ("return_white_mutual", "return_white_mutual_fraction"),
        ("stationary_ready", "stationary_ready"),
        ("stationary_identity", "stationary_identity"),
        ("stationary_patches", "stationary_patch_count"),
        ("stationary_inliers", "stationary_inlier_fraction"),
        ("stationary_white", "stationary_white_mutual_fraction"),
        ("stationary_projection", "stationary_projected_difference"),
        ("zero_slope", "slope_finite_nonzero"),
    ],
)
def test_every_frozen_activation_gate_keeps_profile_candidate(defect, gate):
    """No single failed physical metric can be relabelled as an active model."""
    rows = observations()
    stationary = stationary_observations()
    reference_white = 100.0
    if defect == "grid_not_ready":
        rows[0] = rows[0].model_copy(
            update={"measurement_status": "inconsistent_shifts", "dx": None, "dy": None}
        )
    elif defect == "holdout":
        rows = observations(holdout_error=8.0)
    elif defect == "range":
        rows[0] = rows[0].model_copy(
            update={"measurement_status": "inconsistent_shifts", "dx": None, "dy": None}
        )
    elif defect == "endpoint_white":
        rows[0] = rows[0].model_copy(update={"white_score": 110.0})
        reference_white = 100.0
    elif defect == "neighbour":
        index = next(i for i, row in enumerate(rows) if row.z_um == 2)
        rows[index] = rows[index].model_copy(
            update={"measurement_status": "inconsistent_shifts", "dx": None, "dy": None}
        )
    elif defect == "return_target":
        rows[-2:] = [
            point("above", "approach_above", 0, error=2.0),
            point("below", "approach_below", 0, error=2.0),
        ]
    elif defect == "return_mutual":
        rows[-2:] = [
            point("above", "approach_above", 0, error=1.5),
            point("below", "approach_below", 0, error=-1.5),
        ]
    elif defect == "return_white":
        rows[-2] = rows[-2].model_copy(update={"white_score": 84.0})
    elif defect == "return_white_mutual":
        reference_white = 98.0
        rows[-1] = rows[-1].model_copy(update={"white_score": 84.0})
    elif defect == "stationary_ready":
        stationary = (
            stationary[0].model_copy(
                update={
                    "measurement_status": "inconsistent_shifts",
                    "dx": None,
                    "dy": None,
                }
            ),
            stationary[1],
        )
    elif defect == "stationary_identity":
        stationary = (
            stationary[0],
            stationary[1].model_copy(update={"mask_sha256": "c" * 64}),
        )
    elif defect == "stationary_patches":
        stationary = (
            stationary[0],
            stationary[1].model_copy(update={"accepted_patch_count": 8}),
        )
    elif defect == "stationary_inliers":
        stationary = (
            stationary[0],
            stationary[1].model_copy(update={"inlier_fraction": 0.59}),
        )
    elif defect == "stationary_white":
        stationary = (
            stationary[0],
            stationary[1].model_copy(update={"white_score": 80.0}),
        )
    elif defect == "stationary_projection":
        stationary = stationary_observations(error=2.0)
    else:
        rows = [
            row.model_copy(
                update={"dx": 4.25, "dy": -1.5} if row.measurement_status == "ready" else {}
            )
            for row in rows
        ]
    result = fit(
        rows,
        stationary=stationary,
        reference_white=reference_white,
    )
    assert result.status == "candidate"
    assert gate in result.activation_evidence.failures


def test_saved_valid_profile_without_target_evidence_requires_revalidation():
    """Historical profiles without the new ledger remain incompatible evidence."""
    payload = fit(observations()).model_dump(mode="json")
    payload.pop("activation_evidence")
    with pytest.raises(ValidationError):
        RGFocusModelProfile.model_validate(payload)


@pytest.mark.parametrize(
    "defect", ["failed", "false_gate", "hidden_failure", "excessive_stationary"]
)
def test_valid_status_cannot_bypass_serialized_approach_evidence(defect):
    """The loading/installation boundary checks evidence, not a saved status alone."""
    payload = fit(observations()).model_dump(mode="json")
    evidence = payload["activation_evidence"]
    if defect == "failed":
        evidence["status"] = "failed"
    elif defect == "false_gate":
        evidence["all_grid_and_returns_ready"] = False
    elif defect == "hidden_failure":
        evidence["failures"] = ["forged"]
    else:
        evidence["stationary_prediction_difference_um"] = 3.0
    with pytest.raises(ValidationError):
        RGFocusModelProfile.model_validate(payload)
