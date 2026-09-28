"""Numerical Z estimates and JSON contracts, without camera or stage access."""

import json

import pytest
from pydantic import ValidationError

from fast_ofm_core.focus.z_backlash import (
    ZBacklashEstimate,
    ZBacklashPolicy,
    ZFocusObservation,
    estimate_z_backlash,
)


@pytest.fixture
def policy():
    """Return explicit test tolerances, not deployment defaults."""
    return ZBacklashPolicy(
        minimum_cycles=3,
        minimum_quality_score=0.8,
        maximum_uncertainty_um=0.5,
        preload_margin_um=1.0,
    )


def observations(effects=(4.0, 4.2, 3.8), offsets=None, uncertainty=0.1):
    """Build positive/negative WHITE peak pairs with known signed differences."""
    if offsets is None:
        offsets = [0.0] * len(effects)
    return tuple(
        ZFocusObservation(
            cycle_id=f"cycle-{i}",
            positive_focus_um=float(offset),
            negative_focus_um=float(offset + effect),
            focus_uncertainty_um=uncertainty,
            quality_score=0.9,
        )
        for i, (effect, offset) in enumerate(zip(effects, offsets, strict=True))
    )


@pytest.mark.parametrize("approach", [-1, 1])
@pytest.mark.parametrize("effect_sign", [-1, 1])
def test_known_backlash_in_both_directions(policy, approach, effect_sign):
    """Keep approach choice separate from the measured signed direction effect."""
    result = estimate_z_backlash(
        observations(tuple(effect_sign * value for value in (4.0, 4.2, 3.8))),
        policy=policy,
        preferred_final_approach_sign=approach,
    )
    assert result.signed_direction_effect_um == pytest.approx(4 * effect_sign)
    assert result.backlash_um == pytest.approx(4.0)
    assert result.uncertainty_um == pytest.approx(0.2)
    assert result.preload_candidate_um == pytest.approx(5.2)
    assert result.quality_score == 0.9
    assert result.preferred_final_approach_sign == approach
    assert result.status == "estimate_only"


def test_zero_backlash_keeps_uncertainty_and_margin(policy):
    """A zero observed effect is not proof of a perfectly backlash-free axis."""
    result = estimate_z_backlash(
        observations((0, 0, 0)), policy=policy, preferred_final_approach_sign=1
    )
    assert result.backlash_um == 0
    assert result.uncertainty_um == 0.1
    assert result.preload_candidate_um == pytest.approx(1.1)


def test_replays_original_magnitude_uncertainty_equations(policy):
    """Match the former estimator on consistent-direction observations."""
    cycles = observations((4, 4.2, 3.8), uncertainty=0.3)
    effects = sorted(abs(item.direction_effect_um) for item in cycles)
    old_center = effects[len(effects) // 2]
    focus_uncertainty = max(item.focus_uncertainty_um for item in cycles)
    variability = max(abs(value - old_center) for value in effects)
    old_uncertainty = max(focus_uncertainty, variability)
    result = estimate_z_backlash(cycles, policy=policy, preferred_final_approach_sign=1)
    assert result.backlash_um == old_center
    assert result.uncertainty_um == old_uncertainty
    assert result.preload_candidate_um == (old_center + old_uncertainty + policy.preload_margin_um)


def test_common_focus_offset_does_not_change_backlash(policy):
    """A shared offset of each pair cancels; this does not prove within-pair drift QC."""
    result = estimate_z_backlash(
        observations((4, 4, 4), offsets=(0, 12, 25)),
        policy=policy,
        preferred_final_approach_sign=-1,
    )
    assert result.backlash_um == 4
    assert result.uncertainty_um == 0.1


@pytest.mark.parametrize("effects", [(4, 4, 6), (4, -4, 4), (1, 2, 3)])
def test_scatter_drift_or_inconsistent_directions_rejected(policy, effects):
    """Signed disagreement cannot be hidden by the original absolute-value summary."""
    with pytest.raises(ValueError, match="uncertainty"):
        estimate_z_backlash(observations(effects), policy=policy, preferred_final_approach_sign=1)


def test_uncertain_peaks_rejected(policy):
    """A stable center cannot override uncertain individual observations."""
    with pytest.raises(ValueError, match="uncertainty"):
        estimate_z_backlash(
            observations((4, 4, 4), uncertainty=0.6),
            policy=policy,
            preferred_final_approach_sign=1,
        )


@pytest.mark.parametrize("cycles", [(), observations((4, 4))])
def test_missing_cycles_rejected(policy, cycles):
    """A result requires the explicitly configured number of independent pairs."""
    with pytest.raises(ValueError, match="Insufficient"):
        estimate_z_backlash(cycles, policy=policy, preferred_final_approach_sign=1)


def test_duplicate_cycles_rejected(policy):
    """Copying one cycle does not count as repeated measurement."""
    sample = observations()[0]
    with pytest.raises(ValueError, match="Duplicate"):
        estimate_z_backlash(
            (sample, sample, sample), policy=policy, preferred_final_approach_sign=1
        )


def test_low_quality_rejected(policy):
    """Numerically consistent focus positions still need valid image evidence."""
    cycles = list(observations())
    cycles[1] = ZFocusObservation(**{**cycles[1].model_dump(), "quality_score": 0.7})
    with pytest.raises(ValueError, match="quality"):
        estimate_z_backlash(cycles, policy=policy, preferred_final_approach_sign=1)


@pytest.mark.parametrize("bad", [True, False, 0, 2, 1.0, "1"])
def test_approach_requires_signed_integer(policy, bad):
    """Public and stored direction parameters cannot silently coerce invalid values."""
    with pytest.raises(ValidationError, match="approach sign"):
        estimate_z_backlash(observations(), policy=policy, preferred_final_approach_sign=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_rejected(bad):
    """Non-finite positions never enter aggregation."""
    with pytest.raises(ValidationError):
        ZFocusObservation(
            cycle_id="bad",
            positive_focus_um=bad,
            negative_focus_um=0,
            focus_uncertainty_um=0.1,
            quality_score=0.9,
        )


def test_json_roundtrip_and_immutable_result(policy):
    """The result is ready for OFM storage but is not a commissioned calibration."""
    result = estimate_z_backlash(observations(), policy=policy, preferred_final_approach_sign=1)
    encoded = result.model_dump_json()
    assert ZBacklashEstimate.model_validate_json(encoded) == result
    assert json.loads(encoded)["status"] == "estimate_only"
    with pytest.raises(ValidationError, match="frozen"):
        result.backlash_um = 999


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("backlash_um", 0),
        ("preload_candidate_um", 0.5),
        ("signed_direction_effect_um", -4),
        ("quality_score", 1),
        ("status", "valid"),
        ("axis", "x"),
        ("preferred_final_approach_sign", True),
    ],
)
def test_modified_stored_result_is_rejected(policy, field, value):
    """A modified JSON result must not contradict the evidence it contains."""
    result = estimate_z_backlash(observations(), policy=policy, preferred_final_approach_sign=1)
    data = result.model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        ZBacklashEstimate.model_validate_json(json.dumps(data))


def test_overflow_of_finite_inputs_rejected(policy):
    """Even individually finite coordinates can have an unrepresentable difference."""
    cycles = observations()
    extreme = ZFocusObservation(
        cycle_id="extreme",
        positive_focus_um=-1e308,
        negative_focus_um=1e308,
        focus_uncertainty_um=0.1,
        quality_score=0.9,
    )
    with pytest.raises(ValueError, match="Non-finite"):
        estimate_z_backlash((*cycles, extreme), policy=policy, preferred_final_approach_sign=1)


def test_even_cycle_count(policy):
    """Median aggregation also handles an even number of independently named cycles."""
    result = estimate_z_backlash(
        observations((4, 4.2, 3.8, 4.4)),
        policy=policy,
        preferred_final_approach_sign=1,
    )
    assert result.backlash_um == pytest.approx(4.1)
    assert result.uncertainty_um == pytest.approx(0.3)
    assert result.preload_candidate_um == pytest.approx(5.4)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("minimum_cycles", 0),
        ("minimum_cycles", True),
        ("minimum_quality_score", 0),
        ("minimum_quality_score", 1.1),
        ("maximum_uncertainty_um", -1),
        ("preload_margin_um", 0),
        ("preload_margin_um", float("inf")),
    ],
)
def test_invalid_policy_is_rejected(policy, field, bad):
    """Limits cannot silently become disabled or be expressed in invalid units/types."""
    with pytest.raises(ValidationError):
        ZBacklashPolicy(**{**policy.model_dump(), field: bad})
