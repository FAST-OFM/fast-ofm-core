"""Pure focus-derived Z hysteresis estimates, with no camera or motion imports.

Adapted from the project's earlier motion-calibration implementation at
commit d00a4775b9e4face975f44a2c269417a317f6599 (`_derived_values`).
and backlash_z_contracts.py (ZFocusObservation).

The original median magnitude / conservative uncertainty calculation is retained.
Signed scatter additionally detects inconsistent directions hidden by absolute values.
These are estimates, NOT commissioned motion profiles. Acquisition, tissue/curve
quality, independent preload validation and persistent OFM settings belong to the
calibration action. In particular, constant direction-correlated drift cannot be
distinguished from backlash by these numbers alone.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from statistics import median
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from fast_ofm_core.contracts import StrictModel


class ZFocusObservation(StrictModel):
    """Matched WHITE focus positions approached along positive and negative Z.

    Coordinates are micrometres in one reference frame. The uncertainty is a
    conservative bound on their DIFFERENCE, not one individual peak's uncertainty.
    Acquisition must establish quality and compensate/check temporal drift.
    """

    cycle_id: str = Field(min_length=1)
    positive_focus_um: float
    negative_focus_um: float
    focus_uncertainty_um: float = Field(ge=0)
    quality_score: float = Field(ge=0, le=1)

    @property
    def direction_effect_um(self) -> float:
        """Return negative-approach minus positive-approach focus position."""
        return self.negative_focus_um - self.positive_focus_um


class ZBacklashPolicy(StrictModel):
    """Explicit numerical acceptance parameters; none are hardware motion limits."""

    minimum_cycles: int = Field(ge=1)
    minimum_quality_score: float = Field(gt=0, le=1)
    maximum_uncertainty_um: float = Field(ge=0)
    preload_margin_um: float = Field(gt=0)


class ZBacklashEstimate(StrictModel):
    """Serializable numerical result; never authorises motion or enables correction."""

    status: Literal["estimate_only"] = "estimate_only"
    method: Literal["white_focus_direction_difference"] = "white_focus_direction_difference"
    axis: Literal["z"] = "z"
    preferred_final_approach_sign: Literal[-1, 1]
    observations: tuple[ZFocusObservation, ...] = Field(min_length=1)
    policy: ZBacklashPolicy
    signed_direction_effect_um: float
    backlash_um: float = Field(ge=0)
    uncertainty_um: float = Field(ge=0)
    preload_candidate_um: float = Field(gt=0)
    quality_score: float = Field(ge=0, le=1)

    @field_validator("preferred_final_approach_sign", mode="before")
    @classmethod
    def check_direction(cls, value: object) -> object:
        """Reject bools/strings/floats masquerading as a direction."""
        if type(value) is not int or value not in (-1, 1):
            raise ValueError("Final approach sign must be integer -1 or +1")
        return value

    @model_validator(mode="after")
    def check_derived_result(self) -> Self:
        """Recheck evidence and derived values after JSON loading, not just capture."""
        expected = _derived_values(self.observations, self.policy)
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"Inconsistent derived Z estimate: {name}")
        return self


def estimate_z_backlash(
    observations: Sequence[ZFocusObservation],
    *,
    policy: ZBacklashPolicy,
    preferred_final_approach_sign: Literal[-1, 1],
) -> ZBacklashEstimate:
    """Summarise paired focus observations without acquiring data or moving anything.

    The proposed preload must still fit the full hardware path and pass independent
    verification. Returning an estimate does not satisfy either condition.
    """
    if not isinstance(policy, ZBacklashPolicy):
        raise TypeError("A validated ZBacklashPolicy is required")
    cycles = tuple(observations)
    if any(not isinstance(item, ZFocusObservation) for item in cycles):
        raise TypeError("Validated ZFocusObservation values are required")
    values = _derived_values(cycles, policy)
    return ZBacklashEstimate(
        preferred_final_approach_sign=preferred_final_approach_sign,
        observations=cycles,
        policy=policy,
        signed_direction_effect_um=values["signed_direction_effect_um"],
        backlash_um=values["backlash_um"],
        uncertainty_um=values["uncertainty_um"],
        preload_candidate_um=values["preload_candidate_um"],
        quality_score=values["quality_score"],
    )


def _derived_values(
    observations: tuple[ZFocusObservation, ...], policy: ZBacklashPolicy
) -> dict[str, float]:
    """Reuse conservative direction-difference aggregation, checking signed scatter."""
    if not observations or len(observations) < policy.minimum_cycles:
        raise ValueError("Insufficient Z focus cycles")
    if len({item.cycle_id for item in observations}) != len(observations):
        raise ValueError("Duplicate Z cycle IDs")
    quality = min(item.quality_score for item in observations)
    if quality < policy.minimum_quality_score:
        raise ValueError("Z focus quality is below policy")
    signed = tuple(item.direction_effect_um for item in observations)
    if not all(math.isfinite(value) for value in signed):
        raise ValueError("Non-finite Z direction difference")
    effects = tuple(abs(value) for value in signed)
    center = median(effects)
    signed_center = median(signed)
    variability = max(abs(effect - center) for effect in effects)
    signed_variability = max(abs(value - signed_center) for value in signed)
    focus_uncertainty = max(item.focus_uncertainty_um for item in observations)
    uncertainty = max(
        focus_uncertainty,
        variability,
        signed_variability,
    )
    if not math.isfinite(uncertainty) or uncertainty > policy.maximum_uncertainty_um:
        raise ValueError("Z uncertainty exceeds policy")
    preload = center + uncertainty + policy.preload_margin_um
    if not math.isfinite(preload):
        raise ValueError("Non-finite Z preload estimate")
    return {
        "signed_direction_effect_um": float(signed_center),
        "backlash_um": float(center),
        "uncertainty_um": float(uncertainty),
        "preload_candidate_um": float(preload),
        "quality_score": quality,
    }
