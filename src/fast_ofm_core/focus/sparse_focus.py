"""Small, hardware-free focus-surface scheduler for fast demo scans.

The scheduler deliberately keeps only measured autofocus anchors.  Predicted
heights are never fed back as observations, so a bad estimate cannot teach the
surface to repeat its own error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SparseFocusSettings:
    """Frozen limits for one scan's sparse autofocus schedule."""

    enabled: bool = False
    anchor_interval: int = 8
    minimum_plane_points: int = 4
    maximum_neighbors: int = 12
    neighbor_radius_um: float = 6000.0
    maximum_extrapolation_um: float = 3000.0
    maximum_prediction_delta_um: float = 40.0
    maximum_fit_residual_um: float = 4.0
    maximum_plane_slope_um_per_um: float = 0.05
    maximum_condition_number: float = 1000.0
    maximum_recovery_anchors: int | None = None

    def __post_init__(self) -> None:
        """Reject incomplete or internally inconsistent scan settings."""
        if self.anchor_interval < 1:
            raise ValueError("Anchor interval must be positive")
        if self.maximum_recovery_anchors is not None and self.maximum_recovery_anchors < 1:
            raise ValueError("Recovery anchor limit must be positive")
        if self.minimum_plane_points < 4:
            raise ValueError("A robust local plane needs at least four anchors")
        if self.maximum_neighbors < self.minimum_plane_points:
            raise ValueError("Maximum neighbors cannot be below plane support")
        positive = (
            self.neighbor_radius_um,
            self.maximum_prediction_delta_um,
            self.maximum_fit_residual_um,
            self.maximum_plane_slope_um_per_um,
            self.maximum_condition_number,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Sparse focus limits must be finite and positive")
        if (
            not math.isfinite(self.maximum_extrapolation_um)
            or self.maximum_extrapolation_um < 0
            or self.maximum_extrapolation_um > self.neighbor_radius_um
        ):
            raise ValueError("Extrapolation must fit inside the neighbor radius")


@dataclass(frozen=True)
class FocusAnchor:
    """One autofocus result measured by hardware during this scan."""

    x_um: float
    y_um: float
    z_um: float


@dataclass(frozen=True)
class FocusEstimate:
    """A bounded target height and why it is or is not usable."""

    usable: bool
    target_z_um: float | None
    source: str
    reason: str
    support_count: int = 0
    fit_residual_um: float | None = None
    slope_um_per_um: float | None = None
    extrapolation_um: float | None = None
    validation_error_um: float | None = None


@dataclass(frozen=True)
class SparseFocusDecision:
    """One prepared field decision, frozen until acquisition finishes."""

    x_units: int
    y_units: int
    target_z_units: int
    requires_anchor: bool
    estimate: FocusEstimate
    white_only: bool = False


def _cross(
    origin: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
        second[0] - origin[0]
    )


def _convex_hull(
    points: list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    unique = sorted(set(points))
    if len(unique) <= 1:
        return tuple(unique)
    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return tuple(lower[:-1] + upper[:-1])


def _distance_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    fraction = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_squared
    fraction = min(1.0, max(0.0, fraction))
    return math.hypot(
        point[0] - (start[0] + fraction * dx),
        point[1] - (start[1] + fraction * dy),
    )


def _extrapolation_distance(
    anchors: list[FocusAnchor], target_x_um: float, target_y_um: float
) -> float:
    hull = _convex_hull([(point.x_um, point.y_um) for point in anchors])
    if len(hull) < 3:
        return math.inf
    target = (target_x_um, target_y_um)
    coordinate_scale = max(
        1.0,
        *(max(abs(x), abs(y)) for x, y in hull),
    )
    tolerance = np.finfo(float).eps * coordinate_scale * coordinate_scale * 64
    if all(
        _cross(start, end, target) >= -tolerance
        for start, end in zip(hull, hull[1:] + hull[:1], strict=True)
    ):
        return 0.0
    return min(
        _distance_to_segment(target, start, end)
        for start, end in zip(hull, hull[1:] + hull[:1], strict=True)
    )


def estimate_focus_height(
    anchors: tuple[FocusAnchor, ...],
    *,
    target_x_um: float,
    target_y_um: float,
    current_z_um: float,
    settings: SparseFocusSettings,
) -> FocusEstimate:
    """Choose the widest independently validated local model, never train on it.

    A bent surface need not fit one plane over the entire neighbor radius. Try
    nested spatial neighborhoods without removing interior outliers. A model must
    also predict each of its measured supports when that support is left out.
    """
    values = (target_x_um, target_y_um, current_z_um)
    if not all(math.isfinite(value) for value in values) or any(
        not all(math.isfinite(value) for value in (p.x_um, p.y_um, p.z_um)) for p in anchors
    ):
        return FocusEstimate(False, None, "none", "non-finite focus input")
    # Repeating autofocus at the same XY is not an independent spatial support.
    # Keep the latest actual measurement there; never count duplicates as geometry.
    distinct = {(point.x_um, point.y_um): point for point in anchors}
    distance_rows = sorted(
        [
            (
                math.hypot(point.x_um - target_x_um, point.y_um - target_y_um),
                point,
            )
            for point in distinct.values()
        ],
        key=lambda row: (row[0], row[1].x_um, row[1].y_um, row[1].z_um),
    )
    local = [point for distance, point in distance_rows if distance <= settings.neighbor_radius_um][
        : settings.maximum_neighbors
    ]
    if not local:
        return FocusEstimate(False, None, "none", "no nearby measured anchor")

    # During warm-up a bounded nearest measured height is safer than inventing a
    # plane from collinear points.  Warm-up fields still remain anchors.
    if len(local) < settings.minimum_plane_points:
        return _nearest_estimate(local[0], current_z_um, settings)

    first_failure: FocusEstimate | None = None
    for count in range(len(local), settings.minimum_plane_points - 1, -1):
        result = _estimate_plane(local[:count], target_x_um, target_y_um, current_z_um, settings)
        if result.usable:
            return result
        if first_failure is None:
            first_failure = result
    # A new snake row may have a different local slope from the previous row.
    # Use a measured row tangent only after leaving the initial measured line,
    # with three independently checked supports and a one-spacing horizon.
    if any(point.y_um != target_y_um for point in anchors):
        row = [point for point in local if point.y_um == target_y_um]
        for count in range(len(row), 2, -1):
            result = _estimate_row(row[:count], target_x_um, current_z_um, settings)
            if result.usable:
                return result
    return first_failure or FocusEstimate(False, None, "none", "insufficient measured support")


def _nearest_estimate(
    anchor: FocusAnchor, current_z_um: float, settings: SparseFocusSettings
) -> FocusEstimate:
    """Bound an approach hint; the scheduler still requires an actual measurement."""
    if abs(anchor.z_um - current_z_um) > settings.maximum_prediction_delta_um:
        return FocusEstimate(
            False,
            None,
            "nearest",
            "nearest anchor exceeds the current-Z limit",
            support_count=1,
        )
    return FocusEstimate(
        True, anchor.z_um, "nearest", "bounded nearest measured anchor", support_count=1
    )


def _validation_error(design: npt.NDArray[np.float64], z: npt.NDArray[np.float64]) -> float:
    """Return exact linear leave-one-out errors using PRESS, or refuse leverage one.

    One off-line point can give a perfect plane residual while being the only
    evidence for the cross-row slope. Removing that point makes the model
    singular; such a surface is not independently supported.
    """
    inverse = np.linalg.pinv(design)
    remaining_weight = 1.0 - np.sum(design * inverse.T, axis=1)
    if np.any(remaining_weight <= 1e-9):
        return math.inf
    return float(np.max(np.abs((z - design @ inverse @ z) / remaining_weight)))


def _estimate_plane(
    local: list[FocusAnchor],
    target_x_um: float,
    target_y_um: float,
    current_z_um: float,
    settings: SparseFocusSettings,
) -> FocusEstimate:
    """Fit one particular spatial neighborhood with unchanged physical gates."""
    xy = np.asarray([(point.x_um, point.y_um) for point in local], dtype=float)
    z = np.asarray([point.z_um for point in local], dtype=float)
    centre = np.mean(xy, axis=0)
    centred = xy - centre
    scale = float(np.sqrt(np.mean(np.sum(centred * centred, axis=1))))
    if scale == 0:
        return FocusEstimate(False, None, "plane", "anchor geometry is singular")
    design = np.column_stack((np.ones(len(local)), centred / scale))
    singular_values = np.linalg.svd(design, compute_uv=False)
    rank = int(np.linalg.matrix_rank(design))
    condition = (
        math.inf if singular_values[-1] <= 0 else float(singular_values[0] / singular_values[-1])
    )
    if rank < 3 or condition > settings.maximum_condition_number:
        return FocusEstimate(False, None, "plane", "anchor geometry is ill-conditioned")
    coefficients = np.linalg.lstsq(design, z, rcond=None)[0]
    residual = float(np.max(np.abs(z - design @ coefficients)))
    slope_x = float(coefficients[1] / scale)
    slope_y = float(coefficients[2] / scale)
    slope = math.hypot(slope_x, slope_y)
    if residual > settings.maximum_fit_residual_um:
        return FocusEstimate(
            False,
            None,
            "plane",
            "local focus surface residual is too large",
            support_count=len(local),
            fit_residual_um=residual,
            slope_um_per_um=slope,
        )
    validation_error = _validation_error(design, z)
    if validation_error > settings.maximum_fit_residual_um:
        return FocusEstimate(
            False,
            None,
            "plane",
            "local focus plane fails leave-one-out validation",
            support_count=len(local),
            fit_residual_um=residual,
            slope_um_per_um=slope,
            validation_error_um=validation_error,
        )
    if slope > settings.maximum_plane_slope_um_per_um:
        return FocusEstimate(
            False,
            None,
            "plane",
            "local focus surface slope is too large",
            support_count=len(local),
            fit_residual_um=residual,
            slope_um_per_um=slope,
        )
    target_scaled = (np.asarray((target_x_um, target_y_um)) - centre) / scale
    target_z = float(
        coefficients[0] + coefficients[1] * target_scaled[0] + coefficients[2] * target_scaled[1]
    )
    extrapolation = _extrapolation_distance(local, target_x_um, target_y_um)
    if extrapolation > settings.maximum_extrapolation_um:
        return FocusEstimate(
            False,
            None,
            "plane",
            "target lies too far outside measured support",
            support_count=len(local),
            fit_residual_um=residual,
            slope_um_per_um=slope,
            extrapolation_um=extrapolation,
        )
    if abs(target_z - current_z_um) > settings.maximum_prediction_delta_um:
        return FocusEstimate(
            False,
            None,
            "plane",
            "plane prediction exceeds the current-Z limit",
            support_count=len(local),
            fit_residual_um=residual,
            slope_um_per_um=slope,
            extrapolation_um=extrapolation,
        )
    return FocusEstimate(
        True,
        target_z,
        "plane",
        "bounded local focus plane",
        support_count=len(local),
        fit_residual_um=residual,
        slope_um_per_um=slope,
        extrapolation_um=extrapolation,
        validation_error_um=validation_error,
    )


def _estimate_row(
    points: list[FocusAnchor],
    target_x_um: float,
    current_z_um: float,
    settings: SparseFocusSettings,
) -> FocusEstimate:
    """Validate a local row tangent, never extrapolate across Y or beyond one gap."""
    refused = FocusEstimate(False, None, "row", "no validated local row tangent")
    xs = np.asarray([point.x_um for point in points], dtype=float)
    z = np.asarray([point.z_um for point in points], dtype=float)
    gaps = np.diff(np.sort(xs))
    if np.any(gaps <= 0):
        return refused
    spacing = float(np.min(gaps))
    extrapolation = max(0.0, target_x_um - float(xs.max()), float(xs.min()) - target_x_um)
    if extrapolation > min(spacing, settings.maximum_extrapolation_um):
        return refused
    centre = float(np.mean(xs))
    scale = float(np.std(xs))
    design = np.column_stack((np.ones(len(points)), (xs - centre) / scale))
    if np.linalg.cond(design) > settings.maximum_condition_number:
        return refused
    coefficients = np.linalg.lstsq(design, z, rcond=None)[0]
    residual = float(np.max(np.abs(z - design @ coefficients)))
    # Tighten (never relax) the validation threshold as we leave measured support.
    validation_error = _validation_error(design, z) * (1.0 + extrapolation / spacing)
    slope = float(coefficients[1] / scale)
    target_z = float(coefficients[0] + slope * (target_x_um - centre))
    if (
        not math.isfinite(validation_error)
        or validation_error > settings.maximum_fit_residual_um
        or residual > settings.maximum_fit_residual_um
        or abs(slope) > settings.maximum_plane_slope_um_per_um
        or abs(target_z - current_z_um) > settings.maximum_prediction_delta_um
    ):
        return refused
    return FocusEstimate(
        True,
        target_z,
        "row",
        "bounded independently validated row tangent",
        support_count=len(points),
        fit_residual_um=residual,
        slope_um_per_um=abs(slope),
        extrapolation_um=extrapolation,
        validation_error_um=validation_error,
    )


class SparseFocusScheduler:
    """Run-scoped anchor cadence and pure physical focus prediction."""

    def __init__(
        self,
        settings: SparseFocusSettings,
        *,
        x_um_per_unit: float,
        y_um_per_unit: float,
        z_um_per_unit: float,
    ) -> None:
        """Create an empty run-local scheduler with explicit axis scales."""
        self.settings = settings
        self._scale = (x_um_per_unit, y_um_per_unit, z_um_per_unit)
        self.anchors: list[FocusAnchor] = []
        self.fields_since_anchor = 0
        self.pending: SparseFocusDecision | None = None
        self.anchor_fields = 0
        self.predicted_fields = 0
        self.background_fields = 0
        self.white_fallback_fields = 0
        self.fields_since_rg_probe = 0
        self._initial_row_y_units: int | None = None
        self._recovery_anchors = 0

    def prepare(
        self,
        x_units: int,
        y_units: int,
        current_z_units: int,
        route_z_units: int | None,
    ) -> SparseFocusDecision:
        """Freeze the next field's target and whether hardware AF is required."""
        if self.pending is not None:
            raise RuntimeError("Previous sparse focus decision is still pending")
        if self._initial_row_y_units is None:
            self._initial_row_y_units = y_units
        x_um = x_units * self._scale[0]
        y_um = y_units * self._scale[1]
        current_z_um = current_z_units * self._scale[2]
        estimate = estimate_focus_height(
            tuple(self.anchors),
            target_x_um=x_um,
            target_y_um=y_um,
            current_z_um=current_z_um,
            settings=self.settings,
        )
        warmup = len(self.anchors) < self.settings.minimum_plane_points
        # A refused probe followed by WHITE still satisfies the probe cadence.
        # Otherwise an overdue anchor would re-trigger RG on every later field,
        # even when the retained RG surface gives a validated prediction there.
        cadence_due = self.fields_since_rg_probe >= self.settings.anchor_interval - 1
        validated_prediction = estimate.usable and estimate.source in ("plane", "row")
        requires_anchor = warmup or cadence_due or not validated_prediction
        # A spatially bad model is not a hardware fault. After bounded probes,
        # focus affected fields in WHITE and retry RG only once per interval.
        # Re-evaluate the unchanged RG anchors at every XY: a valid local model
        # resumes normal prediction/cadence immediately, without clearing support.
        recovery_limit = self.settings.maximum_recovery_anchors
        white_only = (
            not validated_prediction
            and self._initial_row_y_units is not None
            and y_units != self._initial_row_y_units
            and recovery_limit is not None
            and self._recovery_anchors >= recovery_limit
            and self.fields_since_rg_probe < self.settings.anchor_interval - 1
        )
        fallback_z = current_z_units if white_only or route_z_units is None else route_z_units
        target_z = (
            fallback_z
            if white_only or not estimate.usable or estimate.target_z_um is None
            else round(estimate.target_z_um / self._scale[2])
        )
        self.pending = SparseFocusDecision(
            x_units=x_units,
            y_units=y_units,
            target_z_units=target_z,
            requires_anchor=requires_anchor,
            estimate=estimate,
            white_only=white_only,
        )
        return self.pending

    def skip_background(self) -> None:
        """Discard a prepared field without advancing the sample cadence."""
        if self.pending is None:
            raise RuntimeError("No sparse focus decision is pending")
        self.background_fields += 1
        self.pending = None

    def _complete_rg_probe(self) -> None:
        """Advance recovery for an RG attempt, even when it handed off to WHITE."""
        if self.pending is None:
            raise RuntimeError("No sparse focus decision is pending")
        self.fields_since_rg_probe = 0
        if self.pending.y_units == self._initial_row_y_units or (
            self.pending.estimate.usable and self.pending.estimate.source in ("plane", "row")
        ):
            self._recovery_anchors = 0
        else:
            self._recovery_anchors += 1

    def complete_anchor(
        self, z_units: int, *, focus_method: str = "rg_simultaneous"
    ) -> FocusAnchor:
        """Record only a successful measured RG result, never WHITE or prediction."""
        if self.pending is None or not self.pending.requires_anchor or self.pending.white_only:
            raise RuntimeError("Pending field is not a focus anchor")
        if focus_method != "rg_simultaneous":
            raise ValueError("Only measured simultaneous RG can train this surface")
        self._complete_rg_probe()
        anchor = FocusAnchor(
            x_um=self.pending.x_units * self._scale[0],
            y_um=self.pending.y_units * self._scale[1],
            z_um=z_units * self._scale[2],
        )
        self.anchors.append(anchor)
        self.anchor_fields += 1
        self.fields_since_anchor = 0
        self.pending = None
        return anchor

    def complete_white_fallback(self, *, rg_attempted: bool) -> None:
        """Finish a WHITE field without promoting its Z or resetting RG cadence."""
        if self.pending is None or not self.pending.requires_anchor:
            raise RuntimeError("Pending field does not require measured focus")
        if self.pending.white_only and rg_attempted:
            raise RuntimeError("WHITE-only field unexpectedly attempted RG")
        if rg_attempted:
            self._complete_rg_probe()
        else:
            self.fields_since_rg_probe += 1
        self.white_fallback_fields += 1
        self.fields_since_anchor += 1
        self.pending = None

    def complete_prediction(self) -> None:
        """Finish one image at predicted Z without promoting it to an anchor."""
        if self.pending is None or self.pending.requires_anchor:
            raise RuntimeError("Pending field is not a predicted focus field")
        self.predicted_fields += 1
        self.fields_since_anchor += 1
        self.fields_since_rg_probe += 1
        self._recovery_anchors = 0
        self.pending = None
