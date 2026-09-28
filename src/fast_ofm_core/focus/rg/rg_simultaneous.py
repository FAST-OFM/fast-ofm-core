"""Linear RAW flat-field and spectral unmixing for simultaneous R/G focus."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal, Self, cast

import cv2
import numpy as np
from pydantic import Field, field_validator, model_validator

from fast_ofm_core.contracts import StrictModel

from .rg_focus_core import (
    PatchBox,
    PatchEvaluation,
    PatchStatus,
    RGFocusCoreSettings,
    candidate_boxes,
    evaluate_patches,
    texture_candidate_mask,
)


class SimultaneousCalibrationSettings(StrictModel):
    """Bounded experimental calibration settings in native RAW-plane pixels."""

    plane_roi: tuple[int, int, int, int] = (314, 60, 1400, 1400)
    frame_timeout_s: float = Field(default=5, gt=0, le=10)
    timeout_s: float = Field(default=300, gt=0, le=600)
    dark_frames: int = Field(default=2, ge=2, le=8)
    fit_cycles: int = Field(default=3, ge=2, le=8)
    validation_cycles: int = Field(default=2, ge=2, le=6)
    smoothing_sigma_px: float = Field(default=4.0, ge=1, le=24)
    minimum_source_norm_dn: float = Field(default=24, ge=4, le=512)
    maximum_saturation_fraction: float = Field(default=0.001, ge=0, le=0.02)
    maximum_invalid_fraction: float = Field(default=0.08, ge=0, le=0.2)
    maximum_condition_number: float = Field(default=5.0, gt=1, le=20)
    maximum_holdout_cv: float = Field(default=0.10, gt=0, le=0.3)
    maximum_cross_leakage_fraction: float = Field(default=0.12, ge=0, le=0.5)
    maximum_mixed_residual_fraction: float = Field(default=0.15, gt=0, le=0.5)
    minimum_mixed_scale: float = Field(default=0.70, gt=0, le=1)
    maximum_mixed_scale: float = Field(default=1.20, ge=1, le=2)
    optics_id: str = Field(default="current", min_length=1, max_length=100)
    brightness_id: str = Field(
        default="arduino-v4-r150-g150-w255-live-unsaved",
        min_length=1,
        max_length=100,
    )

    @field_validator("plane_roi", mode="before")
    @classmethod
    def json_roi(cls, value: object) -> object:
        """Accept the JSON array container while retaining strict integer values."""
        return tuple(value) if isinstance(value, list) else value

    def checked_roi(self, plane_size: list[int] | tuple[int, int]) -> tuple[int, int, int, int]:
        """Require the declared processing window to fit the captured RAW planes."""
        if len(plane_size) != 2:
            raise ValueError("RAW plane size is missing")
        width, height = (int(value) for value in plane_size)
        x, y, roi_width, roi_height = self.plane_roi
        if min(x, y) < 0 or min(roi_width, roi_height) < 16:
            raise ValueError("RAW processing ROI is invalid")
        if x + roi_width > width or y + roi_height > height:
            raise ValueError("RAW processing ROI exceeds the captured planes")
        return self.plane_roi


def _default_focus_core() -> RGFocusCoreSettings:
    """Reuse the proven bounded registration policy after RAW downsampling."""
    return RGFocusCoreSettings(
        registration_highpass_sigma_px=8,
        texture_background_sigma_px=12,
        texture_score_sigma_px=4,
        texture_morphology_kernel_px=5,
        patch_size_px=128,
        patch_stride_px=64,
        minimum_tissue_coverage=0.25,
        minimum_patch_signal=2,
        minimum_patch_std=3,
        minimum_patch_response=0.03,
        minimum_patch_peak_margin=0.02,
        minimum_patch_spectral_correlation=0.03,
        maximum_absolute_shift_px=22,
        maximum_absolute_orthogonal_shift_px=3,
        # The 350 px working crop offers a 4x4 overlapping grid.  Live demo
        # measurements showed that the strongest eight windows cover 95.6% of
        # the unique area covered by twelve, so the remaining overlap adds
        # latency without useful spatial evidence.  Six accepted windows still
        # remain mandatory for robust aggregation and refusal semantics.
        maximum_patch_count=8,
        minimum_patch_count=6,
    )


class SimultaneousFocusSettings(StrictModel):
    """Config-owned one-frame focus curve, movement and registration policy."""

    calibration_positions_um: tuple[float, ...] = Field(
        default=(-32, -24, -16, -8, 0, 8, 16, 24, 32), strict=False
    )
    holdout_positions_um: tuple[float, ...] = Field(default=(-24, -8, 8, 24), strict=False)
    preload_um: float = Field(default=12, gt=0, le=50)
    approach_sign: Literal[-1, 1] = 1
    calibration_timeout_s: float = Field(default=600, gt=0, le=1800)
    autofocus_timeout_s: float = Field(default=20, gt=0, le=120)
    minimum_capture_budget_s: float = Field(default=3, gt=0, le=30)
    maximum_holdout_error_um: float = Field(default=4, gt=0, le=10)
    minimum_slope_norm_px_per_um: float = Field(default=0.05, gt=0)
    focus_tolerance_um: float = Field(default=2, gt=0, le=10)
    maximum_correction_um: float = Field(default=32, gt=0, le=50)
    focus_plane_roi: tuple[int, int, int, int] = (350, 350, 700, 700)
    processing_downsample: Literal[1, 2, 4] = 2
    source_blank_level_dn: float = Field(default=200, gt=32, le=240)
    maximum_component_level: float = Field(default=1.2, gt=1, le=2)
    maximum_frame_saturation_fraction: float = Field(default=0.001, ge=0, le=0.02)
    maximum_frame_residual_fraction: float = Field(default=0.20, gt=0, le=0.5)
    outlier_mad_multiplier: float = Field(default=3.5, ge=1, le=10)
    minimum_outlier_threshold_px: float = Field(default=1, gt=0, le=5)
    maximum_shift_mad_px: float = Field(default=2.5, gt=0, le=10)
    minimum_confidence: float = Field(default=0.03, ge=0, le=1)
    maximum_fit_rmse_um: float = Field(default=4, gt=0, le=10)
    maximum_calibration_cross_track_px: float = Field(default=3, gt=0, le=12)
    core: RGFocusCoreSettings = Field(default_factory=_default_focus_core)

    @field_validator("focus_plane_roi", mode="before")
    @classmethod
    def json_focus_roi(cls, value: object) -> object:
        """Accept a JSON array while retaining strict native-plane integers."""
        return tuple(value) if isinstance(value, list) else value

    def checked_focus_roi(
        self, calibrated_plane_size: tuple[int, int] | list[int]
    ) -> tuple[int, int, int, int]:
        """Require a bounded crop with enough post-downsample registration windows."""
        if len(calibrated_plane_size) != 2:
            raise ValueError("Calibrated simultaneous R/G plane size is missing")
        plane_width, plane_height = (int(value) for value in calibrated_plane_size)
        x, y, width, height = self.focus_plane_roi
        if min(x, y) < 0 or min(width, height) < 16:
            raise ValueError("Simultaneous focus ROI is invalid")
        if x + width > plane_width or y + height > plane_height:
            raise ValueError("Simultaneous focus ROI exceeds the calibrated RAW maps")
        working_shape = (
            height // self.processing_downsample,
            width // self.processing_downsample,
        )
        try:
            available = len(candidate_boxes(working_shape, self.core))
        except ValueError as exc:
            raise ValueError("Simultaneous focus ROI is too small after downsampling") from exc
        if available < self.core.minimum_patch_count:
            raise ValueError("Simultaneous focus ROI cannot provide the minimum patch count")
        return self.focus_plane_roi

    @model_validator(mode="after")
    def valid_grid(self) -> Self:
        """Require an ordered symmetric fit/holdout grid with a central fit point."""
        grid = self.calibration_positions_um
        holdouts = set(self.holdout_positions_um)
        if (
            len(grid) < 7
            or tuple(sorted(grid)) != grid
            or len(set(grid)) != len(grid)
            or 0 not in grid
            or not holdouts
            or not holdouts.issubset(grid)
            or 0 in holdouts
            or not any(value < 0 for value in holdouts)
            or not any(value > 0 for value in holdouts)
            or set(grid) != {-value for value in grid}
            or holdouts != {-value for value in holdouts}
        ):
            raise ValueError("Focus calibration needs an ordered two-sided fit/holdout grid")
        fit = [value for value in grid if value not in holdouts]
        if not any(value < 0 for value in fit) or not any(value > 0 for value in fit):
            raise ValueError("Focus fit points must span both sides of zero")
        proven_correction = min(abs(grid[0]), grid[-1])
        if self.maximum_correction_um > proven_correction:
            raise ValueError("Maximum correction exceeds the two-sided calibration range")
        if self.focus_tolerance_um > self.maximum_correction_um:
            raise ValueError("Focus tolerance exceeds the maximum correction")
        return self


def simultaneous_focus_capture_order(
    settings: SimultaneousFocusSettings,
) -> tuple[float, ...]:
    """Start at focus, then pair both signs from the widest radius inward."""
    magnitudes = sorted(
        {abs(value) for value in settings.calibration_positions_um if value != 0},
        reverse=True,
    )
    return (
        0,
        *(value for magnitude in magnitudes for value in (-magnitude, magnitude)),
    )


class SimultaneousShiftMeasurement(StrictModel):
    """One tissue-aware shift decision from a single spectrally unmixed RAW."""

    status: Literal["ready", "refused"]
    reason: str
    dx: float | None = None
    dy: float | None = None
    confidence: float = Field(ge=0, le=1)
    candidate_patch_count: int = Field(ge=0)
    accepted_patch_count: int = Field(ge=0)
    inlier_patch_count: int = Field(ge=0)
    tissue_coverage: float = Field(ge=0, le=1)
    dx_mad: float = Field(ge=0)
    dy_mad: float = Field(ge=0)
    median_response: float = Field(ge=0)
    patch_status_counts: dict[PatchStatus, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ready_has_shift(self) -> Self:
        """Require coordinates only for measurements accepted by all QC gates."""
        if self.status == "ready" and (self.dx is None or self.dy is None):
            raise ValueError("A ready simultaneous R/G measurement needs dx and dy")
        return self


class SimultaneousFocusObservation(StrictModel):
    """One fixed-grid calibration point measured from one mixed RAW frame."""

    capture_id: str = Field(min_length=1)
    z_um: float
    role: Literal["fit", "holdout"]
    measurement: SimultaneousShiftMeasurement


class SimultaneousFocusCurve(StrictModel):
    """Validated linear two-dimensional shift-to-defocus model."""

    offset_px: tuple[float, float] = Field(strict=False)
    slope_px_per_um: tuple[float, float] = Field(strict=False)
    slope_norm_px_per_um: float = Field(gt=0)
    applicable_z_range_um: tuple[float, float] = Field(strict=False)
    fit_rmse_um: float = Field(ge=0)
    holdout_errors_um: dict[str, float]
    holdout_max_absolute_error_um: float = Field(ge=0)
    cross_track_limit_px: float = Field(gt=0)


def _refused_shift(
    reason: str,
    *,
    candidates: int = 0,
    accepted: int = 0,
    inliers: int = 0,
    coverage: float = 0,
    status_counts: dict[PatchStatus, int] | None = None,
) -> SimultaneousShiftMeasurement:
    return SimultaneousShiftMeasurement(
        status="refused",
        reason=reason,
        confidence=0,
        candidate_patch_count=candidates,
        accepted_patch_count=accepted,
        inlier_patch_count=inliers,
        tissue_coverage=coverage,
        dx_mad=0,
        dy_mad=0,
        median_response=0,
        patch_status_counts=status_counts or {},
    )


def _focus_inputs(
    components: np.ndarray,
    valid: np.ndarray,
    downsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample finite component support without blending invalid map pixels."""
    values = np.asarray(components, dtype=np.float32)
    support = np.array(valid, copy=True)
    if (
        values.ndim != 3
        or values.shape[0] != 2
        or support.shape != values.shape[1:]
        or support.dtype != np.bool_
        or not np.any(support)
    ):
        raise ValueError("Unmixed R/G components and support have invalid geometry")
    support &= np.isfinite(values).all(axis=0)
    if not np.any(support):
        raise ValueError("Unmixed R/G components have no finite common support")
    if downsample == 1:
        return np.where(support[None, ...], np.maximum(values, 0), 0), support
    height, width = support.shape
    output_size = (width // downsample, height // downsample)
    if min(output_size) < 8:
        raise ValueError("Downsampled simultaneous R/G image is too small")
    weights = cv2.resize(support.astype(np.float32), output_size, interpolation=cv2.INTER_AREA)
    reduced = []
    for component in values:
        numerator = cv2.resize(
            np.where(support, component, 0), output_size, interpolation=cv2.INTER_AREA
        )
        reduced.append(
            np.divide(
                numerator,
                weights,
                out=np.zeros_like(numerator),
                where=weights > 0,
            )
        )
    reduced_support = weights >= 1 - 1e-6
    return np.maximum(np.stack(reduced), 0), reduced_support


def _rank_all_focus_boxes(
    source: np.ndarray,
    tissue: np.ndarray,
    settings: SimultaneousFocusSettings,
) -> list[PatchBox]:
    try:
        boxes = candidate_boxes(tissue.shape, settings.core)
    except ValueError:
        return []

    ranked: list[tuple[float, str, PatchBox]] = []
    for box in boxes:
        y_slice = slice(box.y, box.y + box.h)
        x_slice = slice(box.x, box.x + box.w)
        patch_mask = tissue[y_slice, x_slice]
        coverage = float(np.mean(patch_mask))
        if coverage < settings.core.minimum_tissue_coverage:
            continue
        red_values = source[0, y_slice, x_slice][patch_mask]
        green_values = source[1, y_slice, x_slice][patch_mask]
        if red_values.size == 0 or green_values.size == 0:
            continue
        texture_score = min(float(np.std(red_values)), float(np.std(green_values)))
        ranked.append((coverage * texture_score, box.patch_id, box))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [row[2] for row in ranked]


def select_simultaneous_focus_boxes(
    components: np.ndarray,
    valid: np.ndarray,
    settings: SimultaneousFocusSettings,
    *,
    include_reserve: bool = False,
) -> list[PatchBox]:
    """Select the primary windows, optionally followed by ranked reserve windows."""
    values, support = _focus_inputs(components, valid, settings.processing_downsample)
    source = np.rint(
        np.clip(values, 0, settings.maximum_component_level) * settings.source_blank_level_dn
    ).astype(np.uint8)
    texture_source = np.mean(source.astype(np.float32), axis=0)
    tissue, _metrics = texture_candidate_mask(texture_source, support, settings.core)
    ranked = _rank_all_focus_boxes(source, tissue, settings)
    if include_reserve:
        return ranked
    return ranked[: settings.core.maximum_patch_count]


def _evaluate_ranked_focus_boxes(
    source: np.ndarray,
    tissue: np.ndarray,
    support: np.ndarray,
    settings: SimultaneousFocusSettings,
    ranked: list[PatchBox],
    *,
    adaptive_window_selection: bool,
) -> list[PatchEvaluation]:
    """Evaluate the primary budget, then only enough reserve windows to recover."""
    primary = ranked[: settings.core.maximum_patch_count]
    red = source[0].astype(np.float32)
    green = source[1].astype(np.float32)

    def evaluate(boxes: list[PatchBox]) -> list[PatchEvaluation]:
        return evaluate_patches(
            red,
            green,
            tissue,
            settings.core,
            boxes,
            red_source_jpeg8=source[0],
            green_source_jpeg8=source[1],
            source_support=support,
            subpixel_refinement=True,
        )

    rows = evaluate(primary)
    if not adaptive_window_selection:
        return rows
    for reserve_box in ranked[settings.core.maximum_patch_count :]:
        if sum(row.status == "accepted" for row in rows) >= (settings.core.minimum_patch_count):
            break
        rows.extend(evaluate([reserve_box]))
    return rows


def measure_simultaneous_shift(
    components: np.ndarray,
    valid: np.ndarray,
    settings: SimultaneousFocusSettings,
    boxes: Sequence[PatchBox] | None = None,
    *,
    adaptive_window_selection: bool = True,
) -> SimultaneousShiftMeasurement:
    """Measure R/G displacement only on textured support from one unmixed exposure."""
    values, support = _focus_inputs(components, valid, settings.processing_downsample)
    source = np.rint(
        np.clip(values, 0, settings.maximum_component_level) * settings.source_blank_level_dn
    ).astype(np.uint8)
    texture_source = np.mean(source.astype(np.float32), axis=0)
    tissue, tissue_metrics = texture_candidate_mask(texture_source, support, settings.core)
    coverage = tissue_metrics.coverage_inside_valid
    ranked = _rank_all_focus_boxes(source, tissue, settings) if boxes is None else list(boxes)
    selected = ranked[: settings.core.maximum_patch_count] if boxes is None else ranked
    if len(selected) < settings.core.minimum_patch_count:
        return _refused_shift(
            f"Only {len(selected)} tissue windows were available; "
            f"{settings.core.minimum_patch_count} required",
            candidates=len(selected),
            coverage=coverage,
        )
    if len(selected) > settings.core.maximum_patch_count:
        raise ValueError("Simultaneous R/G focus exceeds the configured patch budget")
    rows = _evaluate_ranked_focus_boxes(
        source,
        tissue,
        support,
        settings,
        selected if boxes is not None else ranked,
        adaptive_window_selection=boxes is None and adaptive_window_selection,
    )
    return aggregate_simultaneous_focus_patches(rows, coverage, settings)


def aggregate_simultaneous_focus_patches(
    rows: Sequence[PatchEvaluation],
    coverage: float,
    settings: SimultaneousFocusSettings,
) -> SimultaneousShiftMeasurement:
    """Apply the same robust QC to fixed or adaptively collected tissue windows."""
    status_counts = dict(sorted(Counter(row.status for row in rows).items()))
    accepted = [row for row in rows if row.status == "accepted"]
    if len(accepted) < settings.core.minimum_patch_count:
        return _refused_shift(
            f"Only {len(accepted)} tissue windows passed; {settings.core.minimum_patch_count} required",
            candidates=len(rows),
            accepted=len(accepted),
            coverage=coverage,
            status_counts=status_counts,
        )
    measurements = []
    for row in accepted:
        if row.dx is None or row.dy is None or row.response is None:
            raise RuntimeError("Accepted R/G patch omitted its shift metrics")
        measurements.append(row)
    dx_values = np.asarray([cast(float, row.dx) for row in measurements])
    dy_values = np.asarray([cast(float, row.dy) for row in measurements])
    median_dx = float(np.median(dx_values))
    median_dy = float(np.median(dy_values))
    dx_mad = float(np.median(np.abs(dx_values - median_dx)))
    dy_mad = float(np.median(np.abs(dy_values - median_dy)))
    dx_limit = max(
        settings.minimum_outlier_threshold_px,
        settings.outlier_mad_multiplier * 1.4826 * dx_mad,
    )
    dy_limit = max(
        settings.minimum_outlier_threshold_px,
        settings.outlier_mad_multiplier * 1.4826 * dy_mad,
    )
    inliers = [
        row
        for row in measurements
        if abs(cast(float, row.dx) - median_dx) <= dx_limit
        and abs(cast(float, row.dy) - median_dy) <= dy_limit
    ]
    if len(inliers) < settings.core.minimum_patch_count:
        return _refused_shift(
            "Too few mutually consistent tissue windows",
            candidates=len(rows),
            accepted=len(accepted),
            inliers=len(inliers),
            coverage=coverage,
            status_counts=status_counts,
        )
    final_dx = np.asarray([cast(float, row.dx) for row in inliers])
    final_dy = np.asarray([cast(float, row.dy) for row in inliers])
    final_response = np.asarray([cast(float, row.response) for row in inliers])
    summary_dx = float(np.median(final_dx))
    summary_dy = float(np.median(final_dy))
    summary_dx_mad = float(np.median(np.abs(final_dx - summary_dx)))
    summary_dy_mad = float(np.median(np.abs(final_dy - summary_dy)))
    summary_response = float(np.median(final_response))
    confidence = float(np.clip(summary_response * len(inliers) / len(accepted), 0, 1))
    if (
        summary_dx_mad > settings.maximum_shift_mad_px
        or summary_dy_mad > settings.maximum_shift_mad_px
        or confidence < settings.minimum_confidence
    ):
        return _refused_shift(
            "Tissue-window shift spread or confidence failed",
            candidates=len(rows),
            accepted=len(accepted),
            inliers=len(inliers),
            coverage=coverage,
            status_counts=status_counts,
        )
    return SimultaneousShiftMeasurement(
        status="ready",
        reason="Single mixed RAW passed tissue and registration QC",
        dx=summary_dx,
        dy=summary_dy,
        confidence=confidence,
        candidate_patch_count=len(rows),
        accepted_patch_count=len(accepted),
        inlier_patch_count=len(inliers),
        tissue_coverage=coverage,
        dx_mad=summary_dx_mad,
        dy_mad=summary_dy_mad,
        median_response=summary_response,
        patch_status_counts=status_counts,
    )


class SimultaneousFocusSearchSettings(StrictModel):
    """Diagnostic-only resource limits; never alter the saved Z calibration."""

    maximum_patch_count: int = Field(default=24, ge=8, le=48)
    timeout_s: float = Field(default=8, gt=0, le=20)
    maximum_plane_pixels: int = Field(default=2_000_000, ge=256, le=2_000_000)
    minimum_central_anchor_count: int = Field(default=2, ge=2, le=8)


def inspect_simultaneous_focus_field(
    components: np.ndarray,
    valid: np.ndarray,
    residual: np.ndarray,
    saturated: np.ndarray,
    settings: SimultaneousFocusSettings,
    search: SimultaneousFocusSearchSettings,
    *,
    deadline: float,
) -> dict[str, Any]:
    """Collect bounded peripheral tissue evidence, without authorizing a Z move.

    All arrays cover ONLY the calibrated flat-field rectangle, in native Bayer
    plane coordinates. Central windows are preferred, then the strongest outer
    windows. The caller must first try the unchanged central-crop fast path.
    A spatial flat-field is not proof that the central Z curve works elsewhere.
    """
    shape = valid.shape
    if (
        len(shape) != 2
        or valid.dtype != np.bool_
        or saturated.dtype != np.bool_
        or saturated.shape != shape
        or residual.shape != shape
        or components.shape != (2, *shape)
        or valid.size > search.maximum_plane_pixels
    ):
        raise ValueError("Adaptive focus must fit the bounded calibrated RAW maps")
    if search.maximum_patch_count < settings.core.maximum_patch_count:
        raise ValueError("Adaptive focus budget is smaller than the primary budget")
    if not math.isfinite(deadline):
        raise ValueError("Adaptive focus deadline must be finite")
    roi = settings.checked_focus_roi((shape[1], shape[0]))
    downsample = settings.processing_downsample
    if any(value % downsample for value in (*shape, *roi)):
        raise ValueError("Adaptive RAW coordinates must align with downsampling")
    deadline = min(deadline, time.monotonic() + search.timeout_s)
    report: dict[str, Any] = {
        "diagnostic_only": True,
        "autofocus_authorized": False,
        "z_calibration_scope": "central_crop_only",
        "central_plane_roi": list(roi),
        "search_plane_roi": [0, 0, shape[1], shape[0]],
        "processing_downsample": downsample,
        "limits": search.model_dump(mode="json"),
        "windows": [],
    }
    rows: list[PatchEvaluation] = []
    coverage = 0.0

    def finish(reason: str) -> dict[str, Any]:
        measurement = aggregate_simultaneous_focus_patches(rows, coverage, settings)
        if reason == "deadline":
            measurement = _refused_shift(
                "Adaptive focus search deadline expired",
                candidates=len(rows),
                accepted=measurement.accepted_patch_count,
                inliers=measurement.inlier_patch_count,
                coverage=coverage,
                status_counts=measurement.patch_status_counts,
            )
        return {
            **report,
            "status": measurement.status,
            "stop_reason": reason,
            "measurement": measurement.model_dump(mode="json"),
            "evaluated_patch_count": len(rows),
            "peripheral_patch_count": sum(
                window["region"] == "peripheral" for window in report["windows"]
            ),
        }

    if time.monotonic() >= deadline:
        return finish("deadline")
    finite = np.isfinite(residual)
    support = valid & finite & (residual <= settings.maximum_frame_residual_fraction)
    # Bayer remapping uses neighbours: never count their saturated values as
    # good support merely because the unshifted pixel was below white level.
    unsafe = cv2.dilate(saturated.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    support &= ~unsafe
    support &= np.isfinite(components).all(axis=0)
    if not np.any(support):
        return finish("no_usable_calibrated_support")
    values, support = _focus_inputs(components, support, downsample)
    source = np.rint(
        np.clip(values, 0, settings.maximum_component_level) * settings.source_blank_level_dn
    ).astype(np.uint8)
    tissue, metrics = texture_candidate_mask(
        np.mean(source.astype(np.float32), axis=0), support, settings.core
    )
    coverage = metrics.coverage_inside_valid
    ranked = _rank_all_focus_boxes(source, tissue, settings)
    x, y, width, height = (value // downsample for value in roi)

    def central(box: PatchBox) -> bool:
        return (
            box.x >= x and box.y >= y and box.x + box.w <= x + width and box.y + box.h <= y + height
        )

    # Stable partition preserves tissue/texture ranking within each region.
    ranked = [box for box in ranked if central(box)] + [box for box in ranked if not central(box)]
    report["ranked_patch_count"] = len(ranked)
    report["quality_rejected_patch_count"] = 0
    red, green = (channel.astype(np.float32) for channel in source)
    for box in ranked:
        if time.monotonic() >= deadline:
            return finish("deadline")
        if len(rows) >= search.maximum_patch_count:
            return finish("patch_budget")
        native = [value * downsample for value in (box.x, box.y, box.w, box.h)]
        nx, ny, nw, nh = native
        area = np.s_[ny : ny + nh, nx : nx + nw]
        residual_values = residual[area][valid[area] & finite[area]]
        saturation_fraction = float(np.mean(unsafe[area]))
        residual_p95 = (
            float(np.percentile(residual_values, 95)) if residual_values.size else math.inf
        )
        if (
            saturation_fraction > settings.maximum_frame_saturation_fraction
            or residual_p95 > settings.maximum_frame_residual_fraction
        ):
            report["quality_rejected_patch_count"] += 1
            continue
        evaluated = evaluate_patches(
            red,
            green,
            tissue,
            settings.core,
            [box],
            red_source_jpeg8=source[0],
            green_source_jpeg8=source[1],
            source_support=support,
            subpixel_refinement=True,
        )
        rows.extend(evaluated)
        report["windows"].extend(
            {
                **row.model_dump(mode="json"),
                "region": "central" if central(box) else "peripheral",
                "native_plane_roi": native,
                "saturation_fraction": saturation_fraction,
                "residual_p95": residual_p95,
            }
            for row in evaluated
        )
        # Accepted is not enough: outlier, spread and confidence QC must pass
        # too. Never stop at six mutually inconsistent correlation peaks.
        if aggregate_simultaneous_focus_patches(rows, coverage, settings).status == "ready":
            if time.monotonic() >= deadline:
                # Do not advertise a late result as an in-budget recovery.
                return finish("deadline")
            return finish("sufficient_consistent_windows")
    return finish("candidates_exhausted")


def fit_simultaneous_focus_curve(
    observations: list[SimultaneousFocusObservation],
    settings: SimultaneousFocusSettings,
) -> SimultaneousFocusCurve:
    """Fit on declared points and reject using distinct two-sided holdouts."""
    expected = set(settings.calibration_positions_um)
    if (
        len(observations) != len(expected)
        or {row.z_um for row in observations} != expected
        or len({row.capture_id for row in observations}) != len(observations)
    ):
        raise ValueError("Focus observations do not match the declared calibration grid")
    if any(row.measurement.status != "ready" for row in observations):
        raise ValueError("Every focus calibration point must pass shift QC")
    holdout_positions = set(settings.holdout_positions_um)
    if any(
        row.role != ("holdout" if row.z_um in holdout_positions else "fit") for row in observations
    ):
        raise ValueError("Focus fit/holdout roles differ from the declared grid")
    fit = [row for row in observations if row.role == "fit"]
    holdout = [row for row in observations if row.role == "holdout"]
    z = np.asarray([row.z_um for row in fit], dtype=np.float64)
    shifts = np.asarray([[row.measurement.dx, row.measurement.dy] for row in fit], dtype=np.float64)
    design = np.column_stack((np.ones(len(z)), z))
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(design, shifts, rcond=None)
    offset, slope = coefficients
    slope_norm = float(np.linalg.norm(slope))
    if (
        rank != 2
        or not np.isfinite(coefficients).all()
        or slope_norm < settings.minimum_slope_norm_px_per_um
    ):
        raise ValueError("Simultaneous R/G focus slope is missing or too weak")

    def infer(row: SimultaneousFocusObservation) -> float:
        shift = np.asarray([row.measurement.dx, row.measurement.dy], dtype=float)
        return float(np.dot(shift - offset, slope) / np.dot(slope, slope))

    fit_errors = np.asarray([infer(row) - row.z_um for row in fit])
    fit_rmse = float(np.sqrt(np.mean(fit_errors * fit_errors)))
    if fit_rmse > settings.maximum_fit_rmse_um:
        raise ValueError(f"Simultaneous R/G fit error {fit_rmse:.3f} um exceeds the limit")
    holdout_errors = {row.capture_id: infer(row) - row.z_um for row in holdout}
    holdout_max = max(abs(value) for value in holdout_errors.values())
    if holdout_max > settings.maximum_holdout_error_um:
        raise ValueError(f"Simultaneous R/G holdout error {holdout_max:.3f} um exceeds the limit")
    normal = np.asarray((-slope[1], slope[0])) / slope_norm
    cross_track = [
        abs(
            float(
                np.dot(
                    np.asarray([row.measurement.dx, row.measurement.dy]) - offset,
                    normal,
                )
            )
        )
        for row in observations
    ]
    cross_track_p95 = float(np.percentile(cross_track, 95))
    if max(cross_track) > settings.maximum_calibration_cross_track_px:
        raise ValueError("Simultaneous R/G calibration leaves the optical shift axis")
    return SimultaneousFocusCurve(
        offset_px=(float(offset[0]), float(offset[1])),
        slope_px_per_um=(float(slope[0]), float(slope[1])),
        slope_norm_px_per_um=slope_norm,
        applicable_z_range_um=(min(expected), max(expected)),
        fit_rmse_um=fit_rmse,
        holdout_errors_um=holdout_errors,
        holdout_max_absolute_error_um=float(holdout_max),
        cross_track_limit_px=max(1.0, cross_track_p95 + 0.5),
    )


def infer_simultaneous_defocus(
    curve: SimultaneousFocusCurve,
    measurement: SimultaneousShiftMeasurement,
) -> tuple[float, float]:
    """Project one accepted shift to signed defocus and its cross-track residual."""
    if measurement.status != "ready" or measurement.dx is None or measurement.dy is None:
        raise ValueError("Cannot infer defocus from a refused shift measurement")
    return _project_simultaneous_shift(curve, measurement.dx, measurement.dy)


def _project_simultaneous_shift(
    curve: SimultaneousFocusCurve, dx: float, dy: float
) -> tuple[float, float]:
    """Apply the existing signed calibration to an explicitly checked shift vector."""
    offset = np.asarray(curve.offset_px, dtype=float)
    slope = np.asarray(curve.slope_px_per_um, dtype=float)
    delta = np.asarray([dx, dy]) - offset
    denominator = float(np.dot(slope, slope))
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("Focus curve slope is invalid")
    defocus = float(np.dot(delta, slope) / denominator)
    normal = np.asarray((-slope[1], slope[0])) / math.sqrt(denominator)
    cross_track = abs(float(np.dot(delta, normal)))
    if not curve.applicable_z_range_um[0] <= defocus <= curve.applicable_z_range_um[1]:
        raise ValueError("Measured defocus is outside the calibrated range")
    if cross_track > curve.cross_track_limit_px:
        raise ValueError("Measured shift is outside the calibrated optical axis")
    return defocus, cross_track


def validate_peripheral_focus(
    curve: SimultaneousFocusCurve,
    report: dict[str, Any],
    settings: SimultaneousFocusSettings,
) -> tuple[SimultaneousShiftMeasurement | None, dict[str, Any]]:
    """Authorize a recovery only with contemporaneous calibrated-center agreement.

    Spatial flat-field alone cannot validate off-axis Z. The combined estimate
    still needs the normal six-window robust QC, plus two DISTINCT accepted
    windows wholly inside the original calibrated focus crop. Every accepted
    central witness must independently agree within the existing focus tolerance.
    No center evidence, disagreement, late output or malformed evidence -> WHITE.
    """
    evidence: dict[str, Any] = {
        "authorized": False,
        "central_anchor_count": 0,
        "maximum_central_disagreement_um": None,
        "central_agreement_limit_um": settings.focus_tolerance_um,
        "z_calibration_scope": "same_frame_central_agreement",
    }

    def refuse(reason: str) -> tuple[None, dict[str, Any]]:
        return None, {**evidence, "reason": reason}

    if report.get("status") != "ready":
        return refuse("Peripheral search did not pass bounded measurement QC")
    try:
        search_x, search_y, search_width, search_height = report["search_plane_roi"]
        search_limits = SimultaneousFocusSearchSettings()
        if (
            search_x != 0
            or search_y != 0
            or min(search_width, search_height) <= 0
            or search_width * search_height > search_limits.maximum_plane_pixels
        ):
            return refuse("Peripheral search geometry exceeds calibrated area budget")
        rows = [
            PatchEvaluation.model_validate(
                {name: window[name] for name in PatchEvaluation.model_fields if name in window}
            )
            for window in report["windows"]
        ]
        coordinates = [(row.x, row.y, row.w, row.h) for row in rows]
        if len(set(coordinates)) != len(coordinates) or any(
            min(row.x, row.y) < 0
            or row.w != settings.core.patch_size_px
            or row.h != settings.core.patch_size_px
            or (row.x + row.w) * settings.processing_downsample > search_width
            or (row.y + row.h) * settings.processing_downsample > search_height
            for row in rows
        ):
            return refuse("Peripheral evidence repeats or changes a physical window")
        measurement = aggregate_simultaneous_focus_patches(
            rows, report["measurement"]["tissue_coverage"], settings
        )
        if measurement.status != "ready":
            return refuse(measurement.reason)
        downsample = settings.processing_downsample
        x, y, width, height = settings.focus_plane_roi
        central = [
            row
            for row in rows
            if row.status == "accepted"
            and row.x * downsample >= x
            and row.y * downsample >= y
            and (row.x + row.w) * downsample <= x + width
            and (row.y + row.h) * downsample <= y + height
        ]
        evidence["central_anchor_count"] = len(central)
        if len(central) < SimultaneousFocusSearchSettings().minimum_central_anchor_count:
            return refuse("Peripheral Z lacks two accepted calibrated-center witnesses")
        defocus, cross_track = infer_simultaneous_defocus(curve, measurement)
        differences = []
        for row in central:
            if row.dx is None or row.dy is None or row.response is None:
                return refuse("Central witness is missing accepted shift metrics")
            central_z, _ = _project_simultaneous_shift(curve, row.dx, row.dy)
            differences.append(abs(central_z - defocus))
        disagreement = max(differences)
        evidence["maximum_central_disagreement_um"] = disagreement
        if disagreement > settings.focus_tolerance_um:
            return refuse("Peripheral Z disagrees with calibrated-center witnesses")
    except (KeyError, TypeError, ValueError) as exc:
        return refuse(f"Peripheral focus evidence refused: {exc}")
    return measurement, {
        **evidence,
        "authorized": True,
        "reason": "Six-window QC and same-frame calibrated-center agreement passed",
        "defocus_um": defocus,
        "cross_track_px": cross_track,
    }


def _checked_planes(planes: np.ndarray) -> np.ndarray:
    """Return finite four-channel RAW data as float32 without changing geometry."""
    value = np.asarray(planes)
    if (
        value.ndim != 3
        or value.shape[0] != 4
        or min(value.shape[1:]) < 16
        or not np.isfinite(value).all()
        or np.min(value) < 0
    ):
        raise ValueError("Expected four finite nonnegative RAW Bayer planes")
    return value.astype(np.float32, copy=False)


def crop_planes(planes: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Own one bounded native-plane ROI so full captures can be released promptly."""
    value = _checked_planes(planes)
    x, y, width, height = roi
    if min(x, y) < 0 or min(width, height) < 16:
        raise ValueError("RAW processing ROI is invalid")
    if x + width > value.shape[2] or y + height > value.shape[1]:
        raise ValueError("RAW processing ROI exceeds the captured planes")
    return np.array(value[:, y : y + height, x : x + width], copy=True, order="C")


def crop_spectral_maps(
    maps: dict[str, np.ndarray], roi: tuple[int, int, int, int]
) -> dict[str, np.ndarray]:
    """Own one matching flat-field ROI before spectral correction is evaluated."""
    if set(maps) != {"dark", "red_response", "green_response", "valid"}:
        raise ValueError("Simultaneous R/G calibration map keys changed")
    dark = _checked_planes(maps["dark"])
    if (
        maps["red_response"].shape != dark.shape
        or maps["green_response"].shape != dark.shape
        or maps["valid"].shape != dark.shape[1:]
        or maps["valid"].dtype != np.bool_
    ):
        raise ValueError("Simultaneous R/G calibration map geometry changed")
    x, y, width, height = roi
    cropped = {
        name: crop_planes(maps[name], roi) for name in ("dark", "red_response", "green_response")
    }
    cropped["valid"] = np.array(
        maps["valid"][y : y + height, x : x + width],
        dtype=bool,
        copy=True,
        order="C",
    )
    if cropped["valid"].shape != (height, width):
        raise ValueError("Simultaneous R/G calibration valid map crop changed")
    return cropped


def align_native_planes(
    planes: np.ndarray, offsets_xy: list[list[int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Align all four Bayer lattices to the same sensor-cell-centre grid."""
    value = _checked_planes(planes)
    remaps, valid = _alignment_maps(value.shape[1:], offsets_xy)
    return _remap_planes(value, remaps), valid


def _alignment_maps(
    shape: tuple[int, int], offsets_xy: list[list[int]]
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Prepare the fixed Bayer remaps and their common interpolation support."""
    offsets = np.asarray(offsets_xy, dtype=np.float32)
    if offsets.shape != (4, 2) or not np.isfinite(offsets).all():
        raise ValueError("RAW Bayer channel offsets are invalid")
    height, width = shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    remaps = []
    valid = np.ones((height, width), dtype=bool)
    support_source = np.ones((height, width), dtype=np.float32)
    for dx, dy in offsets:
        map_x = xx + (0.5 - dx) / 2
        map_y = yy + (0.5 - dy) / 2
        remaps.append((map_x, map_y))
        support = cv2.remap(
            support_source,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid &= support >= 1 - 1e-6
    return remaps, valid


def _remap_planes(value: np.ndarray, remaps: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Apply prepared Bayer geometry to this frame, never to a cached frame."""
    aligned = np.empty_like(value, dtype=np.float32)
    for index, (map_x, map_y) in enumerate(remaps):
        aligned[index] = cv2.remap(
            value[index],
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return aligned


def _smooth(planes: np.ndarray, sigma: float) -> np.ndarray:
    return np.stack(
        [
            cv2.GaussianBlur(plane, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
            for plane in planes
        ]
    ).astype(np.float32)


def _condition(
    red: np.ndarray, green: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the 2x2 Gram terms and per-pixel two-column condition number."""
    a = np.sum(red * red, axis=0)
    b = np.sum(red * green, axis=0)
    c = np.sum(green * green, axis=0)
    trace = a + c
    delta = np.sqrt(np.maximum((a - c) ** 2 + 4 * b * b, 0))
    largest = (trace + delta) / 2
    smallest = (trace - delta) / 2
    condition = np.sqrt(
        np.divide(
            largest,
            smallest,
            out=np.full_like(largest, np.inf),
            where=smallest > 0,
        )
    )
    return a, b, c, condition


def fit_spectral_flat_field(
    dark: np.ndarray,
    red: np.ndarray,
    green: np.ndarray,
    *,
    offsets_xy: list[list[int]],
    white_level: float,
    settings: SimultaneousCalibrationSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit spatial four-channel response columns for RED-only and GREEN-only."""
    dark_value = _checked_planes(dark)
    red_value = _checked_planes(red)
    green_value = _checked_planes(green)
    if red_value.shape != dark_value.shape or green_value.shape != dark_value.shape:
        raise ValueError("RAW fit frames have different geometry")
    if not np.isfinite(white_level) or white_level <= 0:
        raise ValueError("RAW white level is invalid")
    saturation = {
        "red": np.mean(red_value >= white_level, axis=(1, 2)).tolist(),
        "green": np.mean(green_value >= white_level, axis=(1, 2)).tolist(),
    }
    if max(*saturation["red"], *saturation["green"]) > settings.maximum_saturation_fraction:
        raise ValueError("A source-only RAW flat-field is saturated")
    dark_aligned, support = align_native_planes(dark_value, offsets_xy)
    red_aligned, red_support = align_native_planes(red_value, offsets_xy)
    green_aligned, green_support = align_native_planes(green_value, offsets_xy)
    red_response = _smooth(np.maximum(red_aligned - dark_aligned, 0), settings.smoothing_sigma_px)
    green_response = _smooth(
        np.maximum(green_aligned - dark_aligned, 0), settings.smoothing_sigma_px
    )
    a, _b, c, condition = _condition(red_response, green_response)
    common_support = support & red_support & green_support
    valid = (
        common_support
        & (np.sqrt(a) >= settings.minimum_source_norm_dn)
        & (np.sqrt(c) >= settings.minimum_source_norm_dn)
        & np.isfinite(condition)
        & (condition <= settings.maximum_condition_number)
    )
    if not np.any(common_support):
        raise ValueError("RAW Bayer planes have no common aligned support")
    invalid_fraction = float(1 - np.mean(valid[common_support]))
    if invalid_fraction > settings.maximum_invalid_fraction:
        raise ValueError(f"Too much invalid simultaneous R/G area: {100 * invalid_fraction:.2f}%")
    condition_values = condition[valid]
    signatures = {
        name: [float(np.median(plane[valid])) for plane in response]
        for name, response in (("red", red_response), ("green", green_response))
    }
    signature_matrix = np.asarray([signatures["red"], signatures["green"]], dtype=np.float64).T
    report: dict[str, Any] = {
        "measurement_domain": "linear-raw12-bayer-common-grid",
        "source_signatures_dn": signatures,
        "signature_condition_number": float(np.linalg.cond(signature_matrix)),
        "condition_number": {
            "median": float(np.median(condition_values)),
            "p95": float(np.percentile(condition_values, 95)),
            "maximum": float(np.max(condition_values)),
        },
        "invalid_fraction": invalid_fraction,
        "alignment_support_fraction": float(np.mean(common_support)),
        "saturation_fraction": saturation,
    }
    return {
        "dark": dark_aligned.astype(np.float32),
        "red_response": red_response,
        "green_response": green_response,
        "valid": valid,
    }, report


def unmix_components(
    planes: np.ndarray,
    offsets_xy: list[list[int]],
    maps: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the spatial four-channel by two-source model in closed form."""
    aligned, support = align_native_planes(planes, offsets_xy)
    a, b, c, _condition_number = _condition(maps["red_response"], maps["green_response"])
    return _unmix_aligned(aligned, support, maps, (a, b, c))


class PreparedSpectralUnmix:
    """Own one immutable ROI calibration and reusable, frame-independent terms.

    The owner must replace this object when the profile, archive identity, ROI or
    Bayer offsets change. There is no global cache and no cached image evidence.
    Inputs are copied so later caller mutations cannot silently stale the Gram terms.
    """

    def __init__(self, maps: dict[str, np.ndarray], offsets_xy: list[list[int]]) -> None:
        """Copy the bounded maps and prepare their geometry and Gram matrix once."""
        self._maps = {
            name: np.array(maps[name], copy=True, order="C")
            for name in ("dark", "red_response", "green_response", "valid")
        }
        shape = self._maps["dark"].shape
        if (
            len(shape) != 3
            or shape[0] != 4
            or min(shape) <= 0
            or self._maps["red_response"].shape != shape
            or self._maps["green_response"].shape != shape
            or self._maps["valid"].shape != shape[1:]
            or self._maps["valid"].dtype != np.bool_
        ):
            raise ValueError("Simultaneous R/G calibration maps are incompatible")
        self._remaps, self._support = _alignment_maps(shape[1:], offsets_xy)
        self._offsets = np.array(offsets_xy, dtype=np.float32, copy=True)
        a, b, c, _condition_number = _condition(
            self._maps["red_response"], self._maps["green_response"]
        )
        self._gram = (a, b, c)
        for value in (
            *self._maps.values(),
            *self._gram,
            self._offsets,
            self._support,
            *(plane for pair in self._remaps for plane in pair),
        ):
            value.setflags(write=False)

    def unmix(
        self, planes: np.ndarray, offsets_xy: list[list[int]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unmix a fresh frame only if its geometry matches this calibration."""
        value = _checked_planes(planes)
        if value.shape != self._maps["dark"].shape or not np.array_equal(
            np.asarray(offsets_xy, dtype=np.float32), self._offsets
        ):
            raise ValueError("Prepared R/G calibration differs from RAW geometry")
        return _unmix_aligned(
            _remap_planes(value, self._remaps), self._support, self._maps, self._gram
        )


def _unmix_aligned(
    aligned: np.ndarray,
    support: np.ndarray,
    maps: dict[str, np.ndarray],
    gram: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve both sources and frame residuals with unchanged arithmetic and gates."""
    dark = maps["dark"]
    red = maps["red_response"]
    green = maps["green_response"]
    valid = maps["valid"] & support
    if (
        aligned.shape != dark.shape
        or red.shape != aligned.shape
        or green.shape != aligned.shape
        or valid.shape != aligned.shape[1:]
        or valid.dtype != np.bool_
    ):
        raise ValueError("Simultaneous R/G calibration maps are incompatible")
    observed = aligned - dark
    a, b, c = gram
    red_projection = np.sum(red * observed, axis=0)
    green_projection = np.sum(green * observed, axis=0)
    determinant = a * c - b * b
    valid &= determinant > np.finfo(np.float32).eps
    red_component = np.divide(
        c * red_projection - b * green_projection,
        determinant,
        out=np.full_like(determinant, np.nan),
        where=valid,
    )
    green_component = np.divide(
        a * green_projection - b * red_projection,
        determinant,
        out=np.full_like(determinant, np.nan),
        where=valid,
    )
    reconstruction = red * red_component[None, ...] + green * green_component[None, ...]
    residual = np.linalg.norm(observed - reconstruction, axis=0)
    residual /= np.maximum(np.linalg.norm(observed, axis=0), 1)
    red_component[~valid] = np.nan
    green_component[~valid] = np.nan
    residual[~valid] = np.nan
    return np.stack((red_component, green_component)), residual, valid


def _component_metrics(
    components: np.ndarray, residual: np.ndarray, valid: np.ndarray
) -> dict[str, Any]:
    values = [component[valid] for component in components]
    medians = [float(np.median(value)) for value in values]
    cvs = [float(np.std(value) / max(abs(np.median(value)), 1e-6)) for value in values]
    return {
        "component_median": medians,
        "component_cv": cvs,
        "reconstruction_residual": {
            "median": float(np.median(residual[valid])),
            "p95": float(np.percentile(residual[valid], 95)),
        },
    }


def validate_spectral_flat_field(
    red_holdout: np.ndarray,
    green_holdout: np.ndarray,
    mixed_holdout: np.ndarray,
    *,
    offsets_xy: list[list[int]],
    white_level: float,
    maps: dict[str, np.ndarray],
    settings: SimultaneousCalibrationSettings,
) -> dict[str, Any]:
    """Validate unused source-only fields and the additive mixed-light exposure."""
    results: dict[str, Any] = {}
    for name, planes in (
        ("red", red_holdout),
        ("green", green_holdout),
        ("mixed", mixed_holdout),
    ):
        value = _checked_planes(planes)
        saturation = np.mean(value >= white_level, axis=(1, 2))
        if np.max(saturation) > settings.maximum_saturation_fraction:
            raise ValueError(f"Held-out {name.upper()} RAW frame is saturated")
        components, residual, valid = unmix_components(value, offsets_xy, maps)
        metrics = _component_metrics(components, residual, valid)
        metrics["saturation_fraction"] = saturation.tolist()
        results[name] = metrics

    red_scale, red_leak = results["red"]["component_median"]
    green_leak, green_scale = results["green"]["component_median"]
    mixed_red, mixed_green = results["mixed"]["component_median"]
    cross_leakage = max(
        abs(red_leak) / max(abs(red_scale), 1e-6),
        abs(green_leak) / max(abs(green_scale), 1e-6),
    )
    holdout_cv = max(
        results["red"]["component_cv"][0],
        results["green"]["component_cv"][1],
    )
    mixed_residual = results["mixed"]["reconstruction_residual"]["p95"]
    if min(red_scale, green_scale) <= 0:
        raise ValueError("A source-only held-out component is not positive")
    if cross_leakage > settings.maximum_cross_leakage_fraction:
        raise ValueError("Source-only held-out spectral leakage is too high")
    if holdout_cv > settings.maximum_holdout_cv:
        raise ValueError("Source-only held-out flat-field residual is too high")
    if mixed_residual > settings.maximum_mixed_residual_fraction:
        raise ValueError("Held-out mixed RAW additivity residual is too high")
    if not all(
        settings.minimum_mixed_scale <= value <= settings.maximum_mixed_scale
        for value in (mixed_red, mixed_green)
    ):
        raise ValueError("Held-out mixed RAW source scale is outside the accepted range")
    return {
        "kind": "unused_source_only_and_mixed_raw_holdouts",
        "red": results["red"],
        "green": results["green"],
        "mixed": results["mixed"],
        "cross_leakage_fraction": float(cross_leakage),
        "maximum_source_holdout_cv": float(holdout_cv),
        "mixed_source_scale": {"red": mixed_red, "green": mixed_green},
        "mixed_reconstruction_residual_p95": float(mixed_residual),
    }
