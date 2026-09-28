"""Pure tissue-mask and spatial R/G mutual-information shift primitives.

This module deliberately has no camera, illumination, stage, file, or workflow imports.
It accepts already corrected, geometrically aligned two-dimensional planes.

Two experimental behaviours are intentionally not preserved: reused boxes do not bypass
mask coverage or correlation-response checks, and source JPEG channel selection is not
hidden inside the estimator. Operational tissue eligibility and failure policy belong to
the next integration layer.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Literal, Self, cast

import cv2
import numpy as np
from pydantic import AliasChoices, Field, model_validator

from fast_ofm_core.contracts import StrictModel

_native_candidates: (
    Callable[
        [np.ndarray, np.ndarray, np.ndarray, int, int, int, float],
        list[tuple[float, int, int]],
    ]
    | None
)
try:
    from ._rg_nmi import candidates as _accelerated_candidates
except ImportError:
    _native_candidates = None
    logging.getLogger(__name__).warning(
        "Native NMI extension unavailable; using the portable exhaustive calculation"
    )
else:
    _native_candidates = _accelerated_candidates

NMI_BACKEND = "native-exhaustive" if _native_candidates is not None else "opencv-exhaustive"

# Execution-only concurrency: it does not change the frozen optical policy or
# the ordered set of patch results. Two workers avoid oversubscribing a Pi 4.
PATCH_EVALUATION_WORKERS = 2


class RGFocusCoreSettings(StrictModel):
    """Explicit tissue, registration, and consistency thresholds."""

    registration_highpass_sigma_px: float = Field(
        default=8.0,
        gt=0,
        validation_alias=AliasChoices("registration_highpass_sigma_px", "phase_highpass_sigma_px"),
    )
    mutual_information_bins: int = Field(default=32, ge=8, le=256)
    texture_background_sigma_px: float = Field(default=12.0, gt=0)
    texture_score_sigma_px: float = Field(default=4.0, gt=0)
    texture_gradient_weight: float = Field(default=0.25, ge=0)
    texture_percentile: float = Field(default=55.0, ge=0, le=100)
    texture_morphology_kernel_px: int = Field(default=7, ge=1, le=63)
    patch_size_px: int = Field(default=128, ge=8)
    patch_stride_px: int = Field(default=64, ge=1)
    minimum_tissue_coverage: float = Field(default=0.35, ge=0, le=1)
    minimum_patch_signal: float = Field(default=16.0, ge=0)
    jpeg8_saturation_level: float = Field(default=250.0, gt=0, le=255)
    maximum_patch_saturation_fraction: float = Field(default=0.01, ge=0, le=1)
    minimum_patch_std: float = Field(default=4.0, ge=0)
    minimum_patch_response: float = Field(default=0.05, ge=0, le=1)
    minimum_patch_peak_margin: float = Field(default=0.03, ge=0, le=1)
    minimum_patch_spectral_correlation: float = Field(default=0.05, ge=-1, le=1)
    maximum_absolute_shift_px: float = Field(default=40.0, gt=0)
    maximum_absolute_orthogonal_shift_px: float = Field(default=4.0, gt=0)
    masked_inpaint_radius_px: float = Field(default=3.0, gt=0, le=32)
    minimum_shifted_support_fraction: float = Field(default=0.50, gt=0, le=1)
    maximum_patch_count: int = Field(default=80, ge=1)
    minimum_patch_count: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def consistent_settings(self) -> Self:
        """Reject parameter combinations that cannot produce a valid grid."""
        if self.patch_stride_px > self.patch_size_px:
            raise ValueError("Patch stride cannot exceed patch size")
        if self.minimum_patch_count > self.maximum_patch_count:
            raise ValueError("Minimum patch count exceeds maximum patch count")
        if self.texture_morphology_kernel_px % 2 == 0:
            raise ValueError("Texture morphology kernel must be odd")
        if self.maximum_absolute_orthogonal_shift_px > self.maximum_absolute_shift_px:
            raise ValueError("Orthogonal shift range exceeds primary shift range")
        return self


class TextureMaskMetrics(StrictModel):
    """Diagnostics from the candidate texture-mask calculation."""

    threshold: float
    coverage_total: float = Field(ge=0, le=1)
    coverage_inside_valid: float = Field(ge=0, le=1)


class PatchBox(StrictModel):
    """One fixed rectangle in the common R/G plane."""

    patch_id: str = Field(min_length=1)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)


class PatchMeasurement(StrictModel):
    """Accepted mutual-information measurement for one common patch."""

    patch_id: str
    x: int
    y: int
    w: int
    h: int
    dx: float
    dy: float
    response: float
    peak_margin: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    median_r: float = Field(ge=0)
    median_g: float = Field(ge=0)
    std_r: float = Field(ge=0)
    std_g: float = Field(ge=0)
    saturation_r: float = Field(ge=0, le=1)
    saturation_g: float = Field(ge=0, le=1)
    spectral_correlation: float = Field(ge=-1, le=1)


PatchStatus = Literal[
    "accepted",
    "insufficient_support",
    "low_signal",
    "saturated",
    "low_texture",
    "correlation_failed",
    "shift_out_of_range",
    "spectral_mismatch",
]


class PatchEvaluation(StrictModel):
    """Accepted or rejected result for every fixed common-plane window."""

    patch_id: str
    x: int
    y: int
    w: int
    h: int
    status: PatchStatus
    reason: str
    coverage: float = Field(ge=0, le=1)
    median_r: float | None = Field(default=None, ge=0)
    median_g: float | None = Field(default=None, ge=0)
    std_r: float | None = Field(default=None, ge=0)
    std_g: float | None = Field(default=None, ge=0)
    saturation_r: float | None = Field(default=None, ge=0, le=1)
    saturation_g: float | None = Field(default=None, ge=0, le=1)
    dx: float | None = None
    dy: float | None = None
    response: float | None = None
    peak_margin: float | None = Field(default=None, ge=0, le=1)
    spectral_correlation: float | None = Field(default=None, ge=-1, le=1)


class ShiftSummary(StrictModel):
    """Robust descriptive statistics; this is not an operational QC decision."""

    dx: float
    dy: float
    response: float
    patch_count: int = Field(ge=1)
    dx_mad: float = Field(ge=0)
    dy_mad: float = Field(ge=0)


def _finite_plane(image: np.ndarray, name: str) -> np.ndarray:
    """Return one finite, non-empty float64 plane."""
    value = np.asarray(image)
    if value.ndim != 2 or value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite non-empty 2D plane")
    return np.array(value, dtype=np.float64, order="C", copy=True)


def _fill_masked_input(
    image: np.ndarray,
    mask: np.ndarray,
    settings: RGFocusCoreSettings,
) -> np.ndarray:
    """Fill excluded pixels independently without adding a shared mask edge.

    Filling is performed independently for each colour from that colour's accepted
    support.  Keeping this step separate also prevents non-finite flat-field samples
    outside support from bleeding into valid pixels during subpixel interpolation.
    """
    value = np.asarray(image)
    support = np.asarray(mask)
    if (
        value.ndim != 2
        or support.dtype != np.bool_
        or support.shape != value.shape
        or not np.any(support)
        or not np.isfinite(value[support]).all()
        or np.min(value[support]) < 0
    ):
        raise ValueError("Masked registration input or support is invalid")
    working: np.ndarray = np.full(value.shape, float(np.median(value[support])), dtype=np.float32)
    working[support] = value[support].astype(np.float32)
    if not np.all(support):
        working = cv2.inpaint(
            working,
            (~support).astype(np.uint8) * 255,
            settings.masked_inpaint_radius_px,
            cv2.INPAINT_TELEA,
        )
    return working


def _quantise_registration_plane(
    image: np.ndarray,
    mask: np.ndarray,
    settings: RGFocusCoreSettings,
) -> tuple[np.ndarray, np.ndarray]:
    """Independently quantise one colour and retain high-pass texture for QC."""
    filled = _fill_masked_input(image, mask, settings).astype(np.float64)
    texture = filled - cv2.GaussianBlur(filled, (0, 0), settings.registration_highpass_sigma_px)
    values = filled[mask]
    low, high = np.percentile(values, (1.0, 99.0))
    if not all(math.isfinite(float(value)) for value in (low, high)) or high <= low:
        raise ValueError("Registration input lacks usable tissue texture")
    scaled = np.clip((filled - low) / (high - low), 0.0, 1.0)
    quantised = np.minimum(
        (scaled * settings.mutual_information_bins).astype(np.int16),
        settings.mutual_information_bins - 1,
    )
    return quantised, texture


@lru_cache(maxsize=8)
def _count_log_count_table(maximum_count: int) -> np.ndarray:
    """Cache ``n * log(n)`` values shared by every displacement of one patch."""
    if maximum_count < 1:
        raise ValueError("Mutual-information count table must be non-empty")
    counts = np.arange(maximum_count + 1, dtype=np.float64)
    table = np.zeros(maximum_count + 1, dtype=np.float64)
    table[1:] = counts[1:] * np.log(counts[1:])
    table.flags.writeable = False
    return table


def _entropy_from_counts(counts: np.ndarray, total: int, count_log_count: np.ndarray) -> float:
    """Return entropy from integer counts without repeated per-candidate logs."""
    return math.log(total) - float(np.sum(count_log_count[counts])) / total


def _normalised_mutual_information_from_joint(
    joint: np.ndarray,
    total: int,
    count_log_count: np.ndarray,
) -> float:
    """Return symmetric NMI from a joint histogram and its sample count."""
    if total < 16:
        raise ValueError("Mutual-information overlap is insufficient")
    if count_log_count.shape[0] <= total:
        raise ValueError("Mutual-information count table is too small")
    first_entropy = _entropy_from_counts(joint.sum(axis=1), total, count_log_count)
    second_entropy = _entropy_from_counts(joint.sum(axis=0), total, count_log_count)
    joint_entropy = _entropy_from_counts(joint, total, count_log_count)
    information = first_entropy + second_entropy - joint_entropy
    denominator = first_entropy + second_entropy
    if not math.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("Mutual-information input lacks entropy")
    return float(np.clip(2 * information / denominator, 0, 1))


def _normalised_mutual_information(
    first: np.ndarray,
    second: np.ndarray,
    bins: int,
    *,
    count_log_count: np.ndarray | None = None,
) -> float:
    """Return symmetric normalised mutual information for equal-length vectors."""
    if first.size < 16 or second.shape != first.shape:
        raise ValueError("Mutual-information overlap is insufficient")
    joint = np.bincount(
        (first.astype(np.int32) * bins + second.astype(np.int32)).ravel(),
        minlength=bins * bins,
    ).reshape(bins, bins)
    total = int(np.sum(joint))
    if total < 16:
        raise ValueError("Mutual-information overlap is insufficient")
    lookup = count_log_count if count_log_count is not None else _count_log_count_table(total)
    return _normalised_mutual_information_from_joint(joint, total, lookup)


def _overlap_slices(
    shape: tuple[int, int], dx: int, dy: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Return first/second slices for a displacement from first to second."""
    height, width = shape
    first_y = slice(max(0, -dy), min(height, height - dy))
    second_y = slice(max(0, dy), min(height, height + dy))
    first_x = slice(max(0, -dx), min(width, width - dx))
    second_x = slice(max(0, dx), min(width, width + dx))
    return (first_y, first_x), (second_y, second_x)


def _parabolic_peak_offset(negative: float, centre: float, positive: float) -> float:
    """Refine one interior maximum without changing the exhaustive search."""
    denominator = negative - 2 * centre + positive
    if not math.isfinite(denominator) or denominator >= -1e-12:
        return 0.0
    offset = 0.5 * (negative - positive) / denominator
    return float(np.clip(offset, -0.75, 0.75)) if math.isfinite(offset) else 0.0


def _refined_peak(
    candidates: Sequence[tuple[float, int, int]],
    response: float,
    integer_dx: int,
    integer_dy: int,
) -> tuple[float, float]:
    """Interpolate independent axes only when both neighbouring scores exist."""
    scores = {(x, y): score for score, x, y in candidates}
    dx = float(integer_dx)
    dy = float(integer_dy)
    left = scores.get((integer_dx - 1, integer_dy))
    right = scores.get((integer_dx + 1, integer_dy))
    above = scores.get((integer_dx, integer_dy - 1))
    below = scores.get((integer_dx, integer_dy + 1))
    if left is not None and right is not None:
        dx += _parabolic_peak_offset(left, response, right)
    if above is not None and below is not None:
        dy += _parabolic_peak_offset(above, response, below)
    return dx, dy


def mutual_information_shift_masked(
    first: np.ndarray,
    second: np.ndarray,
    mask: np.ndarray,
    settings: RGFocusCoreSettings,
    *,
    subpixel_refinement: bool = False,
) -> tuple[float, float, float, float, float]:
    """Register cross-colour tissue using bounded normalised mutual information."""
    a = np.asarray(first)
    b = np.asarray(second)
    support = np.asarray(mask)
    if a.shape != b.shape or support.shape != a.shape or support.dtype != np.bool_:
        raise ValueError("Mutual-information inputs and support must share geometry")
    first_q, first_texture = _quantise_registration_plane(a, support, settings)
    second_q, second_texture = _quantise_registration_plane(b, support, settings)
    maximum_x = int(math.ceil(settings.maximum_absolute_shift_px))
    maximum_y = int(math.ceil(settings.maximum_absolute_orthogonal_shift_px))
    if a.shape[1] <= maximum_x or a.shape[0] <= maximum_y:
        raise ValueError("Configured shift range exceeds the patch geometry")
    bins = settings.mutual_information_bins
    calculate_grid = _native_candidates or _portable_nmi_candidates
    candidates = calculate_grid(
        np.ascontiguousarray(first_q, dtype=np.uint8),
        np.ascontiguousarray(second_q, dtype=np.uint8),
        np.ascontiguousarray(support, dtype=np.uint8),
        bins,
        maximum_x,
        maximum_y,
        settings.minimum_shifted_support_fraction,
    )
    if not candidates:
        raise ValueError("No shift has enough finite tissue overlap")
    candidates.sort(key=lambda row: (-row[0], row[1] * row[1] + row[2] * row[2], row[2], row[1]))
    response, integer_dx, integer_dy = candidates[0]
    separated = [
        score
        for score, other_dx, other_dy in candidates[1:]
        if abs(other_dx - integer_dx) > 1 or abs(other_dy - integer_dy) > 1
    ]
    runner_up = max(separated, default=0.0)
    peak_margin = (response - runner_up) / response if response > 0 else 0.0

    refined_dx, refined_dy = (
        _refined_peak(candidates, response, integer_dx, integer_dy)
        if subpixel_refinement
        else (float(integer_dx), float(integer_dy))
    )

    first_slice, second_slice = _overlap_slices(a.shape, integer_dx, integer_dy)
    common = support[first_slice] & support[second_slice]
    first_values = first_texture[first_slice][common]
    second_values = second_texture[second_slice][common]
    denominator = float(np.std(first_values) * np.std(second_values))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        raise ValueError("Aligned mutual-information result lacks texture")
    spectral_correlation = float(
        np.clip(
            np.mean(
                (first_values - float(np.mean(first_values)))
                * (second_values - float(np.mean(second_values)))
            )
            / denominator,
            -1,
            1,
        )
    )
    if not math.isfinite(spectral_correlation):
        raise ValueError("Non-finite aligned spectral correlation")
    return refined_dx, refined_dy, response, peak_margin, spectral_correlation


def _portable_nmi_candidates(
    first: np.ndarray,
    second: np.ndarray,
    support: np.ndarray,
    bins: int,
    maximum_x: int,
    maximum_y: int,
    required_fraction: float,
) -> list[tuple[float, int, int]]:
    """Retain the same exhaustive calculation on hosts without the extension."""
    required = required_fraction * np.count_nonzero(support)
    count_log_count = _count_log_count_table(first.size)
    candidates: list[tuple[float, int, int]] = []
    for dy in range(-maximum_y, maximum_y + 1):
        for dx in range(-maximum_x, maximum_x + 1):
            first_slice, second_slice = _overlap_slices(first.shape, dx, dy)
            common = cv2.bitwise_and(support[first_slice], support[second_slice])
            count = cv2.countNonZero(common)
            if count < required:
                continue
            joint = cv2.calcHist(
                [first[first_slice], second[second_slice]],
                [0, 1],
                common,
                [bins, bins],
                [0, bins, 0, bins],
            ).astype(np.int64)
            score = _normalised_mutual_information_from_joint(joint, count, count_log_count)
            candidates.append((score, dx, dy))
    return candidates


def texture_candidate_mask(
    white_plane: np.ndarray,
    valid_mask: np.ndarray,
    settings: RGFocusCoreSettings,
) -> tuple[np.ndarray, TextureMaskMetrics]:
    """Calculate the legacy WHITE texture candidate mask within explicit geometry.

    This numerical percentile mask is not, by itself, proof that tissue exists.
    The caller must apply the operational no-tissue and geometry policy.
    """
    gray = _finite_plane(white_plane, "WHITE plane")
    valid = np.asarray(valid_mask)
    if valid.dtype != np.bool_ or valid.shape != gray.shape or not np.any(valid):
        raise ValueError("Valid mask must be a non-empty boolean plane matching WHITE")
    blur = cv2.GaussianBlur(gray, (0, 0), settings.texture_background_sigma_px)
    highpass = np.abs(gray - blur)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    score = cv2.GaussianBlur(
        highpass + settings.texture_gradient_weight * np.sqrt(sx * sx + sy * sy),
        (0, 0),
        settings.texture_score_sigma_px,
    )
    threshold = float(np.percentile(score[valid], settings.texture_percentile))
    candidates = ((score > threshold) & valid).astype(np.uint8) * 255
    kernel = np.ones(
        (
            settings.texture_morphology_kernel_px,
            settings.texture_morphology_kernel_px,
        ),
        dtype=np.uint8,
    )
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, kernel)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, kernel) > 0
    candidates &= valid
    return candidates, TextureMaskMetrics(
        threshold=threshold,
        coverage_total=float(np.mean(candidates)),
        coverage_inside_valid=float(np.mean(candidates[valid])),
    )


def candidate_boxes(shape: tuple[int, int], settings: RGFocusCoreSettings) -> list[PatchBox]:
    """Build the deterministic overlapping patch grid used by AF-010."""
    if len(shape) != 2 or min(shape) < settings.patch_size_px:
        raise ValueError("Plane is smaller than one configured patch")
    height, width = shape
    size = settings.patch_size_px
    stride = settings.patch_stride_px
    return [
        PatchBox(patch_id=f"y{y}_x{x}", x=x, y=y, w=size, h=size)
        for y in range(0, height - size + 1, stride)
        for x in range(0, width - size + 1, stride)
    ]


def _checked_boxes(
    boxes: Sequence[PatchBox],
    shape: tuple[int, int],
    settings: RGFocusCoreSettings,
) -> list[PatchBox]:
    """Require reusable boxes to retain the configured common-plane geometry."""
    height, width = shape
    checked = list(boxes)
    for box in checked:
        if box.w != settings.patch_size_px or box.h != settings.patch_size_px:
            raise ValueError("Fixed box size differs from configured patch size")
        if box.x + box.w > width or box.y + box.h > height:
            raise ValueError("Fixed box lies outside the common plane")
    return checked


def _patch_evaluation(
    values: dict[str, object], status: PatchStatus, reason: str
) -> PatchEvaluation:
    """Validate a progressively populated diagnostic without typed unpacking."""
    return PatchEvaluation.model_validate({**values, "status": status, "reason": reason})


def _required_metric(value: float | None, name: str) -> float:
    """Narrow one metric guaranteed by an accepted patch diagnostic."""
    if value is None:
        raise RuntimeError(f"Accepted patch is missing {name}")
    return value


def evaluate_patches(
    red_plane: np.ndarray,
    green_plane: np.ndarray,
    mask: np.ndarray,
    settings: RGFocusCoreSettings,
    boxes: Sequence[PatchBox] | None = None,
    *,
    red_source_jpeg8: np.ndarray,
    green_source_jpeg8: np.ndarray,
    source_support: np.ndarray,
    subpixel_refinement: bool = False,
) -> list[PatchEvaluation]:
    """Evaluate corrected support while measuring clipping on tissue support."""
    red = np.asarray(red_plane)
    green = np.asarray(green_plane)
    source_r = np.asarray(red_source_jpeg8)
    source_g = np.asarray(green_source_jpeg8)
    candidate_mask = np.asarray(mask)
    source_candidate_mask = np.asarray(source_support)
    if (
        red.ndim != 2
        or green.shape != red.shape
        or source_r.shape != red.shape
        or source_g.shape != red.shape
        or source_r.dtype != np.uint8
        or source_g.dtype != np.uint8
        or candidate_mask.shape != red.shape
        or candidate_mask.dtype != np.bool_
        or source_candidate_mask.shape != red.shape
        or source_candidate_mask.dtype != np.bool_
    ):
        raise ValueError("Corrected/source R/G planes and both support masks must share geometry")
    candidates = (
        candidate_boxes(red.shape, settings)
        if boxes is None
        else _checked_boxes(boxes, red.shape, settings)
    )
    rows: list[PatchEvaluation | None] = []
    pending: list[
        tuple[
            int,
            dict[str, object],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ]
    ] = []
    for box in candidates:
        y_slice = slice(box.y, box.y + box.h)
        x_slice = slice(box.x, box.x + box.w)
        patch_r = red[y_slice, x_slice]
        patch_g = green[y_slice, x_slice]
        patch_source_r = source_r[y_slice, x_slice]
        patch_source_g = source_g[y_slice, x_slice]
        patch_mask = candidate_mask[y_slice, x_slice]
        patch_source_support = source_candidate_mask[y_slice, x_slice]
        coverage = float(np.mean(patch_mask))
        base = {
            "patch_id": box.patch_id,
            "x": box.x,
            "y": box.y,
            "w": box.w,
            "h": box.h,
            "coverage": coverage,
        }
        if not np.any(patch_source_support):
            rows.append(
                _patch_evaluation(
                    base,
                    "insufficient_support",
                    "Tissue source support is empty inside the fixed window",
                )
            )
            continue
        saturation_r = float(
            np.mean(patch_source_r[patch_source_support] >= settings.jpeg8_saturation_level)
        )
        saturation_g = float(
            np.mean(patch_source_g[patch_source_support] >= settings.jpeg8_saturation_level)
        )
        source_metrics = {
            **base,
            "saturation_r": saturation_r,
            "saturation_g": saturation_g,
        }
        if (
            saturation_r > settings.maximum_patch_saturation_fraction
            or saturation_g > settings.maximum_patch_saturation_fraction
        ):
            rows.append(
                _patch_evaluation(
                    source_metrics,
                    "saturated",
                    "Source JPEG8 R/G clipping exceeds the configured maximum",
                )
            )
            continue
        if coverage < settings.minimum_tissue_coverage:
            rows.append(
                _patch_evaluation(
                    source_metrics,
                    "insufficient_support",
                    "Tissue/valid support is below the configured fraction",
                )
            )
            continue
        if not np.isfinite(patch_r[patch_mask]).all() or not np.isfinite(patch_g[patch_mask]).all():
            rows.append(
                _patch_evaluation(
                    base,
                    "low_signal",
                    "Accepted support contains non-finite R/G samples",
                )
            )
            continue
        values_r = patch_r[patch_mask].astype(np.float64)
        values_g = patch_g[patch_mask].astype(np.float64)
        if np.min(values_r) < 0 or np.min(values_g) < 0:
            rows.append(
                _patch_evaluation(
                    base,
                    "low_signal",
                    "Accepted support contains negative corrected samples",
                )
            )
            continue
        median_r = float(np.median(values_r))
        median_g = float(np.median(values_g))
        std_r = float(np.std(values_r))
        std_g = float(np.std(values_g))
        metrics = {
            **source_metrics,
            "median_r": median_r,
            "median_g": median_g,
            "std_r": std_r,
            "std_g": std_g,
        }
        if min(median_r, median_g) < settings.minimum_patch_signal:
            rows.append(
                _patch_evaluation(
                    metrics,
                    "low_signal",
                    "R/G median signal is below the configured minimum",
                )
            )
            continue
        if min(std_r, std_g) < settings.minimum_patch_std:
            rows.append(
                _patch_evaluation(
                    metrics,
                    "low_texture",
                    "R/G texture is below the configured minimum",
                )
            )
            continue
        pending.append((len(rows), metrics, patch_r, patch_g, patch_mask))
        rows.append(None)

    def correlate(
        item: tuple[
            int,
            dict[str, object],
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ],
    ) -> tuple[int, PatchEvaluation]:
        index, metrics, patch_r, patch_g, patch_mask = item
        try:
            dx, dy, response, peak_margin, spectral = mutual_information_shift_masked(
                patch_r,
                patch_g,
                patch_mask,
                settings,
                subpixel_refinement=subpixel_refinement,
            )
        except ValueError as exc:
            return index, _patch_evaluation(metrics, "correlation_failed", str(exc))
        correlation = {
            **metrics,
            "dx": dx,
            "dy": dy,
            "response": response,
            "peak_margin": peak_margin,
            "spectral_correlation": spectral,
        }
        if response < settings.minimum_patch_response:
            return index, (
                _patch_evaluation(
                    correlation,
                    "correlation_failed",
                    "Mutual-information response is below the configured minimum",
                )
            )
        if peak_margin < settings.minimum_patch_peak_margin:
            return index, (
                _patch_evaluation(
                    correlation,
                    "correlation_failed",
                    "Mutual-information peak is not distinct from competing shifts",
                )
            )
        if (
            abs(dx) >= settings.maximum_absolute_shift_px
            or abs(dy) >= settings.maximum_absolute_orthogonal_shift_px
        ):
            return index, (
                _patch_evaluation(
                    correlation,
                    "shift_out_of_range",
                    "R/G optimum reached the configured search boundary",
                )
            )
        if spectral < settings.minimum_patch_spectral_correlation:
            return index, (
                _patch_evaluation(
                    correlation,
                    "spectral_mismatch",
                    "Aligned R/G texture lacks configured spectral agreement",
                )
            )
        return index, (
            _patch_evaluation(
                correlation,
                "accepted",
                "Patch passed support, signal, saturation, and correlation QC",
            )
        )

    if pending:
        workers = min(PATCH_EVALUATION_WORKERS, len(pending))
        if workers == 1:
            completed = map(correlate, pending)
            for index, row in completed:
                rows[index] = row
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rg-patch") as executor:
                for index, row in executor.map(correlate, pending):
                    rows[index] = row
    if any(row is None for row in rows):
        raise RuntimeError("R/G patch evaluation did not complete")
    return [cast(PatchEvaluation, row) for row in rows]


def measure_patches(
    red_plane: np.ndarray,
    green_plane: np.ndarray,
    mask: np.ndarray,
    settings: RGFocusCoreSettings,
    boxes: Sequence[PatchBox] | None = None,
    *,
    red_source_jpeg8: np.ndarray,
    green_source_jpeg8: np.ndarray,
    source_support: np.ndarray,
) -> list[PatchMeasurement]:
    """Return accepted rows while preserving the original pure-core API."""
    evaluations = evaluate_patches(
        red_plane,
        green_plane,
        mask,
        settings,
        boxes,
        red_source_jpeg8=red_source_jpeg8,
        green_source_jpeg8=green_source_jpeg8,
        source_support=source_support,
    )
    return [
        PatchMeasurement(
            patch_id=row.patch_id,
            x=row.x,
            y=row.y,
            w=row.w,
            h=row.h,
            dx=_required_metric(row.dx, "dx"),
            dy=_required_metric(row.dy, "dy"),
            response=_required_metric(row.response, "response"),
            peak_margin=_required_metric(row.peak_margin, "peak_margin"),
            coverage=row.coverage,
            median_r=_required_metric(row.median_r, "median_r"),
            median_g=_required_metric(row.median_g, "median_g"),
            std_r=_required_metric(row.std_r, "std_r"),
            std_g=_required_metric(row.std_g, "std_g"),
            saturation_r=_required_metric(row.saturation_r, "saturation_r"),
            saturation_g=_required_metric(row.saturation_g, "saturation_g"),
            spectral_correlation=_required_metric(row.spectral_correlation, "spectral_correlation"),
        )
        for row in evaluations
        if row.status == "accepted"
    ]


def select_reference_boxes(
    measurements: Sequence[PatchMeasurement],
    settings: RGFocusCoreSettings,
) -> list[PatchBox]:
    """Select a stable bounded set of highest-response reference coordinates."""
    selected = sorted(measurements, key=lambda item: item.response, reverse=True)[
        : settings.maximum_patch_count
    ]
    return [
        PatchBox(
            patch_id=f"ref_{index:03d}_{row.patch_id}",
            x=row.x,
            y=row.y,
            w=row.w,
            h=row.h,
        )
        for index, row in enumerate(selected)
    ]


def summarise_shift(measurements: Sequence[PatchMeasurement]) -> ShiftSummary:
    """Return median displacement, response, and median absolute deviations."""
    if not measurements:
        raise ValueError("Cannot summarise an empty patch set")
    dx_values = np.asarray([row.dx for row in measurements], dtype=np.float64)
    dy_values = np.asarray([row.dy for row in measurements], dtype=np.float64)
    responses = np.asarray([row.response for row in measurements], dtype=np.float64)
    dx = float(np.median(dx_values))
    dy = float(np.median(dy_values))
    return ShiftSummary(
        dx=dx,
        dy=dy,
        response=float(np.median(responses)),
        patch_count=len(measurements),
        dx_mad=float(np.median(np.abs(dx_values - dx))),
        dy_mad=float(np.median(np.abs(dy_values - dy))),
    )
