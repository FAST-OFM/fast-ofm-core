"""Fresh WHITE tissue fields bound to the current OpenFlexure geometry.

This module is the OF-032 integration layer around rg_focus_core. Its mask is
measurement support inside visible tissue, not an exhaustive semantic tissue
outline. It performs no capture, illumination, file, UI, stage, or focus action.
A caller must supply a fresh WHITE frame and explicit frame references;
incompatible or stale references fail closed and never fall back to the whole
image.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Self

import cv2
import numpy as np
from pydantic import Field, ValidationError, model_validator

from fast_ofm_core.contracts import CalibrationStage, StrictModel

from .rg_focus_core import (
    PatchBox,
    RGFocusCoreSettings,
    TextureMaskMetrics,
    candidate_boxes,
    texture_candidate_mask,
)

TissueStatus = Literal["ready", "no_tissue", "insufficient_tissue", "low_signal", "saturated"]
IlluminationMode = Literal["white", "red", "green"]


class TissueFieldError(ValueError):
    """A fail-closed tissue-field contract error with a machine-readable code."""

    def __init__(self, code: Literal["mask_invalid", "mask_stale"], detail: str) -> None:
        """Store the stable error code alongside the human-readable detail."""
        super().__init__(detail)
        self.code = code


class FrameGeometry(StrictModel):
    """Exact processed-JPEG RGB8 geometry shared by WHITE and R/G planes."""

    measurement_space: Literal["processed-jpeg-rgb8"]
    sensor: Literal["imx477"]
    sensor_resolution: tuple[int, int]
    sensor_crop: tuple[int, int, int, int]
    source_image_size: tuple[int, int]
    processing_roi: tuple[int, int, int, int]
    image_size: tuple[int, int]
    plane_size: tuple[int, int]
    white_size: tuple[int, int]
    bit_depth: Literal[8]
    white_level: Literal[255]
    channel_order: tuple[Literal["R"], Literal["G"], Literal["B"]]
    array_axes: Literal["height,width,channel"]
    pixel_to_sensor: tuple[tuple[float, float, float], tuple[float, float, float]]
    white_to_sensor: tuple[tuple[float, float, float], tuple[float, float, float]]
    common_plane_to_sensor: tuple[tuple[float, float, float], tuple[float, float, float]]

    @model_validator(mode="after")
    def exact_jpeg_geometry(self) -> Self:
        """Reject anything except the processing-binding JPEG common grid."""
        sensor_width, sensor_height = self.sensor_resolution
        crop_x, crop_y, crop_width, crop_height = self.sensor_crop
        source_width, source_height = self.source_image_size
        roi_x, roi_y, roi_width, roi_height = self.processing_roi
        if (
            self.sensor_resolution != (4056, 3040)
            or min(crop_x, crop_y) < 0
            or min(crop_width, crop_height, source_width, source_height) <= 0
            or crop_x + crop_width > sensor_width
            or crop_y + crop_height > sensor_height
            or min(roi_x, roi_y) < 0
            or min(roi_width, roi_height) < 16
            or roi_x + roi_width > source_width
            or roi_y + roi_height > source_height
            or crop_width * source_height != crop_height * source_width
            or self.image_size != (roi_width, roi_height)
            or self.plane_size != self.image_size
            or self.white_size != self.image_size
        ):
            raise ValueError("Invalid JPEG sensor crop, source image, or processing ROI")
        x_scale = crop_width / source_width
        y_scale = crop_height / source_height
        expected = (
            (x_scale, 0.0, crop_x + x_scale * (roi_x + 0.5) - 0.5),
            (0.0, y_scale, crop_y + y_scale * (roi_y + 0.5) - 0.5),
        )
        affines = (
            self.pixel_to_sensor,
            self.white_to_sensor,
            self.common_plane_to_sensor,
        )
        if any(not np.isfinite(value).all() for value in map(np.asarray, affines)):
            raise ValueError("JPEG affine contains a non-finite value")
        if any(value != affines[0] for value in affines[1:]) or affines[0] != expected:
            raise ValueError("JPEG common-grid affine does not match the processing ROI")
        return self


class FrameReference(StrictModel):
    """Identity and capture order for one frame at one settled XY field."""

    frame_id: str = Field(min_length=1)
    field_id: str = Field(min_length=1)
    sensor_timestamp_ns: int = Field(gt=0)
    mode: IlluminationMode
    geometry_id: str = Field(min_length=64, max_length=64)


class TissueFieldSettings(StrictModel):
    """Operational WHITE gates; all values are visible parameters, not fallbacks."""

    edge_margin_px: int = Field(default=48, ge=2)
    minimum_white_median: float = Field(default=16.0, gt=0)
    white_saturation_level: float = Field(default=250.0, gt=0)
    maximum_white_saturation_fraction: float = Field(default=0.02, ge=0, le=1)
    coherent_texture_blur_sigma_px: float = Field(default=1.5, gt=0)
    coherent_texture_background_sigma_px: float = Field(default=8.0, gt=0)
    minimum_coherent_texture_fraction: float = Field(default=0.01, gt=0)
    minimum_component_pixels: int = Field(default=2048, ge=1)
    minimum_tissue_fraction: float = Field(default=0.03, gt=0, le=1)

    @model_validator(mode="after")
    def coherent_scales(self) -> Self:
        """Require the background scale to exceed the noise-suppression scale."""
        if self.coherent_texture_background_sigma_px <= self.coherent_texture_blur_sigma_px:
            raise ValueError("Texture background scale must exceed foreground blur")
        return self


class TissueFieldMetrics(StrictModel):
    """Serializable evidence for accepting or refusing the current WHITE field."""

    status: TissueStatus
    reason: str
    white_median: float
    saturation_fraction: float = Field(ge=0, le=1)
    coherent_texture_p95_fraction: float = Field(ge=0)
    candidate_coverage: float = Field(
        ge=0,
        le=1,
        description="Fraction of valid pixels usable as texture support, not tissue area",
    )
    component_count: int = Field(ge=0)
    largest_component_pixels: int = Field(ge=0)
    box_count: int = Field(ge=0)
    core_mask: TextureMaskMetrics | None = None


@dataclass(frozen=True)
class TissueField:
    """Immutable-by-contract arrays and provenance for one current XY field."""

    reference: FrameReference
    geometry: FrameGeometry
    mask: np.ndarray
    white_common: np.ndarray
    boxes: tuple[PatchBox, ...]
    metrics: TissueFieldMetrics
    overlay: np.ndarray


def _fixed_sequence(item: object, length: int) -> list[object] | tuple[object, ...]:
    """Accept only an explicit JSON/Python sequence with the required length."""
    if not isinstance(item, (list, tuple)) or len(item) != length:
        raise ValueError(f"Expected a sequence of length {length}")
    return item


def _integer(item: object) -> int:
    """Accept an integer without bool or string coercion."""
    if isinstance(item, bool) or not isinstance(item, int):
        raise ValueError("Expected an integer")
    return item


def _real(item: object) -> float:
    """Accept a finite JSON number without bool or string coercion."""
    if isinstance(item, bool) or not isinstance(item, (int, float)):
        raise ValueError("Expected a finite number")
    result = float(item)
    if not math.isfinite(result):
        raise ValueError("Expected a finite number")
    return result


def _integers(item: object, length: int) -> tuple[int, ...]:
    """Return one validated fixed-length integer tuple."""
    return tuple(_integer(part) for part in _fixed_sequence(item, length))


def _floats(item: object, length: int) -> tuple[float, ...]:
    """Return one validated fixed-length finite-number tuple."""
    return tuple(_real(part) for part in _fixed_sequence(item, length))


def parse_frame_geometry(value: Mapping[str, object]) -> FrameGeometry:
    """Validate exactly the geometry produced by the JPEG processing binding."""
    try:
        required = {
            "measurement_space",
            "sensor",
            "sensor_resolution",
            "sensor_crop",
            "source_image_size",
            "processing_roi",
            "image_size",
            "plane_size",
            "white_size",
            "bit_depth",
            "white_level",
            "channel_order",
            "array_axes",
            "pixel_to_sensor",
            "white_to_sensor",
            "common_plane_to_sensor",
        }
        if set(value) != required:
            raise ValueError("JPEG geometry fields are missing or unexpected")
        transforms = {
            name: tuple(_floats(row, 3) for row in _fixed_sequence(value[name], 2))
            for name in ("pixel_to_sensor", "white_to_sensor", "common_plane_to_sensor")
        }
        normalized = {
            "measurement_space": value["measurement_space"],
            "sensor": value["sensor"],
            "sensor_resolution": _integers(value["sensor_resolution"], 2),
            "sensor_crop": _integers(value["sensor_crop"], 4),
            "source_image_size": _integers(value["source_image_size"], 2),
            "processing_roi": _integers(value["processing_roi"], 4),
            "image_size": _integers(value["image_size"], 2),
            "plane_size": _integers(value["plane_size"], 2),
            "white_size": _integers(value["white_size"], 2),
            "bit_depth": _integer(value["bit_depth"]),
            "white_level": _integer(value["white_level"]),
            "channel_order": tuple(_fixed_sequence(value["channel_order"], 3)),
            "array_axes": value["array_axes"],
            **transforms,
        }
        return FrameGeometry.model_validate(normalized)
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        raise TissueFieldError("mask_invalid", f"Invalid frame geometry: {exc}") from exc


def geometry_fingerprint(geometry: FrameGeometry) -> str:
    """Return a stable ID for the complete normalized JPEG geometry/domain."""
    payload = json.dumps(
        geometry.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _white_gray(frame: np.ndarray, geometry: FrameGeometry) -> np.ndarray:
    """Convert a supplied WHITE frame to finite grayscale without guessing its size."""
    value = np.asarray(frame)
    expected_shape = (geometry.white_size[1], geometry.white_size[0])
    if value.ndim == 3 and value.shape[2] == 3:
        value = np.mean(value.astype(np.float64), axis=2)
    if (
        value.ndim != 2
        or value.shape != expected_shape
        or not np.isfinite(value).all()
        or np.min(value) < 0
    ):
        raise TissueFieldError("mask_invalid", "WHITE frame does not match its declared geometry")
    return value.astype(np.float64)


def _white_to_common(
    gray: np.ndarray,
    geometry: FrameGeometry,
    edge_margin_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the identical JPEG grid with only edge support removed."""
    width, height = geometry.plane_size
    if edge_margin_px * 2 >= min(width, height):
        raise TissueFieldError("mask_invalid", "Edge margin removes the common plane")
    valid = np.ones((height, width), dtype=bool)
    margin = edge_margin_px
    valid[:margin] = False
    valid[-margin:] = False
    valid[:, :margin] = False
    valid[:, -margin:] = False
    return np.array(gray, dtype=np.float64, copy=True), valid


def _coherent_texture(
    gray: np.ndarray,
    valid: np.ndarray,
    settings: TissueFieldSettings,
) -> tuple[float, np.ndarray]:
    """Measure coherent structure and return its explicit valid-pixel support."""
    foreground = cv2.GaussianBlur(gray, (0, 0), settings.coherent_texture_blur_sigma_px)
    background = cv2.GaussianBlur(foreground, (0, 0), settings.coherent_texture_background_sigma_px)
    median = float(np.median(gray[valid]))
    magnitude = np.abs(foreground - background)
    denominator = max(median, 1e-9)
    fraction = float(np.percentile(magnitude[valid], 95)) / denominator
    support = magnitude >= settings.minimum_coherent_texture_fraction * denominator
    support &= valid
    return fraction, support


def _large_components(mask: np.ndarray, minimum_pixels: int) -> tuple[np.ndarray, int, int]:
    """Remove isolated percentile noise and report retained coherent components."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    sizes = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.asarray([], dtype=int)
    retained_ids = [index + 1 for index, size in enumerate(sizes) if size >= minimum_pixels]
    retained = np.isin(labels, retained_ids)
    return retained, len(retained_ids), int(max(sizes, default=0))


def render_tissue_overlay(
    white_common: np.ndarray,
    valid: np.ndarray,
    mask: np.ndarray,
    boxes: tuple[PatchBox, ...],
) -> np.ndarray:
    """Render texture support and chosen boxes, not a semantic tissue outline."""
    values = white_common[valid]
    low, high = np.percentile(values, [1, 99])
    scaled = np.zeros(white_common.shape, dtype=np.uint8)
    if high > low:
        scaled = np.clip((white_common - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    overlay = np.repeat(scaled[:, :, None], 3, axis=2)
    overlay[mask, 0] //= 3
    overlay[mask, 1] = np.maximum(overlay[mask, 1], 180)
    overlay[mask, 2] //= 3
    for box in boxes:
        cv2.rectangle(
            overlay,
            (box.x, box.y),
            (box.x + box.w - 1, box.y + box.h - 1),
            (255, 0, 255),
            1,
        )
    overlay[~valid] //= 4
    return overlay


def _freeze(value: np.ndarray) -> np.ndarray:
    """Copy one diagnostic array and prevent accidental in-place mutation."""
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def prepare_tissue_field(
    white_frame: np.ndarray,
    reference: FrameReference,
    geometry_value: Mapping[str, object],
    settings: TissueFieldSettings,
    core_settings: RGFocusCoreSettings,
) -> TissueField:
    """Prepare field-bound measurement support and fixed common-plane boxes."""
    geometry = parse_frame_geometry(geometry_value)
    geometry_id = geometry_fingerprint(geometry)
    if reference.mode != "white" or reference.geometry_id != geometry_id:
        raise TissueFieldError(
            "mask_invalid", "WHITE reference mode or geometry binding is invalid"
        )
    if settings.edge_margin_px < math.ceil(core_settings.maximum_absolute_shift_px) + 2:
        raise TissueFieldError(
            "mask_invalid", "Edge margin is smaller than the configured shift support"
        )
    gray = _white_gray(white_frame, geometry)
    common, valid = _white_to_common(gray, geometry, settings.edge_margin_px)
    white_median = float(np.median(common[valid]))
    saturation_fraction = float(np.mean(common[valid] >= settings.white_saturation_level))
    texture_fraction, coherent_support = _coherent_texture(common, valid, settings)

    status: TissueStatus
    reason: str
    mask = np.zeros(common.shape, dtype=bool)
    boxes: tuple[PatchBox, ...] = ()
    core_metrics: TextureMaskMetrics | None = None
    component_count = 0
    largest_component = 0

    if white_median < settings.minimum_white_median:
        status, reason = "low_signal", "WHITE signal is below the configured minimum"
    elif saturation_fraction > settings.maximum_white_saturation_fraction:
        status, reason = "saturated", "WHITE saturation exceeds the configured maximum"
    elif texture_fraction < settings.minimum_coherent_texture_fraction:
        status, reason = "no_tissue", "No coherent WHITE structure above the noise gate"
    else:
        mask, core_metrics = texture_candidate_mask(common, valid, core_settings)
        mask &= coherent_support
        mask, component_count, largest_component = _large_components(
            mask, settings.minimum_component_pixels
        )
        mask &= valid
        coverage = float(np.mean(mask[valid]))
        eligible = [
            box
            for box in candidate_boxes((len(mask), len(mask[0])), core_settings)
            if float(np.mean(mask[box.y : box.y + box.h, box.x : box.x + box.w]))
            >= core_settings.minimum_tissue_coverage
        ]
        eligible.sort(
            key=lambda box: (
                -float(np.mean(mask[box.y : box.y + box.h, box.x : box.x + box.w])),
                box.y,
                box.x,
            )
        )
        boxes = tuple(eligible[: core_settings.maximum_patch_count])
        if (
            coverage < settings.minimum_tissue_fraction
            or len(boxes) < core_settings.minimum_patch_count
        ):
            status = "insufficient_tissue"
            reason = "Coherent tissue support is too small for fixed common patches"
            boxes = ()
        else:
            status, reason = "ready", "Fresh WHITE tissue field is ready"

    coverage = float(np.mean(mask[valid]))
    metrics = TissueFieldMetrics(
        status=status,
        reason=reason,
        white_median=white_median,
        saturation_fraction=saturation_fraction,
        coherent_texture_p95_fraction=texture_fraction,
        candidate_coverage=coverage,
        component_count=component_count,
        largest_component_pixels=largest_component,
        box_count=len(boxes),
        core_mask=core_metrics,
    )
    overlay = render_tissue_overlay(common, valid, mask, boxes)
    return TissueField(
        reference=reference,
        geometry=geometry,
        mask=_freeze(mask),
        white_common=_freeze(common),
        boxes=boxes,
        metrics=metrics,
        overlay=_freeze(overlay),
    )


def require_tissue_field_for_pair(
    field: TissueField | None,
    red: FrameReference,
    green: FrameReference,
    geometry_value: Mapping[str, object],
) -> None:
    """Require a ready WHITE mask from this field before the fresh R/G pair."""
    if field is None:
        raise TissueFieldError("mask_invalid", "R/G measurement has no WHITE mask")
    geometry = parse_frame_geometry(geometry_value)
    current_geometry_id = geometry_fingerprint(geometry)
    references = (field.reference, red, green)
    if field.metrics.status != "ready":
        raise TissueFieldError(
            "mask_invalid", f"WHITE tissue field is not ready: {field.metrics.status}"
        )
    if red.mode != "red" or green.mode != "green":
        raise TissueFieldError("mask_invalid", "R/G frame modes are invalid")
    if (
        any(item.geometry_id != current_geometry_id for item in references)
        or field.geometry != geometry
        or len({item.field_id for item in references}) != 1
        or len({item.frame_id for item in references}) != 3
    ):
        raise TissueFieldError("mask_stale", "Mask, field, frame, or JPEG geometry binding changed")
    if min(red.sensor_timestamp_ns, green.sensor_timestamp_ns) <= (
        field.reference.sensor_timestamp_ns
    ):
        raise TissueFieldError("mask_stale", "R/G pair is not newer than the field WHITE frame")


def tissue_field_manifest_stage() -> CalibrationStage:
    """Return the reusable, real OF-032 stage for a future LED calibration manifest."""
    return CalibrationStage(
        id="tissue_field",
        name="Fresh WHITE tissue field",
        description="Bind a fresh WHITE tissue mask and fixed common R/G patches to this XY field.",
        inputs=["white_frame", "frame_reference", "jpeg_geometry", "parameters"],
        outputs=["tissue_mask", "fixed_boxes", "mask_metrics", "diagnostic_overlay"],
        action="prepare_tissue_field",
        success_criterion="Ready tissue support with matching processed-JPEG provenance",
        timeout_setting="capture_timeout_s",
        cancellation="Discard the unsaved field reference; never use a prior mask or whole-frame fallback.",
        hardware_required=True,
    )
