"""Empirical processed-JPEG R/G flat-field; no camera, motion or AI dependencies."""

from __future__ import annotations

import copy
from typing import Literal, Self

import cv2
import numpy as np
from pydantic import Field, StrictInt, field_validator, model_validator

from fast_ofm_core.contracts import StrictModel

Mode = Literal["red", "green"]
JPEG_DOMAIN = "processed-jpeg-rgb8"


class FlatFieldSettings(StrictModel):
    """One saved parameter set for acquisition and independent holdout checks."""

    measurement_domain: Literal["legacy-raw", "processed-jpeg-rgb8"] = "legacy-raw"
    processing_roi: tuple[StrictInt, StrictInt, StrictInt, StrictInt] | None = None

    dark_frames: int = Field(default=4, ge=2, le=16)
    average_frames: int = Field(default=4, ge=2, le=64, description="Phase-matched fit cycles")
    validation_frames: int = Field(
        default=3, ge=2, le=16, description="Independent phase-matched cycles"
    )
    frame_timeout_s: float = Field(default=5, gt=0, le=10)
    timeout_s: float = Field(default=300, gt=0, le=600)
    smoothing_sigma_px: float = Field(default=1, ge=1, le=64)
    minimum_signal_dn: float = Field(default=64, ge=8, le=1024)
    maximum_gain: float = Field(default=4, ge=1, le=10)
    maximum_mask_fraction: float = Field(default=0.02, ge=0, le=0.1)
    maximum_saturation_fraction: float = Field(default=0.001, ge=0, le=0.05)
    saturation_level_fraction: float = Field(default=0.98, ge=0.9, le=1)
    maximum_dark_signal_dn: float = Field(default=256, ge=1, le=1024)
    maximum_field_texture: float = Field(default=0.04, gt=0, le=0.2)
    maximum_residual_cv: float = Field(default=0.05, gt=0, le=0.2)
    maximum_drift_fraction: float = Field(default=0.05, gt=0, le=0.2)
    maximum_spatial_residual_fraction: float = Field(default=0.05, gt=0, le=0.2)
    optics_id: str = Field(default="current", min_length=1, max_length=100)
    illumination_id: str = Field(default="current", min_length=1, max_length=100)

    @field_validator("processing_roi", mode="before")
    @classmethod
    def json_roi(cls, value: object) -> object:
        """Accept the JSON array container while retaining strict integer coordinates."""
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_domain(self) -> Self:
        """Load historical DN values safely but reject invalid explicit JPEG settings."""
        if self.measurement_domain == JPEG_DOMAIN and (
            self.minimum_signal_dn > 255 or self.maximum_dark_signal_dn > 255
        ):
            raise ValueError("JPEG thresholds are absolute 8-bit DN, at most 255")
        if self.processing_roi is not None:
            x, y, width, height = self.processing_roi
            if min(x, y) < 0 or min(width, height) < 16:
                raise ValueError(
                    "Processing ROI needs nonnegative origin and at least 16x16 pixels"
                )
        return self

    def require_jpeg(self) -> None:
        """Require a deliberate JPEG preset/settings save before acquisition or fitting."""
        if self.measurement_domain != JPEG_DOMAIN:
            raise ValueError("Save the explicit JPEG8 preset before R/G acquisition")
        if self.processing_roi is None:
            raise ValueError("Select and save an explicit common R/G processing ROI")


def processing_binding(configuration: dict, settings: FlatFieldSettings) -> dict:
    """Bind one common JPEG ROI and translate pixel centres without changing sensor FOV."""
    settings.require_jpeg()
    if configuration.get("measurement_space") != JPEG_DOMAIN:
        raise ValueError("Camera does not provide the processed JPEG measurement domain")
    result = copy.deepcopy(configuration)
    geometry = result["geometry"]
    width, height = geometry["image_size"]
    roi = settings.processing_roi
    if roi is None:
        raise ValueError("An explicit processing ROI is required")
    x, y, w, h = roi
    if x + w > width or y + h > height:
        raise ValueError("Processing ROI exceeds the decoded JPEG dimensions")
    result["source_geometry"] = copy.deepcopy(geometry)
    result["processing_roi"] = list(roi)
    geometry.update(
        source_image_size=[width, height],
        processing_roi=list(roi),
        image_size=[w, h],
        plane_size=[w, h],
        white_size=[w, h],
    )
    for key in ("pixel_to_sensor", "white_to_sensor", "common_plane_to_sensor"):
        affine = np.asarray(geometry[key], dtype=float)
        if affine.shape != (2, 3) or not np.isfinite(affine).all():
            raise ValueError("Invalid JPEG pixel-centre geometry")
        affine[:, 2] += affine[:, :2] @ np.array([x, y])
        geometry[key] = affine.tolist()
    return result


def check_processing(processing: dict, reference: dict | None = None) -> None:
    """Validate actual processing against one fixed reference (relative tolerance 1e-5)."""
    for key, shape in (
        ("digital_gain", ()),
        ("colour_gains", (2,)),
        ("colour_correction_matrix", (9,)),
    ):
        values = np.asarray(processing.get(key), dtype=float)
        if (
            values.shape != shape
            or not np.isfinite(values).all()
            or (key != "colour_correction_matrix" and (values <= 0).any())
        ):
            raise ValueError("Invalid actual JPEG processing metadata")
        if reference is not None:
            expected = np.asarray(reference.get(key), dtype=float)
            if (
                expected.shape != shape
                or not np.isfinite(expected).all()
                or not np.allclose(values, expected, rtol=1e-5, atol=0)
            ):
                raise ValueError("Digital gain/white balance/colour processing changed")


def channel_indices(mode: Mode) -> list[int]:
    """Select decoded RGB red or green, on the same pixel grid."""
    if mode not in ("red", "green"):
        raise ValueError("Flat-field mode must be red or green")
    return [0] if mode == "red" else [1]


def checked_array(array: np.ndarray) -> np.ndarray:
    """Require one finite selected plane in decoded JPEG8 levels."""
    if (
        array.ndim != 3
        or array.shape[0] != 1
        or min(array.shape[1:]) < 16
        or not np.isfinite(array).all()
        or np.min(array) < 0
        or np.max(array) > 255
    ):
        raise ValueError("Flat-field requires one finite processed JPEG8 channel plane")
    return array.astype(np.float32)


def smooth(planes: np.ndarray, sigma: float) -> np.ndarray:
    """Suppress shot noise in the illumination map without moving its geometry."""
    return np.stack(
        [cv2.GaussianBlur(p, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101) for p in planes]
    )


def field_metrics(planes: np.ndarray, valid: np.ndarray) -> dict:
    """Robust relative variation on the valid native sensor samples."""
    cvs, ranges, medians = [], [], []
    for plane, mask in zip(planes, valid, strict=True):
        values = plane[mask]
        if values.size == 0 or np.median(values) <= 0:
            raise ValueError("No valid positive flat-field signal")
        reference = float(np.median(values))
        medians.append(reference)
        cvs.append(float(np.std(values) / reference))
        ranges.append(float((np.percentile(values, 95) - np.percentile(values, 5)) / reference))
    return {"cv": cvs, "p95_p5_fraction": ranges, "median_dn": medians}


def fit_flat_field(
    flat: np.ndarray, dark: np.ndarray, settings: FlatFieldSettings
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit measured-dark and smooth gain maps; reject weak, textured or saturated fields.

    Black is included in the measured ALL_OFF frame. It is subtracted once,
    not again using the sensor's nominal black-level constant.
    """
    settings.require_jpeg()
    flat, dark = checked_array(flat), checked_array(dark)
    if flat.shape != dark.shape:
        raise ValueError("Dark and flat geometry differ")
    if np.max(np.median(dark, axis=(1, 2))) > settings.maximum_dark_signal_dn:
        raise ValueError("ALL_OFF signal is too bright; check ambient light and gates")
    saturated = flat >= 255 * settings.saturation_level_fraction
    saturation = np.mean(saturated, axis=(1, 2))
    if np.max(saturation) > settings.maximum_saturation_fraction:
        raise ValueError("Flat-field is saturated; reduce the measurement exposure")
    signal = flat - dark
    illumination = smooth(signal, settings.smoothing_sigma_px)
    reference = np.median(illumination, axis=(1, 2))[:, None, None]
    if np.min(reference) < settings.minimum_signal_dn:
        raise ValueError("Flat-field signal is too weak at the current exposure")
    gain = np.divide(reference, illumination, out=np.zeros_like(signal), where=illumination > 0)
    valid = (
        (signal >= settings.minimum_signal_dn)
        & (illumination >= settings.minimum_signal_dn)
        & (gain <= settings.maximum_gain)
        & (gain >= 1 / settings.maximum_gain)
        & ~saturated
        & (np.abs(signal - illumination) <= settings.maximum_field_texture * reference)
    )
    if np.max(1 - np.mean(valid, axis=(1, 2))) > settings.maximum_mask_fraction:
        raise ValueError(
            "Too much weak/invalid flat-field area: "
            f"{100 * np.max(1 - np.mean(valid, axis=(1, 2))):.2f}% "
            f"(limit {100 * settings.maximum_mask_fraction:.2f}%)"
        )
    texture = float(
        max(
            np.percentile(np.abs(signal[i] - illumination[i])[m] / reference[i, 0, 0], 95)
            for i, m in enumerate(valid)
        )
    )
    if texture > settings.maximum_field_texture:
        raise ValueError("Field contains texture, dirt or excess noise; remove the slide")
    gain[~valid] = 0
    maps = {"dark": dark.copy(), "gain": gain, "valid": valid}
    report = {
        "black_source": "measured_all_off_same_exposure",
        "saturation_fraction": saturation.tolist(),
        "mask_fraction": (1 - np.mean(valid, axis=(1, 2))).tolist(),
        "texture_p95_fraction": texture,
        "gain_min": float(np.min(gain[valid])),
        "gain_max": float(np.max(gain[valid])),
        "before": field_metrics(signal, valid),
    }
    return maps, report


def apply_native_flat_field(
    image: np.ndarray, maps: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply empirical C=(I-D)*gain once on the decoded grid, preserving invalid samples."""
    image = checked_array(image)
    dark, gain, valid = maps["dark"], maps["gain"], maps["valid"]
    if (
        image.shape != dark.shape
        or gain.shape != image.shape
        or valid.shape != image.shape
        or valid.dtype != np.bool_
        or not np.isfinite(dark).all()
        or not np.isfinite(gain).all()
        or np.min(gain) < 0
        or not np.any(valid)
    ):
        raise ValueError("Incompatible or corrupt flat-field maps")
    mask = valid & (image < 255) & (image > dark)
    corrected = (image - dark) * gain
    corrected[~mask] = np.nan
    return corrected, mask


def validate_flat_field(
    heldout: np.ndarray,
    maps: dict[str, np.ndarray],
    fit_report: dict,
    settings: FlatFieldSettings,
) -> dict:
    """Test new, unused exposures; never validate by flattening the fitted image."""
    settings.require_jpeg()
    heldout = checked_array(heldout)
    saturation = np.mean(heldout >= 255 * settings.saturation_level_fraction, axis=(1, 2))
    if np.max(saturation) > settings.maximum_saturation_fraction:
        raise ValueError("Independent validation is saturated")
    corrected, mask = apply_native_flat_field(heldout, maps)
    if np.max(1 - np.mean(mask, axis=(1, 2))) > settings.maximum_mask_fraction:
        raise ValueError("Independent validation has insufficient valid area")
    before = field_metrics(heldout - maps["dark"], mask)
    after = field_metrics(corrected, mask)
    drift = max(
        abs(new / old - 1)
        for new, old in zip(before["median_dn"], fit_report["before"]["median_dn"], strict=True)
    )
    if min(before["median_dn"]) < settings.minimum_signal_dn:
        raise ValueError("Independent validation signal is too weak")
    # A flat-field is a normalised spatial response, not an absolute radiometer:
    # multiplying illumination by a positive scalar does not change its gain map.
    # Keep that scalar in the report, and enforce the bound on spatial residuals.
    spatial = [
        float(np.percentile(np.abs(plane[valid] / median - 1), 95))
        for plane, valid, median in zip(corrected, mask, after["median_dn"], strict=True)
    ]
    if max(spatial) > settings.maximum_spatial_residual_fraction:
        raise ValueError("Independent normalised spatial residual is too high")
    # A near-uniform original is acceptable; correction must not degrade it materially.
    if max(after["cv"]) > settings.maximum_residual_cv or any(
        new > max(old * 1.1, 0.01) for new, old in zip(after["cv"], before["cv"], strict=True)
    ):
        raise ValueError("Independent flat-field residual is too high")
    return {
        "kind": "fresh_holdout_exposures_not_fit_frames",
        "before": before,
        "after": after,
        "quality_basis": "normalised_spatial_response",
        "brightness_drift_fraction": drift,
        "brightness_change_notice": drift > settings.maximum_drift_fraction,
        "spatial_residual_p95_fraction": spatial,
        "saturation_fraction": saturation.tolist(),
    }


def common_measurement_plane(
    corrected: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the selected same-grid JPEG plane without Bayer shifts or resampling."""
    if corrected.shape != valid.shape or corrected.ndim != 3 or len(corrected) != 1:
        raise ValueError("Expected one common JPEG plane and its mask")
    return corrected[0].copy(), valid[0].copy()
