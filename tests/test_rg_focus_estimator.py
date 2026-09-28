"""Strict OF-024 estimator tests with no camera, light, file, or motion access."""

import cv2
import numpy as np
import pytest

from fast_ofm_core.focus.rg.rg_flat_field import (
    apply_native_flat_field,
    common_measurement_plane,
)
from fast_ofm_core.focus.rg.rg_focus_core import (
    PatchMeasurement,
    RGFocusCoreSettings,
)
from fast_ofm_core.focus.rg.rg_focus_estimator import (
    RGFocusEstimatorSettings,
    RGFocusInputs,
    _summarise_inliers,
    estimate_rg_shift,
    rg_shift_manifest_stage,
)
from fast_ofm_core.focus.rg.rg_focus_field import (
    FrameReference,
    TissueFieldError,
    TissueFieldSettings,
    geometry_fingerprint,
    parse_frame_geometry,
    prepare_tissue_field,
)


def geometry(size=192):
    """Return one exact same-grid processed-JPEG IMX477 ROI binding."""
    source_size = size + 16
    roi_offset = 8
    affine = [[2.0, 0.0, 116.5], [0.0, 2.0, 216.5]]
    return {
        "measurement_space": "processed-jpeg-rgb8",
        "sensor": "imx477",
        "sensor_resolution": [4056, 3040],
        "sensor_crop": [100, 200, source_size * 2, source_size * 2],
        "source_image_size": [source_size, source_size],
        "processing_roi": [roi_offset, roi_offset, size, size],
        "image_size": [size, size],
        "plane_size": [size, size],
        "white_size": [size, size],
        "bit_depth": 8,
        "white_level": 255,
        "channel_order": ["R", "G", "B"],
        "array_axes": "height,width,channel",
        "pixel_to_sensor": affine,
        "white_to_sensor": [row.copy() for row in affine],
        "common_plane_to_sensor": [row.copy() for row in affine],
    }


def core_settings(**overrides):
    """Use fixed explicit synthetic thresholds, including signal/saturation QC."""
    values = {
        "phase_highpass_sigma_px": 3,
        "texture_background_sigma_px": 5,
        "texture_score_sigma_px": 2,
        "texture_morphology_kernel_px": 3,
        "patch_size_px": 48,
        "patch_stride_px": 24,
        "minimum_tissue_coverage": 0.35,
        "minimum_patch_signal": 10,
        "jpeg8_saturation_level": 250,
        "maximum_patch_saturation_fraction": 0.01,
        "minimum_patch_std": 4,
        "minimum_patch_response": 0.15,
        "minimum_patch_spectral_correlation": 0.05,
        "maximum_absolute_shift_px": 8,
        "minimum_patch_count": 4,
        "maximum_patch_count": 20,
    }
    values.update(overrides)
    return RGFocusCoreSettings(**values)


def field_settings(**overrides):
    """Keep field classification deterministic for the synthetic image."""
    values = {
        "edge_margin_px": 10,
        "minimum_white_median": 5,
        "white_saturation_level": 250,
        "maximum_white_saturation_fraction": 0.05,
        "coherent_texture_blur_sigma_px": 1,
        "coherent_texture_background_sigma_px": 5,
        "minimum_coherent_texture_fraction": 0.006,
        "minimum_component_pixels": 100,
        "minimum_tissue_fraction": 0.03,
    }
    values.update(overrides)
    return TissueFieldSettings(**values)


def white_image(size=192):
    """Create coherent tissue inside a quiet background."""
    image = np.full((size, size), 80.0)
    yy, xx = np.mgrid[22 : size - 22, 22 : size - 22]
    image[22:-22, 22:-22] += (
        23 * np.sin(xx / 3.7) + 19 * np.cos(yy / 5.1) + 8 * np.sin((xx + yy) / 2.9)
    )
    return image


def references(field_id="field-a"):
    """Build ordered WHITE/RED/GREEN references for one field."""
    geometry_id = geometry_fingerprint(parse_frame_geometry(geometry()))
    return tuple(
        FrameReference(
            frame_id=f"{mode}-{timestamp}",
            field_id=field_id,
            sensor_timestamp_ns=timestamp,
            mode=mode,
            geometry_id=geometry_id,
        )
        for mode, timestamp in (("white", 100), ("red", 200), ("green", 300))
    )


def ready_field(core=None, settings=None):
    """Prepare the actual OF-032 field rather than manufacturing its status."""
    white, _, _ = references()
    return prepare_tissue_field(
        white_image(),
        white,
        geometry(),
        field_settings() if settings is None else settings,
        core_settings() if core is None else core,
    )


def shifted_pair(field, dx=3, dy=-2, backgrounds=(25, 900)):
    """Make one known tissue shift with deliberately unrelated excluded backgrounds."""
    rng = np.random.default_rng(707)
    tissue = field.white_common * 4 + cv2.GaussianBlur(
        rng.normal(0, 25, field.mask.shape), (0, 0), 0.8
    )
    red = rng.normal(backgrounds[0], 2, field.mask.shape)
    red[field.mask] = tissue[field.mask]
    shifted = cv2.warpAffine(
        tissue,
        np.float32([[1, 0, dx], [0, 1, dy]]),
        (tissue.shape[1], tissue.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    green = rng.normal(backgrounds[1], 90, field.mask.shape)
    green[field.mask] = shifted[field.mask] * 1.3 + 20
    return red, green


def estimate(
    red,
    green,
    field,
    *,
    core=None,
    estimator=None,
    valid=None,
    red_valid=None,
    green_valid=None,
    red_source=None,
    green_source=None,
):
    """Call the production estimator with matching provenance."""
    _, red_ref, green_ref = references()
    common_valid = np.ones(field.mask.shape, dtype=bool) if valid is None else valid
    red_valid = common_valid if red_valid is None else red_valid
    green_valid = common_valid if green_valid is None else green_valid
    red_source = (
        np.full(field.mask.shape, 100, dtype=np.uint8) if red_source is None else red_source
    )
    green_source = (
        np.full(field.mask.shape, 100, dtype=np.uint8) if green_source is None else green_source
    )
    return estimate_rg_shift(
        RGFocusInputs(
            red_plane=red,
            green_plane=green,
            red_source_jpeg8=red_source,
            green_source_jpeg8=green_source,
            red_valid=red_valid,
            green_valid=green_valid,
            field=field,
            red_reference=red_ref,
            green_reference=green_ref,
            geometry_value=geometry(),
        ),
        core_settings() if core is None else core,
        RGFocusEstimatorSettings() if estimator is None else estimator,
    )


def test_accepted_integer_windows_are_averaged_after_robust_rejection():
    """A bimodal inlier set retains sub-pixel information instead of median jumps."""

    def patch(index: int, dx: float) -> PatchMeasurement:
        return PatchMeasurement(
            patch_id=f"p{index}",
            x=index,
            y=0,
            w=48,
            h=48,
            dx=dx,
            dy=0,
            response=0.2,
            peak_margin=0.1,
            coverage=0.8,
            median_r=40,
            median_g=40,
            std_r=10,
            std_g=10,
            saturation_r=0,
            saturation_g=0,
            spectral_correlation=0.5,
        )

    result = _summarise_inliers([patch(index, dx) for index, dx in enumerate((0, 0, 0, -2, -2))])
    assert result.dx == pytest.approx(-0.8)
    assert result.dy == 0
    assert result.dx_mad == 0


@pytest.mark.parametrize("backgrounds", [(0, 0), (25, 900), (1200, 30)])
def test_known_shift_survives_different_excluded_backgrounds(backgrounds):
    """Invalid/background pixels and the fixed mask edge do not create a false zero."""
    field = ready_field()
    red, green = shifted_pair(field, backgrounds=backgrounds)
    result = estimate(red, green, field)
    assert result.status == "ready", result.reason
    assert result.dx == pytest.approx(3, abs=0.45)
    assert result.dy == pytest.approx(-2, abs=0.45)
    assert result.inlier_patch_count >= 4
    assert result.confidence >= 0.15


def test_subpixel_shift_is_bounded_by_integer_registration_resolution():
    """Integer mutual-information registration stays within one pixel of truth."""
    field = ready_field()
    red, green = shifted_pair(field, dx=2.4, dy=-1.6)
    result = estimate(red, green, field)
    assert result.status == "ready"
    assert result.dx == pytest.approx(2.4, abs=0.65)
    assert result.dy == pytest.approx(-1.6, abs=0.65)


def test_no_tissue_and_tiny_tissue_refuse_without_whole_frame_fallback():
    """Non-ready WHITE outcomes remain explicit and never reach R/G correlation."""
    white, _, _ = references()
    empty = prepare_tissue_field(
        np.full((192, 192), 80.0),
        white,
        geometry(),
        field_settings(),
        core_settings(),
    )
    dummy = np.full((192, 192), 100.0)
    assert estimate(dummy, dummy, empty).status == "no_tissue"

    tiny = ready_field(settings=field_settings(minimum_tissue_fraction=0.90))
    assert tiny.metrics.status == "insufficient_tissue"
    assert estimate(dummy, dummy, tiny).status == "insufficient_tissue"


def test_saturation_and_spectral_mismatch_have_specific_reasons():
    """Only exact source JPEG8 clipping, not corrected values, drives saturation QC."""
    field = ready_field()
    red, green = shifted_pair(field)
    # Both corrected planes exceed 255 but their source pixels are not clipped.
    assert estimate(red, green, field).status == "ready"
    saturated_source = np.full(field.mask.shape, 255, dtype=np.uint8)
    assert estimate(red, green, field, green_source=saturated_source).status == "saturated"

    rng = np.random.default_rng(99)
    mismatch = rng.normal(600, 100, field.mask.shape)
    strict_spectral = core_settings(
        minimum_patch_response=0,
        minimum_patch_spectral_correlation=0.6,
    )
    result = estimate(red, mismatch, field, core=strict_spectral)
    assert result.status == "spectral_mismatch"
    assert result.dx is None


def test_native_flat_field_mask_cannot_hide_source_jpeg_clipping():
    """Exact JPEG clipping wins even when native correction invalidates those pixels."""
    field = ready_field()
    source_r = np.clip(np.rint(field.white_common), 20, 220).astype(np.uint8)
    source_g = cv2.warpAffine(
        source_r,
        np.float32([[1, 0, 3], [0, 1, -2]]),
        (source_r.shape[1], source_r.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    maps = {
        "dark": np.zeros((1, *source_r.shape), dtype=np.float32),
        "gain": np.full((1, *source_r.shape), 4.0, dtype=np.float32),
        "valid": np.ones((1, *source_r.shape), dtype=bool),
    }

    def correct(source):
        corrected, valid = apply_native_flat_field(source[np.newaxis, ...], maps)
        return common_measurement_plane(corrected, valid)

    red, red_valid = correct(source_r)
    green, green_valid = correct(source_g)
    assert np.nanmax(red) > 255
    assert (
        estimate(
            red,
            green,
            field,
            red_valid=red_valid,
            green_valid=green_valid,
            red_source=source_r,
            green_source=source_g,
        ).status
        == "ready"
    )

    yy, xx = np.indices(source_g.shape)
    clipped = field.mask & ((yy * source_g.shape[1] + xx) % 20 == 0)
    clipped_source_g = source_g.copy()
    clipped_source_g[clipped] = 255
    clipped_green, clipped_valid = correct(clipped_source_g)
    assert not clipped_valid[clipped].any()
    assert np.mean(clipped_source_g[field.mask] == 255) == pytest.approx(0.05, abs=0.002)
    result = estimate(
        red,
        clipped_green,
        field,
        red_valid=red_valid,
        green_valid=clipped_valid,
        red_source=source_r,
        green_source=clipped_source_g,
    )
    assert result.status == "saturated"

    all_clipped = np.full(source_r.shape, 255, dtype=np.uint8)
    clipped_red, no_red_valid = correct(all_clipped)
    clipped_green, no_green_valid = correct(all_clipped)
    result = estimate(
        clipped_red,
        clipped_green,
        field,
        red_valid=no_red_valid,
        green_valid=no_green_valid,
        red_source=all_clipped,
        green_source=all_clipped,
    )
    assert result.status == "saturated"
    assert result.rejection_counts == {"saturated": len(field.boxes)}


def test_conflicting_window_shifts_are_not_aggregated_into_false_focus():
    """Two coherent but contradictory patch populations fail the spread gate."""
    field = ready_field()
    red, positive = shifted_pair(field, dx=3, dy=0, backgrounds=(50, 50))
    _, negative = shifted_pair(field, dx=-3, dy=0, backgrounds=(50, 50))
    green = np.array(positive, copy=True)
    green[:, green.shape[1] // 2 :] = negative[:, green.shape[1] // 2 :]
    result = estimate(
        red,
        green,
        field,
        estimator=RGFocusEstimatorSettings(
            maximum_dx_mad_px=0.6,
            maximum_dy_mad_px=0.6,
            maximum_radial_p90_px=1.0,
        ),
    )
    assert result.status == "inconsistent_shifts"
    assert result.accepted_patch_count >= 4


def test_invalid_geometry_or_valid_mask_fails_closed_and_manifest_names_action():
    """A missing validity plane is mask_invalid, and the manifest has no fallback."""
    field = ready_field()
    red, green = shifted_pair(field)
    with pytest.raises(TissueFieldError) as error:
        estimate(red, green, field, valid=np.ones((10, 10), dtype=bool))
    assert error.value.code == "mask_invalid"
    stage = rg_shift_manifest_stage()
    assert stage.action == estimate_rg_shift.__name__
    assert "sharpness-difference" in stage.cancellation
