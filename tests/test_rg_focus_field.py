"""Fresh field, geometry, tissue, and no-fallback tests for OF-032."""

import numpy as np
import pytest

from fast_ofm_core.focus.rg.rg_focus_core import RGFocusCoreSettings
from fast_ofm_core.focus.rg.rg_focus_field import (
    FrameReference,
    TissueFieldError,
    TissueFieldSettings,
    geometry_fingerprint,
    parse_frame_geometry,
    prepare_tissue_field,
    require_tissue_field_for_pair,
    tissue_field_manifest_stage,
)


def geometry(size=160, crop_x=100, width=None):
    """Create exact rectangular geometry emitted by the JPEG processing binding."""
    width = size if width is None else width
    source_width, source_height = width + 20, size + 12
    crop_width, crop_height = source_width * 4, source_height * 4
    roi_x, roi_y = 10, 6
    affine = [
        [4.0, 0.0, crop_x + 4.0 * (roi_x + 0.5) - 0.5],
        [0.0, 4.0, 200 + 4.0 * (roi_y + 0.5) - 0.5],
    ]
    return {
        "measurement_space": "processed-jpeg-rgb8",
        "sensor": "imx477",
        "sensor_resolution": [4056, 3040],
        "sensor_crop": [crop_x, 200, crop_width, crop_height],
        "source_image_size": [source_width, source_height],
        "processing_roi": [roi_x, roi_y, width, size],
        "image_size": [width, size],
        "plane_size": [width, size],
        "white_size": [width, size],
        "bit_depth": 8,
        "white_level": 255,
        "channel_order": ["R", "G", "B"],
        "array_axes": "height,width,channel",
        "pixel_to_sensor": affine,
        "white_to_sensor": [row.copy() for row in affine],
        "common_plane_to_sensor": [row.copy() for row in affine],
    }


def settings(**overrides):
    """Small deterministic parameters for synthetic field tests."""
    values = {
        "edge_margin_px": 8,
        "minimum_white_median": 5.0,
        "white_saturation_level": 250.0,
        "maximum_white_saturation_fraction": 0.05,
        "coherent_texture_blur_sigma_px": 1.0,
        "coherent_texture_background_sigma_px": 5.0,
        "minimum_coherent_texture_fraction": 0.006,
        "minimum_component_pixels": 80,
        "minimum_tissue_fraction": 0.03,
    }
    values.update(overrides)
    return TissueFieldSettings(**values)


def core_settings(**overrides):
    """Use bounded patches while keeping every core gate enabled."""
    values = {
        "phase_highpass_sigma_px": 3.0,
        "texture_background_sigma_px": 5.0,
        "texture_score_sigma_px": 2.0,
        "texture_morphology_kernel_px": 3,
        "patch_size_px": 32,
        "patch_stride_px": 16,
        "minimum_patch_count": 3,
        "maximum_patch_count": 12,
        "maximum_absolute_shift_px": 4.0,
    }
    values.update(overrides)
    return RGFocusCoreSettings(**values)


def reference(mode, field_id="field-a", timestamp=100, geometry_value=None):
    """Build a frame reference bound to the selected synthetic geometry."""
    geometry_value = geometry() if geometry_value is None else geometry_value
    return FrameReference(
        frame_id=f"{mode}-{timestamp}",
        field_id=field_id,
        sensor_timestamp_ns=timestamp,
        mode=mode,
        geometry_id=geometry_fingerprint(parse_frame_geometry(geometry_value)),
    )


def tissue_image(size=160, bounds=(24, 136)):
    """Create a coherent multiscale tissue-like region on a quiet background."""
    image = np.full((size, size), 80.0)
    low, high = bounds
    yy, xx = np.mgrid[low:high, low:high]
    texture = 24 * np.sin(xx / 3.5) + 18 * np.cos(yy / 5.0)
    image[low:high, low:high] += texture
    return image


def ready_field():
    """Prepare one reusable accepted synthetic tissue field."""
    return prepare_tissue_field(
        tissue_image(),
        reference("white"),
        geometry(),
        settings(),
        core_settings(),
    )


def test_tissue_field_is_bound_and_has_diagnostic_common_boxes():
    """A coherent field produces fixed common boxes and a visible diagnostic overlay."""
    field = ready_field()
    assert field.metrics.status == "ready"
    assert field.metrics.box_count >= 3
    assert field.metrics.candidate_coverage > 0.03
    assert field.mask.shape == (160, 160)
    assert field.overlay.shape == (160, 160, 3)
    assert not field.mask.flags.writeable
    assert not field.overlay.flags.writeable
    assert all(box.w == 32 for box in field.boxes)
    rectangular = parse_frame_geometry(geometry(size=160, width=176))
    assert rectangular.image_size == (176, 160)
    assert rectangular.pixel_to_sensor == rectangular.white_to_sensor
    assert rectangular.pixel_to_sensor == rectangular.common_plane_to_sensor


def test_uniform_and_noisy_empty_fields_do_not_become_full_frame_tissue():
    """Percentile thresholding cannot manufacture tissue from empty background."""
    uniform = np.full((160, 160), 80.0)
    noise = uniform + np.random.default_rng(8).normal(0, 0.5, uniform.shape)
    for image in (uniform, noise):
        field = prepare_tissue_field(
            image,
            reference("white"),
            geometry(),
            settings(),
            core_settings(),
        )
        assert field.metrics.status == "no_tissue"
        assert field.metrics.box_count == 0
        assert not field.mask.any()


def test_small_island_and_weak_or_saturated_white_fail_with_specific_status():
    """Small support, weak signal, and saturation remain distinct non-ready outcomes."""
    small = tissue_image(bounds=(65, 95))
    insufficient = prepare_tissue_field(
        small,
        reference("white"),
        geometry(),
        settings(minimum_tissue_fraction=0.12),
        core_settings(),
    )
    assert insufficient.metrics.status == "insufficient_tissue"
    assert insufficient.boxes == ()

    weak = tissue_image() * 0.03
    weak_field = prepare_tissue_field(
        weak,
        reference("white"),
        geometry(),
        settings(),
        core_settings(),
    )
    assert weak_field.metrics.status == "low_signal"

    saturated = tissue_image()
    saturated[20:140, 20:140] = 255
    saturated_field = prepare_tissue_field(
        saturated,
        reference("white"),
        geometry(),
        settings(),
        core_settings(),
    )
    assert saturated_field.metrics.status == "saturated"


def test_missing_wrong_or_stale_mask_is_not_reported_as_no_tissue():
    """Mask contract errors differ from an honestly empty WHITE field."""
    field = ready_field()
    red = reference("red", timestamp=200)
    green = reference("green", timestamp=201)

    with pytest.raises(TissueFieldError) as missing:
        require_tissue_field_for_pair(None, red, green, geometry())
    assert missing.value.code == "mask_invalid"

    with pytest.raises(TissueFieldError) as wrong_field:
        require_tissue_field_for_pair(
            field,
            reference("red", field_id="field-b", timestamp=200),
            green,
            geometry(),
        )
    assert wrong_field.value.code == "mask_stale"

    with pytest.raises(TissueFieldError) as old_pair:
        require_tissue_field_for_pair(
            field,
            reference("red", timestamp=90),
            green,
            geometry(),
        )
    assert old_pair.value.code == "mask_stale"

    with pytest.raises(TissueFieldError) as changed_crop:
        require_tissue_field_for_pair(field, red, green, geometry(crop_x=102))
    assert changed_crop.value.code == "mask_stale"


def test_current_white_precedes_a_matching_red_green_pair():
    """Only a newer pair from the same field and geometry accepts the mask."""
    require_tissue_field_for_pair(
        ready_field(),
        reference("red", timestamp=200),
        reference("green", timestamp=201),
        geometry(),
    )


@pytest.mark.parametrize(
    "change",
    [
        {"plane_size": [159, 160]},
        {"measurement_space": "linear-raw-bayer12"},
        {"white_to_sensor": [[1, 0, 139.5], [0, 4, 225.5]]},
        {"processing_roi": [11, 6, 160, 160]},
        {"channel_order": ["B", "G", "R"]},
    ],
)
def test_wrong_geometry_fails_as_mask_invalid(change):
    """Domain, ROI, size and affine mismatches fail before mask calculation."""
    bad = geometry()
    bad.update(change)
    with pytest.raises(TissueFieldError) as error:
        parse_frame_geometry(bad)
    assert error.value.code == "mask_invalid"


def test_reference_geometry_and_margin_are_mandatory():
    """A reference cannot be relabelled or measured without shift-safe borders."""
    with pytest.raises(TissueFieldError) as wrong_reference:
        prepare_tissue_field(
            tissue_image(),
            reference("red"),
            geometry(),
            settings(),
            core_settings(),
        )
    assert wrong_reference.value.code == "mask_invalid"

    with pytest.raises(TissueFieldError) as narrow_margin:
        prepare_tissue_field(
            tissue_image(),
            reference("white"),
            geometry(),
            settings(edge_margin_px=5),
            core_settings(),
        )
    assert narrow_margin.value.code == "mask_invalid"


def test_manifest_stage_names_the_real_function_and_no_fallback():
    """The future LED manifest can reuse the actual OF-032 action contract."""
    stage = tissue_field_manifest_stage()
    assert stage.id == "tissue_field"
    assert stage.action == prepare_tissue_field.__name__
    assert "whole-frame fallback" in stage.cancellation
    assert stage.hardware_required
