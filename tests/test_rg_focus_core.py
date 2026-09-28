"""Pure numerical tests for the extracted tissue-aware R/G focus core."""

import subprocess
import sys

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from fast_ofm_core.focus.rg import rg_focus_core
from fast_ofm_core.focus.rg.rg_focus_core import (
    PatchBox,
    RGFocusCoreSettings,
    candidate_boxes,
    measure_patches,
    mutual_information_shift_masked,
    select_reference_boxes,
    summarise_shift,
    texture_candidate_mask,
)


def compact_settings(**overrides):
    """Use small patches while preserving every production threshold explicitly."""
    values = {
        "patch_size_px": 64,
        "patch_stride_px": 32,
        "minimum_patch_count": 2,
        "maximum_patch_count": 8,
    }
    values.update(overrides)
    return RGFocusCoreSettings(**values)


def textured_plane(size=192):
    """Create a deterministic broadband texture with no hardware or saved-data input."""
    noise = np.random.default_rng(314159).normal(0, 40, (size, size))
    return cv2.GaussianBlur(noise, (0, 0), 1.2) + 1000


def jpeg8_source(image, level=100):
    """Return an explicit unsaturated source plane for corrected numerical data."""
    return np.full(np.asarray(image).shape, level, dtype=np.uint8)


def test_import_is_pure():
    """Importing the core must not load camera, light, stage, or Thing modules."""
    code = """
import sys
from fast_ofm_core.focus.rg import rg_focus_core
forbidden = ("picamera2", "openflexure_microscope_server.things")
assert not any(name.startswith(forbidden) for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_known_spatial_shift_is_recovered():
    """Mutual information recovers a known signed cross-colour translation."""
    first = textured_plane()
    expected_dx, expected_dy = 3.0, -2.0
    transform = np.float32([[1, 0, expected_dx], [0, 1, expected_dy]])
    second = cv2.warpAffine(
        np.sqrt(np.maximum(first - np.min(first), 0)) * 40,
        transform,
        (first.shape[1], first.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    dx, dy, response, margin, spectral = mutual_information_shift_masked(
        first,
        second,
        np.ones(first.shape, dtype=bool),
        compact_settings(maximum_absolute_shift_px=6),
    )
    assert dx == expected_dx
    assert dy == expected_dy
    assert response > 0.05
    assert margin > 0.03
    assert spectral > 0.5


def test_optional_subpixel_refinement_preserves_default_and_recovers_fraction():
    """Only an explicit caller gets the local fractional NMI peak estimate."""
    first = textured_plane()
    expected_dx, expected_dy = 3.35, -1.65
    second = cv2.warpAffine(
        np.sqrt(np.maximum(first - np.min(first), 0)) * 40,
        np.float32([[1, 0, expected_dx], [0, 1, expected_dy]]),
        (first.shape[1], first.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    args = (
        first,
        second,
        np.ones(first.shape, dtype=bool),
        compact_settings(maximum_absolute_shift_px=6),
    )

    integer = mutual_information_shift_masked(*args)
    refined = mutual_information_shift_masked(*args, subpixel_refinement=True)

    assert integer[:2] == (3, -2)
    assert refined[0] == pytest.approx(expected_dx, abs=0.35)
    assert refined[1] == pytest.approx(expected_dy, abs=0.35)
    assert refined[2:] == integer[2:]


def test_cached_count_entropy_matches_probability_definition():
    """The faster count identity preserves the original NMI definition."""
    rng = np.random.default_rng(116)
    first = rng.integers(0, 32, 8192, dtype=np.int16)
    second = np.roll(first, 17)
    joint = np.bincount(
        first.astype(np.int32) * 32 + second.astype(np.int32), minlength=32 * 32
    ).reshape(32, 32)
    probability = joint.astype(np.float64) / joint.sum()
    first_probability = probability.sum(axis=1)
    second_probability = probability.sum(axis=0)
    independent = first_probability[:, None] * second_probability[None, :]
    occupied = probability > 0
    information = np.sum(
        probability[occupied] * np.log(probability[occupied] / independent[occupied])
    )
    first_entropy = -np.sum(
        first_probability[first_probability > 0] * np.log(first_probability[first_probability > 0])
    )
    second_entropy = -np.sum(
        second_probability[second_probability > 0]
        * np.log(second_probability[second_probability > 0])
    )
    expected = 2 * information / (first_entropy + second_entropy)

    actual = rg_focus_core._normalised_mutual_information(first, second, 32)

    assert actual == pytest.approx(expected, abs=1e-14)
    table = rg_focus_core._count_log_count_table(first.size)
    assert not table.flags.writeable


@pytest.mark.parametrize("bins", [8, 32, 256])
def test_opencv_joint_histogram_matches_numpy_reference(bins):
    """The accelerated histogram preserves counts for masks and shifted views."""
    rng = np.random.default_rng(930)
    first = rng.integers(0, bins, (71, 93), dtype=np.uint8)
    second = rng.integers(0, bins, first.shape, dtype=np.uint8)
    support = rng.random(first.shape) > 0.31
    first_slice, second_slice = rg_focus_core._overlap_slices(first.shape, 7, -4)
    common = support[first_slice] & support[second_slice]
    expected = np.bincount(
        (
            first[first_slice][common].astype(np.int32) * bins
            + second[second_slice][common].astype(np.int32)
        ),
        minlength=bins * bins,
    ).reshape(bins, bins)
    actual = cv2.calcHist(
        [first[first_slice], second[second_slice]],
        [0, 1],
        common.astype(np.uint8) * 255,
        [bins, bins],
        [0, bins, 0, bins],
    ).astype(np.int64)

    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(("dx", "dy"), [(-3, 2), (3, -2)])
def test_mutual_information_recovers_signed_masked_rectangular_translation(dx, dy):
    """The bounded tissue-only search preserves both shift signs."""
    rng = np.random.default_rng(728)
    first = rng.normal(1000, 40, (64, 96))
    second = cv2.warpAffine(
        first,
        np.float32([[1, 0, dx], [0, 1, dy]]),
        (96, 64),
        borderMode=cv2.BORDER_REFLECT,
    )
    mask = rng.random(first.shape) > 0.2
    measured_dx, measured_dy, response, margin, _spectral = mutual_information_shift_masked(
        first,
        second,
        mask,
        compact_settings(
            maximum_absolute_shift_px=4,
            maximum_absolute_orthogonal_shift_px=3,
        ),
    )
    assert (measured_dx, measured_dy) == (dx, dy)
    assert response > 0.05
    assert margin > 0.03


def test_texture_mask_stays_inside_explicit_valid_geometry():
    """Mask morphology cannot create candidates outside the supplied geometry."""
    white = np.full((160, 160), 1000.0)
    rng = np.random.default_rng(7)
    white[45:115, 45:115] += rng.normal(0, 50, (70, 70))
    valid = np.zeros_like(white, dtype=bool)
    valid[16:-16, 16:-16] = True
    mask, metrics = texture_candidate_mask(white, valid, compact_settings())
    assert mask.dtype == np.bool_
    assert np.any(mask[45:115, 45:115])
    assert not np.any(mask[~valid])
    assert metrics.coverage_total < metrics.coverage_inside_valid
    empty, empty_metrics = texture_candidate_mask(
        np.ones((64, 64)), np.ones((64, 64), dtype=bool), compact_settings()
    )
    assert not empty.any()
    assert empty_metrics.coverage_total == 0


def test_reused_boxes_do_not_bypass_mask_or_response(mocker):
    """The two unsafe AF-010 fixed-box bypasses are deliberately removed."""
    red = textured_plane(64)
    green = np.roll(red, 2, axis=1)
    box = [PatchBox(patch_id="fixed", x=0, y=0, w=64, h=64)]
    settings = compact_settings(minimum_patch_count=1)
    assert (
        measure_patches(
            red,
            green,
            np.zeros((64, 64), dtype=bool),
            settings,
            box,
            red_source_jpeg8=jpeg8_source(red),
            green_source_jpeg8=jpeg8_source(green),
            source_support=np.ones((64, 64), dtype=bool),
        )
        == []
    )
    mocker.patch(
        "fast_ofm_core.focus.rg.rg_focus_core.mutual_information_shift_masked",
        return_value=(1.0, 2.0, 0.01, 0.1, 0.9),
    )
    assert (
        measure_patches(
            red,
            green,
            np.ones((64, 64), dtype=bool),
            settings,
            box,
            red_source_jpeg8=jpeg8_source(red),
            green_source_jpeg8=jpeg8_source(green),
            source_support=np.ones((64, 64), dtype=bool),
        )
        == []
    )


def test_excluded_background_and_mask_outline_cannot_create_a_false_zero_shift():
    """Different invalid fills are removed before tissue-only registration."""
    rng = np.random.default_rng(91)
    size = 96
    yy, xx = np.mgrid[:size, :size]
    support = (xx - 48) ** 2 + (yy - 48) ** 2 < 38**2
    tissue = cv2.GaussianBlur(rng.normal(0, 35, (size, size)), (0, 0), 1.1) + 350
    red = rng.normal(25, 2, (size, size))
    red[support] = tissue[support]
    shifted = cv2.warpAffine(
        tissue,
        np.float32([[1, 0, 4], [0, 1, -3]]),
        (size, size),
        borderMode=cv2.BORDER_REFLECT,
    )
    green = rng.normal(900, 70, (size, size))
    green[support] = shifted[support] * 1.4 + 30
    settings = compact_settings(
        patch_size_px=size,
        patch_stride_px=size,
        minimum_patch_count=1,
        minimum_patch_spectral_correlation=0.3,
    )
    rows = measure_patches(
        red,
        green,
        support,
        settings,
        red_source_jpeg8=jpeg8_source(red),
        green_source_jpeg8=jpeg8_source(green),
        source_support=support,
    )
    assert len(rows) == 1
    assert rows[0].dx == 4
    assert rows[0].dy == -3
    assert rows[0].response > 0.05


def test_nonfinite_samples_outside_support_do_not_bleed_during_alignment():
    """Flat-field NaNs outside support cannot poison a valid subpixel warp."""
    rng = np.random.default_rng(92)
    size = 96
    yy, xx = np.mgrid[:size, :size]
    support = (xx - 48) ** 2 + (yy - 48) ** 2 < 38**2
    first = cv2.GaussianBlur(rng.normal(0, 35, (size, size)), (0, 0), 1.1) + 350
    second = cv2.warpAffine(
        first,
        np.float32([[1, 0, 3], [0, 1, -2]]),
        (size, size),
        borderMode=cv2.BORDER_REFLECT,
    )
    first[~support] = np.nan
    second[~support] = np.nan
    dx, dy, response, margin, spectral = mutual_information_shift_masked(
        first,
        second,
        support,
        compact_settings(maximum_absolute_shift_px=6),
    )
    assert dx == 3
    assert dy == -2
    assert response > 0.05
    assert margin > 0.03
    assert spectral > 0.5


def test_unrelated_colour_textures_fail_registration_qc():
    """Peak search cannot turn independent textured colours into an accepted pair."""
    rng = np.random.default_rng(221)
    size = 128
    red = cv2.GaussianBlur(rng.normal(100, 20, (size, size)), (0, 0), 1.1)
    green = cv2.GaussianBlur(rng.normal(100, 20, (size, size)), (0, 0), 1.1)
    support = np.ones((size, size), dtype=bool)
    rows = measure_patches(
        red,
        green,
        support,
        compact_settings(
            patch_size_px=size,
            patch_stride_px=size,
            minimum_patch_count=1,
        ),
        red_source_jpeg8=jpeg8_source(red),
        green_source_jpeg8=jpeg8_source(green),
        source_support=support,
    )
    assert rows == []


def test_search_boundary_is_a_refusal_not_a_clipped_measurement(mocker):
    """An optimum at the search edge cannot be reported as an in-range shift."""
    size = 64
    red = textured_plane(size)
    green = np.roll(red, 4, axis=1)
    support = np.ones((size, size), dtype=bool)
    settings = compact_settings(
        minimum_patch_count=1,
        maximum_absolute_shift_px=4,
        maximum_absolute_orthogonal_shift_px=3,
    )
    mocker.patch.object(
        rg_focus_core,
        "mutual_information_shift_masked",
        return_value=(4.0, 0.0, 0.2, 0.1, 0.5),
    )
    rows = rg_focus_core.evaluate_patches(
        red,
        green,
        support,
        settings,
        red_source_jpeg8=jpeg8_source(red),
        green_source_jpeg8=jpeg8_source(green),
        source_support=support,
    )
    assert {row.status for row in rows} == {"shift_out_of_range"}


def test_patch_grid_selection_and_summary():
    """Accepted patch geometry stays fixed and robust statistics match the input."""
    red = textured_plane(160)
    transform = np.float32([[1, 0, 2], [0, 1, -1]])
    green = cv2.warpAffine(
        red,
        transform,
        (red.shape[1], red.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    mask = np.ones(red.shape, dtype=bool)
    settings = compact_settings(minimum_patch_response=0.05)
    boxes = candidate_boxes(red.shape, settings)
    rows = measure_patches(
        red,
        green,
        mask,
        settings,
        boxes,
        red_source_jpeg8=jpeg8_source(red),
        green_source_jpeg8=jpeg8_source(green),
        source_support=mask,
    )
    assert len(rows) >= settings.minimum_patch_count
    selected = select_reference_boxes(rows, settings)
    assert len(selected) <= settings.maximum_patch_count
    assert all(box.w == settings.patch_size_px for box in selected)
    repeated = measure_patches(
        red,
        green,
        mask,
        settings,
        selected,
        red_source_jpeg8=jpeg8_source(red),
        green_source_jpeg8=jpeg8_source(green),
        source_support=mask,
    )
    summary = summarise_shift(repeated)
    assert summary.dx == pytest.approx(2, abs=0.4)
    assert summary.dy == pytest.approx(-1, abs=0.4)


def test_parallel_patch_evaluation_is_exactly_ordered_like_serial(monkeypatch):
    """Concurrency changes latency only, never values, failures, or row order."""
    red = textured_plane(160)
    green = cv2.warpAffine(
        red,
        np.float32([[1, 0, 2], [0, 1, -1]]),
        (red.shape[1], red.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    mask = np.ones(red.shape, dtype=bool)
    settings = compact_settings(minimum_patch_response=0.05)
    kwargs = {
        "red_source_jpeg8": jpeg8_source(red),
        "green_source_jpeg8": jpeg8_source(green),
        "source_support": mask,
    }
    monkeypatch.setattr(rg_focus_core, "PATCH_EVALUATION_WORKERS", 1)
    serial = rg_focus_core.evaluate_patches(red, green, mask, settings, **kwargs)
    monkeypatch.setattr(rg_focus_core, "PATCH_EVALUATION_WORKERS", 2)
    parallel = rg_focus_core.evaluate_patches(red, green, mask, settings, **kwargs)
    assert parallel == serial
    assert [row.patch_id for row in parallel] == [
        box.patch_id for box in candidate_boxes(red.shape, settings)
    ]


@pytest.mark.parametrize(
    "values",
    [
        {"texture_morphology_kernel_px": 4},
        {"patch_size_px": 32, "patch_stride_px": 33},
        {"minimum_patch_count": 9, "maximum_patch_count": 8},
    ],
)
def test_inconsistent_settings_are_rejected(values):
    """Every copied threshold is validated rather than being a hidden constant."""
    with pytest.raises(ValidationError):
        RGFocusCoreSettings(**values)


def test_geometry_and_empty_summary_fail_closed():
    """Invalid shapes and boxes fail instead of being silently clipped."""
    settings = compact_settings()
    with pytest.raises(ValueError, match="smaller"):
        candidate_boxes((32, 32), settings)
    red = textured_plane(64)
    bad = [PatchBox(patch_id="outside", x=1, y=0, w=64, h=64)]
    with pytest.raises(ValueError, match="outside"):
        measure_patches(
            red,
            red,
            np.ones((64, 64), dtype=bool),
            settings,
            bad,
            red_source_jpeg8=jpeg8_source(red),
            green_source_jpeg8=jpeg8_source(red),
            source_support=np.ones((64, 64), dtype=bool),
        )
    with pytest.raises(ValueError, match="empty"):
        summarise_shift([])
