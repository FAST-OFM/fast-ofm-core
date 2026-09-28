"""Pure numerical tests for simultaneous R/G RAW flat-field unmixing."""

import cv2
import numpy as np
import pytest

from fast_ofm_core.focus.rg import rg_simultaneous
from fast_ofm_core.focus.rg.rg_focus_core import (
    PatchBox,
    PatchEvaluation,
    candidate_boxes,
)
from fast_ofm_core.focus.rg.rg_simultaneous import (
    SimultaneousCalibrationSettings,
    SimultaneousFocusObservation,
    SimultaneousFocusSettings,
    SimultaneousShiftMeasurement,
    crop_spectral_maps,
    fit_simultaneous_focus_curve,
    fit_spectral_flat_field,
    infer_simultaneous_defocus,
    measure_simultaneous_shift,
    select_simultaneous_focus_boxes,
    simultaneous_focus_capture_order,
    unmix_components,
    validate_spectral_flat_field,
)

OFFSETS = [[0, 0], [1, 0], [0, 1], [1, 1]]


@pytest.mark.parametrize("mode", ["mixed", "off", "singular", "masked"])
def test_prepared_unmix_is_bit_exact_and_owns_calibration(mode):
    """Cached geometry/Gram terms preserve every output and cannot stale by mutation."""
    dark, red, green = fields(32)
    maps = {
        "dark": dark.copy(),
        "red_response": red.astype(np.float32),
        "green_response": green.astype(np.float32),
        "valid": np.ones((32, 32), dtype=bool),
    }
    if mode == "singular":
        maps["green_response"] = maps["red_response"].copy()
    if mode == "masked":
        maps["valid"][8:16, 7:14] = False
    planes = dark if mode == "off" else (dark + 0.8 * red + 0.9 * green)
    expected = unmix_components(planes, OFFSETS, maps)
    prepared = rg_simultaneous.PreparedSpectralUnmix(maps, OFFSETS)
    for source in maps.values():
        source[...] = 0
    for _ in range(3):
        actual = prepared.unmix(planes, OFFSETS)
        for first, second in zip(expected, actual, strict=True):
            np.testing.assert_array_equal(first, second)
        # A caller may reuse/mutate returned arrays without corrupting cached support.
        actual[2][...] = False


def test_prepared_unmix_rejects_changed_geometry():
    """A previous ROI or Bayer order must never be silently applied to new RAW."""
    dark, red, green = fields(32)
    maps = {
        "dark": dark,
        "red_response": red,
        "green_response": green,
        "valid": np.ones((32, 32), dtype=bool),
    }
    prepared = rg_simultaneous.PreparedSpectralUnmix(maps, OFFSETS)
    with pytest.raises(ValueError, match="differs from RAW geometry"):
        prepared.unmix(dark[:, :16], OFFSETS)
    with pytest.raises(ValueError, match="differs from RAW geometry"):
        prepared.unmix(dark, list(reversed(OFFSETS)))


def fields(size: int = 64):
    """Build blank source fields with spatial shading and spectral cross-talk."""
    yy, xx = np.mgrid[:size, :size]
    red_shading = 0.82 + 0.18 * (xx + yy) / (2 * (size - 1))
    green_shading = 0.78 + 0.22 * (2 * xx + yy) / (3 * (size - 1))
    dark = np.full((4, size, size), 128, dtype=np.float32)
    red = np.array([32, 220, 230, 860], dtype=np.float32)[:, None, None] * red_shading
    green = np.array([170, 720, 710, 58], dtype=np.float32)[:, None, None] * green_shading
    return dark, red, green


def test_fit_and_validate_source_maps_and_mixed_holdout():
    """Unused R/G fields and a slightly dimmer additive mixed field pass."""
    dark, red_signal, green_signal = fields()
    settings = SimultaneousCalibrationSettings(plane_roi=(0, 0, 64, 64), smoothing_sigma_px=2)
    maps, report = fit_spectral_flat_field(
        dark,
        dark + red_signal,
        dark + green_signal,
        offsets_xy=OFFSETS,
        white_level=4095,
        settings=settings,
    )
    validation = validate_spectral_flat_field(
        dark + red_signal,
        dark + green_signal,
        dark + 0.94 * (red_signal + green_signal),
        offsets_xy=OFFSETS,
        white_level=4095,
        maps=maps,
        settings=settings,
    )
    assert report["signature_condition_number"] < 2
    assert validation["cross_leakage_fraction"] < 0.01
    assert validation["mixed_reconstruction_residual_p95"] < 0.03
    assert validation["mixed_source_scale"]["red"] == pytest.approx(0.94, abs=0.01)
    assert validation["mixed_source_scale"]["green"] == pytest.approx(0.94, abs=0.01)


def test_unmix_returns_two_flat_components():
    """Spatial source response maps flatten a synthetic mixed empty field."""
    dark, red_signal, green_signal = fields()
    settings = SimultaneousCalibrationSettings(plane_roi=(0, 0, 64, 64), smoothing_sigma_px=2)
    maps, _report = fit_spectral_flat_field(
        dark,
        dark + red_signal,
        dark + green_signal,
        offsets_xy=OFFSETS,
        white_level=4095,
        settings=settings,
    )
    components, residual, valid = unmix_components(
        dark + 0.9 * red_signal + 0.8 * green_signal, OFFSETS, maps
    )
    assert np.median(components[0][valid]) == pytest.approx(0.9, abs=0.01)
    assert np.median(components[1][valid]) == pytest.approx(0.8, abs=0.01)
    assert np.percentile(residual[valid], 95) < 0.03


def test_focus_crop_slices_every_calibration_map_before_unmixing():
    """A relative focus ROI owns matching contiguous RAW-map data only."""
    shape = (4, 24, 28)
    source = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    maps = {
        "dark": source,
        "red_response": source + 1000,
        "green_response": source + 2000,
        "valid": np.ones(shape[1:], dtype=bool),
    }
    maps["valid"][5, 7] = False
    cropped = crop_spectral_maps(maps, (3, 2, 16, 16))
    assert cropped["dark"].shape == (4, 16, 16)
    assert cropped["valid"].shape == (16, 16)
    assert cropped["dark"].flags.c_contiguous
    assert cropped["valid"].flags.c_contiguous
    assert cropped["dark"][0, 0, 0] == source[0, 2, 3]
    assert cropped["red_response"][0, 0, 0] == source[0, 2, 3] + 1000
    assert not cropped["valid"][3, 4]
    cropped["dark"][0, 0, 0] = -1
    assert source[0, 2, 3] >= 0


def test_collinear_source_spectra_are_rejected():
    """Two indistinguishable spectra can never create an accepted calibration."""
    dark, red_signal, _green_signal = fields()
    settings = SimultaneousCalibrationSettings(plane_roi=(0, 0, 64, 64), smoothing_sigma_px=2)
    with pytest.raises(ValueError, match="invalid simultaneous R/G area"):
        fit_spectral_flat_field(
            dark,
            dark + red_signal,
            dark + 0.8 * red_signal,
            offsets_xy=OFFSETS,
            white_level=4095,
            settings=settings,
        )


def test_single_mixed_components_produce_tissue_shift():
    """A textured common field yields one robust displacement from two components."""
    size = 640
    noise = np.random.default_rng(42).normal(0, 1, (size, size)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 1.2)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    red = 0.35 + 0.55 * texture
    green = np.roll(red, shift=(2, -5), axis=(0, 1)) * 0.92 + 0.03
    measurement = measure_simultaneous_shift(
        np.stack((red, green)),
        np.ones((size, size), dtype=bool),
        SimultaneousFocusSettings(),
    )
    assert measurement.status == "ready"
    assert measurement.dx is not None
    assert measurement.dy is not None
    assert abs(abs(measurement.dx) - 2.5) <= 0.6
    assert abs(abs(measurement.dy) - 1) <= 0.6
    assert measurement.inlier_patch_count >= 6
    assert sum(measurement.patch_status_counts.values()) == measurement.candidate_patch_count
    assert measurement.patch_status_counts["accepted"] == measurement.accepted_patch_count


def _focus_patch_result(box: PatchBox, *, accepted: bool) -> PatchEvaluation:
    """Build a deterministic per-window result for adaptive-selection tests."""
    values = {
        **box.model_dump(),
        "coverage": 1,
        "status": "accepted" if accepted else "correlation_failed",
        "reason": "synthetic accepted" if accepted else "synthetic refusal",
    }
    if accepted:
        values.update(dx=2.0, dy=0.25, response=0.5)
    return PatchEvaluation.model_validate(values)


def _adaptive_focus_fixture(
    monkeypatch: pytest.MonkeyPatch,
    accepted_ids: set[str],
) -> tuple[np.ndarray, np.ndarray, SimultaneousFocusSettings, list[PatchBox], list[int]]:
    """Install a ranked 4x4 grid and record each correlation batch size."""
    settings = SimultaneousFocusSettings()
    ranked = candidate_boxes((350, 350), settings.core)
    calls: list[int] = []

    monkeypatch.setattr(
        rg_simultaneous,
        "_rank_all_focus_boxes",
        lambda _source, _tissue, _settings: ranked,
    )

    def evaluate(*args, **_kwargs):
        boxes = list(args[4])
        calls.append(len(boxes))
        return [_focus_patch_result(box, accepted=box.patch_id in accepted_ids) for box in boxes]

    monkeypatch.setattr(rg_simultaneous, "evaluate_patches", evaluate)
    components = np.full((2, 700, 700), 0.5, dtype=np.float32)
    valid = np.ones((700, 700), dtype=bool)
    return components, valid, settings, ranked, calls


def test_focus_uses_one_ranked_reserve_window_only_after_primary_refusal(
    monkeypatch: pytest.MonkeyPatch,
):
    """Five primary successes are rescued by one window, then evaluation stops."""
    settings = SimultaneousFocusSettings()
    ranked = candidate_boxes((350, 350), settings.core)
    accepted_ids = {box.patch_id for box in ranked[:5]}
    accepted_ids.add(ranked[8].patch_id)
    components, valid, settings, _ranked, calls = _adaptive_focus_fixture(monkeypatch, accepted_ids)

    measurement = measure_simultaneous_shift(components, valid, settings)

    assert measurement.status == "ready"
    assert measurement.candidate_patch_count == 9
    assert measurement.accepted_patch_count == 6
    assert calls == [8, 1]


def test_focus_does_not_touch_reserve_after_primary_success(
    monkeypatch: pytest.MonkeyPatch,
):
    """Six primary successes retain the existing eight-window fast path exactly."""
    settings = SimultaneousFocusSettings()
    ranked = candidate_boxes((350, 350), settings.core)
    accepted_ids = {box.patch_id for box in ranked[:6]}
    components, valid, settings, _ranked, calls = _adaptive_focus_fixture(monkeypatch, accepted_ids)

    measurement = measure_simultaneous_shift(components, valid, settings)

    assert measurement.status == "ready"
    assert measurement.candidate_patch_count == 8
    assert measurement.accepted_patch_count == 6
    assert calls == [8]


def test_focus_exhausts_reserve_once_then_keeps_normal_refusal(
    monkeypatch: pytest.MonkeyPatch,
):
    """An unrescuable frame refuses once after all remaining grid windows."""
    settings = SimultaneousFocusSettings()
    ranked = candidate_boxes((350, 350), settings.core)
    accepted_ids = {box.patch_id for box in ranked[:5]}
    components, valid, settings, _ranked, calls = _adaptive_focus_fixture(monkeypatch, accepted_ids)

    measurement = measure_simultaneous_shift(components, valid, settings)

    assert measurement.status == "refused"
    assert measurement.reason == "Only 5 tissue windows passed; 6 required"
    assert measurement.candidate_patch_count == 16
    assert measurement.accepted_patch_count == 5
    assert calls == [8, *([1] * 8)]


def test_fixed_calibration_windows_never_use_adaptive_reserve(
    monkeypatch: pytest.MonkeyPatch,
):
    """A fixed calibration grid retains its exact eight-window contract."""
    settings = SimultaneousFocusSettings()
    ranked = candidate_boxes((350, 350), settings.core)
    accepted_ids = {box.patch_id for box in ranked[:5]}
    components, valid, settings, ranked, calls = _adaptive_focus_fixture(monkeypatch, accepted_ids)

    measurement = measure_simultaneous_shift(
        components,
        valid,
        settings,
        ranked[:8],
        adaptive_window_selection=False,
    )

    assert measurement.status == "refused"
    assert measurement.candidate_patch_count == 8
    assert calls == [8]


def _ready_shift(dx: float, dy: float) -> SimultaneousShiftMeasurement:
    """Build one accepted synthetic shift without optical image generation."""
    return SimultaneousShiftMeasurement(
        status="ready",
        reason="synthetic",
        dx=dx,
        dy=dy,
        confidence=0.8,
        candidate_patch_count=12,
        accepted_patch_count=10,
        inlier_patch_count=10,
        tissue_coverage=0.4,
        dx_mad=0.2,
        dy_mad=0.1,
        median_response=0.5,
    )


def test_focus_curve_uses_distinct_holdouts_and_signed_projection():
    """The declared ±32 um curve predicts signed defocus on its held-out points."""
    settings = SimultaneousFocusSettings()
    observations = []
    for index, z_um in enumerate(settings.calibration_positions_um):
        holdout = z_um in settings.holdout_positions_um
        noise = 0.4 if holdout and z_um > 0 else (-0.3 if holdout else 0)
        observations.append(
            SimultaneousFocusObservation(
                capture_id=f"point-{index}",
                z_um=z_um,
                role="holdout" if holdout else "fit",
                measurement=_ready_shift(1.5 + 0.8 * z_um + noise, -0.5 + 0.1 * z_um),
            )
        )
    curve = fit_simultaneous_focus_curve(observations, settings)
    inferred, cross_track = infer_simultaneous_defocus(
        curve, _ready_shift(1.5 + 0.8 * 13, -0.5 + 0.1 * 13)
    )
    assert curve.holdout_max_absolute_error_um < 1
    assert inferred == pytest.approx(13, abs=0.1)
    assert cross_track < 0.1


def test_focus_curve_refuses_defocus_outside_calibrated_range():
    """A runtime estimate cannot extrapolate past either calibrated Z endpoint."""
    settings = SimultaneousFocusSettings()
    observations = [
        SimultaneousFocusObservation(
            capture_id=f"point-{index}",
            z_um=z_um,
            role="holdout" if z_um in settings.holdout_positions_um else "fit",
            measurement=_ready_shift(0.75 * z_um, 0.05 * z_um),
        )
        for index, z_um in enumerate(settings.calibration_positions_um)
    ]
    curve = fit_simultaneous_focus_curve(observations, settings)
    with pytest.raises(ValueError, match="outside the calibrated range"):
        infer_simultaneous_defocus(curve, _ready_shift(0.75 * 40, 0.05 * 40))


def test_focus_selection_is_bounded_and_does_not_mutate_support():
    """RAW focus work stays within its configured window budget and input ownership."""
    size = 640
    yy, xx = np.mgrid[:size, :size]
    texture = (np.sin(xx / 5) + np.cos(yy / 7) + 2) / 4
    components = np.stack((0.25 + 0.7 * texture, 0.3 + 0.6 * texture))
    valid = np.ones((size, size), dtype=bool)
    valid[:2] = False
    original = valid.copy()
    settings = SimultaneousFocusSettings()
    boxes = select_simultaneous_focus_boxes(components, valid, settings)
    assert settings.core.minimum_patch_count <= len(boxes) <= 16
    assert np.array_equal(valid, original)


def test_focus_roi_is_json_roundtrippable_and_has_a_bounded_patch_grid():
    """The default central crop yields sixteen possible downsampled windows."""
    settings = SimultaneousFocusSettings(focus_plane_roi=[350, 350, 700, 700])
    assert settings.checked_focus_roi((1400, 1400)) == (350, 350, 700, 700)
    shape = (700 // settings.processing_downsample,) * 2
    assert len(candidate_boxes(shape, settings.core)) == 16


@pytest.mark.parametrize(
    ("roi", "message"),
    [
        ((800, 800, 700, 700), "exceeds"),
        ((0, 0, 200, 200), "too small"),
    ],
)
def test_focus_roi_rejects_unusable_geometry(roi, message):
    """A focus crop cannot escape its maps or underfill the patch contract."""
    settings = SimultaneousFocusSettings(focus_plane_roi=roi)
    with pytest.raises(ValueError, match=message):
        settings.checked_focus_roi((1400, 1400))


def test_focus_settings_reject_correction_beyond_two_sided_grid():
    """Runtime correction can never exceed the range proved by calibration."""
    with pytest.raises(ValueError, match="correction exceeds"):
        SimultaneousFocusSettings(maximum_correction_um=33)


def test_focus_capture_order_pairs_both_signs_from_focus():
    """The acquisition route starts at zero and balances sign at every radius."""
    assert simultaneous_focus_capture_order(SimultaneousFocusSettings()) == (
        0,
        -32,
        32,
        -24,
        24,
        -16,
        16,
        -8,
        8,
    )


def test_focus_curve_rejects_failed_holdout():
    """Independent holdout error blocks activation instead of widening runtime QC."""
    settings = SimultaneousFocusSettings()
    observations = [
        SimultaneousFocusObservation(
            capture_id=f"point-{index}",
            z_um=z_um,
            role="holdout" if z_um in settings.holdout_positions_um else "fit",
            measurement=_ready_shift(
                z_um + (10 if z_um == settings.holdout_positions_um[-1] else 0),
                0,
            ),
        )
        for index, z_um in enumerate(settings.calibration_positions_um)
    ]
    with pytest.raises(ValueError, match="holdout error"):
        fit_simultaneous_focus_curve(observations, settings)
