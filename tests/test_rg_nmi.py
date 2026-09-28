"""Parity and boundary tests for the production optional NMI extension."""

import re
import subprocess
import sys

import cv2
import numpy as np
import pytest

from fast_ofm_core.focus.rg import rg_focus_core as core
from fast_ofm_core.focus.rg.rg_simultaneous import (
    SimultaneousFocusSettings,
)


def edge_cases():
    """Reproduce the 39 bounded cases used for the Pi candidate acceptance."""
    settings = SimultaneousFocusSettings().core.model_copy(
        update={
            "maximum_absolute_shift_px": 12,
            "maximum_absolute_orthogonal_shift_px": 3,
        }
    )
    rng = np.random.default_rng(20260909)
    image = cv2.GaussianBlur(rng.uniform(20, 220, (64, 64)).astype(np.float32), (0, 0), 0.8)
    full = np.ones(image.shape, dtype=bool)
    cases = []
    for dx in (-12, -5, 0, 5, 12):
        for dy in (-3, 0, 3):
            other = cv2.warpAffine(
                image,
                np.float32([[1, 0, dx], [0, 1, dy]]),
                (64, 64),
                borderMode=cv2.BORDER_REFLECT,
            )
            cases.append((f"shift_{dx}_{dy}", image, other, full, settings))
    yy, xx = np.mgrid[:64, :64]
    for name, mask in (
        ("holes", rng.random((64, 64)) > 0.15),
        ("edge_tissue", xx < 28),
        ("thin_tissue", xx < 3),
        ("empty", np.zeros((64, 64), dtype=bool)),
    ):
        cases.append((name, image, np.roll(image, 3, axis=1), mask, settings))
    for count in (15, 16, 20, 32, 48):
        mask = np.zeros((64, 64), dtype=bool)
        mask.ravel()[rng.choice(4096, count, replace=False)] = True
        cases.append((f"sparse_{count}", image, image.copy(), mask, settings))
    for name, plane in (
        ("constant", np.full((64, 64), 80, dtype=np.float32)),
        ("saturated", np.full((64, 64), 255, dtype=np.float32)),
        ("stripes", ((xx % 8 < 4) * 100 + 50).astype(np.float32)),
        ("checker_ties", (((xx // 4 + yy // 4) % 2) * 100 + 50).astype(np.float32)),
    ):
        cases.append((name, plane, plane.copy(), full, settings))
    bad = image.copy()
    bad[20, 20] = np.nan
    cases.append(("nan_inside", bad, image, full, settings))
    hidden = full.copy()
    hidden[20, 20] = False
    cases.append(("nan_outside", bad, image, hidden, settings))
    cases.append(("wrong_mask_dtype", image, image, full.astype(np.uint8), settings))
    cases.append(("wrong_shape", image, image[:, :-1], full, settings))
    cases.append(("range_exceeds_patch", image[:8, :8], image[:8, :8], full[:8, :8], settings))
    cases.append(("negative_contrast", image, 255 - image, full, settings))
    cases.append(
        (
            "unrelated",
            image,
            rng.uniform(20, 220, (64, 64)).astype(np.float32),
            full,
            settings,
        )
    )
    # Enough total variation but a uniform admissible overlap in the first candidate.
    corner = np.full((64, 64), 80, dtype=np.float32)
    corner[:3] = rng.uniform(20, 220, (3, 64))
    corner[:, :12] = rng.uniform(20, 220, (64, 12))
    cases.append(("zero_entropy_overlap", corner, corner, full, settings))
    cluster = np.zeros_like(full)
    cluster[28:32, 28:33] = True
    cases.append(("small_connected_overlap", image, image.copy(), cluster, settings))
    # Two-valued quantisation globally, but the first admissible overlap is constant.
    tiny_texture = np.full((64, 64), 80, dtype=np.float32)
    tiny_texture[:2, :24] = 180
    entropy_settings = settings.model_copy(update={"minimum_shifted_support_fraction": 0.5})
    cases.append(
        (
            "constant_admissible_overlap",
            tiny_texture,
            tiny_texture,
            full,
            entropy_settings,
        )
    )
    cases.append(
        (
            "zero_entropy_first_overlap",
            tiny_texture,
            tiny_texture[::-1, ::-1],
            full,
            entropy_settings,
        )
    )

    return cases


@pytest.mark.parametrize("case", edge_cases(), ids=lambda case: case[0])
def test_native_matches_portable_shift_and_refusal(case, monkeypatch):
    """Keep the whole registration result and first refusal unchanged."""
    native = pytest.importorskip("fast_ofm_core.focus.rg._rg_nmi")
    name, first, second, mask, settings = case
    monkeypatch.setattr(core, "_native_candidates", None)
    try:
        expected = core.mutual_information_shift_masked(
            first, second, mask, settings, subpixel_refinement=True
        )
    except (ValueError, TypeError) as exc:
        monkeypatch.setattr(core, "_native_candidates", native.candidates)
        with pytest.raises(type(exc), match="^" + re.escape(str(exc)) + "$"):
            core.mutual_information_shift_masked(
                first, second, mask, settings, subpixel_refinement=True
            )
    else:
        monkeypatch.setattr(core, "_native_candidates", native.candidates)
        actual = core.mutual_information_shift_masked(
            first, second, mask, settings, subpixel_refinement=True
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-9, err_msg=name)


@pytest.mark.parametrize("bins", [8, 32, 256])
def test_native_grid_matches_every_portable_candidate(bins):
    """Check every score and coordinate, not only the selected peak."""
    native = pytest.importorskip("fast_ofm_core.focus.rg._rg_nmi")
    rng = np.random.default_rng(194)
    first = rng.integers(0, bins, (48, 64), dtype=np.uint8)
    second = rng.integers(0, bins, (48, 64), dtype=np.uint8)
    support = (rng.random(first.shape) > 0.1).astype(np.uint8)
    args = (first, second, support, bins, 12, 3, 0.5)
    np.testing.assert_allclose(
        native.candidates(*args),
        core._portable_nmi_candidates(*args),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "case",
    [
        "dtype",
        "strided",
        "empty",
        "shape",
        "bins_small",
        "bins_large",
        "negative_range",
        "large_range",
        "fraction_nan",
        "fraction_zero",
        "fraction_large",
        "value_outside_bins",
        "dimensions",
    ],
)
def test_native_rejects_invalid_buffers_and_parameters(case):
    """Reject unsafe extension inputs before native indexing or allocation."""
    native = pytest.importorskip("fast_ofm_core.focus.rg._rg_nmi")
    plane = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64) % 32
    args = [plane, plane.copy(), np.ones_like(plane), 32, 12, 3, 0.5]
    changes = {
        "dtype": (0, plane.astype(np.float32)),
        "strided": (0, plane[:, ::-1]),
        "empty": (0, plane[:0]),
        "shape": (1, plane[:32]),
        "bins_small": (3, 0),
        "bins_large": (3, 257),
        "negative_range": (4, -1),
        "large_range": (4, 2**30),
        "fraction_nan": (6, np.nan),
        "fraction_zero": (6, 0),
        "fraction_large": (6, 1.1),
        "value_outside_bins": (0, np.full_like(plane, 255)),
        "dimensions": (0, plane.ravel()),
    }
    index, value = changes[case]
    args[index] = value
    with pytest.raises((ValueError, BufferError)):
        native.candidates(*args)


def test_native_error_is_not_silently_retried(monkeypatch):
    """A loaded extension's computation failure must not fall back or pass QC."""

    def fail(*_args):
        raise MemoryError("bounded allocation failed")

    monkeypatch.setattr(core, "_native_candidates", fail)
    image = np.random.default_rng(21).uniform(20, 200, (64, 64))
    with pytest.raises(MemoryError, match="bounded allocation failed"):
        core.mutual_information_shift_masked(
            image,
            image,
            np.ones(image.shape, dtype=bool),
            SimultaneousFocusSettings().core,
        )


def test_package_without_extension_keeps_portable_calculation():
    """A missing optional binary is explicit and does not prevent package use."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules['fast_ofm_core.focus.rg._rg_nmi'] = None
import numpy as np
from fast_ofm_core.focus.rg import rg_focus_core as core
assert core.NMI_BACKEND == 'opencv-exhaustive'
settings = core.RGFocusCoreSettings(maximum_absolute_shift_px=12)
image = np.random.default_rng(21).uniform(20, 200, (64, 64))
result = core.mutual_information_shift_masked(image, image, np.ones(image.shape, bool), settings)
assert result[:2] == (0, 0)
""",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Native NMI extension unavailable" in result.stderr
