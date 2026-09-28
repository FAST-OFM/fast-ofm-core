from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np
import pytest

from fast_ofm_core.focus.rg.flat_field_service import (
    INPUT_MEDIA_TYPE,
    RESULT_MEDIA_TYPE,
)
from fast_ofm_core.focus.rg.rg_flat_field import FlatFieldSettings
from fast_ofm_core.service import dispatch


def _descriptor(path: Path, media_type: str) -> dict[str, object]:
    return {
        "uri": path.as_uri(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "media_type": media_type,
        "size_bytes": path.stat().st_size,
        "role": "test-input",
    }


def _request(operation: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "protocol_version": "1.0",
        "request_id": str(uuid.uuid4()),
        "operation": operation,
        "timeout_ms": 30_000,
        "payload": payload,
    }


def _field() -> tuple[np.ndarray, np.ndarray, FlatFieldSettings]:
    yy, xx = np.mgrid[:96, :96]
    shading = 90 + 70 * np.exp(-((xx - 48) ** 2 + (yy - 48) ** 2) / 4000)
    dark = np.full((1, 96, 96), 5, dtype=np.float32)
    flat = dark + shading[None].astype(np.float32)
    settings = FlatFieldSettings(
        measurement_domain="processed-jpeg-rgb8",
        processing_roi=(0, 0, 96, 96),
        minimum_signal_dn=8,
        maximum_dark_signal_dn=32,
        smoothing_sigma_px=2.0,
    )
    return flat, dark, settings


def test_flat_field_fit_apply_and_validate_cross_the_artifact_boundary(
    tmp_path: Path,
) -> None:
    flat, dark, settings = _field()
    fit_input = tmp_path / "fit-input.npz"
    np.savez_compressed(fit_input, flat=flat, dark=dark)
    fitted = dispatch(
        _request(
            "rg.flat_field.fit",
            {
                "input": _descriptor(fit_input, INPUT_MEDIA_TYPE),
                "settings": settings.model_dump(mode="json"),
            },
        ),
        artifact_roots=(tmp_path,),
    )
    assert fitted["status"] == "completed", fitted
    maps_descriptor = fitted["result"]["maps"]
    assert maps_descriptor["media_type"] == RESULT_MEDIA_TYPE
    maps_path = Path(maps_descriptor["uri"].removeprefix("file://"))
    with np.load(maps_path, allow_pickle=False) as archive:
        maps = {name: np.array(archive[name], copy=True) for name in archive.files}
    assert set(maps) == {"dark", "gain", "valid"}

    apply_input = tmp_path / "apply-input.npz"
    np.savez_compressed(apply_input, image=flat, **maps)
    applied = dispatch(
        _request(
            "rg.flat_field.apply",
            {"input": _descriptor(apply_input, INPUT_MEDIA_TYPE)},
        ),
        artifact_roots=(tmp_path,),
    )
    assert applied["status"] == "completed", applied
    corrected_path = Path(applied["result"]["corrected"]["uri"].removeprefix("file://"))
    with np.load(corrected_path, allow_pickle=False) as archive:
        assert np.asarray(archive["valid"]).all()
        assert np.median(archive["corrected"]) == pytest.approx(np.median(flat - dark), rel=0.01)

    heldout = flat + np.random.default_rng(4).normal(0, 0.1, flat.shape)
    validation_input = tmp_path / "validation-input.npz"
    np.savez_compressed(validation_input, heldout=heldout, **maps)
    validated = dispatch(
        _request(
            "rg.flat_field.validate",
            {
                "input": _descriptor(validation_input, INPUT_MEDIA_TYPE),
                "settings": settings.model_dump(mode="json"),
                "fit_report": fitted["result"]["report"],
            },
        ),
        artifact_roots=(tmp_path,),
    )
    assert validated["status"] == "completed", validated
    assert max(validated["result"]["quality"]["after"]["cv"]) < 0.005


def test_flat_field_service_refuses_digest_mismatch(tmp_path: Path) -> None:
    flat, dark, settings = _field()
    path = tmp_path / "fit-input.npz"
    np.savez_compressed(path, flat=flat, dark=dark)
    descriptor = _descriptor(path, INPUT_MEDIA_TYPE)
    descriptor["sha256"] = "0" * 64
    response = dispatch(
        _request(
            "rg.flat_field.fit",
            {"input": descriptor, "settings": settings.model_dump(mode="json")},
        ),
        artifact_roots=(tmp_path,),
    )
    assert response["status"] == "refused"
    assert response["error"]["code"] == "ARTIFACT_DIGEST_MISMATCH"
