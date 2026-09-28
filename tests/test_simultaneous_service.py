from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np
from test_rg_simultaneous import OFFSETS, _ready_shift, fields

from fast_ofm_core.focus.rg.rg_simultaneous import (
    SimultaneousCalibrationSettings,
    SimultaneousFocusObservation,
    SimultaneousFocusSettings,
)
from fast_ofm_core.focus.rg.simultaneous_service import (
    INPUT_MEDIA_TYPE,
    RESULT_MEDIA_TYPE,
)
from fast_ofm_core.service import dispatch


def _descriptor(path: Path) -> dict[str, object]:
    return {
        "uri": path.as_uri(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "media_type": INPUT_MEDIA_TYPE,
        "size_bytes": path.stat().st_size,
        "role": "rg-simultaneous-input",
    }


def _request(operation: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "protocol_version": "1.0",
        "request_id": str(uuid.uuid4()),
        "operation": operation,
        "timeout_ms": 30_000,
        "payload": payload,
    }


def test_simultaneous_calibration_returns_verified_maps_and_holdout_report(
    tmp_path: Path,
) -> None:
    dark, red, green = fields(64)
    path = tmp_path / "simultaneous-calibration.npz"
    np.savez_compressed(
        path,
        dark=dark,
        red_fit=dark + red,
        green_fit=dark + green,
        red_holdout=dark + red,
        green_holdout=dark + green,
        mixed_holdout=dark + 0.94 * (red + green),
    )
    settings = SimultaneousCalibrationSettings(plane_roi=(0, 0, 64, 64), smoothing_sigma_px=2)
    response = dispatch(
        _request(
            "rg.simultaneous.calibrate",
            {
                "input": _descriptor(path),
                "settings": settings.model_dump(mode="json"),
                "channel_offsets_xy": OFFSETS,
                "white_level": 4095,
            },
        ),
        artifact_roots=(tmp_path,),
    )
    assert response["status"] == "completed", response
    result = response["result"]
    assert result["arrays"]["media_type"] == RESULT_MEDIA_TYPE
    assert result["validation"]["mixed_source_scale"]["red"] > 0.9
    output = Path(result["arrays"]["uri"].removeprefix("file://"))
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "dark",
            "red_response",
            "green_response",
            "valid",
            "components",
            "component_valid",
        }


def test_simultaneous_focus_plan_fit_and_evaluate_are_json_only() -> None:
    settings = SimultaneousFocusSettings()
    planned = dispatch(
        _request(
            "rg.simultaneous.focus.plan",
            {"settings": settings.model_dump(mode="json")},
        )
    )
    assert planned["status"] == "completed"
    assert planned["result"]["capture_order_um"][:3] == [0, -32, 32]

    observations = [
        SimultaneousFocusObservation(
            capture_id=f"point-{index}",
            z_um=z_um,
            role="holdout" if z_um in settings.holdout_positions_um else "fit",
            measurement=_ready_shift(1.5 + 0.8 * z_um, -0.5 + 0.1 * z_um),
        )
        for index, z_um in enumerate(settings.calibration_positions_um)
    ]
    fitted = dispatch(
        _request(
            "rg.simultaneous.focus.fit",
            {
                "settings": settings.model_dump(mode="json"),
                "observations": [row.model_dump(mode="json") for row in observations],
            },
        )
    )
    assert fitted["status"] == "completed", fitted
    evaluated = dispatch(
        _request(
            "rg.simultaneous.focus.evaluate",
            {
                "curve": fitted["result"]["curve"],
                "measurement": _ready_shift(1.5 + 0.8 * 13, -0.5 + 0.1 * 13).model_dump(
                    mode="json"
                ),
                "settings": settings.model_dump(mode="json"),
                "peripheral_search": None,
            },
        )
    )
    assert evaluated["status"] == "completed", evaluated
    assert abs(evaluated["result"]["defocus_um"] - 13) < 0.1
