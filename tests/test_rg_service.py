from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

import numpy as np
from test_rg_focus_estimator import (
    core_settings,
    field_settings,
    geometry,
    ready_field,
    references,
    shifted_pair,
    white_image,
)

from fast_ofm_core.focus.rg.rg_focus_estimator import RGFocusEstimatorSettings
from fast_ofm_core.focus.rg.rg_focus_model import RGFocusMeasurementPolicy
from fast_ofm_core.focus.rg.rg_service import CALIBRATION_MEDIA_TYPE, FRAME_MEDIA_TYPE
from fast_ofm_core.service import dispatch


def _descriptor(path: Path, media_type: str) -> dict[str, object]:
    return {
        "uri": path.as_uri(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "media_type": media_type,
        "size_bytes": path.stat().st_size,
    }


def _artifacts(directory: Path) -> tuple[Path, Path]:
    field = ready_field()
    red, green = shifted_pair(field)
    white_reference, red_reference, green_reference = references()
    metadata = {
        "white_reference": white_reference.model_dump(mode="json"),
        "red_reference": red_reference.model_dump(mode="json"),
        "green_reference": green_reference.model_dump(mode="json"),
    }
    frame_path = directory / "frame.npz"
    np.savez_compressed(
        frame_path,
        white_frame=white_image(),
        red_plane=red,
        green_plane=green,
        red_source_jpeg8=np.full(red.shape, 100, dtype=np.uint8),
        green_source_jpeg8=np.full(green.shape, 100, dtype=np.uint8),
        red_valid=np.ones(red.shape, dtype=bool),
        green_valid=np.ones(green.shape, dtype=bool),
        metadata_json=np.asarray(json.dumps(metadata, separators=(",", ":"))),
    )
    policy = RGFocusMeasurementPolicy(
        measurement_domain="processed-jpeg-rgb8",
        core=core_settings(),
        estimator=RGFocusEstimatorSettings(),
        tissue_field=field_settings(),
    )
    calibration_path = directory / "calibration.json"
    calibration_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "calibration_id": "synthetic-calibration",
                "geometry": geometry(),
                "measurement_policy": policy.model_dump(mode="json"),
            }
        )
    )
    return frame_path, calibration_path


def _request(frame: Path, calibration: Path) -> dict[str, object]:
    return {
        "protocol_version": "1.0",
        "request_id": str(uuid.uuid4()),
        "operation": "rg.measure",
        "timeout_ms": 30_000,
        "payload": {
            "frame": _descriptor(frame, FRAME_MEDIA_TYPE),
            "calibration": _descriptor(calibration, CALIBRATION_MEDIA_TYPE),
            "position_um": {"x_um": 10.0, "y_um": 20.0, "z_um": 30.0},
        },
    }


def test_rg_measure_accepts_verified_artifacts(tmp_path: Path) -> None:
    frame, calibration = _artifacts(tmp_path)
    response = dispatch(_request(frame, calibration), artifact_roots=(tmp_path,))
    assert response["status"] == "completed", response
    result = response["result"]
    assert result["measurement_status"] == "accepted"
    assert result["shift_px"]["dx_px"] == 3
    assert result["shift_px"]["dy_px"] == -2
    assert result["correction_status"] == "not_requested"


def test_rg_measure_refuses_unconfigured_root_and_digest_mismatch(tmp_path: Path) -> None:
    frame, calibration = _artifacts(tmp_path)
    denied = dispatch(_request(frame, calibration))
    assert denied["status"] == "refused"
    assert denied["error"]["code"] == "ARTIFACT_ACCESS_DENIED"

    request = _request(frame, calibration)
    request["payload"]["frame"]["sha256"] = "0" * 64
    mismatch = dispatch(request, artifact_roots=(tmp_path,))
    assert mismatch["status"] == "refused"
    assert mismatch["error"]["code"] == "ARTIFACT_DIGEST_MISMATCH"
