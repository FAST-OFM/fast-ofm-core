from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path

from test_rg_focus_model import observations, stationary_observations

from fast_ofm_core.focus.rg.calibration_service import SERIES_MEDIA_TYPE
from fast_ofm_core.focus.rg.rg_focus_model import (
    RGFocusModelSettings,
    jpeg_measurement_draft_policy,
)
from fast_ofm_core.service import dispatch


def _series(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "profile_id": "profile-v1",
                "source_series_id": "series-v1",
                "points": [row.model_dump(mode="json") for row in observations()],
                "settings": RGFocusModelSettings(maximum_holdout_error_um=2.0).model_dump(
                    mode="json"
                ),
                "reference_white_score": 100.0,
                "stationary_observations": [
                    row.model_dump(mode="json") for row in stationary_observations()
                ],
                "compatibility": {"geometry_id": "geometry-a"},
                "source_evidence": {"capture_manifest": "sha256:synthetic"},
                "measurement_policy": jpeg_measurement_draft_policy().model_dump(mode="json"),
            }
        )
    )


def test_calibration_fit_returns_reloadable_valid_profile(tmp_path: Path) -> None:
    path = tmp_path / "series.json"
    _series(path)
    request = {
        "protocol_version": "1.0",
        "request_id": str(uuid.uuid4()),
        "operation": "calibration.fit",
        "timeout_ms": 30_000,
        "payload": {
            "series": {
                "uri": path.as_uri(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "media_type": SERIES_MEDIA_TYPE,
                "size_bytes": path.stat().st_size,
            }
        },
    }
    response = dispatch(request, artifact_roots=(tmp_path,))
    assert response["status"] == "completed", response
    assert response["result"]["calibration_id"] == "profile-v1"
    assert response["result"]["calibration_status"] == "valid"
    assert response["result"]["profile"]["measurement_policy"]["measurement_domain"] == (
        "processed-jpeg-rgb8"
    )


def test_calibration_validate_recomputes_profile_evidence_and_focus_budget(
    tmp_path: Path,
) -> None:
    path = tmp_path / "series.json"
    _series(path)
    fit = dispatch(
        {
            "protocol_version": "1.0",
            "request_id": str(uuid.uuid4()),
            "operation": "calibration.fit",
            "timeout_ms": 30_000,
            "payload": {
                "series": {
                    "uri": path.as_uri(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "media_type": SERIES_MEDIA_TYPE,
                    "size_bytes": path.stat().st_size,
                }
            },
        },
        artifact_roots=(tmp_path,),
    )
    profile = fit["result"]["profile"]
    response = dispatch(
        {
            "protocol_version": "1.0",
            "request_id": str(uuid.uuid4()),
            "operation": "calibration.validate",
            "timeout_ms": 5_000,
            "payload": {"profile": profile, "focus_tolerance_um": 2.0},
        }
    )
    assert response["status"] == "completed", response
    assert response["result"]["profile"] == profile
    assert response["result"]["profile_status"] == "valid"
    assert 0 < response["result"]["effective_residual_tolerance_um"] <= 2.0


def test_calibration_validate_refuses_tampered_empirical_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "series.json"
    _series(path)
    fit_request = {
        "protocol_version": "1.0",
        "request_id": str(uuid.uuid4()),
        "operation": "calibration.fit",
        "timeout_ms": 30_000,
        "payload": {
            "series": {
                "uri": path.as_uri(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "media_type": SERIES_MEDIA_TYPE,
                "size_bytes": path.stat().st_size,
            }
        },
    }
    profile = dispatch(fit_request, artifact_roots=(tmp_path,))["result"]["profile"]
    profile["empirical_cross_track_limit_px"] += 1.0
    response = dispatch(
        {
            "protocol_version": "1.0",
            "request_id": str(uuid.uuid4()),
            "operation": "calibration.validate",
            "timeout_ms": 5_000,
            "payload": {"profile": profile},
        }
    )
    assert response["status"] == "refused"
    assert response["error"]["code"] == "INVALID_PROFILE"
