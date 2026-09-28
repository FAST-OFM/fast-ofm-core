from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np
from test_rg_focus_estimator import (
    core_settings,
    field_settings,
    geometry,
    references,
    white_image,
)

from fast_ofm_core.focus.rg.tissue_field_service import (
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
        "role": "rg-tissue-field-input",
    }


def _request(path: Path) -> dict[str, object]:
    reference, _, _ = references()
    return {
        "protocol_version": "1.0",
        "request_id": str(uuid.uuid4()),
        "operation": "rg.tissue_field.prepare",
        "timeout_ms": 30_000,
        "payload": {
            "input": _descriptor(path),
            "reference": reference.model_dump(mode="json"),
            "geometry": geometry(),
            "settings": field_settings().model_dump(mode="json"),
            "core_settings": core_settings().model_dump(mode="json"),
        },
    }


def test_tissue_field_prepare_crosses_verified_artifact_boundary(tmp_path: Path) -> None:
    path = tmp_path / "white.npz"
    np.savez_compressed(path, white_frame=white_image())
    response = dispatch(_request(path), artifact_roots=(tmp_path,))
    assert response["status"] == "completed", response
    result = response["result"]
    assert result["field"]["metrics"]["status"] == "ready"
    assert result["field"]["boxes"]
    assert result["arrays"]["media_type"] == RESULT_MEDIA_TYPE
    output = Path(result["arrays"]["uri"].removeprefix("file://"))
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == {"mask", "white_common", "overlay"}
        assert archive["mask"].dtype == np.bool_
        assert archive["overlay"].shape[-1] == 3


def test_tissue_field_prepare_refuses_digest_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "white.npz"
    np.savez_compressed(path, white_frame=white_image())
    request = _request(path)
    request["payload"]["input"]["sha256"] = "0" * 64
    response = dispatch(request, artifact_roots=(tmp_path,))
    assert response["status"] == "refused"
    assert response["error"]["code"] == "ARTIFACT_DIGEST_MISMATCH"
