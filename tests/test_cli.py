from __future__ import annotations

import json
import uuid

import pytest

from fast_ofm_core.cli import main


def request(operation: str) -> dict[str, object]:
    return {
        "protocol_version": "1.0",
        "request_id": str(uuid.uuid4()),
        "operation": operation,
        "timeout_ms": 1000,
        "payload": {},
    }


def test_run_request_executes_one_protocol_envelope(tmp_path, capsys) -> None:
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request("core.capabilities")), encoding="utf-8")

    main(["run-request", str(path), "--artifact-root", str(tmp_path)])

    response = json.loads(capsys.readouterr().out)
    assert response["status"] == "completed"
    assert response["result"]["protocol_versions"] == ["1.0"]


def test_run_request_returns_nonzero_for_protocol_failure(tmp_path, capsys) -> None:
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request("stitching.run")), encoding="utf-8")

    with pytest.raises(SystemExit) as captured:
        main(["run-request", str(path), "--artifact-root", str(tmp_path)])

    assert captured.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] in {"failed", "refused"}
