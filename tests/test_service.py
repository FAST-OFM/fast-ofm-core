from __future__ import annotations

import io
import json
import uuid

from fast_ofm_core.service import dispatch, serve_jsonl


def request(operation: str = "core.capabilities") -> dict[str, object]:
    return {
        "protocol_version": "1.0",
        "request_id": str(uuid.uuid4()),
        "operation": operation,
        "timeout_ms": 1000,
        "payload": {},
    }


def test_capability_handshake_is_independent(monkeypatch) -> None:
    from fast_ofm_core.stitching import service as stitching_service

    monkeypatch.setattr(stitching_service, "find_worker", lambda: None)
    response = dispatch(request())
    assert response["status"] == "completed"
    assert response["result"]["operations"] == [
        "core.capabilities",
        "rg.measure",
        "rg.decide",
        "rg.flat_field.fit",
        "rg.flat_field.apply",
        "rg.flat_field.validate",
        "rg.tissue_field.prepare",
        "rg.simultaneous.calibrate",
        "rg.simultaneous.focus.sample",
        "rg.simultaneous.focus.plan",
        "rg.simultaneous.focus.fit",
        "rg.simultaneous.focus.evaluate",
        "calibration.fit",
        "calibration.validate",
        "focus.surface.fit",
        "focus.surface.predict",
        "planning.route",
    ]


def test_unavailable_operation_fails_closed(monkeypatch) -> None:
    from fast_ofm_core.stitching import service as stitching_service

    monkeypatch.setattr(stitching_service, "find_worker", lambda: None)
    response = dispatch(request("stitching.run"))
    assert response["status"] == "failed"
    assert response["error"]["code"] == "CAPABILITY_UNAVAILABLE"


def test_stitching_retry_policy_crosses_protocol_boundary(monkeypatch) -> None:
    """Transient worker failures retain their retryable marker."""
    from fast_ofm_core import stitching
    from fast_ofm_core.stitching import service as stitching_service

    monkeypatch.setattr(stitching_service, "find_worker", lambda: "/worker")

    def fail(*_args, **_kwargs):
        raise stitching.StitchingServiceError(
            "STITCHING_TIMEOUT", "worker timed out", retryable=True
        )

    monkeypatch.setattr(stitching, "run_stitching_payload", fail)
    response = dispatch(request("stitching.run"))

    assert response["status"] == "failed"
    assert response["error"] == {
        "code": "STITCHING_TIMEOUT",
        "message": "worker timed out",
        "retryable": True,
    }


def test_jsonl_keeps_one_response_per_input_line() -> None:
    source = io.StringIO(json.dumps(request()) + "\nnot-json\n")
    destination = io.StringIO()
    assert serve_jsonl(source, destination) == 0
    responses = [json.loads(line) for line in destination.getvalue().splitlines()]
    assert [item["status"] for item in responses] == ["completed", "failed"]
    assert responses[1]["error"]["code"] == "INVALID_JSON"
