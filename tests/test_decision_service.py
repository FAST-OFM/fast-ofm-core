from __future__ import annotations

import uuid

from test_rg_focus_control import measurement
from test_rg_focus_model import fit, observations

from fast_ofm_core.focus.rg.rg_focus_control import RGFocusControlSettings
from fast_ofm_core.service import dispatch


def test_rg_decide_returns_bounded_move_from_json_models() -> None:
    request = {
        "protocol_version": "1.0",
        "request_id": str(uuid.uuid4()),
        "operation": "rg.decide",
        "timeout_ms": 1000,
        "payload": {
            "profile": fit(observations()).model_dump(mode="json"),
            "measurement": measurement(dx=1.85, dy=-1.2).model_dump(mode="json"),
            "control": RGFocusControlSettings().model_dump(mode="json"),
            "iteration": 0,
            "total_correction_um": 0.0,
        },
    }
    response = dispatch(request)
    assert response["status"] == "completed"
    assert response["result"]["status"] == "move"
