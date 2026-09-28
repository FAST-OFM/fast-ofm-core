from __future__ import annotations

import json
from pathlib import Path

import pytest

from fast_ofm_core.service import dispatch

EXAMPLES = Path(__file__).parents[1] / "examples" / "protocol" / "v1"


@pytest.mark.parametrize("name", ["planning-route", "focus-surface"])
def test_reference_request_response_pair(name: str) -> None:
    """Published synthetic fixtures remain executable and specimen-free."""
    request = json.loads((EXAMPLES / f"{name}.request.json").read_text())
    expected = json.loads((EXAMPLES / f"{name}.response.json").read_text())
    actual = dispatch(request)
    if name == "focus-surface":
        for field in ("slope_um_per_um", "validation_error_um"):
            assert actual["result"].pop(field) == pytest.approx(
                expected["result"].pop(field), abs=1e-12
            )
    assert actual == expected
    serialized = json.dumps({"request": request, "response": expected}).lower()
    for forbidden in ("patient", "specimen", "hostname", "password", "token"):
        assert forbidden not in serialized
