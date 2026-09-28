from __future__ import annotations

import uuid

import pytest
from test_focus_surface import observation, prediction_request

from fast_ofm_core.service import dispatch


def payload() -> dict[str, object]:
    return {
        "anchors": [
            {
                "anchor_id": f"a-{index}",
                "position_um": {"x_um": x, "y_um": y, "z_um": 10 + 0.001 * x + 0.002 * y},
                "source": "rg",
                "confidence": 0.95,
            }
            for index, (x, y) in enumerate(
                ((-500.0, -500.0), (500.0, -500.0), (-500.0, 500.0), (500.0, 500.0))
            )
        ],
        "limits": {
            "maximum_support_distance_um": 2000.0,
            "maximum_abs_slope": 0.05,
            "maximum_residual_um": 2.0,
            "maximum_extrapolation_um": 500.0,
            "maximum_prediction_delta_um": 40.0,
        },
    }


def call(operation: str, value: dict[str, object]) -> dict[str, object]:
    return dispatch(
        {
            "protocol_version": "1.0",
            "request_id": str(uuid.uuid4()),
            "operation": operation,
            "timeout_ms": 1000,
            "payload": value,
        }
    )


def test_fit_identifies_validated_support_without_global_model() -> None:
    response = call("focus.surface.fit", payload())
    assert response["status"] == "completed"
    assert response["result"]["anchor_count"] == 4
    assert response["result"]["method"] == "bounded_local_on_demand"


def test_predict_returns_bounded_plane_result() -> None:
    value = payload()
    value["target_um"] = {"x_um": 0.0, "y_um": 0.0, "z_um": 10.0}
    response = call("focus.surface.predict", value)
    assert response["status"] == "completed"
    assert response["result"]["status"] == "predicted"
    assert response["result"]["predicted_z_um"] == 10.0


def test_predict_refuses_missing_support_without_process_failure() -> None:
    value = payload()
    value["target_um"] = {"x_um": 10000.0, "y_um": 10000.0, "z_um": 10.0}
    response = call("focus.surface.predict", value)
    assert response["status"] == "completed"
    assert response["result"]["status"] == "refused"
    assert response["result"]["refusal_code"] == "FOCUS_SURFACE_UNAVAILABLE"


def test_predict_refuses_empty_warmup_support_without_protocol_failure() -> None:
    value = payload()
    value["anchors"] = []
    value["target_um"] = {"x_um": 0.0, "y_um": 0.0, "z_um": 10.0}
    response = call("focus.surface.predict", value)
    assert response["status"] == "completed"
    assert response["result"] == {
        "position_um": {"x_um": 0.0, "y_um": 0.0},
        "status": "refused",
        "support_count": 0,
        "reason": "no nearby measured anchor",
        "refusal_code": "FOCUS_SURFACE_UNAVAILABLE",
    }


def test_predict_reports_diagnostics_used_by_the_gpl_adapter() -> None:
    value = payload()
    value["limits"].update(
        minimum_plane_points=4,
        maximum_neighbors=8,
        maximum_condition_number=500.0,
    )
    value["target_um"] = {"x_um": 0.0, "y_um": 0.0, "z_um": 10.0}
    response = call("focus.surface.predict", value)
    result = response["result"]
    assert response["status"] == "completed"
    assert result["fit_residual_um"] == 0.0
    assert result["validation_error_um"] == pytest.approx(0.0, abs=1e-12)
    assert result["extrapolation_um"] == 0.0
    assert result["slope_um_per_um"] > 0


def test_exact_focus_surface_contract_roundtrips_through_process_service() -> None:
    value = {
        "observations": [observation().model_dump(mode="json")],
        "request": prediction_request(
            target_x_um=-125.5,
            target_y_um=250.25,
            predicted_monotonic_s=20.0,
        ).model_dump(mode="json"),
    }
    response = call("focus.surface.predict", value)
    assert response["status"] == "completed", response
    assert response["result"]["status"] == "usable"
    assert response["result"]["target_z_um"] == 0.0
    assert response["result"]["support_observation_ids"] == ["observation-1"]
