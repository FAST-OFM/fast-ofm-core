from __future__ import annotations

import json
import uuid
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from test_focus_surface import observation, prediction_request
from test_surface_service import payload as surface_payload

from fast_ofm_core.service import OPERATIONS, capabilities, dispatch

SCHEMA_DIRECTORY = Path(__file__).parents[1] / "src" / "fast_ofm_core" / "protocol" / "v1"


def schemas() -> dict[str, dict[str, object]]:
    return {
        path.name: json.loads(path.read_text()) for path in sorted(SCHEMA_DIRECTORY.glob("*.json"))
    }


def registry_for(values: dict[str, dict[str, object]]) -> Registry:
    return Registry().with_resources(
        (value["$id"], Resource.from_contents(value)) for value in values.values()
    )


def validate(name: str, instance: object) -> None:
    values = schemas()
    validator = Draft202012Validator(
        values[name],
        registry=registry_for(values),
        format_checker=FormatChecker(),
    )
    validator.validate(instance)


def validate_definition(name: str, definition: str, instance: object) -> None:
    values = schemas()
    validator = Draft202012Validator(
        {"$ref": f"{values[name]['$id']}#/$defs/{definition}"},
        registry=registry_for(values),
        format_checker=FormatChecker(),
    )
    validator.validate(instance)


def test_every_protocol_schema_is_valid_draft_2020_12() -> None:
    for schema in schemas().values():
        Draft202012Validator.check_schema(schema)


def test_request_schema_lists_every_dispatcher_operation() -> None:
    """The executable dispatcher and published envelope cannot drift apart."""
    operations = schemas()["request.schema.json"]["properties"]["operation"]["enum"]
    assert tuple(operations) == OPERATIONS


def test_capabilities_and_envelopes_validate_together() -> None:
    request = {
        "protocol_version": "1.0",
        "request_id": str(uuid.uuid4()),
        "operation": "core.capabilities",
        "timeout_ms": 1000,
        "payload": {},
    }
    response = dispatch(request)
    validate("request.schema.json", request)
    validate("response.schema.json", response)
    validate("capabilities.schema.json", capabilities())


def test_new_rg_process_payload_schemas_cover_the_published_boundaries() -> None:
    """Tissue-field and simultaneous RAW definitions accept their exact shapes."""
    artifact = {
        "uri": "file:///tmp/input.npz",
        "sha256": "0" * 64,
        "media_type": "application/vnd.fast-ofm.test+npz",
        "size_bytes": 123,
    }
    validate_definition(
        "rg-tissue-field.schema.json",
        "request",
        {
            "input": artifact,
            "reference": {},
            "geometry": {},
            "settings": {},
            "core_settings": {},
        },
    )
    validate_definition(
        "rg-tissue-field.schema.json",
        "result",
        {"field": {}, "arrays": artifact},
    )
    settings = {}
    offsets = [[0, 0], [1, 0], [0, 1], [1, 1]]
    validate_definition(
        "rg-simultaneous.schema.json",
        "calibrate_request",
        {
            "input": artifact,
            "settings": settings,
            "channel_offsets_xy": offsets,
            "white_level": 4095,
        },
    )
    validate_definition(
        "rg-simultaneous.schema.json",
        "focus_sample_request",
        {
            "input": artifact,
            "settings": settings,
            "channel_offsets_xy": offsets,
            "white_level": 4095,
            "boxes": None,
            "adaptive_window_selection": True,
            "inspect_periphery": False,
        },
    )
    validate_definition(
        "rg-simultaneous.schema.json",
        "focus_plan_result",
        {"capture_order_um": [0, -32, 32]},
    )
    validate_definition(
        "rg-simultaneous.schema.json",
        "focus_evaluate_result",
        {
            "measurement": {},
            "peripheral_validation": None,
            "defocus_um": 1.25,
            "cross_track_px": 0.1,
        },
    )


def test_route_request_and_result_validate_together() -> None:
    payload = {
        "bounds": {
            "minimum_um": {"x_um": 0.0, "y_um": 0.0},
            "maximum_um": {"x_um": 2500.0, "y_um": 1500.0},
        },
        "field_size_um": {"x_um": 1000.0, "y_um": 1000.0},
        "overlap_fraction": 0.2,
        "traversal": "serpentine",
        "focus_anchor_spacing_um": 1500.0,
    }
    request = {
        "protocol_version": "1.0",
        "request_id": str(uuid.uuid4()),
        "operation": "planning.route",
        "timeout_ms": 1000,
        "payload": payload,
    }
    response = dispatch(request)
    validate("request.schema.json", request)
    validate_definition("route-plan.schema.json", "request", payload)
    validate("response.schema.json", response)
    validate_definition("route-plan.schema.json", "result", response["result"])


def test_exact_grid_route_request_and_result_validate_together() -> None:
    payload = {
        "traversal": "exact_grid_serpentine",
        "grid": {
            "origin_um": {"x_um": 100.0, "y_um": 200.0},
            "step_um": {"x_um": 10.0, "y_um": -20.0},
            "columns": 3,
            "rows": 2,
        },
    }
    response = dispatch(
        {
            "protocol_version": "1.0",
            "request_id": str(uuid.uuid4()),
            "operation": "planning.route",
            "timeout_ms": 1000,
            "payload": payload,
        }
    )
    validate_definition("route-plan.schema.json", "request", payload)
    validate_definition("route-plan.schema.json", "result", response["result"])


def test_focus_surface_fit_and_prediction_validate_together() -> None:
    fit_payload = surface_payload()
    fit_request = {
        "protocol_version": "1.0",
        "request_id": str(uuid.uuid4()),
        "operation": "focus.surface.fit",
        "timeout_ms": 1000,
        "payload": fit_payload,
    }
    fit_response = dispatch(fit_request)
    validate_definition("focus-surface.schema.json", "fit_request", fit_payload)
    validate_definition("focus-surface.schema.json", "fit_result", fit_response["result"])

    predict_payload = surface_payload()
    predict_payload["target_um"] = {"x_um": 0.0, "y_um": 0.0, "z_um": 10.0}
    predict_request = {
        "protocol_version": "1.0",
        "request_id": str(uuid.uuid4()),
        "operation": "focus.surface.predict",
        "timeout_ms": 1000,
        "payload": predict_payload,
    }
    predict_response = dispatch(predict_request)
    validate_definition("focus-surface.schema.json", "predict_request", predict_payload)
    validate_definition("focus-surface.schema.json", "prediction", predict_response["result"])


def test_exact_focus_surface_request_and_prediction_validate_together() -> None:
    payload = {
        "observations": [observation().model_dump(mode="json")],
        "request": prediction_request(
            target_x_um=-125.5,
            target_y_um=250.25,
            predicted_monotonic_s=20.0,
        ).model_dump(mode="json"),
    }
    response = dispatch(
        {
            "protocol_version": "1.0",
            "request_id": str(uuid.uuid4()),
            "operation": "focus.surface.predict",
            "timeout_ms": 1000,
            "payload": payload,
        }
    )
    validate_definition("focus-surface.schema.json", "exact_predict_request", payload)
    validate_definition("focus-surface.schema.json", "exact_prediction", response["result"])


def test_stitch_manifest_request_and_result_schemas_are_coherent() -> None:
    artifact = {
        "uri": "file:///tmp/example",
        "sha256": "0" * 64,
        "media_type": "image/jpeg",
        "size_bytes": 123,
    }
    validate(
        "stitch-manifest.schema.json",
        {"schema_version": "1.0", "tiles": [artifact]},
    )
    request = {
        "tile_manifest": {
            **artifact,
            "media_type": "application/vnd.fast-ofm.stitch-manifest+json",
        },
        "output_name": "scan-1",
        "workers": 3,
        "cache_bytes": 4 * 1024 * 1024 * 1024,
        "registration_mode": "full_correlation",
        "pyramid": True,
        "minimum_overlap": 0.14,
        "correlation_resize": 0.2,
        "work_tile_size": 8192,
        "maximum_runtime_s": 21600,
    }
    validate_definition("stitching-run.schema.json", "request", request)
    result = {
        "ome_bigtiff": {**artifact, "media_type": "image/tiff"},
        "registration_report": {**artifact, "media_type": "application/json"},
        "resource_report": {**artifact, "media_type": "application/json"},
    }
    validate_definition("stitching-run.schema.json", "result", result)
