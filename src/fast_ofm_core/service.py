"""Dependency-free protocol dispatcher for the standalone core process."""

from __future__ import annotations

import json
import platform
import sys
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from fast_ofm_core import __version__

PROTOCOL_VERSION = "1.0"
OPERATIONS = (
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
    "stitching.run",
    "job.status",
    "job.cancel",
)


def capabilities() -> dict[str, Any]:
    """Return stable capability data and probe only the optional native backend."""
    try:
        from fast_ofm_core.focus.rg import _rg_nmi  # noqa: F401
    except ImportError:
        native_rg_nmi = False
    else:
        native_rg_nmi = True
    from fast_ofm_core.focus.rg.rg_focus_core import NMI_BACKEND
    from fast_ofm_core.stitching.service import find_worker

    stitching_worker = find_worker() is not None
    operations = [
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
    if stitching_worker:
        operations.append("stitching.run")
    return {
        "core_version": __version__,
        "protocol_versions": [PROTOCOL_VERSION],
        "operations": operations,
        "implementation": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "native_rg_nmi": native_rg_nmi,
            "rg_nmi_backend": NMI_BACKEND,
            "stitching_worker": stitching_worker,
        },
        "experimental_operations": [],
    }


def _error(
    request_id: str,
    code: str,
    message: str,
    *,
    status: str = "failed",
    retryable: bool = False,
) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": status,
        "error": {"code": code, "message": message, "retryable": retryable},
    }


def dispatch(value: object, *, artifact_roots: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Validate the envelope required for safe dispatch and execute one request."""
    if not isinstance(value, Mapping):
        return _error(str(uuid.uuid4()), "INVALID_REQUEST", "Request must be an object")
    request_id = value.get("request_id")
    if not isinstance(request_id, str):
        request_id = str(uuid.uuid4())
        return _error(request_id, "INVALID_REQUEST_ID", "request_id must be a UUID string")
    try:
        uuid.UUID(request_id)
    except ValueError:
        return _error(request_id, "INVALID_REQUEST_ID", "request_id must be a UUID string")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        return _error(request_id, "UNSUPPORTED_PROTOCOL", "Only protocol version 1.0 is supported")
    operation = value.get("operation")
    if operation not in OPERATIONS:
        return _error(request_id, "UNKNOWN_OPERATION", "Unknown operation")
    if operation == "core.capabilities":
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "status": "completed",
            "result": capabilities(),
        }
    timeout_ms = value.get("timeout_ms")
    if (
        isinstance(timeout_ms, bool)
        or not isinstance(timeout_ms, int)
        or not 1 <= timeout_ms <= 3_600_000
    ):
        return _error(
            request_id, "INVALID_TIMEOUT", "timeout_ms must be an integer from 1 to 3600000"
        )
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        return _error(request_id, "INVALID_PAYLOAD", "payload must be an object", status="refused")
    if operation == "planning.route":
        from fast_ofm_core.planning import PlanningError, plan_route

        try:
            result = plan_route(payload)
        except PlanningError as error:
            return _error(request_id, "INVALID_PAYLOAD", str(error), status="refused")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "status": "completed",
            "result": result,
        }
    if operation == "stitching.run":
        from fast_ofm_core.stitching import StitchingServiceError, run_stitching_payload
        from fast_ofm_core.stitching.service import find_worker

        if find_worker() is None:
            return _error(
                request_id,
                "CAPABILITY_UNAVAILABLE",
                "The replaceable OpenFlexure stitching worker is not installed",
            )
        try:
            result = run_stitching_payload(payload, artifact_roots=artifact_roots)
        except StitchingServiceError as error:
            return _error(
                request_id,
                error.code,
                str(error),
                retryable=error.retryable,
                status=("failed" if error.code.startswith("STITCHING_") else "refused"),
            )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "status": "completed",
            "result": result,
        }
    if operation == "rg.measure":
        from fast_ofm_core.focus.rg.rg_service import RGServiceError, measure_payload

        try:
            result = measure_payload(payload, artifact_roots=artifact_roots)
        except RGServiceError as error:
            return _error(request_id, error.code, str(error), status="refused")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "status": "completed",
            "result": result,
        }
    if operation in {
        "rg.flat_field.fit",
        "rg.flat_field.apply",
        "rg.flat_field.validate",
    }:
        from fast_ofm_core.focus.rg.flat_field_service import (
            FlatFieldServiceError,
            apply_payload,
            fit_payload,
            validate_payload,
        )

        handlers = {
            "rg.flat_field.fit": fit_payload,
            "rg.flat_field.apply": apply_payload,
            "rg.flat_field.validate": validate_payload,
        }
        try:
            result = handlers[operation](payload, artifact_roots=artifact_roots)
        except FlatFieldServiceError as error:
            return _error(request_id, error.code, str(error), status="refused")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "status": "completed",
            "result": result,
        }
    if operation == "rg.tissue_field.prepare":
        from fast_ofm_core.focus.rg.tissue_field_service import (
            TissueFieldServiceError,
            prepare_payload,
        )

        try:
            result = prepare_payload(payload, artifact_roots=artifact_roots)
        except TissueFieldServiceError as error:
            return _error(request_id, error.code, str(error), status="refused")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "status": "completed",
            "result": result,
        }
    if operation.startswith("rg.simultaneous."):
        from fast_ofm_core.focus.rg.simultaneous_service import (
            SimultaneousServiceError,
            calibrate_payload,
            focus_evaluate_payload,
            focus_fit_payload,
            focus_plan_payload,
            focus_sample_payload,
        )

        try:
            if operation == "rg.simultaneous.calibrate":
                result = calibrate_payload(payload, artifact_roots=artifact_roots)
            elif operation == "rg.simultaneous.focus.sample":
                result = focus_sample_payload(payload, artifact_roots=artifact_roots)
            elif operation == "rg.simultaneous.focus.plan":
                result = focus_plan_payload(payload)
            elif operation == "rg.simultaneous.focus.fit":
                result = focus_fit_payload(payload)
            else:
                result = focus_evaluate_payload(payload)
        except SimultaneousServiceError as error:
            return _error(request_id, error.code, str(error), status="refused")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "status": "completed",
            "result": result,
        }
    if operation == "calibration.fit":
        from fast_ofm_core.focus.rg.calibration_service import (
            CalibrationServiceError,
            fit_payload,
        )

        try:
            result = fit_payload(payload, artifact_roots=artifact_roots)
        except CalibrationServiceError as error:
            return _error(request_id, error.code, str(error), status="refused")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "status": "completed",
            "result": result,
        }
    if operation == "calibration.validate":
        from fast_ofm_core.focus.rg.calibration_service import (
            CalibrationServiceError,
            validate_profile_payload,
        )

        try:
            result = validate_profile_payload(payload)
        except CalibrationServiceError as error:
            return _error(request_id, error.code, str(error), status="refused")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "status": "completed",
            "result": result,
        }
    if operation == "rg.decide":
        from fast_ofm_core.focus.rg.decision_service import (
            DecisionServiceError,
            decide_payload,
        )

        try:
            result = decide_payload(payload)
        except DecisionServiceError as error:
            return _error(request_id, "INVALID_PAYLOAD", str(error), status="refused")
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "status": "completed",
            "result": result,
        }
    if operation in {"focus.surface.fit", "focus.surface.predict"}:
        from fast_ofm_core.focus.surface_service import (
            FocusSurfacePayloadError,
            fit_surface_payload,
            predict_surface_payload,
        )

        try:
            result = (
                fit_surface_payload(payload)
                if operation == "focus.surface.fit"
                else predict_surface_payload(payload)
            )
        except FocusSurfacePayloadError as error:
            return _error(
                request_id,
                "INVALID_PAYLOAD",
                str(error),
                status="refused",
            )
        return {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request_id,
            "status": "completed",
            "result": result,
        }
    return _error(
        request_id,
        "CAPABILITY_UNAVAILABLE",
        f"Operation {operation} is defined by the protocol but not enabled in this build",
    )


def serve_jsonl(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    artifact_roots: tuple[Path, ...] = (),
) -> int:
    """Serve independent request lines until EOF without protocol contamination."""
    for raw_line in input_stream:
        try:
            request = json.loads(raw_line)
        except json.JSONDecodeError:
            response = _error(str(uuid.uuid4()), "INVALID_JSON", "Input line is not valid JSON")
        else:
            response = dispatch(request, artifact_roots=artifact_roots)
        output_stream.write(json.dumps(response, separators=(",", ":")) + "\n")
        output_stream.flush()
    return 0


def stdio_service(*, artifact_roots: tuple[Path, ...] = ()) -> int:
    """Run the JSON-lines service on standard streams."""
    return serve_jsonl(sys.stdin, sys.stdout, artifact_roots=artifact_roots)
