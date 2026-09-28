"""Build the protocol schema for the strict persisted focus-surface contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from fast_ofm_core.focus.focus_surface import (
    FocusObservation,
    FocusPrediction,
    FocusPredictionRequest,
)


def build_schema() -> dict[str, Any]:
    """Combine Pydantic's exact nested definitions under stable protocol names."""
    definitions: dict[str, Any] = {}
    roots: dict[str, dict[str, Any]] = {}
    for name, model in (
        ("observations", list[FocusObservation]),
        ("request", FocusPredictionRequest),
        ("prediction", FocusPrediction),
    ):
        schema = TypeAdapter(model).json_schema()
        for definition_name, definition in schema.pop("$defs", {}).items():
            previous = definitions.setdefault(definition_name, definition)
            if previous != definition:
                raise RuntimeError(f"Conflicting generated definition: {definition_name}")
        roots[name] = schema
    definitions["exact_predict_request"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["observations", "request"],
        "properties": {
            "observations": roots["observations"],
            "request": roots["request"],
        },
    }
    definitions["exact_prediction"] = roots["prediction"]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://fast-ofm.org/schema/protocol/v1/focus-surface-exact.schema.json",
        "title": "Fast OFM exact persisted focus-surface contract",
        "$defs": definitions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("outputs", nargs="+", type=Path)
    arguments = parser.parse_args()
    encoded = json.dumps(build_schema(), indent=2, sort_keys=True) + "\n"
    for output in arguments.outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
