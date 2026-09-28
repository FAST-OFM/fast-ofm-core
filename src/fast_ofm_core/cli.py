"""Command-line entry points for independent use and process integration."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from fast_ofm_core.service import capabilities, dispatch, stdio_service


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fast-ofm-core")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("capabilities", help="print the supported protocol capabilities")
    service = commands.add_parser(
        "serve-jsonl", help="serve protocol v1 as JSON lines on stdin/stdout"
    )
    service.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        type=Path,
        help="allow verified file artifacts below this directory (repeatable)",
    )
    one_shot = commands.add_parser(
        "run-request", help="execute one protocol-v1 JSON request file and exit"
    )
    one_shot.add_argument("request", type=Path)
    one_shot.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        type=Path,
        help="allow verified file artifacts below this directory (repeatable)",
    )
    args = parser.parse_args(argv)
    if args.command == "capabilities":
        print(json.dumps(capabilities(), indent=2, sort_keys=True))
        return
    if args.command == "run-request":
        try:
            request = json.loads(args.request.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            parser.error(f"could not read a JSON request: {error}")
        response = dispatch(request, artifact_roots=tuple(args.artifact_root))
        print(json.dumps(response, separators=(",", ":"), allow_nan=False))
        if response.get("status") not in {"completed", "accepted"}:
            raise SystemExit(2)
        return
    raise SystemExit(stdio_service(artifact_roots=tuple(args.artifact_root)))
