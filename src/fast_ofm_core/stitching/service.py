"""Validate a tile manifest and invoke a separately licensed stitching worker."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fast_ofm_core.artifacts import ArtifactError, load_json_artifact, resolve_artifact

MANIFEST_MEDIA_TYPE = "application/vnd.fast-ofm.stitch-manifest+json"
IMAGE_MEDIA_TYPES = {"image/jpeg", "image/png", "image/tiff"}
DEFAULT_WORKER = "fast-ofm-stitch-openflexure"


class StitchingServiceError(ValueError):
    """A stable stitching request, worker, or output failure."""

    def __init__(self, code: str, detail: str, *, retryable: bool = False) -> None:
        super().__init__(detail)
        self.code = code
        self.retryable = retryable


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise StitchingServiceError(
            "INVALID_PAYLOAD", f"{name} must be an integer in [{minimum}, {maximum}]"
        )
    return value


def _number(value: object, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StitchingServiceError("INVALID_PAYLOAD", f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise StitchingServiceError("INVALID_PAYLOAD", f"{name} must be in [{minimum}, {maximum}]")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _descriptor(path: Path, media_type: str, role: str) -> dict[str, object]:
    return {
        "uri": path.resolve().as_uri(),
        "sha256": _sha256(path),
        "media_type": media_type,
        "size_bytes": path.stat().st_size,
        "role": role,
    }


def _worker_path(worker_command: Sequence[str] | None) -> tuple[str, ...]:
    if worker_command is not None:
        if not worker_command or any(
            not isinstance(value, str) or not value for value in worker_command
        ):
            raise StitchingServiceError("INVALID_WORKER", "Worker command is empty")
        return tuple(worker_command)
    executable = find_worker()
    if executable is None:
        raise StitchingServiceError(
            "STITCHING_WORKER_UNAVAILABLE",
            "The replaceable OpenFlexure stitching worker is not installed",
            retryable=True,
        )
    return (executable,)


def find_worker() -> str | None:
    """Locate an explicit, PATH-installed, or same-environment worker executable."""
    configured = os.environ.get("FAST_OFM_STITCH_WORKER")
    if configured:
        return shutil.which(configured)
    executable = shutil.which(DEFAULT_WORKER)
    if executable is not None:
        return executable
    sibling = Path(sys.executable).with_name(DEFAULT_WORKER)
    return str(sibling) if sibling.is_file() and os.access(sibling, os.X_OK) else None


def _validated_image_directory(
    manifest: Mapping[str, object],
    *,
    roots: Sequence[Path],
    ignored_names: set[str] | None = None,
) -> tuple[Path, int]:
    tiles = manifest.get("tiles")
    if not isinstance(tiles, list) or not 1 <= len(tiles) <= 10_000:
        raise StitchingServiceError("INVALID_MANIFEST", "tiles must contain 1 to 10000 items")
    paths: list[Path] = []
    for index, descriptor in enumerate(tiles):
        if not isinstance(descriptor, Mapping):
            raise StitchingServiceError("INVALID_MANIFEST", f"tiles[{index}] is not an artifact")
        media_type = descriptor.get("media_type")
        if media_type not in IMAGE_MEDIA_TYPES:
            raise StitchingServiceError(
                "INVALID_MANIFEST", f"tiles[{index}] has an unsupported media type"
            )
        try:
            paths.append(
                resolve_artifact(
                    descriptor,
                    roots=roots,
                    media_type=str(media_type),
                    maximum_size_bytes=256 * 1024 * 1024,
                )
            )
        except ArtifactError as error:
            raise StitchingServiceError(error.code, str(error)) from error
    directories = {path.parent for path in paths}
    if len(directories) != 1:
        raise StitchingServiceError("INVALID_MANIFEST", "all tiles must share one directory")
    directory = directories.pop()
    declared = {path.resolve() for path in paths}
    present = {
        path.resolve()
        for pattern in ("img_*.jpeg", "img_*.jpg", "img_*.png", "img_*.tif", "img_*.tiff")
        for path in directory.glob(pattern)
        if path.is_file() and not path.is_symlink()
    }
    ignored = {directory / name for name in (ignored_names or set())}
    unexpected = present - declared - {path.resolve() for path in ignored}
    if unexpected:
        raise StitchingServiceError(
            "UNDECLARED_TILE", "the image directory contains files absent from the manifest"
        )
    return directory, len(paths)


def run_stitching_payload(
    payload: Mapping[str, object],
    *,
    artifact_roots: Sequence[Path],
    worker_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run one bounded synchronous stitch through a replaceable executable."""
    try:
        manifest_path = resolve_artifact(
            payload.get("tile_manifest"),
            roots=artifact_roots,
            media_type=MANIFEST_MEDIA_TYPE,
            maximum_size_bytes=16 * 1024 * 1024,
        )
        manifest = load_json_artifact(manifest_path)
    except ArtifactError as error:
        raise StitchingServiceError(error.code, str(error)) from error
    if manifest.get("schema_version") != "1.0":
        raise StitchingServiceError("INVALID_MANIFEST", "unsupported tile-manifest version")
    output_name = payload.get("output_name")
    if (
        not isinstance(output_name, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", output_name) is None
    ):
        raise StitchingServiceError("INVALID_PAYLOAD", "output_name is invalid")
    ignored_names = {
        f"{output_name}.ome.tiff",
        f"{output_name}.registration.json",
        f"{output_name}.resources.json",
        f"{output_name}.worker.log",
    }
    directory, tile_count = _validated_image_directory(
        manifest, roots=artifact_roots, ignored_names=ignored_names
    )
    workers = _integer(payload.get("workers"), "workers", 1, 64)
    cache_bytes = _integer(payload.get("cache_bytes"), "cache_bytes", 64 * 1024 * 1024, 1 << 50)
    ram_cache_mb = max(64, (cache_bytes + 1024 * 1024 - 1) // (1024 * 1024))
    mode = payload.get("registration_mode")
    mode_argument = {
        "full_correlation": "full",
        "experimental_overlap_strip": "overlap-strip",
    }.get(mode)
    if mode_argument is None:
        raise StitchingServiceError("INVALID_PAYLOAD", "registration_mode is invalid")
    if payload.get("pyramid", True) is not True:
        raise StitchingServiceError("INVALID_PAYLOAD", "pyramid must be true")
    viewer_dzi = payload.get("viewer_dzi", False)
    if not isinstance(viewer_dzi, bool):
        raise StitchingServiceError("INVALID_PAYLOAD", "viewer_dzi must be a boolean")
    minimum_overlap = _number(payload.get("minimum_overlap", 0.1), "minimum_overlap", 0.01, 0.99)
    resize = _number(payload.get("correlation_resize", 0.2), "correlation_resize", 0.01, 1.0)
    tile_size = _integer(payload.get("work_tile_size", 2048), "work_tile_size", 128, 16384)
    maximum_runtime_s = _integer(
        payload.get("maximum_runtime_s", 21_600), "maximum_runtime_s", 60, 86_400
    )
    output_path = directory / f"{output_name}.ome.tiff"
    report_path = directory / f"{output_name}.registration.json"
    resource_path = directory / f"{output_name}.resources.json"
    log_path = directory / f"{output_name}.worker.log"
    dzi_path = directory / f"{output_name}.dzi"
    dzi_tiles = directory / f"{output_name}_files"
    guarded_outputs = [output_path, report_path, resource_path, log_path]
    if viewer_dzi:
        guarded_outputs.extend((dzi_path, dzi_tiles))
    if any(path.exists() for path in guarded_outputs):
        raise StitchingServiceError("OUTPUT_EXISTS", "refusing to overwrite stitch output")
    command_parts = [
        *_worker_path(worker_command),
        "--stitch-tiff",
    ]
    if viewer_dzi:
        command_parts.append("--stitch-dzi")
    command = (
        *command_parts,
        "--tile-size",
        str(tile_size),
        "--workers",
        str(workers),
        "--ram-cache-mb",
        str(ram_cache_mb),
        "--correlation-mode",
        mode_argument,
        "--minimum-overlap",
        str(minimum_overlap),
        "--resize",
        str(resize),
        "--output-name",
        output_name,
        "--tile-manifest",
        str(manifest_path),
        str(directory),
    )
    started = time.monotonic()
    try:
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=maximum_runtime_s,
                text=True,
            )
    except subprocess.TimeoutExpired as error:
        raise StitchingServiceError(
            "STITCHING_TIMEOUT", "stitching worker timed out", retryable=True
        ) from error
    except OSError as error:
        raise StitchingServiceError(
            "STITCHING_WORKER_FAILED", str(error), retryable=True
        ) from error
    elapsed = time.monotonic() - started
    if completed.returncode != 0 or not output_path.is_file() or output_path.is_symlink():
        raise StitchingServiceError(
            "STITCHING_WORKER_FAILED",
            f"stitching worker exited with status {completed.returncode}; see {log_path.name}",
        )
    if viewer_dzi and (
        not dzi_path.is_file()
        or dzi_path.is_symlink()
        or not dzi_tiles.is_dir()
        or not any(
            path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            for path in dzi_tiles.rglob("*")
        )
    ):
        raise StitchingServiceError(
            "STITCHING_WORKER_FAILED", "stitching worker produced an incomplete DZI pyramid"
        )
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "tile_count": tile_count,
                "registration_mode": mode,
                "minimum_overlap": minimum_overlap,
                "correlation_resize": resize,
                "worker_log": log_path.name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    resource_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "workers": workers,
                "cache_bytes": cache_bytes,
                "work_tile_size": tile_size,
                "elapsed_s": elapsed,
                "output_size_bytes": output_path.stat().st_size,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "ome_bigtiff": _descriptor(output_path, "image/tiff", "ome-bigtiff"),
        "registration_report": _descriptor(report_path, "application/json", "registration-report"),
        "resource_report": _descriptor(resource_path, "application/json", "resource-report"),
    }
    if viewer_dzi:
        result["viewer_dzi"] = _descriptor(dzi_path, "application/xml", "viewer-dzi-descriptor")
    return result
