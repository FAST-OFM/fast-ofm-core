"""Verified, allow-listed file artifacts for the process protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


class ArtifactError(ValueError):
    """A stable artifact boundary failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def describe_artifact(path: Path, media_type: str, *, role: str) -> dict[str, object]:
    """Describe a core-produced regular file using the public artifact contract."""
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ArtifactError("INVALID_ARTIFACT", "Produced artifact is not a regular file")
    return {
        "uri": resolved.as_uri(),
        "sha256": _sha256(resolved),
        "media_type": media_type,
        "size_bytes": resolved.stat().st_size,
        "role": role,
    }


def resolve_artifact(
    descriptor: object,
    *,
    roots: Sequence[Path],
    media_type: str,
    maximum_size_bytes: int,
) -> Path:
    """Resolve and verify one regular file strictly inside an allowed root."""
    if not isinstance(descriptor, Mapping):
        raise ArtifactError("INVALID_ARTIFACT", "Artifact descriptor must be an object")
    if set(descriptor) - {"uri", "sha256", "media_type", "size_bytes", "role"}:
        raise ArtifactError("INVALID_ARTIFACT", "Artifact descriptor has unknown fields")
    uri = descriptor.get("uri")
    expected_digest = descriptor.get("sha256")
    if not isinstance(uri, str) or not isinstance(expected_digest, str):
        raise ArtifactError("INVALID_ARTIFACT", "Artifact URI and SHA-256 are required")
    if descriptor.get("media_type") != media_type:
        raise ArtifactError("INVALID_MEDIA_TYPE", f"Expected {media_type}")
    parsed = urlparse(uri)
    if (
        parsed.scheme != "file"
        or parsed.netloc not in ("", "localhost")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ArtifactError("INVALID_ARTIFACT_URI", "Only local file URIs are accepted")
    requested = Path(unquote(parsed.path))
    if not requested.is_absolute():
        raise ArtifactError("INVALID_ARTIFACT_URI", "Artifact path must be absolute")
    if not roots:
        raise ArtifactError("ARTIFACT_ACCESS_DENIED", "No artifact roots were configured")
    try:
        path = requested.resolve(strict=True)
    except OSError as error:
        raise ArtifactError("ARTIFACT_NOT_FOUND", "Artifact does not exist") from error
    allowed = tuple(Path(root).resolve(strict=True) for root in roots)
    if not any(path == root or root in path.parents for root in allowed):
        raise ArtifactError("ARTIFACT_ACCESS_DENIED", "Artifact is outside configured roots")
    if not path.is_file() or path.is_symlink():
        raise ArtifactError("INVALID_ARTIFACT", "Artifact must resolve to a regular file")
    size = path.stat().st_size
    declared_size = descriptor.get("size_bytes")
    if isinstance(declared_size, bool) or (
        declared_size is not None and not isinstance(declared_size, int)
    ):
        raise ArtifactError("INVALID_ARTIFACT", "Artifact size must be an integer")
    if declared_size is not None and declared_size != size:
        raise ArtifactError("ARTIFACT_SIZE_MISMATCH", "Artifact size differs from descriptor")
    if size > maximum_size_bytes:
        raise ArtifactError("ARTIFACT_TOO_LARGE", "Artifact exceeds the operation limit")
    if len(expected_digest) != 64 or _sha256(path) != expected_digest:
        raise ArtifactError("ARTIFACT_DIGEST_MISMATCH", "Artifact SHA-256 verification failed")
    return path


def load_json_artifact(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object after the file boundary has been verified."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(
            "INVALID_ARTIFACT_CONTENT", "Artifact is not valid UTF-8 JSON"
        ) from error
    if not isinstance(value, dict):
        raise ArtifactError("INVALID_ARTIFACT_CONTENT", "JSON artifact must contain an object")
    return value
