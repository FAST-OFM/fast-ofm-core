"""Artifact-only service for empirical processed-JPEG flat-field operations."""

from __future__ import annotations

import json
import os
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from fast_ofm_core.artifacts import (
    ArtifactError,
    describe_artifact,
    resolve_artifact,
)

from .rg_flat_field import (
    FlatFieldSettings,
    apply_native_flat_field,
    fit_flat_field,
    validate_flat_field,
)

INPUT_MEDIA_TYPE = "application/vnd.fast-ofm.rg-flat-field-input+npz"
RESULT_MEDIA_TYPE = "application/vnd.fast-ofm.rg-flat-field-result+npz"
MAXIMUM_BUNDLE_BYTES = 1024 * 1024 * 1024
MAXIMUM_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


class FlatFieldServiceError(ValueError):
    """A stable flat-field request or artifact failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _arrays(path: Path, expected: set[str]) -> dict[str, np.ndarray]:
    """Load one bounded, non-pickle NPZ with an exact member set."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if (
                any(member.flag_bits & 0x1 for member in members)
                or sum(member.file_size for member in members) > MAXIMUM_UNCOMPRESSED_BYTES
            ):
                raise FlatFieldServiceError(
                    "INVALID_FLAT_FIELD_ARTIFACT", "Flat-field archive is unsafe"
                )
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise FlatFieldServiceError(
                    "INVALID_FLAT_FIELD_ARTIFACT",
                    "Flat-field archive has missing or unknown arrays",
                )
            return {name: np.array(archive[name], copy=True) for name in expected}
    except FlatFieldServiceError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise FlatFieldServiceError(
            "INVALID_FLAT_FIELD_ARTIFACT", "Flat-field archive is not a valid NPZ"
        ) from error


def _write_result(path: Path, **arrays: np.ndarray) -> dict[str, object]:
    """Atomically publish core-owned arrays next to the verified input artifact."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return describe_artifact(path, RESULT_MEDIA_TYPE, role="rg-flat-field-result")


def _settings(value: object) -> FlatFieldSettings:
    try:
        settings = FlatFieldSettings.model_validate(value)
        settings.require_jpeg()
        return settings
    except (ValidationError, ValueError) as error:
        raise FlatFieldServiceError("INVALID_FLAT_FIELD_SETTINGS", str(error)) from error


def _input(
    payload: Mapping[str, object],
    *,
    artifact_roots: Sequence[Path],
    expected: set[str],
) -> tuple[Path, dict[str, np.ndarray]]:
    try:
        path = resolve_artifact(
            payload.get("input"),
            roots=artifact_roots,
            media_type=INPUT_MEDIA_TYPE,
            maximum_size_bytes=MAXIMUM_BUNDLE_BYTES,
        )
    except ArtifactError as error:
        raise FlatFieldServiceError(error.code, str(error)) from error
    return path, _arrays(path, expected)


def fit_payload(
    payload: Mapping[str, object], *, artifact_roots: Sequence[Path]
) -> dict[str, object]:
    """Fit maps from phase-matched flat and dark arrays."""
    if set(payload) != {"input", "settings"}:
        raise FlatFieldServiceError("INVALID_PAYLOAD", "Flat-field fit fields changed")
    path, arrays = _input(payload, artifact_roots=artifact_roots, expected={"flat", "dark"})
    settings = _settings(payload.get("settings"))
    try:
        maps, report = fit_flat_field(arrays["flat"], arrays["dark"], settings)
    except ValueError as error:
        raise FlatFieldServiceError("FLAT_FIELD_REFUSED", str(error)) from error
    result_path = path.parent / "flat-field-fit-result.npz"
    descriptor = _write_result(
        result_path,
        dark=maps["dark"],
        gain=maps["gain"],
        valid=maps["valid"],
    )
    return {"maps": descriptor, "report": report}


def apply_payload(
    payload: Mapping[str, object], *, artifact_roots: Sequence[Path]
) -> dict[str, object]:
    """Apply one verified map bundle to one decoded JPEG channel plane."""
    if set(payload) != {"input"}:
        raise FlatFieldServiceError("INVALID_PAYLOAD", "Flat-field apply fields changed")
    path, arrays = _input(
        payload,
        artifact_roots=artifact_roots,
        expected={"image", "dark", "gain", "valid"},
    )
    try:
        corrected, valid = apply_native_flat_field(
            arrays["image"],
            {key: arrays[key] for key in ("dark", "gain", "valid")},
        )
    except (KeyError, ValueError) as error:
        raise FlatFieldServiceError("FLAT_FIELD_REFUSED", str(error)) from error
    descriptor = _write_result(
        path.parent / "flat-field-apply-result.npz",
        corrected=corrected,
        valid=valid,
    )
    return {"corrected": descriptor}


def validate_payload(
    payload: Mapping[str, object], *, artifact_roots: Sequence[Path]
) -> dict[str, object]:
    """Validate maps against independent held-out exposures."""
    if set(payload) != {"input", "settings", "fit_report"}:
        raise FlatFieldServiceError("INVALID_PAYLOAD", "Flat-field validation fields changed")
    _path, arrays = _input(
        payload,
        artifact_roots=artifact_roots,
        expected={"heldout", "dark", "gain", "valid"},
    )
    fit_report = payload.get("fit_report")
    if not isinstance(fit_report, Mapping):
        raise FlatFieldServiceError("INVALID_PAYLOAD", "fit_report must be an object")
    settings = _settings(payload.get("settings"))
    try:
        quality = validate_flat_field(
            arrays["heldout"],
            {key: arrays[key] for key in ("dark", "gain", "valid")},
            dict(fit_report),
            settings,
        )
        json.dumps(quality, allow_nan=False)
    except (KeyError, TypeError, ValueError) as error:
        raise FlatFieldServiceError("FLAT_FIELD_REFUSED", str(error)) from error
    return {"quality": quality}
