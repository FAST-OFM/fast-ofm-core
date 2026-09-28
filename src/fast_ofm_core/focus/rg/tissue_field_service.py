"""Artifact boundary for preparing one fresh WHITE tissue field."""

from __future__ import annotations

import os
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from fast_ofm_core.artifacts import ArtifactError, describe_artifact, resolve_artifact

from .rg_focus_core import RGFocusCoreSettings
from .rg_focus_field import (
    FrameReference,
    TissueFieldError,
    TissueFieldSettings,
    prepare_tissue_field,
)

INPUT_MEDIA_TYPE = "application/vnd.fast-ofm.rg-tissue-field-input+npz"
RESULT_MEDIA_TYPE = "application/vnd.fast-ofm.rg-tissue-field-result+npz"
MAXIMUM_BUNDLE_BYTES = 512 * 1024 * 1024
MAXIMUM_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024


class TissueFieldServiceError(ValueError):
    """A stable tissue-field request or artifact failure."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


def _white_frame(path: Path) -> np.ndarray:
    """Load one bounded NPZ containing exactly one WHITE frame."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if (
                any(member.flag_bits & 0x1 for member in members)
                or sum(member.file_size for member in members) > MAXIMUM_UNCOMPRESSED_BYTES
            ):
                raise TissueFieldServiceError(
                    "INVALID_TISSUE_FIELD_ARTIFACT", "Tissue-field archive is unsafe"
                )
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"white_frame"}:
                raise TissueFieldServiceError(
                    "INVALID_TISSUE_FIELD_ARTIFACT",
                    "Tissue-field archive has missing or unknown arrays",
                )
            return np.array(archive["white_frame"], copy=True)
    except TissueFieldServiceError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise TissueFieldServiceError(
            "INVALID_TISSUE_FIELD_ARTIFACT",
            "Tissue-field archive is not a valid NPZ",
        ) from error


def _write_result(path: Path, **arrays: np.ndarray) -> dict[str, object]:
    """Atomically publish immutable tissue arrays next to the verified input."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary, **{name: np.asarray(value) for name, value in arrays.items()}
        )
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return describe_artifact(path, RESULT_MEDIA_TYPE, role="rg-tissue-field-result")


def prepare_payload(
    payload: Mapping[str, object], *, artifact_roots: Sequence[Path]
) -> dict[str, object]:
    """Prepare one field and return its arrays plus JSON-safe contracts."""
    expected = {"input", "reference", "geometry", "settings", "core_settings"}
    if set(payload) != expected:
        raise TissueFieldServiceError("INVALID_PAYLOAD", "Tissue-field request fields changed")
    try:
        path = resolve_artifact(
            payload.get("input"),
            roots=artifact_roots,
            media_type=INPUT_MEDIA_TYPE,
            maximum_size_bytes=MAXIMUM_BUNDLE_BYTES,
        )
        reference = FrameReference.model_validate(payload.get("reference"))
        geometry = payload.get("geometry")
        if not isinstance(geometry, Mapping):
            raise TissueFieldServiceError(
                "INVALID_PAYLOAD", "Tissue-field geometry must be an object"
            )
        settings = TissueFieldSettings.model_validate(payload.get("settings"))
        core_settings = RGFocusCoreSettings.model_validate(payload.get("core_settings"))
        field = prepare_tissue_field(
            _white_frame(path), reference, geometry, settings, core_settings
        )
    except ArtifactError as error:
        raise TissueFieldServiceError(error.code, str(error)) from error
    except TissueFieldServiceError:
        raise
    except (ValidationError, TissueFieldError, ValueError) as error:
        code = error.code.upper() if isinstance(error, TissueFieldError) else "INVALID_PAYLOAD"
        raise TissueFieldServiceError(code, str(error)) from error

    descriptor = _write_result(
        path.parent / "tissue-field-result.npz",
        mask=field.mask,
        white_common=field.white_common,
        overlay=field.overlay,
    )
    return {
        "arrays": descriptor,
        "field": {
            "reference": field.reference.model_dump(mode="json"),
            "geometry": field.geometry.model_dump(mode="json"),
            "boxes": [box.model_dump(mode="json") for box in field.boxes],
            "metrics": field.metrics.model_dump(mode="json"),
        },
    }
