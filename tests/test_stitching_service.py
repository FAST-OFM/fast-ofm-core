from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

from fast_ofm_core.stitching.service import (
    MANIFEST_MEDIA_TYPE,
    StitchingServiceError,
    run_stitching_payload,
)


def descriptor(path: Path, media_type: str) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "uri": path.resolve().as_uri(),
        "sha256": hashlib.sha256(content).hexdigest(),
        "media_type": media_type,
        "size_bytes": len(content),
    }


def request(tmp_path: Path) -> tuple[dict[str, object], tuple[str, ...]]:
    tiles = []
    for index in range(2):
        path = tmp_path / f"img_{index}_0_0.jpeg"
        Image.new("RGB", (8, 8), (index * 20, 0, 0)).save(path)
        tiles.append(descriptor(path, "image/jpeg"))
    manifest = tmp_path / "stitch-manifest.json"
    manifest.write_text(json.dumps({"schema_version": "1.0", "tiles": tiles}), encoding="utf-8")
    worker = tmp_path / "fake_worker.py"
    worker.write_text(
        """\
import pathlib
import sys
name = sys.argv[sys.argv.index('--output-name') + 1]
manifest = pathlib.Path(sys.argv[sys.argv.index('--tile-manifest') + 1])
assert manifest.name == 'stitch-manifest.json'
directory = pathlib.Path(sys.argv[-1])
(directory / f'{name}.ome.tiff').write_bytes(b'II*\\x00fake-ome')
if '--stitch-dzi' in sys.argv:
    (directory / f'{name}.dzi').write_text('<Image/>')
    tile = directory / f'{name}_files' / '0' / '0_0.jpg'
    tile.parent.mkdir(parents=True)
    tile.write_bytes(b'fake-tile')
""",
        encoding="utf-8",
    )
    return (
        {
            "tile_manifest": descriptor(manifest, MANIFEST_MEDIA_TYPE),
            "output_name": "synthetic",
            "workers": 3,
            "cache_bytes": 4 * 1024 * 1024 * 1024,
            "registration_mode": "full_correlation",
            "pyramid": True,
        },
        (sys.executable, str(worker)),
    )


def test_stitching_job_validates_tiles_and_records_bounded_resources(tmp_path: Path) -> None:
    payload, worker = request(tmp_path)

    result = run_stitching_payload(payload, artifact_roots=(tmp_path,), worker_command=worker)

    assert result["ome_bigtiff"]["media_type"] == "image/tiff"
    assert Path(str(result["ome_bigtiff"]["uri"])[7:]).read_bytes().endswith(b"fake-ome")
    resources = json.loads((tmp_path / "synthetic.resources.json").read_text())
    assert resources["workers"] == 3
    assert resources["cache_bytes"] == 4 * 1024 * 1024 * 1024
    registration = json.loads((tmp_path / "synthetic.registration.json").read_text())
    assert registration["tile_count"] == 2
    assert registration["registration_mode"] == "full_correlation"


def test_stitching_job_refuses_undeclared_image(tmp_path: Path) -> None:
    payload, worker = request(tmp_path)
    Image.new("RGB", (8, 8)).save(tmp_path / "img_undeclared_0_0.jpeg")

    try:
        run_stitching_payload(payload, artifact_roots=(tmp_path,), worker_command=worker)
    except StitchingServiceError as error:
        assert error.code == "UNDECLARED_TILE"
    else:
        raise AssertionError("undeclared image was accepted")


def test_stitching_job_can_require_a_complete_viewer_pyramid(tmp_path: Path) -> None:
    payload, worker = request(tmp_path)
    payload["viewer_dzi"] = True

    result = run_stitching_payload(payload, artifact_roots=(tmp_path,), worker_command=worker)

    assert result["viewer_dzi"]["media_type"] == "application/xml"
    assert (tmp_path / "synthetic_files" / "0" / "0_0.jpg").is_file()


def test_stitching_job_refuses_overwrite(tmp_path: Path) -> None:
    payload, worker = request(tmp_path)
    (tmp_path / "synthetic.ome.tiff").write_bytes(b"existing")

    try:
        run_stitching_payload(payload, artifact_roots=(tmp_path,), worker_command=worker)
    except StitchingServiceError as error:
        assert error.code == "OUTPUT_EXISTS"
    else:
        raise AssertionError("existing output was overwritten")
