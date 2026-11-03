"""Deterministic integration tests for the unified `reality run` CLI command."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from apps.cli.main import main
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructionResult,
    ReconstructedPoint,
)

_BACKEND_SPEC = "tests.test_cli_runner:_TestCliBackend"


@dataclass(frozen=True)
class _TestCliBackend:
    """Deterministic backend stand-in providing registered cameras and coplanar floor points."""

    def reconstruct(self, evidence):
        ids = [item.id for item in evidence]
        if len(ids) < 2:
            raise ValueError("need >= 2 images")
        poses = [
            ReconstructedCameraPose(evidence_id=ids[0], position=(0.0, 0.0, 0.0), rotation=(1.0, 0.0, 0.0, 0.0)),
            ReconstructedCameraPose(evidence_id=ids[1], position=(1.0, 0.0, 0.0), rotation=(1.0, 0.0, 0.0, 0.0)),
        ]
        if len(ids) > 2:
            poses.append(
                ReconstructedCameraPose(evidence_id=ids[2], position=(2.0, 0.0, 0.0), rotation=(1.0, 0.0, 0.0, 0.0))
            )
        points = [
            ReconstructedPoint(
                position=(float(x), -1.0, float(z)),
                track_id=f"trk-{i:03d}",
                source_evidence_ids=[ids[0], ids[1]],
                color=(200, 200, 200),
            )
            for i, (x, z) in enumerate(
                (x, z)
                for x in np.linspace(-2.0, 2.0, 5)
                for z in np.linspace(-2.0, 2.0, 5)
            )
        ]
        return ReconstructionResult(
            points=points,
            camera_poses=poses,
            registration_status="success",
        )


@pytest.fixture
def capture_dataset(tmp_path: Path) -> Path:
    """Fixture creating a valid capture dataset with 3 images and manifest."""
    dataset = tmp_path / "dataset"
    images = dataset / "images"
    images.mkdir(parents=True)

    for i in range(3):
        im = Image.new("RGB", (640, 480), color=(128 + i * 10, 128, 128))
        im.save(images / f"img_{i:03d}.jpg")

    manifest = {
        "dataset": "run_cli_test",
        "images": [{"file": f"img_{i:03d}.jpg"} for i in range(3)],
        "image_size": [640, 480],
        "measured_baselines": [
            {"evidence_id_a": "img_000", "evidence_id_b": "img_001", "distance_m": 1.0}
        ],
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset


@pytest.fixture
def raw_images_folder(tmp_path: Path) -> Path:
    """Folder of raw photos without any manifest.json."""
    folder = tmp_path / "raw_photos"
    folder.mkdir(parents=True)
    for i in range(3):
        im = Image.new("RGB", (320, 240), color=(100 + i * 20, 100, 100))
        im.save(folder / f"photo_{i:02d}.jpg")
    return folder


def test_reality_run_with_manifest_and_exports(capture_dataset: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REALITY_TEST_BACKEND", _BACKEND_SPEC)
    out_dir = tmp_path / "out_run"

    # Run unified CLI with exports
    code = main([
        "run",
        str(capture_dataset),
        "-o", str(out_dir),
        "--no-depth",
        "--no-mesh",
        "--export", "gltf",
        "--export", "obj",
        "--json",
    ])
    assert code == 0

    assert (out_dir / "worldir.json").exists()
    assert (out_dir / "report.json").exists()
    assert (out_dir / "points.ply").exists()
    assert (out_dir / "cameras.json").exists()

    # Verify exports were generated
    exports_dir = out_dir / "exports"
    assert exports_dir.exists()
    assert (exports_dir / "model.gltf").exists()
    assert (exports_dir / "model.obj").exists()

    # Check report
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "SUCCESS"
    assert report["images_ingested"] == 3
    assert report["stages"]["reconstruction"]["cameras_registered"] == 3


def test_reality_run_raw_photos_auto_discovery(raw_images_folder: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REALITY_TEST_BACKEND", _BACKEND_SPEC)
    out_dir = tmp_path / "out_raw"

    code = main([
        "run",
        str(raw_images_folder),
        "-o", str(out_dir),
        "--no-depth",
        "--no-mesh",
    ])
    assert code == 0

    assert (out_dir / "worldir.json").exists()
    world_data = json.loads((out_dir / "worldir.json").read_text(encoding="utf-8"))
    assert len(world_data.get("entities", {})) >= 0


def test_reality_run_with_terrain_fusion(capture_dataset: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REALITY_TEST_BACKEND", _BACKEND_SPEC)
    out_dir = tmp_path / "out_terrain"

    # Create terrain elevation JSON
    terrain_file = tmp_path / "terrain.json"
    terrain_data = {
        "elevation": [
            [5.0, 5.2, 5.4],
            [5.1, 5.3, 5.5],
            [5.0, 5.2, 5.4],
        ],
        "bounds": [-10.0, -10.0, 10.0, 10.0],
        "resolution_m": 10.0,
    }
    terrain_file.write_text(json.dumps(terrain_data), encoding="utf-8")

    code = main([
        "run",
        str(capture_dataset),
        "-o", str(out_dir),
        "--no-depth",
        "--no-mesh",
        "--terrain", str(terrain_file),
    ])
    assert code == 0

    world_data = json.loads((out_dir / "worldir.json").read_text(encoding="utf-8"))
    assert "terrain" in world_data.get("metadata", {})
    assert world_data["metadata"]["terrain"]["is_watertight"] is True
    assert "terrain-ground" in world_data.get("entities", {})


def test_reality_run_empty_folder_fails_honestly(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("REALITY_TEST_BACKEND", _BACKEND_SPEC)
    empty_folder = tmp_path / "empty_dir"
    empty_folder.mkdir()

    code = main([
        "run",
        str(empty_folder),
        "-o", str(tmp_path / "out_fail"),
    ])
    assert code == 1
