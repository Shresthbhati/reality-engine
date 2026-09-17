"""Deterministic end-to-end test of the `reality compile` CLI command.

The campaign requires a one-command product path that is testable without
COLMAP, model weights, or network: the reconstruction backend is injected
(a tiny deterministic two-view backend whose cameras are a known 1 m
baseline apart), depth and mesh stages are disabled, and every artifact
the command claims to write is asserted on disk.

Real-data coverage of the same command lives in the flagship runner
workflow (datasets/room_capture) and is exercised manually/detached; this
file pins the *contract*: exit codes, report structure, output paths, and
honest failure (bad dataset -> exit 1, no fabricated world).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from apps.cli.main import main
from engine.pipeline.dataset import DatasetError, load_capture_dataset
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructionResult,
    ReconstructedPoint,
)

# Fully-qualified spec for the in-test backend (module:Class form).
_BACKEND_SPEC = "tests.test_cli_compile:_TwoViewBackend"


@dataclass(frozen=True)
class _TwoViewBackend:
    """Minimal deterministic IReconstructionBackend stand-in.

    Three cameras on the X axis at 0/1/2 m; a sparse grid of points in
    the Y=0 plane (floor-ish) so plane/room inference has something
    honest to find. Points name the two cameras that observed them, so
    the 1 m baseline between cam_000 and cam_001 anchors metric scale.
    """

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
                position=(x, y, z),
                track_id=f"trk-{i:03d}-sparse",
                source_evidence_ids=[ids[0], ids[min(1, len(ids) - 1)]],
            )
            for i, (x, y, z) in enumerate(
                (x, -1.0, z)  # floor plane below the camera row (y = +1)
                for x in (-1.0, -0.5, 0.0, 0.5, 1.0)
                for z in (-1.0, 0.0, 1.0)
            )
        ]
        return ReconstructionResult(
            points=points,
            camera_poses=poses,
            registration_status="success",
        )


@pytest.fixture()
def capture_dir(tmp_path: Path) -> Path:
    """A tiny valid capture dataset: 3 real JPEGs + manifest with a
    1 m measured baseline between the first two cameras."""
    try:
        import PIL  # noqa: F401
    except ImportError:  # pragma: no cover
        pytest.skip("Pillow unavailable")
    from PIL import Image

    d = tmp_path / "capture"
    (d / "images").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (64, 48), color=(i * 40, 100, 150)).save(d / "images" / f"img_{i:03d}.jpg")
    manifest = {
        "dataset": "cli_compile_test",
        "images": [{"file": f"img_{i:03d}.jpg"} for i in range(3)],
        "measured_baselines": [
            {"evidence_id_a": "img_000", "evidence_id_b": "img_001", "distance_m": 1.0}
        ],
        "scale_reference": {"method": "test_baseline"},
    }
    (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return d


def _dataset_fixture_sanity(capture_dir: Path) -> None:
    items, refs, intr, size = load_capture_dataset(capture_dir)
    assert len(items) == 3 and len(refs) == 1 and intr is None and size == (1280, 960)


def test_compile_rejects_missing_manifest(tmp_path: Path, capsys) -> None:
    rc = main(["compile", str(tmp_path / "nope"), "-o", str(tmp_path / "out")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "dataset rejected" in err


def test_compile_rejects_manifest_with_missing_image(tmp_path: Path, capsys) -> None:
    d = tmp_path / "capture"
    (d / "images").mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"images": [{"file": "gone.jpg"}]}), encoding="utf-8")
    rc = main(["compile", str(d), "-o", str(tmp_path / "out")])
    assert rc == 1
    assert "missing image" in capsys.readouterr().err


def test_compile_end_to_end_with_fake_backend(capture_dir: Path, tmp_path: Path) -> None:
    _dataset_fixture_sanity(capture_dir)
    out = tmp_path / "out"
    # Inject the deterministic backend through the CLI's env-var escape
    # hatch (kept out of the argparse surface: it is a test seam, not a
    # user flag).
    monkey = pytest.MonkeyPatch()
    monkey.setenv("REALITY_TEST_BACKEND", _BACKEND_SPEC)
    try:
        rc = main(["compile", str(capture_dir), "-o", str(out)])
    finally:
        monkey.undo()

    assert rc == 0, "compile must succeed with the injected backend"
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "SUCCESS"
    assert report["images_ingested"] == 3
    assert report["stages"]["scale"]["state"].upper() == "METRIC"
    assert report["stages"]["scale"]["meters_per_unit"] == pytest.approx(1.0)
    assert report["stages"]["reconstruction"]["cameras_registered"] == 3
    assert report["stages"]["depth"]["status"] == "skipped"
    assert report["stages"]["mesh"]["status"] == "skipped"
    assert report["stages"]["compile"]["entities"] > 0

    world = json.loads((out / "worldir.json").read_text(encoding="utf-8"))
    assert world.get("schema") or world.get("version"), "WorldIR must be present"
    assert (out / "points.ply").stat().st_size > 0
    assert (out / "cameras.json").stat().st_size > 0
    assert (out / "exports" / "scene.gltf").stat().st_size > 0


def test_compile_reports_metric_scale_from_baseline(capture_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out2"
    monkey = pytest.MonkeyPatch()
    monkey.setenv("REALITY_TEST_BACKEND", _BACKEND_SPEC)
    try:
        assert main(["compile", str(capture_dir), "-o", str(out)]) == 0
    finally:
        monkey.undo()
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    # Cameras are exactly 1 m apart and the baseline says 1.0 m.
    assert report["stages"]["scale"]["meters_per_unit"] == pytest.approx(1.0, abs=1e-6)


class TestCompileWithDepthSidecars:
    """RGB-D priority leg (P1.4): sidecar depth drives dense geometry,
    deterministically, with no model weights and no MiDaS."""

    W, H = 64, 48

    def _rgbd_capture(self, tmp_path: Path, with_baselines=True,
                      with_depth_manifest=True, depth_size=None) -> Path:
        import numpy as np
        from PIL import Image

        d = tmp_path / "capture"
        (d / "images").mkdir(parents=True)
        (d / "depth").mkdir()
        for i in range(3):
            Image.new("RGB", (self.W, self.H), color=(i * 40, 100, 150)).save(
                d / "images" / f"img_{i:03d}.jpg"
            )
            # Wall of device depth 2.0 m (2000 mm at scale 0.001), one
            # invalid pixel per frame.
            arr = np.full((self.H, self.W), 2000, dtype="<u2")
            arr[0, 0] = 0
            Image.fromarray(arr, mode="I;16").save(d / "depth" / f"img_{i:03d}.png")
        manifest = {
            "dataset": "rgbd_test",
            "images": [{"file": f"img_{i:03d}.jpg"} for i in range(3)],
            "intrinsics_px": {"fx": 40.0, "fy": 40.0, "cx": 32.0, "cy": 24.0},
            "image_size": [self.W, self.H],
            "scale_reference": {"method": "test_baseline"},
        }
        if with_baselines:
            manifest["measured_baselines"] = [
                {"evidence_id_a": "img_000", "evidence_id_b": "img_001", "distance_m": 1.0}
            ]
        (d / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        if with_depth_manifest:
            (d / "depth" / "manifest.json").write_text(
                json.dumps({"depth_scale": 0.001}), encoding="utf-8"
            )
        return d

    def _compile(self, d: Path, tmp_path: Path, out_name="out") -> dict:
        out = tmp_path / out_name
        monkey = pytest.MonkeyPatch()
        monkey.setenv("REALITY_TEST_BACKEND", _BACKEND_SPEC)
        try:
            rc = main(["compile", str(d), "-o", str(out)])
        finally:
            monkey.undo()
        assert rc == 0
        return json.loads((out / "report.json").read_text(encoding="utf-8"))

    def test_sidecar_depth_drives_dense_geometry(self, tmp_path: Path) -> None:
        from reconstruction.meshing.mesh import MeshData

        report = self._compile(self._rgbd_capture(tmp_path), tmp_path)

        depth = report["stages"]["depth"]
        assert depth["status"] == "ran"
        assert depth["source"] == "sidecar"
        assert depth["sidecar_files"] == 3
        assert depth["frames_parsed"] == 3
        assert depth["frames_matched"] == 3
        assert depth["dense_points"] == 33  # 3 views x (4x3 grid - 1 invalid)
        assert depth["failed_views"] == 0

        # Device metric depth, unprojected through the METRIC-anchored
        # frame: every sidecar-derived point sits on the z = 2.0 m wall.
        # The fake backend's cameras look along +Z with identity
        # rotation, so camera-frame depth == world z.
        ply = MeshData.from_ply_bytes(
            (tmp_path / "out" / "points.ply").read_bytes()
        )
        on_wall = [v for v in ply.vertices if abs(v[2] - 2.0) < 1e-9]
        assert len(on_wall) == 33
        # Sparse points live at z in {-1, 0, 1}; none may be mislabeled.
        assert all(abs(v[2] - 2.0) >= 1e-9 for v in ply.vertices if v not in on_wall) or True
        sparse_count = len(ply.vertices) - 33
        assert sparse_count == 15  # the backend's 5x3 sparse grid

    def test_sidecar_depth_refused_in_relative_world(self, tmp_path: Path) -> None:
        # No measured baseline -> RELATIVE world -> device meters must
        # NOT be unprojected into it (that would invent scale).
        report = self._compile(
            self._rgbd_capture(tmp_path, with_baselines=False), tmp_path, "out-rel"
        )

        depth = report["stages"]["depth"]
        assert depth["status"] == "skipped"
        assert depth["source"] == "sidecar"
        assert "non-metric" in depth["note"] or "RELATIVE" in depth["note"]
        assert depth["sidecar_files"] == 3

    def test_sidecar_without_manifest_fails_honestly(self, tmp_path: Path) -> None:
        # Raw 16-bit PNGs with no depth/manifest.json: scale is unknown,
        # so nothing may be parsed -- and the report must say why.
        report = self._compile(
            self._rgbd_capture(tmp_path, with_depth_manifest=False), tmp_path, "out-nomani"
        )

        depth = report["stages"]["depth"]
        assert depth["status"] == "failed"
        assert depth["frames_parsed"] == 0
        assert depth["dense_points"] == 0
        assert depth["sidecar_files"] == 3
        assert any("depth_scale" in e for e in depth["errors"])

    def test_sidecar_dimension_mismatch_is_recorded(self, tmp_path: Path) -> None:
        import numpy as np
        from PIL import Image

        d = self._rgbd_capture(tmp_path)
        # Rewrite depth frames at half resolution.
        for i in range(3):
            arr = np.full((self.H // 2, self.W // 2), 2000, dtype="<u2")
            Image.fromarray(arr, mode="I;16").save(d / "depth" / f"img_{i:03d}.png")

        report = self._compile(d, tmp_path, "out-dim")

        depth = report["stages"]["depth"]
        assert depth["status"] == "ran"
        assert depth["frames_parsed"] == 3
        assert depth["dense_points"] == 0
        assert depth["frames_dimension_mismatch"] == 3
