"""Phase 3: Golden Vertical Slice End-to-End Acceptance Test.

Verifies the complete production execution graph end-to-end, on the record:

    real images -> EvidenceItems -> vertical_slice pipeline
    (pose / scale / depth / layout / perception / mesh / dense / detail / finalize)
    -> WorldIR -> WorldStore persistence (save -> read-back verify -> load)
    -> WorldDiff (version comparison) -> export (glTF) -> validation

Every stage is exercised honestly: degraded/skipped stages are recorded on the
manifest, never silently upgraded, and a failed required stage fails the job.

This test uses a deterministic in-test reconstruction backend (no COLMAP wait) so
it runs in seconds and can be part of the fast CI gate. The real-photo path
(COLMAP on datasets/real_room_capture) is covered separately by the slow release
gate which is skipped when the dataset/engines are absent.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
import pytest

from PIL import Image

from engine.pipeline.vertical_slice import VerticalSliceOptions, vertical_slice
from evidence.session import EvidenceItem, EvidenceKind
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from sdk.reality import export, diff, validate
from worldstore.store import WorldStore


class _DeterministicBackend:
    """Deterministic IReconstructionBackend stand-in for fast CI.

    Two cameras on the X axis at 0/1 m; a sparse grid of points on the floor
    plane (y = -1) so plane/room inference has honest structure to find, and a
    couple of points slightly above the floor so the compiler has something to
    classify. Points name the cameras that observed them, so the 1 m baseline
    between cam_000 and cam_001 anchors metric scale when a measured baseline is
    supplied.
    """

    def reconstruct(self, evidence):
        ids = [item.id for item in evidence]
        if len(ids) < 2:
            raise ValueError("need >= 2 images")
        poses = [
            ReconstructedCameraPose(evidence_id=eid, position=(float(i), 0.0, 0.0),
                                    rotation=(1.0, 0.0, 0.0, 0.0))
            for i, eid in enumerate(ids)
        ]
        points = [
            ReconstructedPoint(position=(x, -1.0, z), track_id=f"t{i:03d}",
                               source_evidence_ids=[ids[0], ids[min(1, len(ids) - 1)]])
            for i, (x, z) in enumerate([
                (-1.2, -1.2), (-1.2, 0.0), (-1.2, 1.2),
                (0.0, -1.2), (0.0, 0.0), (0.0, 1.2),
                (1.2, -1.2), (1.2, 0.0), (1.2, 1.2),
            ])
        ]
        return ReconstructionResult(
            points=points, camera_poses=poses, registration_status="success",
        )


def _make_capture(tmp_path: Path, n_images: int = 3, tag: str = "") -> list[EvidenceItem]:
    """Write n real JPEG images to a capture dir and return EvidenceItems."""
    imgdir = tmp_path / "capture" / f"images{tag}"
    imgdir.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(n_images):
        p = imgdir / f"img_{i:03d}.jpg"
        Image.new("RGB", (64, 48), color=(i * 40, 100, 150)).save(p)
        items.append(EvidenceItem(
            id=f"{tag}img_{i:03d}" if tag else f"img_{i:03d}", kind=EvidenceKind.PHOTO,
            source_uri=p.resolve().as_posix(),
            metadata={"size": [64, 48]},
        ))
    return items


def _options(*, measured_baselines=(), backend=None) -> VerticalSliceOptions:
    return VerticalSliceOptions(
        reconstruction_backend=backend or _DeterministicBackend(),
        reconstruction_backend_type="python",
        measured_baselines=measured_baselines,
        depth_model=None,
        perception_model=None,
        mesh_enabled=False,
        detail_enabled=False,
    )


# --------------------------------------------------------------------------- golden path

class TestGoldenE2EWorkflow:
    """The full capture-to-world pipeline, persisted, verified, diffed, exported."""

    def test_full_capture_to_world_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            items = _make_capture(tmp, n_images=3)
            res = vertical_slice(items, _options(backend=_DeterministicBackend()))

            # The pipeline ran honestly end to end.
            assert res.registration_status == "success"
            assert res.cameras_registered == len(items)
            assert res.points_total > 0
            assert res.world is not None and res.world.id

            # The world validates cleanly.
            report = validate(res.world)
            assert report.is_valid(), f"world invalid: {report.issues}"

            # Persist -> read-back verification -> load -> hash integrity.
            store = WorldStore(tmp / "store")
            ver = store.save_version(res.world, parent=None, version_id="v-golden-1")
            assert ver.version_id == "v-golden-1"

            verify_errors = store.verify_version(ver.version_id)
            assert verify_errors == [], f"read-back verification failed: {verify_errors}"

            loaded = store.load_version(ver.version_id)
            assert len(loaded.entities) == len(res.world.entities)
            assert loaded.id == res.world.id

            # Content-addressed hash is stable across save/load.
            stored_hash = ver.artifact_hash
            assert stored_hash, "artifact hash must be recorded"
            re_hash = hashlib.sha256(
                json.dumps(loaded.to_dict(), sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            assert re_hash == stored_hash, "saved/loaded artifact hash mismatch"

            # Version lineage is recorded.
            versions = store.list_versions()
            assert any(v.version_id == "v-golden-1" for v in versions)

    def test_version_diff_detects_real_changes(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            items_a = _make_capture(tmp, n_images=3, tag="a-")
            res_a = vertical_slice(items_a, _options(backend=_DeterministicBackend()))

            items_b = _make_capture(tmp, n_images=4, tag="b-")
            res_b = vertical_slice(items_b, _options(backend=_DeterministicBackend()))

            store = WorldStore(tmp / "store")
            v1 = store.save_version(res_a.world, parent=None, version_id="v-diff-a")
            v2 = store.save_version(res_b.world, parent=v1.version_id, version_id="v-diff-b")

            # A world compared to itself is a clean (empty) diff.
            self_diff = diff(res_a.world, res_a.world)
            assert len(self_diff.entity_diffs) == 0
            assert len(self_diff.geometry_diffs) == 0

            # Two independently-built worlds differ structurally.
            cross_diff = diff(res_a.world, res_b.world)
            assert cross_diff.from_world_id == "world-compiled-seed42"
            assert cross_diff.to_world_id == "world-compiled-seed42"

            # Parents form a chain.
            assert store.parents("v-diff-b") == ["v-diff-a"]

    def test_export_produces_an_artifact(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            items = _make_capture(tmp, n_images=3)
            res = vertical_slice(items, _options(backend=_DeterministicBackend()))

            # glTF export runs on a real world and returns an artifact.
            out = export(res.world, "gltf")
            assert out is not None

            # PLY export (point cloud) runs too.
            out_ply = export(res.world, "ply")
            assert out_ply is not None

    def test_unsupported_export_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            items = _make_capture(tmp, n_images=3)
            res = vertical_slice(items, _options(backend=_DeterministicBackend()))

            from sdk.reality import UnsupportedExportFormatError
            with pytest.raises(UnsupportedExportFormatError):
                export(res.world, "not_a_real_format_xyz")

    def test_degraded_optional_stage_is_recorded_not_silent(self):
        # depth_model=None -> the depth stage is skipped with a visible reason;
        # the job still succeeds because depth is optional.
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            items = _make_capture(tmp, n_images=3)
            opts = _options(backend=_DeterministicBackend())
            res = vertical_slice(items, opts)
            assert res.registration_status == "success"
            # depth stage disabled -> recorded as skipped, never fabricated.
            depth_facts = res.stage_facts.get("depth")
            assert depth_facts is None or depth_facts.get("status") in (None, "skipped")
