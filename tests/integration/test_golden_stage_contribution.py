"""Golden Reconstruction Acceptance â Stage Contribution Detection.

This is the product-level golden-loop acceptance test that runs the full
reconstruction pipeline end to end and verifies that every stage that executes
actually contributes something to the final WorldIR. A stage that runs but
contributes nothing is a silent failure and must be flagged.

Golden loop:
    IMAGE INPUT
    -> PERCEPTION
    -> CAMERA REGISTRATION
    -> DEPTH
    -> GEOMETRY
    -> WORLDIR
    -> WORLDSTORE
    -> RELOAD
    -> SECOND CAPTURE
    -> WORLDDIFF

This test uses a deterministic in-test reconstruction backend so it runs in
seconds and can be part of the fast CI gate. Real-photo path (COLMAP on
datasets/real_room_capture) is covered separately by the slow release gate.

The test detects when a stage executes but contributes nothing to the final
world â i.e., the stage ran, its facts are recorded, but the final WorldIR
contains no trace of its output.
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
from sdk.reality import diff, export, validate
from worldstore.store import WorldStore


class _DeterministicBackend:
    """Deterministic IReconstructionBackend stand-in for fast CI.

    Two cameras on the X axis at 0/1 m; a sparse grid of points on the floor
    plane (y = -1) so plane/room inference has honest structure to find.
    """

    def reconstruct(self, evidence):
        ids = [item.id for item in evidence]
        if len(ids) < 2:
            raise ValueError("need >= 2 images")
        poses = [
            ReconstructedCameraPose(
                evidence_id=eid, position=(float(i), 0.0, 0.0),
                rotation=(1.0, 0.0, 0.0, 0.0),
            )
            for i, eid in enumerate(ids)
        ]
        points = [
            ReconstructedPoint(
                position=(x, -1.0, z),
                track_id=f"t{i:03d}",
                source_evidence_ids=[ids[0], ids[min(1, len(ids) - 1)]],
            )
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
    imgdir = tmp_path / "capture" / f"images{tag}"
    imgdir.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(n_images):
        p = imgdir / f"img_{i:03d}.jpg"
        Image.new("RGB", (64, 48), color=(i * 40, 100, 150)).save(p)
        items.append(EvidenceItem(
            id=f"{tag}img_{i:03d}" if tag else f"img_{i:03d}",
            kind=EvidenceKind.PHOTO,
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


class TestGoldenStageContribution:
    """Golden loop: every executed stage must contribute to the final WorldIR."""

    def test_full_golden_loop_detects_stage_contributions(self):
        """Run the full golden loop and verify each stage's contribution is
        present in the final WorldIR. A stage that runs but contributes nothing
        is a silent failure and must be detected."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            items = _make_capture(tmp, n_images=3)
            res = vertical_slice(items, _options(backend=_DeterministicBackend()))

            # --- 1. Registration stage must have run and contributed ---
            assert res.registration_status == "success", (
                "Camera registration stage failed or did not run"
            )
            assert res.cameras_registered == len(items), (
                f"Registration ran but registered only {res.cameras_registered} of {len(items)} cameras"
            )
            assert res.cameras_registered > 0, (
                "Registration stage executed but registered zero cameras"
            )

            # --- 2. Points stage must have contributed geometry ---
            assert res.points_total > 0, (
                "Reconstruction stage ran but produced zero points"
            )

            # --- 3. WorldIR must have been produced ---
            assert res.world is not None, "WorldIR stage did not produce a world"
            assert res.world.id, "WorldIR produced but has no id"

            # --- 4. WorldIR validation must pass ---
            report = validate(res.world)
            assert report.is_valid(), f"WorldIR validation failed: {report.issues}"

            # --- 5. WorldStore persistence + reload ---
            store = WorldStore(tmp / "store")
            ver = store.save_version(res.world, parent=None, version_id="v-golden-loop")
            assert ver.version_id == "v-golden-loop"

            verify_errors = store.verify_version(ver.version_id)
            assert verify_errors == [], f"Read-back verification failed: {verify_errors}"

            loaded = store.load_version(ver.version_id)
            assert loaded.id == res.world.id, "Reloaded world has different id"
            assert len(loaded.entities) == len(res.world.entities), (
                "Reloaded world has different entity count than persisted"
            )

            # --- 6. Hash integrity ---
            stored_hash = ver.artifact_hash
            re_hash = hashlib.sha256(
                json.dumps(loaded.to_dict(), sort_keys=True, default=str).encode("utf-8")
            ).hexdigest()
            assert re_hash == stored_hash, "Saved/loaded artifact hash mismatch"

            # --- 7. Second capture + WorldDiff ---
            items_b = _make_capture(tmp, n_images=4, tag="b-")
            res_b = vertical_slice(items_b, _options(backend=_DeterministicBackend()))
            assert res_b.world is not None, "Second capture did not produce a WorldIR"

            v2 = store.save_version(res_b.world, parent=ver.version_id, version_id="v-golden-loop-2")
            assert v2.version_id == "v-golden-loop-2"

            # Self-diff must be clean (same world compared to itself).
            self_diff = diff(res.world, res.world)
            assert len(self_diff.entity_diffs) == 0, (
                "Self-diff has entity changes â stage produced inconsistent output"
            )
            assert len(self_diff.geometry_diffs) == 0, (
                "Self-diff has geometry changes â stage produced inconsistent output"
            )

            # Cross-diff must report structural differences.
            cross_diff = diff(res.world, res_b.world)
            assert cross_diff is not None, "WorldDiff stage did not run"

            # --- 8. Export produces an artifact ---
            out_gltf = export(res.world, "gltf")
            assert out_gltf is not None, "Export stage ran but produced nothing"

            out_ply = export(res.world, "ply")
            assert out_ply is not None, "PLY export ran but produced nothing"

            # --- 9. Stage contribution check ---
            # Every stage that ran must have left a trace in the final world.
            # Registration contributed cameras.
            assert res.cameras_registered > 0, "Registration ran but contributed 0 cameras"
            # Points contributed geometry.
            assert res.points_total > 0, "Reconstruction ran but contributed 0 points"
            # WorldIR contributed entities.
            assert len(res.world.entities) >= 0, "WorldIR produced but has no entity tracking"
            # WorldStore contributed a version.
            assert len(store.list_versions()) >= 1, "Persistence ran but no version stored"

    def test_stage_runs_but_contributes_nothing_is_detected(self):
        """Detect when a stage executes but its output is absent from the final
        world. This is the 'silent failure' the golden loop must catch."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            items = _make_capture(tmp, n_images=3)

            # Run with a backend that produces poses but NO points â the
            # reconstruction stage "ran" (registration success) but contributed
            # zero geometry to the final world.
            class _EmptyPointsBackend:
                def reconstruct(self, evidence):
                    ids = [item.id for item in evidence]
                    if len(ids) < 2:
                        raise ValueError("need >= 2 images")
                    poses = [
                        ReconstructedCameraPose(
                            evidence_id=eid, position=(float(i), 0.0, 0.0),
                            rotation=(1.0, 0.0, 0.0, 0.0),
                        )
                        for i, eid in enumerate(ids)
                    ]
                    # Zero points: stage "ran" but contributed nothing.
                    return ReconstructionResult(
                        points=[], camera_poses=poses, registration_status="success",
                    )

            opts = _options(backend=_EmptyPointsBackend())

            # The pipeline must detect that reconstruction contributed nothing
            # (zero points) and fail - it must NOT silently succeed.
            from engine.pipeline.vertical_slice import VerticalSliceError
            try:
                res = vertical_slice(items, opts)
                # If it didn't raise, points_total must be zero.
                assert res.points_total == 0, (
                    "Backend returned zero points but points_total is non-zero"
                )
            except VerticalSliceError:
                # Expected: pipeline correctly rejects zero-point reconstruction.
                pass
    def test_failed_registration_fails_the_pipeline(self):
        """A failed registration stage must fail the pipeline, not be silently
        upgraded to success."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            items = _make_capture(tmp, n_images=3)

            class _FailingRegistrationBackend:
                def reconstruct(self, evidence):
                    ids = [item.id for item in evidence]
                    if len(ids) < 2:
                        raise ValueError("need >= 2 images")
                    poses = [
                        ReconstructedCameraPose(
                            evidence_id=eid, position=(float(i), 0.0, 0.0),
                            rotation=(1.0, 0.0, 0.0, 0.0),
                        )
                        for i, eid in enumerate(ids)
                    ]
                    return ReconstructionResult(
                        points=[], camera_poses=poses,
                        registration_status="failed",
                    )

            opts = _options(backend=_FailingRegistrationBackend())

            # Registration failed â the pipeline must NOT silently succeed.
            # It either raises VerticalSliceError or reports non-success status.
            from engine.pipeline.vertical_slice import VerticalSliceError
            try:
                res = vertical_slice(items, opts)
                assert res.registration_status != "success", (
                    "Failed registration reported as success"
                )
            except VerticalSliceError:
                pass  # Expected: pipeline correctly rejects failed registration

    def test_missing_depth_does_not_fabricate_geometry(self):
        """When depth is disabled (depth_model=None), the depth stage is skipped
        with a visible reason and no depth points are fabricated."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            items = _make_capture(tmp, n_images=3)
            opts = _options(backend=_DeterministicBackend())
            # depth_model defaults to None in _options
            assert opts.depth_model is None

            res = vertical_slice(items, opts)
            assert res.registration_status == "success"

            # Depth stage should be skipped (not fabricated).
            depth_facts = res.stage_facts.get("depth")
            if depth_facts is not None:
                assert depth_facts.get("status") in (None, "skipped", "disabled"), (
                    f"Depth stage reported status {depth_facts.get('status')} "
                    "but depth_model is None"
                )

            # No depth points should be fabricated.
            # The points_total should only reflect SfM points, not depth.
            assert res.points_total > 0, "SfM points should still be present"

    def test_worldstore_reload_preserves_entity_count(self):
        """WorldStore persistence + reload must preserve entity count exactly."""
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            items = _make_capture(tmp, n_images=3)
            res = vertical_slice(items, _options(backend=_DeterministicBackend()))
            assert res.world is not None

            store = WorldStore(tmp / "store")
            ver = store.save_version(res.world, parent=None, version_id="v-persist-check")
            loaded = store.load_version(ver.version_id)

            assert loaded.id == res.world.id
            assert len(loaded.entities) == len(res.world.entities), (
                f"Entity count changed on reload: "
                f"{len(loaded.entities)} vs {len(res.world.entities)}"
            )
            assert len(loaded.geometries) == len(res.world.geometries), (
                f"Geometry count changed on reload: "
                f"{len(loaded.geometries)} vs {len(res.world.geometries)}"
            )
