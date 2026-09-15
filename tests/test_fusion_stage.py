"""Pipeline-level tests for the P6-02 fusion stage in
engine/pipeline/vertical_slice.py (_fusion_stage): plural-source
fusion -> WorldIR POINTCLOUD geometry write-back, with honest skip
reasons and artifact persistence.

Mirrors test_meshing_pipeline.py's fake-result fixture style; the real
COLMAP path stays a dependency-gated concern of the mesh stage, not
faked here.
"""

import types

import pytest

from world_ir.artifact_store import MemoryArtifactStore


def _metric_world():
    from world_ir.world_v1 import WorldIR

    w = WorldIR(id="w-fusion")
    w.metadata["scale"] = {"state": "metric", "meters_per_unit": 1.0}
    return w


def _points(n_sparse=30, n_depth=30, seed=3):
    """A cloud with both track-id families, deterministic."""
    import numpy as np

    from provenance import Uncertainty

    unc = Uncertainty(confidence=0.8, note="test fixture: sigma ~0.02 m")
    rng = np.random.default_rng(seed)
    pts = []
    for i in range(n_sparse):
        pts.append(types.SimpleNamespace(
            position=(float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)), float(rng.uniform(1.5, 2.5))),
            track_id=f"track-{i}",
            uncertainty=unc,
            source_evidence_ids=["img1"],
        ))
    for i in range(n_depth):
        pts.append(types.SimpleNamespace(
            position=(float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)), float(rng.uniform(1.5, 2.5))),
            track_id=f"depth-cam0-{i:05d}-00001",
            uncertainty=unc,
            source_evidence_ids=["depth-cam0"],
        ))
    return pts


def _fake_result(points):
    return types.SimpleNamespace(
        points=points,
        camera_poses=[types.SimpleNamespace(position=(0.0, 0.0, 0.0))],
    )


class TestFusionStage:
    def test_writeback_creates_pointcloud_geometry(self):
        import engine.pipeline.vertical_slice as vs

        world = _metric_world()
        options = vs.VerticalSliceOptions(artifact_store=MemoryArtifactStore())
        facts = vs._fusion_stage(
            _fake_result(_points()), world, options, "metric"
        )
        assert facts["status"] == "ran"
        assert facts["n_associated"] >= 1
        geom = world.geometries["geom-fused-pointcloud"]
        assert geom.type.value == "pointcloud"
        assert geom.data_hash, "geometry must carry the artifact hash"
        assert geom.observations[0].metadata["n_conflicts"] == facts["n_conflicts"]

    def test_relative_world_skips(self):
        import engine.pipeline.vertical_slice as vs

        world = _metric_world()
        options = vs.VerticalSliceOptions(artifact_store=MemoryArtifactStore())
        facts = vs._fusion_stage(
            _fake_result(_points()), world, options, "relative"
        )
        assert facts["status"] == "skipped"
        assert "METRIC" in facts["note"]
        assert "geom-fused-pointcloud" not in world.geometries

    def test_single_source_skips(self):
        import engine.pipeline.vertical_slice as vs

        world = _metric_world()
        options = vs.VerticalSliceOptions(artifact_store=MemoryArtifactStore())
        only_sparse = [p for p in _points() if not p.track_id.startswith("depth-")]
        facts = vs._fusion_stage(
            _fake_result(only_sparse), world, options, "metric"
        )
        assert facts["status"] == "skipped"
        assert "plural" in facts["note"]

    def test_no_store_skips(self):
        import engine.pipeline.vertical_slice as vs

        world = _metric_world()
        options = vs.VerticalSliceOptions(artifact_store=None)
        facts = vs._fusion_stage(
            _fake_result(_points()), world, options, "metric"
        )
        assert facts["status"] == "skipped"
        assert "artifact_store" in facts["note"]
