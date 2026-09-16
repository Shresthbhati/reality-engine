"""Integration tests: the REAL dense-MVS capture (GPU COLMAP run of
2026-09-15, datasets/room_capture_mvs/dense/fused.ply -- 294,345 fused
points from patch_match_stereo + stereo_fusion) driven through the
full canonical ingestion chain:

    real fused.ply bytes
      -> parse_fused_ply          (reconstruction/backend/dense_output.py)
      -> ingest_fused_ply         (reconstruction/dense_ingest.py)
      -> PointCloudData + ArtifactStore (content-addressed)
      -> WorldIR Geometry(type=POINTCLOUD) with dense-MVS provenance

Honesty rules:
- the dataset is REAL captured output (a local artifact of the
  verified GPU run), never regenerated synthetically for the test;
- when the dataset is absent the whole module SKIPS with the exact
  path checked -- the integration is real-data-gated, not faked, and
  the skip reason names it;
- assertions are measured facts (counts, extents, provenance
  strings), tolerating whatever the run actually produced.

This closes ledger P6-01's PENDING verification item: "parse-to-fusion
integration on a real dense run".
"""

from __future__ import annotations

import os

import pytest

from reconstruction.backend.dense_output import parse_fused_ply
from reconstruction.dense_ingest import ingest_fused_ply
from world_ir.artifact_store import MemoryArtifactStore, _digest

REAL_FUSED_PLY = os.path.join(
    "datasets", "room_capture_mvs", "dense", "fused.ply"
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(REAL_FUSED_PLY),
    reason=(
        f"real dense-MVS capture not present at {REAL_FUSED_PLY} "
        "(gitignored artifact of the 2026-09-15 GPU COLMAP run) -- "
        "integration is real-data-gated, not simulated"
    ),
)


@pytest.fixture(scope="module")
def real_bytes():
    with open(REAL_FUSED_PLY, "rb") as f:
        return f.read()


class TestRealDenseCaptureIntegration:
    def test_real_run_parses_to_expected_count(self, real_bytes):
        pts = parse_fused_ply(real_bytes)
        # The run's recorded output: 294,345 fused points.
        assert len(pts) == 294345
        assert all(p.track_id.startswith("dense:") for p in pts[:10])

    def test_real_run_extent_is_plausible(self, real_bytes):
        # The SfM-scale room (uncalibrated scale, expected ~9x8x7
        # extent per the ledger) -- measured, not assumed exact.
        pts = parse_fused_ply(real_bytes)
        xs = [p.position[0] for p in pts]
        ys = [p.position[1] for p in pts]
        zs = [p.position[2] for p in pts]
        for axis in (xs, ys, zs):
            extent = max(axis) - min(axis)
            assert 0.1 < extent < 100.0, f"implausible extent {extent}"

    def test_real_run_ingests_to_canonical_geometry_with_provenance(
        self, real_bytes
    ):
        store = MemoryArtifactStore()
        geometry = ingest_fused_ply(
            real_bytes, artifact_store=store, source_evidence_ids=["img-real"]
        )
        assert geometry.type.value == "pointcloud"
        obs = geometry.observations[0]
        assert obs.sensor_type == "dense_mvs_fused"
        assert obs.metadata["source"] == "colmap_stereo_fusion"
        assert obs.metadata["n_points"] == 294345
        assert geometry.bounds_min is not None
        assert geometry.bounds_min.x <= geometry.bounds_max.x

    def test_real_run_artifact_is_content_addressed_and_roundtrips(
        self, real_bytes
    ):
        store = MemoryArtifactStore()
        geometry = ingest_fused_ply(
            real_bytes, artifact_store=store, source_evidence_ids=["img-real"]
        )
        payload = store.get(geometry.data_uri)
        assert geometry.data_hash == _digest(payload)
        from world_ir.geometry_data import PointCloudData

        cloud = PointCloudData.from_bytes(payload)
        assert len(cloud.points) == 294345
        # Same bytes -> same content hash: re-ingest dedupes.
        geometry2 = ingest_fused_ply(
            real_bytes, artifact_store=store, source_evidence_ids=["img-real"]
        )
        assert geometry2.data_hash == geometry.data_hash
