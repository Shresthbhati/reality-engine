"""End-to-end tests for the dense-MVS ingestion path (directive P5):
COLMAP's fused.ply enters the CANONICAL pipeline --

    fused.ply bytes
      -> parse_fused_ply (ReconstructedPoints)
      -> PointCloudData (deterministic canonical encoding)
      -> ArtifactStore (content-addressed)
      -> WorldIR Geometry(POINTCLOUD) with dense provenance

BACKEND EXISTS != FEATURE EXISTS: the CUDA COLMAP run is external
evidence; this chain is what makes it a Reality Engine capability.
"""

import struct

import pytest

from world_ir.artifact_store import MemoryArtifactStore
from world_ir.geometry_data import PointCloudData


def _fused_ply(n=500):
    """Synthetic fused.ply: a real-looking point cloud (a floor plane
    with jitter), binary little endian, vertex-only."""
    import random

    rng = random.Random(31)
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex %d\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n" % n
    ).encode("ascii")
    body = bytearray()
    for _ in range(n):
        x, y = rng.uniform(-2, 2), rng.uniform(-2, 2)
        z = 0.01 * rng.gauss(0, 1)
        body += struct.pack("<3f3B", x, y, z, 200, 200, 200)
    return header + bytes(body)


class TestFusedPlyToCanonical:
    def test_end_to_end_chain(self):
        from reconstruction.backend.dense_output import parse_fused_ply
        from reconstruction.dense_ingest import ingest_fused_ply

        data = _fused_ply()
        store = MemoryArtifactStore()
        geometry = ingest_fused_ply(
            data, artifact_store=store, source_evidence_ids=["img1", "img2", "img3"]
        )
        # Geometry is canonical POINTCLOUD, content-addressed.
        assert geometry.type.value == "pointcloud"
        assert geometry.data_uri and geometry.data_hash
        # The artifact round-trips to the canonical encoding.
        payload = store.get(geometry.data_uri)
        cloud = PointCloudData.from_bytes(payload)
        assert len(cloud.points) == 500
        # Hash equals the payload's digest (content-addressed store).
        from world_ir.artifact_store import _digest

        assert geometry.data_hash == _digest(payload)

    def test_provenance_names_dense_run_not_sparse(self):
        from reconstruction.backend.dense_output import parse_fused_ply
        from reconstruction.dense_ingest import ingest_fused_ply

        store = MemoryArtifactStore()
        geometry = ingest_fused_ply(
            _fused_ply(), artifact_store=store, source_evidence_ids=["img1"]
        )
        obs = geometry.observations[0]
        assert obs.sensor_type == "dense_mvs_fused"
        assert obs.metadata["source"] == "colmap_stereo_fusion"
        assert obs.metadata["n_points"] == 500
        # The MiDaS/monocular prior must NOT be claimed here: this is
        # true multi-view stereo output, and the record says so.
        assert "monocular" not in obs.metadata["source"]

    def test_bounds_recorded(self):
        from reconstruction.dense_ingest import ingest_fused_ply

        geometry = ingest_fused_ply(
            _fused_ply(200), artifact_store=MemoryArtifactStore(),
            source_evidence_ids=["img1"],
        )
        assert geometry.bounds_min is not None and geometry.bounds_max is not None
        assert geometry.bounds_min.x <= geometry.bounds_max.x
        # The floor plane spans roughly [-2, 2] in x.
        assert geometry.bounds_max.x > 1.5

    def test_empty_cloud_refused(self):
        from reconstruction.dense_ingest import ingest_fused_ply
        from reconstruction.backend.dense_output import DenseOutputError

        header = (
            b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
            b"property float x\nproperty float y\nproperty float z\n"
            b"end_header\n"
        )
        with pytest.raises(DenseOutputError):
            ingest_fused_ply(header, artifact_store=MemoryArtifactStore(),
                             source_evidence_ids=["img1"])

    def test_no_store_refused(self):
        from reconstruction.dense_ingest import ingest_fused_ply
        from reconstruction.dense_ingest import DenseIngestError

        with pytest.raises(DenseIngestError, match="artifact_store"):
            ingest_fused_ply(_fused_ply(10), artifact_store=None,
                             source_evidence_ids=["img1"])
