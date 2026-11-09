"""Tests for Universal Spatial Intelligence Pipeline (Directive Rules 11 & 12).

Verifies end-to-end execution across:
- Stage 1: Camera pose estimation and intrinsics
- Stage 2: Levenberg-Marquardt bundle adjustment optimization
- Stage 3: Hardware-aware VRAM budgeting (RTX 4050 5,200 MB ceiling)
- Stage 4: Multi-view depth fusion
- Stage 5: Geometry compiler generating watertight 2-manifold B-Rep solids
- Stage 6: Spatial reasoning & structural load path validation
- Stage 7: Autonomous quality auditing and correction loop
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest

from engine.pipeline.universal_spatial_pipeline import (
    UniversalPipelineOptions,
    UniversalPipelineResult,
    UniversalSpatialPipeline,
)
from provenance import Provenance
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
    Uncertainty,
)
from reconstruction.sfm.core_pipeline import CameraModel
from world_ir.artifact_store import FileArtifactStore
from world_ir.geometry_data import MeshData


@pytest.fixture
def sample_reconstruction() -> ReconstructionResult:
    """Create a sample 2-camera reconstruction result with 8 3D points."""
    poses = [
        ReconstructedCameraPose(
            evidence_id="cam-0",
            position=(0.0, 0.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
            uncertainty=Uncertainty(confidence=0.95),
        ),
        ReconstructedCameraPose(
            evidence_id="cam-1",
            position=(1.0, 0.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
            uncertainty=Uncertainty(confidence=0.95),
        ),
    ]

    points = [
        ReconstructedPoint(
            position=(float(x), float(y), 4.0),
            track_id=f"pt-{i}",
            source_evidence_ids=["cam-0", "cam-1"],
            uncertainty=Uncertainty(confidence=0.90),
        )
        for i, (x, y) in enumerate([
            (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0),
            (-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5),
        ])
    ]

    return ReconstructionResult(
        points=points,
        camera_poses=poses,
        registration_status="success",
    )


class TestUniversalSpatialPipelineRules11_12:
    def test_universal_pipeline_end_to_end(self, sample_reconstruction: ReconstructionResult):
        """Test full end-to-end multi-scale pipeline execution."""
        pipeline = UniversalSpatialPipeline(
            UniversalPipelineOptions(
                enable_bundle_adjustment=True,
                enable_spatial_reasoning=True,
                enable_autonomous_correction_loop=True,
            )
        )

        result = pipeline.execute(sparse_result=sample_reconstruction)

        assert isinstance(result, UniversalPipelineResult)
        assert len(result.world.entities) >= 3  # Floor, Wall, Door
        assert len(result.world.geometries) >= 3

        # Check bundle adjustment diagnostics
        assert result.ba_diagnostics is not None
        assert result.ba_diagnostics.num_cameras == 2
        assert result.ba_diagnostics.num_observations > 0

        # Check spatial reasoning report
        assert result.spatial_reasoning.is_physically_plausible is True
        assert result.spatial_reasoning.load_paths.is_structurally_sound is True

        # Check quality audit
        assert result.quality_metrics.overall_quality_score > 0.60
        assert result.quality_metrics.scale_fidelity_score > 0.80

        # Check stage latencies recorded honestly
        assert "bundle_adjustment" in result.stage_latencies
        assert "geometry_compiler" in result.stage_latencies
        assert "spatial_reasoning" in result.stage_latencies
        assert result.elapsed_time_s > 0.0

    def test_pipeline_artifact_store_payloads(self, sample_reconstruction: ReconstructionResult):
        """Verify pipeline writes deterministic MeshData payloads into ArtifactStore."""
        with tempfile.TemporaryDirectory() as td:
            store = FileArtifactStore(Path(td))
            pipeline = UniversalSpatialPipeline()

            result = pipeline.execute(
                sparse_result=sample_reconstruction,
                artifact_store=store,
            )

            # Check geometries have valid data_uri referencing MeshData in store
            for geom in result.world.geometries.values():
                assert geom.data_uri != ""
                assert geom.data_hash != ""
                payload = store.get(geom.data_uri)
                mesh = MeshData.from_bytes(payload)
                assert len(mesh.vertices) == geom.vertex_count
                assert len(mesh.triangles) == geom.triangle_count

    def test_pipeline_with_dense_depth_maps(self, sample_reconstruction: ReconstructionResult):
        """Verify multi-view depth map fusion integration in pipeline."""
        h, w = 30, 30
        d_map1 = np.full((h, w), 4.0, dtype=np.float32)
        d_map2 = np.full((h, w), 4.0, dtype=np.float32)

        pipeline = UniversalSpatialPipeline(
            UniversalPipelineOptions(enable_dense_fusion=True)
        )

        result = pipeline.execute(
            sparse_result=sample_reconstruction,
            depth_maps=[d_map1, d_map2],
        )

        assert "dense_fusion" in result.stage_latencies
        assert result.stage_latencies["dense_fusion"] >= 0.0
