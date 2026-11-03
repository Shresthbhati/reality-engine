"""Tests for Universal Pipeline Orchestrator (Universal Intelligence Expansion Epic E01-1).

Validates:
1. Autonomous execution from image directory with zero technical configuration.
2. Complete pipeline: Ingestion -> Registration -> Scale -> Compilation -> Spatial Reasoning -> Validation.
3. Metric vs Relative scale handling.
4. Material hypotheses enrichment.
5. Result export and atomic WorldStore persistence.
6. Telemetry and error recovery.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from engine.orchestrator.universal_pipeline import (
    PipelineMode,
    UniversalPipeline,
    UniversalPipelineOptions,
    UniversalPipelineResult,
)
from evidence.packages import DeterministicPackageBuilder
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.scale import ScaleReference, ScaleState
from sdk import reality
from world_ir.schema_v1 import EntityType, RelationshipKind
from worldstore.store import WorldStore


def _write_synthetic_frame(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    arr = rng.integers(60, 200, size=(48, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path, format="JPEG", quality=90)


class MockReconstructionBackend:
    backend_name = "test_recon_backend"

    def reconstruct(self, evidence):
        # Generate synthetic 4-wall room geometry with floor and ceiling (UP is y)
        poses = [
            ReconstructedCameraPose(
                evidence_id=e.id,
                position=(1.0 + float(i) * 0.1, 1.0, 1.0),
                rotation=(1.0, 0.0, 0.0, 0.0),
            )
            for i, e in enumerate(evidence)
        ]
        pts = []
        pid = 0
        ev_ids = [e.id for e in evidence[:2]]
        density_step = 0.1
        xs = np.arange(0.0, 2.5, density_step)
        zs = np.arange(0.0, 2.5, density_step)
        ys = np.arange(0.0, 2.1, 0.2)

        # Floor points (y = 0.0)
        for x in xs:
            for z in zs:
                pts.append(ReconstructedPoint(position=(float(x), 0.0, float(z)), track_id=f"p{pid}", source_evidence_ids=ev_ids, color=(100, 100, 100)))
                pid += 1

        # Ceiling points (y = 2.0)
        for x in xs:
            for z in zs:
                pts.append(ReconstructedPoint(position=(float(x), 2.0, float(z)), track_id=f"p{pid}", source_evidence_ids=ev_ids, color=(200, 200, 200)))
                pid += 1

        # Wall: z = 0.0
        for x in xs:
            for y in ys:
                pts.append(ReconstructedPoint(position=(float(x), float(y), 0.0), track_id=f"p{pid}", source_evidence_ids=ev_ids, color=(150, 120, 100)))
                pid += 1

        # Wall: z = 2.4
        for x in xs:
            for y in ys:
                pts.append(ReconstructedPoint(position=(float(x), float(y), 2.4), track_id=f"p{pid}", source_evidence_ids=ev_ids, color=(150, 120, 100)))
                pid += 1

        # Wall: x = 0.0
        for z in zs:
            for y in ys:
                pts.append(ReconstructedPoint(position=(0.0, float(y), float(z)), track_id=f"p{pid}", source_evidence_ids=ev_ids, color=(150, 120, 100)))
                pid += 1

        # Wall: x = 2.4
        for z in zs:
            for y in ys:
                pts.append(ReconstructedPoint(position=(2.4, float(y), float(z)), track_id=f"p{pid}", source_evidence_ids=ev_ids, color=(150, 120, 100)))
                pid += 1

        return ReconstructionResult(
            points=pts,
            camera_poses=poses,
            registration_status="success",
            registration_stats={"mean_reprojection_error_px": 0.25},
        )


def test_universal_pipeline_autonomous_run_from_folder(tmp_path):
    """Test full single-call reconstruction from an evidence directory."""
    img_dir = tmp_path / "photos"
    img_dir.mkdir()
    for i in range(5):
        _write_synthetic_frame(img_dir / f"img_{i:03d}.jpg", seed=i)

    backend = MockReconstructionBackend()
    options = UniversalPipelineOptions(
        session_name="test-autonomous-session",
        backends=[backend],
        enable_spatial_reasoning=True,
        enable_materials=True,
    )

    result = reality.reconstruct(img_dir, options=options)

    assert isinstance(result, UniversalPipelineResult)
    assert result.world is not None
    assert len(result.world.entities) >= 4  # Floor, ceiling, walls promoted

    # Check validation report
    assert result.validation_report is not None
    assert result.validation_report.is_valid

    # Check telemetry
    assert "total_runtime_s" in result.telemetry
    assert "device" in result.telemetry
    assert result.telemetry["total_runtime_s"] < 30.0

    # Check human summary
    summary = result.summary()
    assert "UniversalPipelineResult" in summary
    assert "entities" in summary


def test_universal_pipeline_scale_anchoring_and_export(tmp_path):
    """Test scale anchoring, export, and atomic persistence on pipeline result."""
    img_dir = tmp_path / "photos"
    img_dir.mkdir()
    for i in range(4):
        _write_synthetic_frame(img_dir / f"frame_{i:02d}.jpg", seed=10 + i)

    backend = MockReconstructionBackend()
    # Provide a measured scale reference between first two frames
    scale_ref = ScaleReference(
        evidence_id_a="frame_00.jpg",
        evidence_id_b="frame_01.jpg",
        distance_m=0.10,
    )

    options = UniversalPipelineOptions(
        backends=[backend],
        scale_reference=scale_ref,
        enable_spatial_reasoning=True,
    )

    result = UniversalPipeline.reconstruct(img_dir, options=options)

    assert result.scale_state == ScaleState.METRIC
    assert result.world.metadata["scale"]["state"] == "metric"

    # Test export to glTF
    gltf_data, report = result.export("gltf")
    assert report.format == "gltf"
    assert len(report.content_hash) == 64

    # Test export to floorplan SVG
    svg_data, svg_report = result.export("floorplan_svg")
    assert svg_report.format == "floorplan_svg"
    assert "<svg" in svg_data

    # Test atomic persistence to WorldStore
    store_dir = tmp_path / "store"
    version_id = result.save_to_store(store_dir)
    assert version_id is not None

    # Load back from store and verify integrity
    store = WorldStore(store_dir)
    loaded = store.load_version(version_id)
    assert loaded.id == result.world.id
    assert len(loaded.entities) == len(result.world.entities)



def test_universal_pipeline_with_gnss_georeferencing(tmp_path):
    """Test that the universal pipeline automatically detects GNSS metadata and georeferences the world."""
    from evidence.session import EvidenceItem, EvidenceKind
    from world_ir.coordinates import Frame

    # Create 4 synthetic evidence items with GNSS metadata
    evidence_items = []
    ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 20.0

    for i in range(4):
        item = EvidenceItem(
            id=f"gps_item_{i:02d}",
            kind=EvidenceKind.PHOTO,
            source_uri=f"file:///dummy/gps_{i:02d}.jpg",
            metadata={
                "latitude_deg": ref_lat + float(i) * 0.0001,
                "longitude_deg": ref_lon + float(i) * 0.0001,
                "altitude_m": ref_alt + float(i) * 0.5,
            },
        )
        evidence_items.append(item)

    backend = MockReconstructionBackend()
    options = UniversalPipelineOptions(
        backends=[backend],
        enable_spatial_reasoning=True,
    )

    result = UniversalPipeline.reconstruct(evidence_items, options=options)

    # Verify georeferencing was triggered and registered
    assert "georeferencing" in result.telemetry["stages"]
    assert result.telemetry["stages"]["georeferencing"]["success"] is True
    assert result.world.coordinate_frame == Frame.ENU
    assert "world_to_enu" in result.world.transforms
    assert "georeference" in result.world.metadata
    assert result.world.metadata["georeference"]["status"] == "georeferenced"
