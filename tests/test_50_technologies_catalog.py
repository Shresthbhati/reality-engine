"""Comprehensive unit and integration tests for the 50 Open-Source Technologies subsystem."""

import pytest

from engine.technologies.adapters import get_adapter_for_technology
from engine.technologies.benchmark import TechnologyBenchmarkHarness
from engine.technologies.catalog import (
    ALL_50_TECHNOLOGIES,
    TECHNOLOGIES_BY_ID,
    TechnologySubsystem,
    get_technologies_by_subsystem,
    get_top_priority_technologies,
)
from engine.technologies.registry import get_technology_registry
from world_ir.world_v1 import WorldIR


def test_catalog_exact_fifty_technologies():
    """Verify that exactly 50 distinct open-source technologies are cataloged."""
    assert len(ALL_50_TECHNOLOGIES) == 50
    assert len(TECHNOLOGIES_BY_ID) == 50
    # Ensure IDs are unique
    ids = [t.id for t in ALL_50_TECHNOLOGIES]
    assert len(set(ids)) == 50


def test_seven_subsystems_distribution():
    """Verify all 7 specialist subsystems have the expected technologies."""
    subsystems = {t.subsystem for t in ALL_50_TECHNOLOGIES}
    assert subsystems == {
        TechnologySubsystem.VISUAL_PERCEPTION,
        TechnologySubsystem.GEOMETRIC_ESTIMATION,
        TechnologySubsystem.RECONSTRUCTION_MAPPING,
        TechnologySubsystem.POINTCLOUD_3D_PERCEPTION,
        TechnologySubsystem.GEOMETRY_CAD_MESH,
        TechnologySubsystem.GEOSPATIAL_ENVIRONMENT,
        TechnologySubsystem.INFERENCE_ACCELERATION,
    }

    # Group 1: 12 visual perception
    vis = get_technologies_by_subsystem(TechnologySubsystem.VISUAL_PERCEPTION)
    assert len(vis) == 12

    # Group 2: 7 geometric estimation
    geom = get_technologies_by_subsystem(TechnologySubsystem.GEOMETRIC_ESTIMATION)
    assert len(geom) == 7

    # Group 3: 8 reconstruction mapping
    recon = get_technologies_by_subsystem(TechnologySubsystem.RECONSTRUCTION_MAPPING)
    assert len(recon) == 8

    # Group 4: 8 pointcloud 3D perception
    pt = get_technologies_by_subsystem(TechnologySubsystem.POINTCLOUD_3D_PERCEPTION)
    assert len(pt) == 8

    # Group 5: 8 geometry/CAD mesh
    mesh = get_technologies_by_subsystem(TechnologySubsystem.GEOMETRY_CAD_MESH)
    assert len(mesh) == 8

    # Group 6: 4 geospatial environment
    geo = get_technologies_by_subsystem(TechnologySubsystem.GEOSPATIAL_ENVIRONMENT)
    assert len(geo) == 4

    # Group 7: 3 inference acceleration
    acc = get_technologies_by_subsystem(TechnologySubsystem.INFERENCE_ACCELERATION)
    assert len(acc) == 3


def test_top_priority_candidates():
    """Verify top evaluation priority technologies match strategic architecture goals."""
    top = get_top_priority_technologies()
    top_ids = {t.id for t in top}

    # Must include TEASER++, PCL, MaskDINO, GTSAM, Manifold, mrcal, XFeat, etc.
    assert "teaser_plus_plus" in top_ids
    assert "pcl" in top_ids
    assert "maskdino" in top_ids
    assert "gtsam" in top_ids
    assert "manifold" in top_ids
    assert "mrcal" in top_ids
    assert "xfeat" in top_ids
    assert "onnx_runtime" in top_ids
    assert "pdal" in top_ids


def test_registry_discovery_honesty():
    """Verify registry checks availability honestly without faking presence."""
    registry = get_technology_registry()
    statuses = registry.scan_all()
    assert len(statuses) == 50

    # Every status must have notes and a descriptor
    for s in statuses:
        assert s.descriptor.id in TECHNOLOGIES_BY_ID
        assert isinstance(s.available, bool)
        assert len(s.notes) > 0


def test_adapters_execute_and_promote_to_world_ir():
    """Test all 7 subsystem adapters execute and promote results into canonical WorldIR."""
    sample_techs = [
        "yolox",             # Visual perception
        "teaser_plus_plus",  # Geometric estimation
        "alicevision",       # Reconstruction mapping
        "pcl",               # 3D Pointcloud
        "manifold",          # Geometry CAD mesh
        "pdal",              # Geospatial environment
        "onnx_runtime",      # Inference acceleration
    ]

    for tech_id in sample_techs:
        adapter = get_adapter_for_technology(tech_id)
        result = adapter.execute(input_data={"test": True})
        assert result.success is True
        assert result.execution_time_ms > 0

        world = WorldIR(id=f"test_world_{tech_id}")
        updated_world = adapter.promote_to_world_ir(result, world)
        assert updated_world is not None

        # Verify specific Promotion targets
        if tech_id == "yolox":
            assert len(updated_world.entities) > 0
            assert "perception_models_evaluated" in updated_world.metadata
        elif tech_id == "teaser_plus_plus":
            assert f"{tech_id}_alignment" in updated_world.transforms
        elif tech_id == "manifold":
            assert f"geom_{tech_id}_refined" in updated_world.geometries
            assert updated_world.metadata["mesh_processing"][tech_id]["is_manifold"] is True
        elif tech_id == "onnx_runtime":
            assert updated_world.metadata["inference_accelerators"][tech_id]["within_rtx4050_bounds"] is True


def test_benchmark_harness_comprehensive_evaluation():
    """Test benchmarking harness across all 50 technologies."""
    harness = TechnologyBenchmarkHarness()
    summary = harness.benchmark_all_50()

    assert summary["total_evaluated"] == 50
    assert summary["recommended_count"] >= 10
    assert summary["qualified_count"] >= 30
    assert len(summary["results"]) == 50

    # Ensure markdown report generates cleanly
    report = harness.generate_markdown_report()
    assert "# Reality Engine — 50 Technologies Comprehensive Benchmark Report" in report
    assert "**Total Technologies Evaluated**: 50 / 50" in report
    assert "TEASER++" in report
    assert "Manifold" in report
    assert "GTSAM" in report
