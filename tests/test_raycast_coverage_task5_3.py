"""Unit tests for Ray-cast surface coverage and evidence state evaluation (Blueprint Task 5.3).

Verifies:
1. Ray-cast coverage correctly identifies OBSERVED, PARTIALLY_OBSERVED, OCCLUDED, and UNOBSERVED surfaces.
2. In a synthetic room with an obstacle (wardrobe) in front of the wall, the hidden zone is detected as occluded.
3. Surfaces behind the camera field of view are correctly classified as UNOBSERVED.
4. Updates world.hierarchy nodes, world.surfaces, and surface entities with exact coverage fractions and evidence states.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from engine.pipeline.coverage import (
    AABB,
    CameraView,
    Ray,
    compute_surface_coverage,
    compute_world_surface_coverage,
)
from perception.architecture.adapters import adapt_architecture_to_contract
from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import Entity, EntityType


def test_ray_aabb_intersection():
    """Verify slab ray-AABB intersection math."""
    box = AABB(min_pt=np.array([1.0, 1.0, 0.0]), max_pt=np.array([2.0, 2.0, 2.0]))

    # Ray hitting box directly
    ray_hit = Ray(origin=np.array([0.0, 1.5, 1.0]), direction=np.array([1.0, 0.0, 0.0]))
    t = box.intersect(ray_hit)
    assert t is not None
    assert t == pytest.approx(1.0)

    # Ray pointing away from box
    ray_miss = Ray(origin=np.array([0.0, 1.5, 1.0]), direction=np.array([-1.0, 0.0, 0.0]))
    assert box.intersect(ray_miss) is None

    # Ray glancing above box
    ray_above = Ray(origin=np.array([0.0, 1.5, 3.0]), direction=np.array([1.0, 0.0, 0.0]))
    assert box.intersect(ray_above) is None


def test_surface_coverage_with_known_hidden_region():
    """Verify ray-cast coverage on a synthetic room with a known hidden region."""
    # North wall at y = 3.0, spanning x in [-2, 2], z in [0, 2.5]
    # Normal points South: (0.0, -1.0, 0.0)
    wall_bmin = (-2.0, 3.0, 0.0)
    wall_bmax = (2.0, 3.0, 2.5)
    wall_normal = (0.0, -1.0, 0.0)

    # Camera at (0.0, 0.0, 1.2) looking North (0.0, 1.0, 0.0)
    cam = CameraView.from_pose(
        camera_id="cam-01",
        position=(0.0, 0.0, 1.2),
        forward_vector=(0.0, 1.0, 0.0),
        fov_deg=80.0,
    )

    # 1. Unobstructed case
    clean_res = compute_surface_coverage(
        surface_id="surf-north",
        entity_id="wall-north",
        bounds_min=wall_bmin,
        bounds_max=wall_bmax,
        normal=wall_normal,
        cameras=[cam],
        occluders=[],
        grid_size=6,
    )
    assert clean_res.coverage_fraction >= 0.90
    assert clean_res.evidence_state == "OBSERVED"
    assert clean_res.occluded_samples == 0

    # 2. Obstructed case: Wardrobe placed at (0.0, 1.5, 0.0) blocking center of wall
    wardrobe_box = AABB(
        min_pt=np.array([-1.0, 1.2, 0.0]),
        max_pt=np.array([1.0, 1.8, 2.2]),
    )
    occluded_res = compute_surface_coverage(
        surface_id="surf-north",
        entity_id="wall-north",
        bounds_min=wall_bmin,
        bounds_max=wall_bmax,
        normal=wall_normal,
        cameras=[cam],
        occluders=[("wardrobe-01", wardrobe_box)],
        grid_size=6,
    )
    assert occluded_res.occluded_samples > 0
    assert "wardrobe-01" in occluded_res.occluding_entity_ids
    assert occluded_res.coverage_fraction < clean_res.coverage_fraction
    assert occluded_res.evidence_state in ("PARTIALLY_OBSERVED", "OCCLUDED")

    # 3. Behind camera case: South wall at y = -3.0
    south_bmin = (-2.0, -3.0, 0.0)
    south_bmax = (2.0, -3.0, 2.5)
    south_normal = (0.0, 1.0, 0.0)

    unobserved_res = compute_surface_coverage(
        surface_id="surf-south",
        entity_id="wall-south",
        bounds_min=south_bmin,
        bounds_max=south_bmax,
        normal=south_normal,
        cameras=[cam],
        occluders=[],
        grid_size=6,
    )
    assert unobserved_res.observed_samples == 0
    assert unobserved_res.coverage_fraction == 0.0
    assert unobserved_res.evidence_state == "UNOBSERVED"


def test_world_surface_coverage_pipeline_integration():
    """Verify compute_world_surface_coverage populates WorldIR hierarchy and surfaces."""
    world = WorldIR(id="world-coverage-test")

    # Add North wall (facing South) and South wall (facing North)
    world.entities["wall-north"] = Entity(
        id="wall-north",
        name="North Wall",
        type=EntityType.WALL,
        confidence=0.9,
        provenance=Provenance.OBSERVED,
        custom_properties={
            "bounds_min": [-2.0, 3.0, 0.0],
            "bounds_max": [2.0, 3.0, 2.5],
            "normal": [0.0, -1.0, 0.0],
            "evidence_state": "OBSERVED",
        },
    )

    world.entities["wall-south"] = Entity(
        id="wall-south",
        name="South Wall",
        type=EntityType.WALL,
        confidence=0.9,
        provenance=Provenance.OBSERVED,
        custom_properties={
            "bounds_min": [-2.0, -3.0, 0.0],
            "bounds_max": [2.0, -3.0, 2.5],
            "normal": [0.0, 1.0, 0.0],
            "evidence_state": "OBSERVED",
        },
    )

    # Add Wardrobe occluder in front of North wall
    world.entities["wardrobe-01"] = Entity(
        id="wardrobe-01",
        name="Wardrobe",
        type=EntityType.STRUCTURE,
        confidence=0.9,
        provenance=Provenance.OBSERVED,
        custom_properties={
            "bounds_min": [-1.0, 1.2, 0.0],
            "bounds_max": [1.0, 1.8, 2.2],
        },
    )

    # Adapt into contract hierarchy first
    adapt_architecture_to_contract(world)

    # Run ray-cast coverage with camera at origin looking North
    cameras = [("cam-01", (0.0, 0.0, 1.2), (0.0, 1.0, 0.0))]
    report = compute_world_surface_coverage(world, cameras, fov_deg=80.0, grid_size=6)

    assert "wall-north" in report.surfaces
    assert "wall-south" in report.surfaces

    north_cov = report.surfaces["wall-north"]
    south_cov = report.surfaces["wall-south"]

    # North wall is partially occluded by wardrobe
    assert north_cov.occluded_samples > 0
    assert north_cov.evidence_state in ("PARTIALLY_OBSERVED", "OCCLUDED")

    # South wall is unobserved
    assert south_cov.coverage_fraction == 0.0
    assert south_cov.evidence_state == "UNOBSERVED"

    # In-place updates on world.hierarchy
    north_node = world.hierarchy.nodes["node-wall-north"]
    assert north_node.coverage == pytest.approx(north_cov.coverage_fraction)
    assert north_node.evidence_state == north_cov.evidence_state

    south_node = world.hierarchy.nodes["node-wall-south"]
    assert south_node.coverage == 0.0
    assert south_node.evidence_state == "UNOBSERVED"

    # In-place updates on world.surfaces
    assert world.surfaces["surf-wall-north"].custom_attributes["coverage"] == north_cov.coverage_fraction
    assert world.surfaces["surf-wall-south"].custom_attributes["evidence_state"] == "UNOBSERVED"
