"""Tests for Intelligent Next-Best-View Capture Planner (Epic E08-1)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from planning.capture_planner import (
    CameraPose,
    CapturePlanResult,
    RecommendedViewpoint,
    SurfacePatch,
    _is_visible,
    plan_next_best_view_capture,
)
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType
from world_ir.world_v1 import WorldIR


def test_visibility_incidence_and_fov_pruning():
    cam_pos = np.array([0.0, 0.0, 1.5])
    cam_dir = np.array([0.0, 1.0, 0.0])  # pointing +Y

    # Patch in front of camera, normal pointing back at camera (-Y)
    patch_pos = np.array([0.0, 3.0, 1.5])
    patch_norm = np.array([0.0, -1.0, 0.0])
    assert _is_visible(patch_pos, patch_norm, cam_pos, cam_dir, fov_deg=65.0) is True

    # Back-facing patch (normal pointing +Y, away from camera) -> invisible
    patch_norm_back = np.array([0.0, 1.0, 0.0])
    assert _is_visible(patch_pos, patch_norm_back, cam_pos, cam_dir, fov_deg=65.0) is False

    # Outside FOV (to the side at 80 degrees)
    patch_pos_side = np.array([5.0, 0.5, 1.5])
    assert _is_visible(patch_pos_side, patch_norm, cam_pos, cam_dir, fov_deg=65.0) is False

    # Beyond max range (15m > 8m)
    patch_pos_far = np.array([0.0, 15.0, 1.5])
    assert _is_visible(patch_pos_far, patch_norm, cam_pos, cam_dir, max_dist=8.0) is False


def test_next_best_view_uncertainty_reduction_ge_40_percent():
    """Verify that capture recommendations reduce scene reconstruction uncertainty by >= 40%."""
    world = WorldIR(id="w-sparse-capture")

    # 4 walls and floor forming a 6x6 room
    walls = [
        Entity(id="wall-north", name="North Wall", type=EntityType.WALL, transform={"position": {"x": 0.0, "y": 3.0, "z": 0.0}}),
        Entity(id="wall-south", name="South Wall", type=EntityType.WALL, transform={"position": {"x": 0.0, "y": -3.0, "z": 0.0}}),
        Entity(id="wall-east", name="East Wall", type=EntityType.WALL, transform={"position": {"x": 3.0, "y": 0.0, "z": 0.0}}),
        Entity(id="wall-west", name="West Wall", type=EntityType.WALL, transform={"position": {"x": -3.0, "y": 0.0, "z": 0.0}}),
        Entity(id="floor", name="Floor", type=EntityType.FLOOR, transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}}),
        Entity(id="table", name="Table", type=EntityType.STRUCTURE, transform={"position": {"x": 0.5, "y": 0.5, "z": 0.0}}),
    ]
    for w in walls:
        world.entities[w.id] = w

    # Initial capture has only 1 camera pointing at the North Wall
    # leaving South, East, West walls and corners largely unobserved
    existing_cams = [
        CameraPose(camera_id="cam-1", position=(0.0, 0.0, 1.5), look_direction=(0.0, 1.0, 0.0), fov_deg=60.0)
    ]

    result = plan_next_best_view_capture(
        world=world,
        existing_cameras=existing_cams,
        max_recommendations=5,
        camera_height_m=1.6,
        camera_fov_deg=65.0,
    )

    # Initial uncertainty should be substantial due to unobserved walls
    assert result.initial_uncertainty > 0.50
    assert result.unobserved_patches > 0
    assert len(result.recommendations) >= 3

    # Success criteria: recommendations reduce reconstruction uncertainty by >= 40%
    print(f"\nCapture Planning Uncertainty Reduction: {result.uncertainty_reduction_ratio * 100:.2f}%")
    assert result.uncertainty_reduction_ratio >= 0.40, (
        f"Uncertainty reduction {result.uncertainty_reduction_ratio} is below 40% threshold"
    )

    # Verify recommendations are properly structured
    for rec in result.recommendations:
        assert rec.information_gain > 0.0
        assert len(rec.position) == 3
        assert len(rec.look_at) == 3
        assert len(rec.look_direction) == 3
        assert len(rec.target_entity_ids) > 0
        assert "Stand at" in rec.instruction

    # Verify JSON serialization
    res_dict = result.to_dict()
    assert "initial_uncertainty" in res_dict
    assert "uncertainty_reduction_ratio" in res_dict
    assert "recommendations" in res_dict
    assert len(res_dict["recommendations"]) == len(result.recommendations)
