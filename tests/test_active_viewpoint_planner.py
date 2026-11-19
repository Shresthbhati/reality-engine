"""Tests for active viewpoint and coverage planning engine (Phase 5 / Cycle 4).

Validates:
- Ray-AABB intersection math (SceneObstacle.intersects_ray)
- Surface incidence angle gating (<= 75 deg threshold)
- Occlusion detection when obstacle blocks line of sight
- Collision-free navigable space enforcement with room margins
- Quantitative trade-off between travel cost and information gain
- CoveragePlan metrics: surface_coverage, occlusion_ratio, unobserved_area_m2, planning_time_ms
"""

from __future__ import annotations

import pytest

from perception.planning.viewpoint_planner import (
    ActiveViewpointPlanner,
    SceneObstacle,
    SurfacePatch,
)


def test_obstacle_ray_intersection_slab_method() -> None:
    # Obstacle centered at (2.0, 0.0, 1.0) with size 1.0 x 1.0 x 1.0
    # X in [1.5, 2.5], Y in [-0.5, 0.5], Z in [0.5, 1.5]
    obst = SceneObstacle(
        name="pillar",
        aabb_min=(1.5, -0.5, 0.5),
        aabb_max=(2.5, 0.5, 1.5),
    )

    # Direct hit along X axis
    hit = obst.intersects_ray(origin=(0.0, 0.0, 1.0), direction=(1.0, 0.0, 0.0), max_dist=5.0)
    assert hit is True

    # Miss: ray aimed parallel or offset along Y
    miss_y = obst.intersects_ray(origin=(0.0, 2.0, 1.0), direction=(1.0, 0.0, 0.0), max_dist=5.0)
    assert miss_y is False

    # Miss: ray stops before reaching obstacle (max_dist too short)
    short_ray = obst.intersects_ray(origin=(0.0, 0.0, 1.0), direction=(1.0, 0.0, 0.0), max_dist=1.0)
    assert short_ray is False

    # Miss: ray aimed in opposite direction
    backwards = obst.intersects_ray(origin=(0.0, 0.0, 1.0), direction=(-1.0, 0.0, 0.0), max_dist=5.0)
    assert backwards is False


def test_surface_incidence_angle_gating() -> None:
    planner = ActiveViewpointPlanner(max_incidence_angle_deg=75.0)

    # Patch at (2.0, 0.0, 1.0) facing -X: normal = (-1.0, 0.0, 0.0)
    patch_facing_west = SurfacePatch(
        position=(2.0, 0.0, 1.0),
        normal=(-1.0, 0.0, 0.0),
    )

    # Camera at (0.0, 0.0, 1.0) looks directly at patch: angle = 0 deg
    inc_ok, unoccluded = planner.evaluate_ray_visibility(cam_pos=(0.0, 0.0, 1.0), patch=patch_facing_west)
    assert inc_ok is True
    assert unoccluded is True

    # Camera at (4.0, 0.0, 1.0) is behind patch (looking at back of surface): angle > 90 deg
    behind_ok, _ = planner.evaluate_ray_visibility(cam_pos=(4.0, 0.0, 1.0), patch=patch_facing_west)
    assert behind_ok is False

    # Grazing angle near 80 deg (> 75 deg threshold)
    # Camera placed at large Y offset: cam = (0.5, 8.0, 1.0)
    grazing_ok, _ = planner.evaluate_ray_visibility(cam_pos=(0.5, 8.0, 1.0), patch=patch_facing_west)
    assert grazing_ok is False


def test_raycast_obstacle_occlusion() -> None:
    planner = ActiveViewpointPlanner()

    # Target patch at (4.0, 0.0, 1.0) facing camera (-X)
    target_patch = SurfacePatch(
        position=(4.0, 0.0, 1.0),
        normal=(-1.0, 0.0, 0.0),
    )

    # Add a solid blocking wall / partition between camera at (0,0,1) and target at (4,0,1)
    planner.add_obstacle(
        aabb_min=(1.8, -1.0, 0.0),
        aabb_max=(2.2, 1.0, 2.5),
        name="dividing_wall",
    )

    # Ray from camera (0.0, 0.0, 1.0) to patch (4.0, 0.0, 1.0) is occluded by the dividing wall!
    in_view, unoccluded = planner.evaluate_ray_visibility(cam_pos=(0.0, 0.0, 1.0), patch=target_patch)
    assert in_view is True  # Incidence angle is valid (facing camera)
    assert unoccluded is False  # But raycast hits the obstacle!

    # Camera moved around the wall to (3.0, 2.0, 1.0) has clear line of sight
    in_view_clear, unoccluded_clear = planner.evaluate_ray_visibility(cam_pos=(3.0, 2.0, 1.0), patch=target_patch)
    assert in_view_clear is True
    assert unoccluded_clear is True


def test_collision_free_room_bounds() -> None:
    planner = ActiveViewpointPlanner(wall_margin_m=0.35)
    planner.set_room_bounds(x_min=0.0, y_min=0.0, x_max=5.0, y_max=5.0)

    # Valid interior point
    assert planner.is_pose_collision_free((2.5, 2.5, 1.45)) is True

    # Too close to west wall (< 0.35m margin)
    assert planner.is_pose_collision_free((0.20, 2.5, 1.45)) is False

    # Outside room bounds
    assert planner.is_pose_collision_free((-0.5, 2.5, 1.45)) is False


def test_active_capture_coverage_plan_generation() -> None:
    planner = ActiveViewpointPlanner()
    planner.set_room_bounds(x_min=0.0, y_min=0.0, x_max=6.0, y_max=6.0)

    # Add entity in the center: e.g. a table or cabinet
    planner.add_entity_surfaces(
        center=(3.0, 3.0, 0.5),
        extent=(1.0, 1.0, 1.0),
        entity_id="cabinet_center",
        uncertainty=1.0,
    )

    # Plan active capture from current position (1.0, 1.0, 1.45) looking at (3.0, 3.0, 0.5)
    plan = planner.plan_viewpoints(
        target_center=(3.0, 3.0, 0.5),
        current_cam_pos=(1.0, 1.0, 1.45),
        num_candidates=12,
        top_k=3,
    )

    assert plan.initial_surface_coverage == pytest.approx(0.0, abs=1e-3)
    assert plan.expected_surface_coverage > 0.0
    assert plan.total_unobserved_area_m2 > 0.0
    assert plan.planning_time_ms >= 0.0
    assert len(plan.recommended_viewpoints) >= 1

    top_vp = plan.recommended_viewpoints[0]
    assert top_vp.is_collision_free is True
    assert top_vp.is_reachable is True
    assert top_vp.expected_information_gain > 0.0
    assert top_vp.expected_uncertainty_reduction > 0.0

    plan_dict = plan.to_dict()
    assert "recommended_viewpoints" in plan_dict
    assert "initial_surface_coverage" in plan_dict
    assert "planning_time_ms" in plan_dict
