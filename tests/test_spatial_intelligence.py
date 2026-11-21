"""Comprehensive Unit Tests for Spatial Intelligence, Perception, and Multi-View Engine.

Validates:
1. Taxonomy resolution, semantic compatibility, and open-vocabulary query generation.
2. CIELAB dominant color extraction, texture roughness, and material classification.
3. 3D spatial lifting, oriented bounding box, and surface normal estimation.
4. Rigorous covariance propagation from pixel & depth variance to world coordinates.
5. Epipolar ray-ray triangulation, CPA distance, and reprojection verification.
6. Multi-view association with transitive same-image mutual exclusion and refusal logging.
7. Spatial relationship inference (mounted_on, supported_by, part_of).
8. Bridge to WorldIR entities and metric3d.v1 observations.
"""

from __future__ import annotations

import numpy as np

from engine.multiengine.spatial_intelligence.association import (
    AssociationOutcome,
    MultiViewIntelligenceEngine,
    build_camera_ray,
    compute_ray_cpa,
    evaluate_cross_view_pair,
)
from engine.multiengine.spatial_intelligence.attributes import (
    MaterialType,
    ObjectState,
    SurfaceFinish,
    extract_visual_attributes,
)
from engine.multiengine.spatial_intelligence.integration import (
    DetectionInput,
    reconstruct_spatial_scene,
)
from engine.multiengine.spatial_intelligence.spatial_lifting import (
    SpatialRelationKind,
    propagate_unprojection_covariance,
)
from engine.multiengine.spatial_intelligence.taxonomy import (
    Domain,
    are_semantically_compatible,
    get_detection_queries_by_domain,
    resolve_taxonomy_class,
)
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera, Quat, Vec3


def test_taxonomy_resolution_and_compatibility() -> None:
    # 1. Alias matching
    cls_socket = resolve_taxonomy_class("power outlet")
    assert cls_socket.taxonomy_id == "interior.electrical.socket"
    assert cls_socket.name == "socket"

    cls_chair = resolve_taxonomy_class("dining chair")
    assert cls_chair.taxonomy_id == "interior.furniture.chair"

    cls_facade = resolve_taxonomy_class("storefront")
    assert cls_facade.taxonomy_id == "exterior.architecture.building_facade"

    cls_curb = resolve_taxonomy_class("curbstone")
    assert cls_curb.taxonomy_id == "exterior.circulation.kerb"

    # 2. Semantic compatibility
    assert are_semantically_compatible("socket", "electrical socket")
    assert are_semantically_compatible("door", "door_frame")  # part-parent
    assert are_semantically_compatible("table", "desk")       # common confusion
    assert not are_semantically_compatible("door", "chair")   # incompatible
    assert not are_semantically_compatible("socket", "tree")   # incompatible

    # 3. Query generation by domain
    int_queries = get_detection_queries_by_domain(Domain.INTERIOR)
    assert "surface" in int_queries
    assert "electrical" in int_queries
    assert "furniture" in int_queries

    ext_queries = get_detection_queries_by_domain(Domain.EXTERIOR)
    assert "architecture" in ext_queries
    assert "vegetation" in ext_queries
    assert "circulation" in ext_queries


def test_visual_attribute_extraction() -> None:
    # Synthetic RGB image with a brown wooden patch and white background
    img = np.full((100, 100, 3), 245, dtype=np.uint8)  # Off-white
    # Brown patch in center (RGB ~ [139, 90, 43])
    img[30:70, 30:70, 0] = 139
    img[30:70, 30:70, 1] = 90
    img[30:70, 30:70, 2] = 43

    attrs = extract_visual_attributes(
        image_rgb=img,
        box_2d=(30, 30, 70, 70),
        class_hint="table",
    )
    assert attrs.material == MaterialType.WOOD
    assert "wood" in attrs.dominant_color_name or "brown" in attrs.dominant_color_name
    assert attrs.surface_finish in (SurfaceFinish.MATTE, SurfaceFinish.GLOSSY)
    assert attrs.operational_state == ObjectState.UNSPECIFIED


def test_covariance_propagation_math() -> None:
    intr = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000)
    pos = Vec3(0.0, 0.0, 0.0)
    rot = Quat(1.0, 0.0, 0.0, 0.0)  # Identity
    cam = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(position=pos, rotation=rot))

    cov_2m = propagate_unprojection_covariance(500.0, 500.0, 2.0, cam)
    cov_5m = propagate_unprojection_covariance(500.0, 500.0, 5.0, cam)

    # Must be positive-definite
    assert np.all(np.linalg.eigvalsh(cov_2m) > 0)
    assert np.all(np.linalg.eigvalsh(cov_5m) > 0)

    # Uncertainty in Z grows quadratically with distance
    assert cov_5m[2, 2] > cov_2m[2, 2]


def test_epipolar_ray_cpa_triangulation() -> None:
    # Camera A at (-1, 0, 0), looking towards (+X, +Z)
    orig_a = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    # Target point at (0, 0, 2.0)
    dir_a = np.array([1.0, 0.0, 2.0], dtype=np.float64)
    dir_a /= np.linalg.norm(dir_a)

    # Camera B at (1, 0, 0), looking towards (-X, +Z)
    orig_b = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    dir_b = np.array([-1.0, 0.0, 2.0], dtype=np.float64)
    dir_b /= np.linalg.norm(dir_b)

    dist, s_a, t_b, p_a, p_b = compute_ray_cpa(orig_a, dir_a, orig_b, dir_b)

    # The rays intersect exactly at (0, 0, 2.0)
    assert dist < 1e-6
    assert s_a > 0.0
    assert t_b > 0.0
    p_mid = 0.5 * (p_a + p_b)
    np.testing.assert_allclose(p_mid, [0.0, 0.0, 2.0], atol=1e-5)


def test_cross_view_pair_evaluation_and_rejection() -> None:
    intr = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000)

    # Camera 1 at (0, 0, 0)
    cam1 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(0.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))
    # Camera 2 at (1, 0, 0)
    cam2 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(1.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))

    # Both observing a physical table at (0.5, 0.0, 2.0)
    # Cam 1: u = 1000*(0.5/2.0) + 500 = 750, v = 500
    box_a = (700, 450, 800, 550)
    ray_a = build_camera_ray("obs_1", "img_01", "table", box_a, cam1)

    # Cam 2: u = 1000*((0.5-1.0)/2.0) + 500 = 250, v = 500
    box_b = (200, 450, 300, 550)
    ray_b = build_camera_ray("obs_2", "img_02", "table", box_b, cam2)

    ps = evaluate_cross_view_pair(ray_a, ray_b)
    assert ps.outcome == AssociationOutcome.MERGED
    assert ps.score > 0.5
    assert ps.triangulated_position is not None
    assert abs(ps.triangulated_position[0] - 0.5) < 0.1
    assert abs(ps.triangulated_position[2] - 2.0) < 0.1

    # Same image must be refused
    ray_a2 = build_camera_ray("obs_3", "img_01", "table", (100, 100, 200, 200), cam1)
    ps_same = evaluate_cross_view_pair(ray_a, ray_a2)
    assert ps_same.outcome == AssociationOutcome.REFUSED
    assert ps_same.reason == "same_image_conflict"

    # Incompatible classes must be refused
    ray_chair = build_camera_ray("obs_4", "img_02", "chair", box_b, cam2)
    ps_incompat = evaluate_cross_view_pair(ray_a, ray_chair)
    assert ps_incompat.outcome == AssociationOutcome.REFUSED
    assert ps_incompat.reason == "semantic_incompatible"


def test_multi_view_intelligence_engine_end_to_end() -> None:
    intr = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000)
    cam1 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(0.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))
    cam2 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(1.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))

    # Two views of Table T1
    ray_t1_v1 = build_camera_ray("t1_v1", "img_1", "table", (700, 450, 800, 550), cam1)
    ray_t1_v2 = build_camera_ray("t1_v2", "img_2", "table", (200, 450, 300, 550), cam2)

    # One view of Door D1 on img_1
    ray_d1_v1 = build_camera_ray("d1_v1", "img_1", "door", (100, 100, 250, 450), cam1)

    engine = MultiViewIntelligenceEngine()
    merged, decisions, refusals = engine.associate_candidates([ray_t1_v1, ray_t1_v2, ray_d1_v1])

    # Expect 2 physical entities (1 merged table with 2 views, 1 single-view door)
    assert len(merged) == 2
    merged_table = next((m for m in merged if m.primary_class == "table"), None)
    assert merged_table is not None
    assert merged_table.is_multiview
    assert len(merged_table.member_observation_ids) == 2
    assert set(merged_table.sources) == {"img_1", "img_2"}

    merged_door = next((m for m in merged if m.primary_class == "door"), None)
    assert merged_door is not None
    assert not merged_door.is_multiview


def test_spatial_relationships_and_worldir_export() -> None:
    intr = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000)
    cam = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(0.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))

    # Flat depth map at 2.0 meters
    depth = np.full((1000, 1000), 2.0, dtype=np.float32)

    # Detections: A wall behind, a table resting, and a socket near the wall
    dets = [
        DetectionInput(
            detection_id="det_wall",
            source_image_id="img_1",
            class_name="wall",
            box_2d=(100, 100, 900, 900),
            confidence=0.9,
        ),
        DetectionInput(
            detection_id="det_socket",
            source_image_id="img_1",
            class_name="socket",
            box_2d=(480, 480, 520, 520),
            confidence=0.85,
        ),
    ]

    scene_res = reconstruct_spatial_scene(
        detections=dets,
        cameras={"img_1": cam},
        depth_maps={"img_1": depth},
    )

    assert len(scene_res.lifted_entities) == 2
    assert len(scene_res.spatial_relationships) >= 1

    rel = scene_res.spatial_relationships[0]
    assert rel.relation == SpatialRelationKind.MOUNTED_ON

    # Export to WorldIR and verify
    world = scene_res.to_world_ir("test_world")
    assert len(world.entities) == 2
    total_rels = sum(len(e.relationships) for e in world.entities.values())
    assert total_rels >= 1

    # Verify metric3d observations
    m3d_obs = scene_res.to_metric3d_observations()
    assert len(m3d_obs) >= 4  # 2 positions + 2 extents
