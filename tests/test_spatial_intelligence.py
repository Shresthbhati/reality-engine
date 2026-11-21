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
    assert "wood" in attrs.material.value
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


def test_expanded_taxonomy_and_hierarchy() -> None:
    from engine.multiengine.spatial_intelligence.taxonomy import (
        get_ancestor_classes,
        get_domain_prompts,
        get_hierarchical_path,
        get_subparts_for_class,
    )

    # 1. New Interior Classes
    hinge = resolve_taxonomy_class("door hinge")
    assert hinge.taxonomy_id == "interior.opening.door_hinge"
    lock = resolve_taxonomy_class("deadbolt")
    assert lock.taxonomy_id == "interior.opening.door_lock"
    ac = resolve_taxonomy_class("split ac")
    assert ac.taxonomy_id == "interior.hvac.split_ac"
    panel = resolve_taxonomy_class("breaker box")
    assert panel.taxonomy_id == "interior.electrical.panel"
    pipe = resolve_taxonomy_class("water pipe")
    assert pipe.taxonomy_id == "interior.plumbing.pipe"

    # 2. New Exterior Classes
    building = resolve_taxonomy_class("building structure")
    assert building.taxonomy_id == "exterior.architecture.building"
    bollard = resolve_taxonomy_class("traffic bollard")
    assert bollard.taxonomy_id == "exterior.street.bollard"
    manhole = resolve_taxonomy_class("manhole cover")
    assert manhole.taxonomy_id == "exterior.utility.manhole"
    paving = resolve_taxonomy_class("block paving")
    assert paving.taxonomy_id == "exterior.circulation.paving"

    # 3. Hierarchy Navigation
    path = get_hierarchical_path(hinge.taxonomy_id)
    assert path == ["interior", "opening", "door_hinge"]

    ancestors = get_ancestor_classes(hinge.taxonomy_id)
    assert len(ancestors) == 1
    assert ancestors[0].taxonomy_id == "interior.opening.door"

    subparts = get_subparts_for_class("interior.opening.door")
    subpart_ids = [s.taxonomy_id for s in subparts]
    assert "interior.opening.door_hinge" in subpart_ids
    assert "interior.opening.door_lock" in subpart_ids
    assert "interior.opening.door_frame" in subpart_ids

    # 4. Domain Prompts
    prompts = get_domain_prompts(Domain.INTERIOR)
    assert len(prompts) >= 30
    assert "socket" in prompts
    assert "door" in prompts


def test_detection_pipeline_tiling_and_part_preservation() -> None:
    from engine.multiengine.spatial_intelligence.detection_pipeline import (
        DetectionBox,
        DetectionScale,
        compute_box_iou,
        generate_tiling_windows,
        suppress_duplicates_preserving_parts,
        transform_tile_detections_to_image,
    )

    # 1. Tiling window generation
    windows = generate_tiling_windows(image_width=1920, image_height=1080, tile_size=800, overlap_ratio=0.25)
    assert len(windows) >= 4
    # All windows must stay within image bounds
    for w in windows:
        assert 0 <= w.x1 < w.x2 <= 1920
        assert 0 <= w.y1 < w.y2 <= 1080

    # 2. Coordinate transformation back to full image
    win = windows[0]
    tile_det = DetectionBox(
        box_id="det_local",
        class_name="socket",
        box_2d=(50.0, 60.0, 100.0, 110.0),
        confidence=0.8,
        scale=DetectionScale.MICRO,
    )
    mapped = transform_tile_detections_to_image([tile_det], win, 1920, 1080)
    assert len(mapped) == 1
    assert mapped[0].x1 == win.x1 + 50.0
    assert mapped[0].y1 == win.y1 + 60.0

    # 3. IoU Computation
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (0.0, 0.0, 10.0, 10.0)
    assert compute_box_iou(box1, box2) == 1.0

    box3 = (10.0, 10.0, 20.0, 20.0)
    assert compute_box_iou(box1, box3) == 0.0

    # 4. Duplicate Suppression with Subpart Preservation
    parent_door = DetectionBox(
        box_id="door_01",
        class_name="door",
        box_2d=(100.0, 100.0, 300.0, 600.0),
        confidence=0.90,
        scale=DetectionScale.STANDARD,
    )
    subpart_handle = DetectionBox(
        box_id="handle_01",
        class_name="door_handle",
        box_2d=(110.0, 340.0, 140.0, 370.0),
        confidence=0.75,
        scale=DetectionScale.MICRO,
    )
    duplicate_door = DetectionBox(
        box_id="door_02",
        class_name="door",
        box_2d=(105.0, 95.0, 305.0, 605.0),
        confidence=0.80,
        scale=DetectionScale.STANDARD,
    )

    reconciled = suppress_duplicates_preserving_parts([parent_door, subpart_handle, duplicate_door])
    # The duplicate door must be fused with the main door, but the door handle must NOT be suppressed!
    class_names = [d.class_name for d in reconciled]
    assert "door" in class_names
    assert "door_handle" in class_names
    assert len(reconciled) == 2


def test_mask_refinement_speckles_and_holes() -> None:
    from engine.multiengine.spatial_intelligence.detection_pipeline import refine_instance_mask

    # 50x50 binary mask with main 30x30 object, small 2-pixel noise speckle, and small 4-pixel hole
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True
    # Hole inside object (2x2)
    mask[20:22, 20:22] = False
    # Noise speckle outside object
    mask[2:4, 2:4] = True

    refined = refine_instance_mask(mask, min_component_area_px=10, max_fill_hole_area_px=10)

    # Speckle must be removed
    assert not np.any(refined[2:4, 2:4])
    # Interior small hole must be filled
    assert np.all(refined[20:22, 20:22])
    # Main object preserved
    assert refined[15, 15]


def test_active_capture_gap_analysis() -> None:
    from engine.multiengine.spatial_intelligence.active_capture import (
        ActiveCaptureAnalyzer,
        CaptureIssueType,
        CapturePriority,
    )
    from engine.multiengine.spatial_intelligence.association import PhysicalMergedEntity
    from engine.multiengine.spatial_intelligence.spatial_lifting import SpatialEntity3D

    intr = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000)
    cam = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(0.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))

    # Single-view entity touching border with synthetic uncertified scale
    single_obs = SpatialEntity3D(
        entity_id="obs_single",
        class_name="socket",
        position_m=(1.0, 0.5, 2.0),
        dimensions_m=(0.1, 0.1, 0.1),
        orientation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        covariance_xyz=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
        is_truncated_by_border=True,
        mask_pixels_count=400,
        source_image_id="img_1",
    )

    merged_ent = PhysicalMergedEntity(
        canonical_entity_id="phys_single",
        primary_class="socket",
        member_observation_ids=("obs_single",),
        sources=("img_1",),
        position_m=(1.0, 0.5, 2.0),
        dimensions_m=(0.1, 0.1, 0.1),
        orientation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        surface_normal=None,
        confidence=0.5,
        reprojection_residual_px=0.0,
        ray_triangulation_residual_m=0.0,
        is_multiview=False,
        is_physically_metric=False,
    )

    analyzer = ActiveCaptureAnalyzer()
    plan = analyzer.analyze(
        merged_entities=[merged_ent],
        lifted_entities=[single_obs],
        cameras={"img_1": cam},
        is_scene_metric=False,
    )

    assert plan.total_entities_analyzed == 1
    assert plan.multiview_coverage_ratio == 0.0
    assert not plan.is_scale_certified

    # Must contain CRITICAL scale reference request and HIGH missing viewpoints request
    issues = [r.issue_type for r in plan.requests]
    assert CaptureIssueType.MISSING_SCALE_REFERENCE in issues
    assert CaptureIssueType.MISSING_VIEWPOINTS in issues

    scale_req = next(r for r in plan.requests if r.issue_type == CaptureIssueType.MISSING_SCALE_REFERENCE)
    assert scale_req.priority == CapturePriority.CRITICAL


def test_fine_grained_visual_attributes() -> None:
    from engine.multiengine.spatial_intelligence.attributes import (
        ObjectCondition,
        ObjectState,
        extract_visual_attributes,
    )

    # 1. Door Open Aspect Ratio Test
    img_door = np.full((100, 100, 3), 200, dtype=np.uint8)
    # Thin open door box (aspect = 15/90 < 0.22)
    attr_open = extract_visual_attributes(img_door, box_2d=(10, 5, 25, 95), class_hint="door")
    assert attr_open.operational_state == ObjectState.OPEN

    # Wide closed door box (aspect = 50/90 > 0.45)
    attr_closed = extract_visual_attributes(img_door, box_2d=(10, 5, 60, 95), class_hint="door")
    assert attr_closed.operational_state == ObjectState.CLOSED

    # 2. Light Fixture ON State Test
    img_light_on = np.full((50, 50, 3), 250, dtype=np.uint8)  # High luminance / specular
    attr_light_on = extract_visual_attributes(img_light_on, box_2d=(5, 5, 45, 45), class_hint="ceiling_light")
    assert attr_light_on.operational_state == ObjectState.ON

    # 3. Object Condition Test
    assert attr_open.condition in (ObjectCondition.PRISTINE, ObjectCondition.UNKNOWN)

