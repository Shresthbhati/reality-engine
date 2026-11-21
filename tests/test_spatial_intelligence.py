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
    CrossViewRayCandidate,
    MultiViewIntelligenceEngine,
    build_camera_ray,
    compute_ray_cpa,
    evaluate_cross_view_pair,
    triangulate_multiview_rays,
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


def test_scale_integrity_and_zero_geometry_fabrication() -> None:
    """Verifies that missing 3D coordinates and dimensions remain None, with zero fabricated values."""
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    cam = PinholeCamera(
        intrinsics=intr,
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
    )

    # Observation without depth map or lifted entity
    ray = build_camera_ray(
        observation_id="unlifted_obs_1",
        source_image_id="img_1",
        class_name="chair",
        box_2d=(100, 100, 200, 200),
        camera=cam,
        lifted_entity=None,
    )
    # Default scale must be relative_depth, never synthetic or metric
    assert ray.scale_state == "relative_depth"
    assert ray.scale_units == "relative"
    assert not ray.is_physically_metric

    engine = MultiViewIntelligenceEngine()
    merged, decisions, refusals = engine.associate_candidates([ray])

    assert len(merged) == 1
    ent = merged[0]
    # Invariant: NEVER invent position by stepping 2.0m along ray
    assert ent.position_m is None
    # Invariant: NEVER invent dimensions from taxonomy minimums
    assert ent.dimensions_m is None
    assert ent.dimension_status == "UNKNOWN"
    assert ent.scale_state == "relative_depth"
    assert ent.scale_units == "relative"
    assert not ent.is_physically_metric


def test_scale_state_incompatibility_refusal() -> None:
    """Verifies that cross-view matching strictly rejects merging incompatible scale states."""
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    cam1 = PinholeCamera(
        intrinsics=intr,
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
    )
    cam2 = PinholeCamera(
        intrinsics=intr,
        extrinsics=CameraExtrinsics(position=Vec3(1.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
    )

    ray_relative = build_camera_ray(
        observation_id="obs_rel",
        source_image_id="img_1",
        class_name="desk",
        box_2d=(250, 150, 390, 330),
        camera=cam1,
        scale_state="relative_depth",
        scale_units="relative",
    )
    ray_metric = build_camera_ray(
        observation_id="obs_met",
        source_image_id="img_2",
        class_name="desk",
        box_2d=(150, 150, 290, 330),
        camera=cam2,
        scale_state="metric_anchored",
        scale_units="m",
        is_physically_metric=True,
    )

    score = evaluate_cross_view_pair(ray_relative, ray_metric)
    assert score.outcome == AssociationOutcome.REFUSED
    assert score.reason == "incompatible_scale_states"


def test_robust_multiview_triangulation_svd_huber() -> None:
    """Verifies N-view ray triangulation using SVD/least-squares with Huber outlier resistance."""
    from engine.multiengine.spatial_intelligence.association import triangulate_multiview_rays

    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    # Known 3D target point
    true_pt = Vec3(0.3, -0.1, 3.5)

    cameras = [
        PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(position=Vec3(-0.6, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0))),
        PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0))),
        PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(position=Vec3(0.6, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0))),
    ]

    rays: list[CrossViewRayCandidate] = []
    for i, c in enumerate(cameras):
        uv = c.project(true_pt)
        assert uv is not None
        ray = build_camera_ray(
            observation_id=f"ray_{i}",
            source_image_id=f"view_{i}",
            class_name="bookshelf",
            box_2d=(uv[0] - 20, uv[1] - 40, uv[0] + 20, uv[1] + 40),
            camera=c,
            scale_state="synthetic_scale_aligned",
        )
        rays.append(ray)

    # Add a 4th view with slight measurement noise
    cam4 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.5, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)))
    uv4 = cam4.project(true_pt)
    assert uv4 is not None
    # Add noisy outlier offset to 4th view
    ray4 = build_camera_ray(
        observation_id="ray_3",
        source_image_id="view_3",
        class_name="bookshelf",
        box_2d=(uv4[0] - 10, uv4[1] - 30, uv4[0] + 30, uv4[1] + 50),
        camera=cam4,
        scale_state="synthetic_scale_aligned",
    )
    rays.append(ray4)

    p_sol, ray_res, reproj_res, cov_3x3, meta = triangulate_multiview_rays(rays)
    assert p_sol is not None
    assert abs(p_sol[0] - true_pt.x) < 0.08
    assert abs(p_sol[1] - true_pt.y) < 0.08
    assert abs(p_sol[2] - true_pt.z) < 0.12
    assert ray_res < 0.15
    assert cov_3x3 is not None
    assert np.all(np.linalg.eigvals(cov_3x3) > 0.0)  # Positive definite covariance


def test_adversarial_triangulation_rejections() -> None:
    """Verifies that degenerate geometry (near-parallel, negative depth, ill-conditioned) is rejected."""
    from engine.multiengine.spatial_intelligence.association import triangulate_multiview_rays

    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)

    # 1. Near-parallel rays (<2 degrees)
    cam1 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)))
    cam2 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(position=Vec3(0.005, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)))

    ray1 = build_camera_ray("r1", "v1", "wall", (315, 235, 325, 245), cam1)
    ray2 = build_camera_ray("r2", "v2", "wall", (315, 235, 325, 245), cam2)

    pair_score = evaluate_cross_view_pair(ray1, ray2)
    assert pair_score.outcome == AssociationOutcome.REFUSED
    assert pair_score.reason in ("near_parallel_rays", "zero_camera_baseline")

    # 2. Negative depth (point behind camera)
    cam_back1 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 2.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)))
    cam_back2 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(position=Vec3(1.0, 0.0, 2.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)))
    # Point at Z = -2.0 is behind cameras looking along +Z
    ray_b1 = build_camera_ray("rb1", "vb1", "table", (310, 230, 330, 250), cam_back1)
    ray_b2 = build_camera_ray("rb2", "vb2", "table", (330, 230, 350, 250), cam_back2)

    p_sol, _, _, _, meta = triangulate_multiview_rays([ray_b1, ray_b2])
    assert p_sol is None
    assert meta.get("fail_reason") in ("ray_intersection_behind_camera", "near_parallel_rays")


def test_detection_pipeline_full_execution() -> None:
    """Verifies that the DetectionPipeline runs end-to-end with tiling, subpart detection, and mask refinement."""
    from engine.multiengine.spatial_intelligence.detection_pipeline import (
        DetectionPipeline,
        RuleBasedDetectorAdapter,
        ThresholdSegmenterAdapter,
    )

    # High-resolution image (1000 x 1200) triggering tiling
    img_rgb = np.full((1000, 1200, 3), 180, dtype=np.uint8)

    pipeline = DetectionPipeline(
        detector=RuleBasedDetectorAdapter(),
        segmenter=ThresholdSegmenterAdapter(),
        tile_size=600,
        tile_overlap=0.25,
    )

    res = pipeline.run_pipeline(img_rgb, domain=Domain.INTERIOR, image_id="test_img_001")

    assert res.image_id == "test_img_001"
    assert len(res.raw_detections) > 0
    assert len(res.refined_boxes) > 0
    assert res.execution_stats["tiles_processed"] > 0
    # Every refined box must have a valid refined mask
    for b in res.refined_boxes:
        assert b.mask_2d is not None
        assert b.mask_2d.shape[:2] == (1000, 1200)


def test_active_capture_honesty_uncalibrated_scale() -> None:
    """Verifies that active capture never promises metric distances when scale is uncalibrated."""
    from engine.multiengine.spatial_intelligence.active_capture import (
        ActiveCaptureAnalyzer,
        CaptureIssueType,
    )
    from engine.multiengine.spatial_intelligence.association import PhysicalMergedEntity

    cam = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
    )

    # Entity with uncalibrated relative depth and unpositioned (position_m=None)
    uncalibrated_ent = PhysicalMergedEntity(
        canonical_entity_id="phys_uncalibrated_001",
        primary_class="cabinet",
        member_observation_ids=("obs_1",),
        sources=("img_1",),
        position_m=None,
        dimensions_m=None,
        orientation_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        surface_normal=None,
        confidence=0.5,
        reprojection_residual_px=0.0,
        ray_triangulation_residual_m=0.0,
        is_multiview=False,
        scale_state="relative_depth",
        scale_units="relative",
        is_physically_metric=False,
    )

    analyzer = ActiveCaptureAnalyzer()
    plan = analyzer.analyze(
        merged_entities=[uncalibrated_ent],
        lifted_entities=[],
        cameras={"img_1": cam},
        is_scene_metric=False,
    )

    assert not plan.is_scale_certified
    scale_req = next(r for r in plan.requests if r.issue_type == CaptureIssueType.MISSING_SCALE_REFERENCE)
    # When uncalibrated relative depth, suggested distance must be 0.0 (honest about unknown gauge)
    assert scale_req.suggested_distance_m == 0.0
    assert "uncalibrated" in scale_req.guidance_text.lower()


def test_visual_attribute_custom_predictor_protocol() -> None:
    """Verifies the VisualAttributePredictor protocol can be implemented by custom model backends."""
    from engine.multiengine.spatial_intelligence.attributes import (
        DeterministicAttributePredictor,
        EntityVisualAttributes,
        VisualAttributePredictor,
    )

    predictor = DeterministicAttributePredictor()
    assert isinstance(predictor, VisualAttributePredictor)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    attrs = predictor.predict_attributes(img, box_2d=(10, 10, 80, 80), class_hint="table")
    assert isinstance(attrs, EntityVisualAttributes)
    assert attrs.confidence_basis == "deterministic_cielab_texture_analysis"


def test_adversarial_transitive_scale_bridge_prevention() -> None:
    """Verifies that an unknown-scale observation CANNOT bridge incompatible clusters (A -> B -> C)."""
    intr = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    cam_a = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(0.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))
    cam_b = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(0.5, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))
    cam_c = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(1.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))

    # Candidate A: METRIC_ANCHORED
    ray_a = build_camera_ray(
        observation_id="obs_a",
        source_image_id="img_a",
        class_name="chair",
        box_2d=(300, 220, 340, 260),
        camera=cam_a,
        scale_state="metric_anchored",
        scale_units="m",
        is_physically_metric=True,
    )

    # Candidate B: UNKNOWN scale (potential bridge attack)
    ray_b = build_camera_ray(
        observation_id="obs_b",
        source_image_id="img_b",
        class_name="chair",
        box_2d=(300, 220, 340, 260),
        camera=cam_b,
        scale_state="unknown",
        scale_units="unknown",
        is_physically_metric=False,
    )

    # Candidate C: SFM_GAUGE
    ray_c = build_camera_ray(
        observation_id="obs_c",
        source_image_id="img_c",
        class_name="chair",
        box_2d=(300, 220, 340, 260),
        camera=cam_c,
        scale_state="sfm_gauge",
        scale_units="sfm_units",
        is_physically_metric=False,
    )

    # 1. Pairwise checks must refuse any pair involving unknown or mismatched scales
    score_ab = evaluate_cross_view_pair(ray_a, ray_b)
    assert score_ab.outcome == AssociationOutcome.REFUSED
    assert score_ab.reason == "unknown_scale_refusal"

    score_bc = evaluate_cross_view_pair(ray_b, ray_c)
    assert score_bc.outcome == AssociationOutcome.REFUSED
    assert score_bc.reason == "unknown_scale_refusal"

    score_ac = evaluate_cross_view_pair(ray_a, ray_c)
    assert score_ac.outcome == AssociationOutcome.REFUSED
    assert score_ac.reason == "incompatible_scale_states"

    # 2. Cluster-wide association must yield 3 separate entities (NO transitive merge)
    engine = MultiViewIntelligenceEngine()
    merged, decisions, refusals = engine.associate_candidates([ray_a, ray_b, ray_c])

    assert len(merged) == 3
    # Verify no cluster contains mixed scales
    for ent in merged:
        assert len(ent.member_observation_ids) == 1
        assert not ent.is_multiview


def test_sfm_gauge_scale_invariance_and_unit_truthfulness() -> None:
    """Verifies that SfM gauge coordinates work invariantly across scene scales without metre assumptions."""
    intr = CameraIntrinsics(fx=800.0, fy=800.0, cx=400.0, cy=300.0, width=800, height=600)

    # Test two scales: large scale (k=100.0) and small scale (k=0.005)
    for k in (100.0, 0.005):
        true_pos = Vec3(0.2 * k, -0.1 * k, 3.0 * k)
        cam1 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(-0.4 * k, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))
        cam2 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(0.4 * k, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))

        uv1 = cam1.project(true_pos)
        uv2 = cam2.project(true_pos)
        assert uv1 is not None and uv2 is not None

        ray1 = build_camera_ray("g1", "v1", "desk", (uv1[0] - 10, uv1[1] - 10, uv1[0] + 10, uv1[1] + 10), cam1, scale_state="sfm_gauge")
        ray2 = build_camera_ray("g2", "v2", "desk", (uv2[0] - 10, uv2[1] - 10, uv2[0] + 10, uv2[1] + 10), cam2, scale_state="sfm_gauge")

        # 1. Pairwise check must succeed in SfM gauge without hitting metre limits
        score = evaluate_cross_view_pair(ray1, ray2)
        assert score.outcome == AssociationOutcome.MERGED
        assert score.ray_distance_m is None  # Gauge residual NOT labeled as metres

        # 2. Multi-ray triangulation in gauge scale
        p_sol, ray_res, rep_res, cov_3x3, meta = triangulate_multiview_rays([ray1, ray2])
        assert p_sol is not None
        assert abs(p_sol[0] - true_pos.x) < 0.05 * k
        assert abs(p_sol[2] - true_pos.z) < 0.05 * k
        assert meta["scale_state"] == "sfm_gauge"
        assert meta["residual_units"] == "sfm_units"
        assert meta["covariance_units"] == "sfm_units^2"
        assert "ray_residuals_m" not in meta  # Honest about non-metric units

        # 3. Association and entity export
        engine = MultiViewIntelligenceEngine()
        merged, _, _ = engine.associate_candidates([ray1, ray2])
        assert len(merged) == 1
        m_ent = merged[0]
        assert m_ent.scale_state == "sfm_gauge"
        assert m_ent.scale_units == "sfm_units"
        assert not m_ent.is_physically_metric

        ent_dict = m_ent.to_dict()
        assert ent_dict["position_m"] is None  # Do NOT claim metres
        assert ent_dict["ray_triangulation_residual_m"] is None
        assert ent_dict["position"] is not None
        assert ent_dict["residual_units"] == "sfm_units"


def test_active_capture_unknown_position_honesty_and_scale_flag() -> None:
    """Verifies that active capture never invents (0,0,0) target positions and flags scale anchor requests."""
    from engine.multiengine.spatial_intelligence.active_capture import (
        ActiveCaptureAnalyzer,
        CaptureIssueType,
    )
    from engine.multiengine.spatial_intelligence.association import PhysicalMergedEntity

    cam = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
    )

    # Entity without 3D position
    unpositioned_ent = PhysicalMergedEntity(
        canonical_entity_id="phys_unpos_001",
        primary_class="door",
        member_observation_ids=("obs_u1",),
        sources=("img_u1",),
        position=None,
        dimensions=None,
        is_multiview=False,
        scale_state="synthetic_scale_aligned",
        scale_units="synthetic_m",
        is_physically_metric=False,
    )

    analyzer = ActiveCaptureAnalyzer()
    plan = analyzer.analyze(
        merged_entities=[unpositioned_ent],
        lifted_entities=[],
        cameras={"img_u1": cam},
        is_scene_metric=False,
    )

    # Check requests
    for req in plan.requests:
        # Zero fabrication: Target position must be None, NEVER (0.0, 0.0, 0.0)
        assert req.target_position is None
        assert req.target_position_m is None
        r_dict = req.to_dict()
        assert r_dict["target_position"] is None
        assert r_dict["target_position_m"] is None

        if req.issue_type == CaptureIssueType.MISSING_SCALE_REFERENCE:
            assert req.is_scale_request is True
            # Synthetic scale is not physical metric calibration: suggested_distance_m must be 0.0
            assert req.suggested_distance_m == 0.0
        else:
            assert req.is_scale_request is False
            # Viewpoint request on non-metric scene must not specify metric distance
            assert req.suggested_distance_m == 0.0


def test_triangulation_many_views_and_outlier_rejection() -> None:
    """Verifies 5-view triangulation with outlier resistance and positive depth validation."""
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    true_pt = Vec3(0.1, -0.2, 2.5)

    # 4 consistent cameras + 1 outlier camera
    cameras = [
        PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(-0.6, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0))),
        PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(-0.2, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0))),
        PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(0.2, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0))),
        PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(0.6, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0))),
    ]

    rays: list[CrossViewRayCandidate] = []
    for i, c in enumerate(cameras):
        uv = c.project(true_pt)
        assert uv is not None
        ray = build_camera_ray(
            observation_id=f"ray_5v_{i}",
            source_image_id=f"view_5v_{i}",
            class_name="lamp",
            box_2d=(uv[0] - 15, uv[1] - 15, uv[0] + 15, uv[1] + 15),
            camera=c,
            scale_state="metric_anchored",
            scale_units="m",
            is_physically_metric=True,
        )
        rays.append(ray)

    # Add 5th camera with perturbed box (outlier)
    cam5 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(0.0, 0.4, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))
    uv5 = cam5.project(true_pt)
    assert uv5 is not None
    outlier_ray = build_camera_ray(
        observation_id="ray_5v_outlier",
        source_image_id="view_5v_4",
        class_name="lamp",
        box_2d=(uv5[0] + 60, uv5[1] + 60, uv5[0] + 90, uv5[1] + 90),
        camera=cam5,
        scale_state="metric_anchored",
        scale_units="m",
        is_physically_metric=True,
    )
    rays.append(outlier_ray)

    p_sol, ray_res, rep_res, cov_3x3, meta = triangulate_multiview_rays(rays)
    assert p_sol is not None
    # Position should still be accurately recovered despite 1 outlier ray out of 5
    assert abs(p_sol[0] - true_pt.x) < 0.10
    assert abs(p_sol[1] - true_pt.y) < 0.10
    assert abs(p_sol[2] - true_pt.z) < 0.20
    assert meta["scale_state"] == "metric_anchored"
    assert meta["residual_units"] == "m"


def test_cluster_wide_coordinate_frame_and_units_integrity() -> None:
    """Verifies that observations in different coordinate frames or units are rejected from clustering."""
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    cam1 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(0.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))
    cam2 = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(Vec3(0.5, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))

    ray_world = build_camera_ray(
        "obs_w", "v1", "sofa", (300, 220, 340, 260), cam1,
        scale_state="synthetic_scale_aligned",
        coordinate_frame="world:canonical",
    )
    ray_local = build_camera_ray(
        "obs_l", "v2", "sofa", (300, 220, 340, 260), cam2,
        scale_state="synthetic_scale_aligned",
        coordinate_frame="camera_local_frame",
    )

    score = evaluate_cross_view_pair(ray_world, ray_local)
    assert score.outcome == AssociationOutcome.REFUSED
    assert score.reason == "coordinate_frame_mismatch"


