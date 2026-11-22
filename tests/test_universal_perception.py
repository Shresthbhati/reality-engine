"""Comprehensive Unit and Integration Tests for Universal Perception and Reconstruction (PR #362).

Validates:
1. Video Ingestion, Temporal Coverage, Shot Boundary, and Image Quality Assessment.
2. Fine-Grained Entity Detection, Part Hierarchy (parent-child linking), and Polygon Masking.
3. Box Truncation, Occlusion Level Estimation, and Coordinate Preservation.
4. Temporal Identity Tracking Across Video Frames.
5. Multi-View Repeated Object Discrimination and Same-Image Mutual Exclusion.
6. Perception Model Inventory, License Provenance, and Adapter Wrappers.
7. End-to-End Scene Reconstruction from Multi-View Images and Video Sequences into WorldIR and Metric3D.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from engine.multiengine.spatial_intelligence.association import (
    AssociationOutcome,
    MultiViewIntelligenceEngine,
    TemporalIdentityTracker,
    build_camera_ray,
    evaluate_cross_view_pair,
)
from engine.multiengine.spatial_intelligence.detection_pipeline import (
    AdapterDetectorWrapper,
    DetectionBox,
    DetectionPipeline,
    DetectionScale,
    detect_box_truncation,
    estimate_box_occlusion,
    extract_polygon_from_mask,
)
from engine.multiengine.spatial_intelligence.integration import (
    DetectionInput,
    get_perception_model_inventory,
    reconstruct_spatial_scene,
)
from engine.multiengine.spatial_intelligence.video_ingestion import (
    VideoIngestionPipeline,
    compute_image_quality,
    compute_interframe_divergence,
)
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera, Quat, Vec3

# ============================================================================
# 1. Video Ingestion, Shot Boundary, and Image Quality Assessment Tests
# ============================================================================

def test_image_quality_sharp_vs_blurry() -> None:
    # 1. Sharp textured synthetic image (checkerboard)
    h, w = 320, 320
    sharp_img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, 20):
        for x in range(0, w, 20):
            if ((x // 20) + (y // 20)) % 2 == 0:
                sharp_img[y:y+20, x:x+20] = [200, 200, 200]
            else:
                sharp_img[y:y+20, x:x+20] = [50, 50, 50]

    qr_sharp = compute_image_quality(sharp_img)
    assert qr_sharp.is_usable is True
    assert qr_sharp.verdict in ("ok", "warn")
    assert qr_sharp.blur_laplacian_var > 100.0
    assert qr_sharp.quality_score > 0.40

    # 2. Heavily blurred image (uniform flat grey)
    flat_img = np.full((h, w, 3), 128, dtype=np.uint8)
    qr_flat = compute_image_quality(flat_img)
    assert qr_flat.blur_laplacian_var < 5.0
    assert qr_flat.is_usable is False
    assert qr_flat.verdict == "will_be_dropped"
    assert any("motion_blur" in iss or "blur" in iss for iss in qr_flat.issues)

    # 3. Overexposed image (blown highlights)
    blown_img = np.full((h, w, 3), 254, dtype=np.uint8)
    qr_blown = compute_image_quality(blown_img)
    assert qr_blown.clipped_high_fraction > 0.90
    assert qr_blown.is_usable is False
    assert any("clipping" in iss or "overexposed" in iss for iss in qr_blown.issues)


def test_shot_boundary_divergence_and_image_sequence_ingestion() -> None:
    h, w = 120, 160
    frame_a = np.full((h, w, 3), 40, dtype=np.uint8)
    frame_b = np.full((h, w, 3), 45, dtype=np.uint8)  # Smooth transition within same shot
    frame_c = np.full((h, w, 3), 220, dtype=np.uint8)  # Sudden scene change / cut

    div_same_shot = compute_interframe_divergence(frame_a, frame_b)
    assert div_same_shot < 15.0

    div_cut = compute_interframe_divergence(frame_b, frame_c)
    assert div_cut > 100.0

    # Ingest sequence of frames via VideoIngestionPipeline
    pipeline = VideoIngestionPipeline(
        target_fps=2.0,
        max_frame_budget=10,
        shot_divergence_threshold=40.0,
        min_quality_score=0.10,
    )

    # Textured images with good lighting and a shot cut
    img_seq = []
    for i in range(5):
        img = np.full((h, w, 3), 100 + i * 4, dtype=np.uint8)
        img[::2, ::2] = 160 + i * 4  # Texture so Laplacian var > 25
        img_seq.append(img)

    # Scene cut at frame 3
    img_seq[3] = np.full((h, w, 3), 30, dtype=np.uint8)
    img_seq[3][::2, ::2] = 230

    sampled = pipeline.ingest_image_sequence(img_seq, source_id="test_video", fps=2.0)
    assert len(sampled) >= 3
    for sf in sampled:
        assert sf.provenance.source_video_id == "test_video"
        assert sf.provenance.quality.is_usable is True


# ============================================================================
# 2. Fine-Grained Entity Detection & Part Hierarchy Tests
# ============================================================================

def test_polygon_mask_and_truncation_occlusion() -> None:
    # 1. Polygon extraction from mask
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 30:70] = True
    poly = extract_polygon_from_mask(mask, max_points=16)
    assert poly is not None
    assert len(poly) >= 4
    for pt in poly:
        assert 25.0 <= pt[0] <= 75.0 or 20.0 <= pt[1] <= 65.0

    # 2. Border truncation check
    # Box touching image boundary
    box_truncated = (0.0, 10.0, 50.0, 80.0)
    assert detect_box_truncation(box_truncated, image_width=100, image_height=100, margin=2.0) is True

    # Box safely inside interior
    box_interior = (20.0, 20.0, 60.0, 60.0)
    assert detect_box_truncation(box_interior, image_width=100, image_height=100, margin=2.0) is False

    # 3. Occlusion level estimation
    # Box area = 40 * 40 = 1600. Mask has 20*20 = 400 (ratio = 0.25 -> partial)
    mask_sparse = np.zeros((100, 100), dtype=bool)
    mask_sparse[20:40, 20:40] = True
    occ = estimate_box_occlusion(box_interior, mask_sparse)
    assert occ == "partial"

    # Fully covered box (ratio = 1.0 -> none)
    mask_full = np.zeros((100, 100), dtype=bool)
    mask_full[20:60, 20:60] = True
    occ_none = estimate_box_occlusion(box_interior, mask_full)
    assert occ_none == "none"


def test_parent_child_part_hierarchy_linking() -> None:
    # Build detector that returns door and door_handle
    class MockDoorPartDetector:
        def detect(self, image_rgb: np.ndarray, text_prompts: list[str]) -> list[DetectionBox]:
            h, w = image_rgb.shape[:2]
            clean = [p.replace(" ", "_") for p in text_prompts]
            if "door_handle" in clean:
                # Detected on crop
                return [
                    DetectionBox(
                        box_id="handle_001",
                        class_name="door_handle",
                        box_2d=(60.0, 300.0, 100.0, 360.0),
                        confidence=0.88,
                        scale=DetectionScale.MICRO,
                    )
                ]
            elif "door" in clean:
                return [
                    DetectionBox(
                        box_id="door_001",
                        class_name="door",
                        box_2d=(50.0, 50.0, 350.0, 750.0),
                        confidence=0.92,
                        scale=DetectionScale.STANDARD,
                    )
                ]
            return []

    pipeline = DetectionPipeline(detector=MockDoorPartDetector())
    test_img = np.zeros((800, 800, 3), dtype=np.uint8)
    res = pipeline.run_pipeline(
        image_rgb=test_img,
        image_id="frame_001",
        run_tiling=False,
        detect_subparts=True,
    )

    box_dict = {b.class_name: b for b in res.refined_boxes}
    assert "door" in box_dict
    assert "door_handle" in box_dict

    door_box = box_dict["door"]
    handle_box = box_dict["door_handle"]

    # Verify parent-child linkage
    assert handle_box.parent_box_id == door_box.box_id
    assert handle_box.box_id in door_box.child_box_ids
    assert door_box.box_id in res.parent_part_tree
    assert handle_box.box_id in res.parent_part_tree[door_box.box_id]

    # Coordinate preservation: handle must be placed within door coordinates in full image
    assert door_box.x1 <= handle_box.x1 <= handle_box.x2 <= door_box.x2 + 50.0
    assert door_box.y1 <= handle_box.y1 <= handle_box.y2 <= door_box.y2 + 50.0


# ============================================================================
# 3. Temporal Identity Tracking and Repeated Object Discrimination Tests
# ============================================================================

def test_temporal_identity_tracker_continuity() -> None:
    tracker = TemporalIdentityTracker(max_missed_frames=3, min_iou_match=0.30)

    # Frame 0: Chair at (100, 100, 200, 200)
    obs_f0 = [
        DetectionBox(
            box_id="chair_f0",
            class_name="chair",
            box_2d=(100.0, 100.0, 200.0, 200.0),
            confidence=0.90,
        )
    ]
    assignments_f0 = tracker.process_frame_observations(0, 0.0, obs_f0)
    track_id = assignments_f0["chair_f0"]
    assert track_id.startswith("track_")

    # Frame 1: Chair moved slightly to (105, 102, 205, 202) -> IoU ~ 0.85
    obs_f1 = [
        DetectionBox(
            box_id="chair_f1",
            class_name="chair",
            box_2d=(105.0, 102.0, 205.0, 202.0),
            confidence=0.88,
        )
    ]
    assignments_f1 = tracker.process_frame_observations(1, 0.5, obs_f1)
    assert assignments_f1["chair_f1"] == track_id  # Track continuity preserved!

    track = tracker.active_tracks[track_id]
    assert track.age_frames == 2
    assert track.observation_ids == ["chair_f0", "chair_f1"]
    assert track.frame_indices == [0, 1]


def test_repeated_object_discrimination_multiview() -> None:
    """Proves that multiple physical objects of the same class (e.g. Chair A and Chair B)
    are NOT incorrectly merged into a single entity across views when their 3D positions diverge."""
    # Camera 1 at origin facing +Z
    cam1 = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(
            position=Vec3(0.0, 0.0, 0.0),
            rotation=Quat(1.0, 0.0, 0.0, 0.0),
        ),
    )
    # Camera 2 translated to the right along X by 1.0m
    cam2 = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(
            position=Vec3(1.0, 0.0, 0.0),
            rotation=Quat(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Physical Chair 1 at (0.0, 0.0, 4.0)
    # In Cam 1: u = 320, v = 240
    # In Cam 2: x_rel = -1.0, u = 500*(-1)/4 + 320 = 195, v = 240
    ray_chair1_cam1 = build_camera_ray(
        observation_id="chair1_view1",
        source_image_id="img1",
        class_name="chair",
        box_2d=(300.0, 220.0, 340.0, 260.0),
        camera=cam1,
        scale_state="synthetic_scale_aligned",
    )
    ray_chair1_cam2 = build_camera_ray(
        observation_id="chair1_view2",
        source_image_id="img2",
        class_name="chair",
        box_2d=(175.0, 220.0, 215.0, 260.0),
        camera=cam2,
        scale_state="synthetic_scale_aligned",
    )

    # Physical Chair 2 at (2.0, 0.0, 4.0) (2 metres to the right of Chair 1!)
    # In Cam 1: x_rel = 2.0, u = 500*(2)/4 + 320 = 570, v = 240
    # In Cam 2: x_rel = 1.0, u = 500*(1)/4 + 320 = 445, v = 240
    ray_chair2_cam1 = build_camera_ray(
        observation_id="chair2_view1",
        source_image_id="img1",
        class_name="chair",
        box_2d=(550.0, 220.0, 590.0, 260.0),
        camera=cam1,
        scale_state="synthetic_scale_aligned",
    )
    ray_chair2_cam2 = build_camera_ray(
        observation_id="chair2_view2",
        source_image_id="img2",
        class_name="chair",
        box_2d=(425.0, 220.0, 465.0, 260.0),
        camera=cam2,
        scale_state="synthetic_scale_aligned",
    )

    # 1. Direct cross-pair evaluation
    # Chair 1 View 1 should merge with Chair 1 View 2
    ps_c1 = evaluate_cross_view_pair(ray_chair1_cam1, ray_chair1_cam2)
    assert ps_c1.outcome == AssociationOutcome.MERGED
    assert ps_c1.score > 0.50

    # Chair 2 View 1 should merge with Chair 2 View 2
    ps_c2 = evaluate_cross_view_pair(ray_chair2_cam1, ray_chair2_cam2)
    assert ps_c2.outcome == AssociationOutcome.MERGED
    assert ps_c2.score > 0.50

    # Chair 1 View 1 must REFUSE Chair 2 View 2 due to ray divergence!
    ps_cross = evaluate_cross_view_pair(ray_chair1_cam1, ray_chair2_cam2)
    assert ps_cross.outcome == AssociationOutcome.REFUSED
    assert ps_cross.reason in ("ray_intersection_behind_camera", "excessive_ray_distance", "excessive_reprojection_error")

    # 2. Multi-view clustering engine with all 4 observations
    engine = MultiViewIntelligenceEngine(min_match_score=0.25)
    merged, decisions, refusals = engine.associate_candidates([
        ray_chair1_cam1, ray_chair1_cam2,
        ray_chair2_cam1, ray_chair2_cam2,
    ])

    # Exactly 2 distinct physical chair entities must be produced!
    assert len(merged) == 2
    c1_members = next(m.member_observation_ids for m in merged if "chair1_view1" in m.member_observation_ids)
    c2_members = next(m.member_observation_ids for m in merged if "chair2_view1" in m.member_observation_ids)

    assert set(c1_members) == {"chair1_view1", "chair1_view2"}
    assert set(c2_members) == {"chair2_view1", "chair2_view2"}

    # No false merge!
    assert not (set(c1_members) & set(c2_members))


# ============================================================================
# 4. Perception Model Inventory and Adapter Lifecycle Tests
# ============================================================================

def test_perception_model_inventory_and_licenses() -> None:
    inv = get_perception_model_inventory()
    assert "owlvit_detector" in inv
    assert "sam_prompted_segmenter" in inv
    assert "depth_anything_v2" in inv
    assert "pycolmap_sfm" in inv

    # Verify verified permissive licenses (Apache-2.0, MIT, BSD-3-Clause)
    for model_name, info in inv.items():
        assert info["licence"] in ("Apache-2.0", "MIT", "BSD-3-Clause")
        assert "weights_hash" in info
        assert info["status"] == "IMPLEMENTED_AND_VERIFIED"


def test_adapter_wrapper_protocols() -> None:
    class MockAdapter:
        engine_name = "mock_owlvit"

        def predict(self, inputs: list[np.ndarray], options: dict) -> Any:
            class DummyResult:
                success = True
                predictions = [
                    {
                        "detections": [
                            {
                                "label": "socket",
                                "score": 0.82,
                                "box_xyxy": [10.0, 20.0, 40.0, 50.0],
                            }
                        ]
                    }
                ]
            return DummyResult()

    wrapper = AdapterDetectorWrapper(MockAdapter())
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dets = wrapper.detect(test_img, ["socket"])
    assert len(dets) == 1
    assert dets[0].class_name == "socket"
    assert dets[0].confidence == 0.82
    assert dets[0].source_engine == "mock_owlvit"


# ============================================================================
# 5. End-to-End Scene Reconstruction Tests
# ============================================================================

def test_end_to_end_scene_reconstruction_from_detections() -> None:
    cam1 = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
    )
    cam2 = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.5, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
    )

    # Table seen from Cam 1 and Cam 2
    det1 = DetectionInput(
        detection_id="table_v1",
        source_image_id="view_1",
        class_name="table",
        box_2d=(250.0, 180.0, 390.0, 300.0),
        confidence=0.90,
    )
    det2 = DetectionInput(
        detection_id="table_v2",
        source_image_id="view_2",
        class_name="table",
        box_2d=(200.0, 180.0, 340.0, 300.0),
        confidence=0.88,
    )

    cameras = {"view_1": cam1, "view_2": cam2}
    result = reconstruct_spatial_scene(
        detections=[det1, det2],
        cameras=cameras,
        scale_state="synthetic_scale_aligned",
        scale_units="synthetic_m",
    )

    assert len(result.merged_entities) == 1
    table_ent = result.merged_entities[0]
    assert table_ent.primary_class == "table"
    assert table_ent.is_multiview is True
    assert set(table_ent.member_observation_ids) == {"table_v1", "table_v2"}

    # Export to WorldIR
    world = result.to_world_ir("test_reconstruction_world")
    assert world.id == "test_reconstruction_world"
    assert len(world.entities) == 1
    w_ent = next(iter(world.entities.values()))
    assert w_ent.name == "Table"
    assert w_ent.custom_properties["is_multiview"] is True
    assert w_ent.custom_properties["scale_state"] == "synthetic_scale_aligned"

    # Export to Metric3D
    m3d_obs = result.to_metric3d_observations()
    assert len(m3d_obs) >= 1
    assert all(o.units == "synthetic_m" for o in m3d_obs)
    # Zero fabrication: synthetic scale residuals are NEVER reported as metres
    for o in m3d_obs:
        assert o.units != "m"


def test_reconstruct_spatial_scene_auto_detection() -> None:
    """Tests automatic detection and tracking pipeline execution when images are provided directly."""
    cam1 = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
    )

    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    test_img[100:300, 100:300] = [180, 180, 180]

    result = reconstruct_spatial_scene(
        detections=None,  # triggers auto-detection
        cameras={"cam_01": cam1},
        images_rgb={"cam_01": test_img},
        scale_state="sfm_gauge",
        scale_units="sfm_units",
    )

    assert result.execution_telemetry["total_detections"] >= 1
    assert len(result.merged_entities) >= 1
    assert "owlvit_detector" in result.model_inventory
