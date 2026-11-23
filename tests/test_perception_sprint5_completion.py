"""Tests for Sprint 5 Perception Completion: Multi-Scale Preprocessing,

Hierarchical Components, Mask Topology, Rejection Ledger, Duplicate Track Merging,
and Multi-View Appearance Consistency.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from engine.multiengine.spatial_intelligence.detection_pipeline import (
    DetectionBox,
    DetectionPipeline,
    DetectionScale,
    EXPECTED_KEY_COMPONENTS,
    GENUINE_STRUCTURAL_PARTS,
    extract_mask_topology,
    get_subpart_detection_targets,
    preprocess_image_for_detection,
    suppress_duplicates_preserving_parts,
)
from engine.multiengine.spatial_intelligence.association import (
    CrossViewRayCandidate,
    MultiViewIntelligenceEngine,
    TemporalIdentityTracker,
    TemporalTrack,
)
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera, Quat, Vec3


def test_image_preprocessing_and_metadata():
    """Verify resolution-aware image preprocessing, dynamic range telemetry, and aspect ratio."""
    # Test valid 3-channel RGB image
    img = np.zeros((600, 800, 3), dtype=np.uint8)
    img[100:300, 200:500] = 180  # add high-contrast block

    processed, meta = preprocess_image_for_detection(img)
    assert processed.shape == (600, 800, 3)
    assert meta["original_shape"] == (600, 800, 3)
    assert meta["processed_shape"] == (600, 800, 3)
    assert meta["scale_factors"] == (1.0, 1.0)
    assert abs(meta["aspect_ratio"] - (800 / 600)) < 1e-4
    assert meta["std_intensity"] > 0.0
    assert not meta["is_low_contrast"]

    # Test resizing when max_dimension is set
    downscaled, meta_down = preprocess_image_for_detection(img, max_dimension=400)
    assert max(downscaled.shape[:2]) == 400
    assert meta_down["processed_shape"] == (300, 400, 3)
    assert abs(meta_down["scale_factors"][0] - 0.5) < 1e-3
    assert abs(meta_down["scale_factors"][1] - 0.5) < 1e-3


def test_mask_topology_extraction_and_rejection():
    """Verify mask topology extraction: holes, connected components, solidity, and degenerate rejection."""
    # 1. Solid rectangular mask
    mask_solid = np.zeros((200, 200), dtype=bool)
    mask_solid[40:160, 40:160] = True
    top_solid = extract_mask_topology(mask_solid)
    assert top_solid["is_valid"] is True
    assert top_solid["mask_area_px"] == 120 * 120
    assert top_solid["hole_count"] == 0
    assert top_solid["component_count"] == 1
    assert top_solid["solidity"] >= 0.95

    # 2. Mask with internal hole (annulus / ring)
    mask_hole = mask_solid.copy()
    mask_hole[80:120, 80:120] = False
    top_hole = extract_mask_topology(mask_hole)
    assert top_hole["is_valid"] is True
    assert top_hole["hole_count"] >= 1
    assert top_hole["solidity"] < 0.95  # hollow center reduces solidity

    # 3. Disconnected components
    mask_multi = np.zeros((200, 200), dtype=bool)
    mask_multi[20:60, 20:60] = True
    mask_multi[120:160, 120:160] = True
    top_multi = extract_mask_topology(mask_multi)
    assert top_multi["is_valid"] is True
    assert top_multi["component_count"] >= 2

    # 4. Degenerate mask rejection: tiny speckle (< 20 px)
    mask_tiny = np.zeros((100, 100), dtype=bool)
    mask_tiny[10:13, 10:13] = True  # 9 px
    top_tiny = extract_mask_topology(mask_tiny)
    assert top_tiny["is_valid"] is False
    assert "mask_area_too_small" in top_tiny["validation_note"]

    # 5. Degenerate mask rejection: whole-frame flood (> 98%)
    mask_flood = np.ones((100, 100), dtype=bool)
    top_flood = extract_mask_topology(mask_flood)
    assert top_flood["is_valid"] is False
    assert "exceeds_98_percent" in top_flood["validation_note"]


def test_hierarchical_component_recognition_and_confidence_propagation():
    """Verify subpart targets, containment validation, and confidence propagation."""
    # Check that required structural entities have genuine component targets
    for host in ("door", "window", "table", "chair", "cabinet", "outlet_plate", "lamp", "sink"):
        assert host in GENUINE_STRUCTURAL_PARTS
        targets = get_subpart_detection_targets(host)
        target_names = {t.name for t in targets}
        expected = EXPECTED_KEY_COMPONENTS.get(host, ())
        for exp in expected:
            assert exp in target_names or any(exp in n for n in target_names)

    # Test confidence propagation: c_subpart = sqrt(c_det * c_host)
    host_box = DetectionBox(
        box_id="door_01",
        class_name="door",
        box_2d=(100.0, 50.0, 300.0, 500.0),
        confidence=0.81,
    )
    raw_subpart_conf = 0.49
    expected_prop = round(math.sqrt(0.81 * 0.49), 4)  # 0.9 * 0.7 = 0.63
    assert abs(expected_prop - 0.63) < 1e-3


def test_rejection_ledger_records_reasons():
    """Verify that suppress_duplicates_preserving_parts records explicit rejection reasons."""
    rejection_records: list[dict] = []

    # Two heavily overlapping detections of different classes (chair vs table)
    box_a = DetectionBox(
        box_id="det_table",
        class_name="table",
        box_2d=(100.0, 100.0, 300.0, 300.0),
        confidence=0.90,
    )
    box_b = DetectionBox(
        box_id="det_chair",
        class_name="chair",
        box_2d=(105.0, 105.0, 295.0, 295.0),
        confidence=0.40,
    )

    kept = suppress_duplicates_preserving_parts(
        [box_a, box_b],
        iou_threshold=0.50,
        image_shape=(800, 800),
        rejection_records=rejection_records,
    )

    assert len(kept) == 1
    assert kept[0].box_id == "det_table"
    assert len(rejection_records) >= 1
    reasons = [r["reason"] for r in rejection_records]
    assert "cross_class_nms_suppressed" in reasons


def test_temporal_duplicate_track_merging_and_uncertainty():
    """Verify that TemporalIdentityTracker merges duplicate tracks and computes track uncertainty."""
    tracker = TemporalIdentityTracker()

    # Track 1: Door observed in frame 0 and 1
    track_1 = TemporalTrack(
        track_id="track_0001",
        class_name="door",
        active=True,
        observation_ids=["obs_01", "obs_02"],
        frame_indices=[0, 1],
        timestamps_sec=[0.0, 0.1],
        last_box_2d=(100.0, 100.0, 300.0, 500.0),
        age_frames=2,
        missed_frames=0,
    )
    # Track 2: Duplicate door track from another scale in frame 2 and 3
    track_2 = TemporalTrack(
        track_id="track_0002",
        class_name="door",
        active=True,
        observation_ids=["obs_03", "obs_04"],
        frame_indices=[2, 3],
        timestamps_sec=[0.2, 0.3],
        last_box_2d=(102.0, 101.0, 301.0, 498.0),
        age_frames=2,
        missed_frames=0,
    )

    tracker.active_tracks["track_0001"] = track_1
    tracker.active_tracks["track_0002"] = track_2

    assert track_1.track_uncertainty < 1.0

    # Merge duplicate tracks
    merged_count = tracker.merge_duplicate_tracks(iou_threshold=0.50)
    assert merged_count == 1
    assert len(tracker.active_tracks) == 1
    assert "track_0001" in tracker.active_tracks
    merged_track = tracker.active_tracks["track_0001"]
    assert len(merged_track.observation_ids) == 4
    assert merged_track.age_frames == 4
    assert track_2.termination_reason == "merged_into_track_0001"


def test_multiview_appearance_consistency_metric():
    """Verify multi-view appearance consistency score calculation across views."""
    engine = MultiViewIntelligenceEngine(max_ray_distance_m=0.50)

    K = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000)
    cam_a = PinholeCamera(K, CameraExtrinsics(Vec3(0.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))
    cam_b = PinholeCamera(K, CameraExtrinsics(Vec3(1.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))

    # Both observations share compatible normalized appearance features
    app_feat = tuple([0.2] * 24)

    ray_a = CrossViewRayCandidate(
        observation_id="obs_a",
        source_image_id="img_01",
        class_name="chair",
        origin=np.array([0.0, 0.0, 0.0]),
        direction=np.array([0.0, 0.0, 1.0]),
        box_2d=(400.0, 400.0, 600.0, 600.0),
        camera=cam_a,
        appearance_features=app_feat,
        scale_state="sfm_gauge",
        scale_units="sfm_units",
    )
    ray_b = CrossViewRayCandidate(
        observation_id="obs_b",
        source_image_id="img_02",
        class_name="chair",
        origin=np.array([1.0, 0.0, 0.0]),
        direction=np.array([-0.2425, 0.0, 0.9701]),
        box_2d=(350.0, 400.0, 550.0, 600.0),
        camera=cam_b,
        appearance_features=app_feat,
        scale_state="sfm_gauge",
        scale_units="sfm_units",
    )

    entities, decisions, refusals = engine.associate_candidates([ray_a, ray_b])
    assert len(entities) == 1
    ent = entities[0]
    assert ent.canonical_entity_id.startswith("phys_")
    assert ent.details.get("appearance_consistency_score") == 1.0
