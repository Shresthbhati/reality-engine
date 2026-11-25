"""Comprehensive tests & real-photo RTX 4050 benchmarks for the Perception Engineering Mega Sprint."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from perception.detection.grounded_arch import FULL_ARCHITECTURAL_VOCABULARY, GroundingDinoDetector
from perception.detection.normalized import CoordinateConvention, normalize_detection
from perception.evidence_enrichment import (
    associate_cross_view_detections,
    compute_observation_depth_evidence,
    deduplicate_video_frames,
)
from perception.scene_intelligence import (
    VisualSceneGraph,
    assess_scene_observability,
    bridge_scene_to_world,
    extract_appearance_and_materials,
    run_sequential_perception_pipeline,
)
from perception.segmentation.arch_segmenter import evaluate_mask_quality
from perception.world_bridge import assess_image_quality
from world_ir.observations import DepthSource
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore


def test_depth_evidence_mask_vs_box_and_never_fabricates_metric_scale():
    depth = np.full((120, 160), 4.0, dtype=np.float32)
    mask = np.zeros((120, 160), dtype=bool)
    mask[30:80, 40:100] = True
    depth[mask] = 1.5

    ev = compute_observation_depth_evidence(
        "obs_1",
        depth,
        (40.0, 30.0, 100.0, 80.0),
        mask=mask,
        depth_source=DepthSource.RELATIVE,
    )
    assert ev.median_depth == pytest.approx(1.5, abs=1e-3)
    assert ev.valid_depth_fraction == pytest.approx(1.0)
    assert ev.boundary_depth_discontinuity == pytest.approx(2.5, abs=1e-2)
    assert ev.foreground_background_boundary is True
    assert ev.metric_scale_status == "relative_uncalibrated"

    # Missing / NaN depth stays missing
    nan_depth = np.full((120, 160), np.nan, dtype=np.float32)
    ev_nan = compute_observation_depth_evidence("obs_2", nan_depth, (40, 30, 100, 80), mask=mask)
    assert ev_nan.median_depth is None
    assert ev_nan.metric_scale_status == "missing"
    assert ev_nan.unreliable_reason == "no_valid_depth_pixels"


def test_adversarial_image_quality_dark_motion_blur_and_noise():
    # Dark underexposed room
    dark = np.full((240, 320, 3), 12, dtype=np.uint8)
    q_dark = assess_image_quality(dark)
    assert q_dark.is_underexposed is True
    assert q_dark.underexposed_fraction == pytest.approx(1.0)

    # Directional motion blur (horizontal stripes blurred horizontally)
    base = np.zeros((240, 320, 3), dtype=np.uint8)
    for x in range(0, 320, 20):
        cv2.line(base, (x, 0), (x, 239), (220, 220, 220), 2)
    blurred_h = cv2.GaussianBlur(base, (31, 1), 0)
    q_motion = assess_image_quality(blurred_h, blur_threshold=5000.0)
    assert q_motion.is_blurry is True
    assert q_motion.motion_blur_score > 0.25
    assert q_motion.is_motion_blurred is True

    # High sensor noise
    rng = np.random.default_rng(7)
    noisy = np.clip(128 + rng.normal(0, 35, size=(240, 320, 3)), 0, 255).astype(np.uint8)
    q_noisy = assess_image_quality(noisy)
    assert q_noisy.is_noisy is True


def test_mask_quality_never_presents_box_as_valid_mask():
    rgb = np.full((200, 200, 3), 120, dtype=np.uint8)
    # Exact full-box fill -> flagged as degenerate_box_fill
    box_fill = np.zeros((200, 200), dtype=bool)
    box_fill[20:120, 20:120] = True
    val_fill, _, _, ratio, _, _ = evaluate_mask_quality(box_fill, (20, 20, 120, 120), rgb)
    assert val_fill == "degenerate_box_fill"
    assert ratio == pytest.approx(1.0)

    # Unavailable mask
    val_none, area_none, _, _, _, unc_none = evaluate_mask_quality(None, (20, 20, 120, 120), rgb)
    assert val_none == "unavailable"
    assert area_none == 0
    assert unc_none == pytest.approx(1.0)


def test_adversarial_repeated_structures_and_contradictory_cross_view_association():
    # Two identical-looking doors in the same view -> never merged
    d1 = normalize_detection(
        {"label": "door", "bbox_px": (20, 40, 100, 240), "score": 0.9},
        image_width=640,
        image_height=480,
        image_id="view_0",
    )
    d2 = normalize_detection(
        {"label": "door", "bbox_px": (320, 40, 400, 240), "score": 0.89},
        image_width=640,
        image_height=480,
        image_id="view_0",
    )
    # Cross-view door vs window with identical embedding -> contradictory
    d3 = normalize_detection(
        {"label": "window", "bbox_px": (30, 40, 110, 240), "score": 0.85},
        image_width=640,
        image_height=480,
        image_id="view_1",
    )
    # Cross-view door without camera geometry -> ambiguous_repeated_structure
    d4 = normalize_detection(
        {"label": "door", "bbox_px": (35, 45, 115, 245), "score": 0.88},
        image_width=640,
        image_height=480,
        image_id="view_2",
    )
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    embs = {d.detection_id: vec for d in (d1, d2, d3, d4)}
    assocs = associate_cross_view_detections([d1, d2, d3, d4], embeddings_by_id=embs)

    statuses = {(a.label_a, a.label_b, a.status) for a in assocs}
    assert ("door", "door", "ambiguous_repeated_structure") in statuses
    assert ("door", "window", "contradictory") in statuses


def test_video_frame_deduplication_preserves_timestamps_and_indices():
    f0 = np.full((120, 160, 3), 40, dtype=np.uint8)
    f1 = f0.copy()  # duplicate frame
    f2 = np.full((120, 160, 3), 180, dtype=np.uint8)  # distinct keyframe
    res = deduplicate_video_frames([f0, f1, f2], [0.0, 0.033, 1.5])
    assert res["kept_indices"] == [0, 2]
    assert res["dropped_duplicate_indices"] == [1]
    assert res["keyframes"] == [
        {"frame_index": 0, "timestamp_s": 0.0},
        {"frame_index": 2, "timestamp_s": 1.5},
    ]


def test_mask_cropped_material_hypotheses_stay_unverified():
    rgb = np.full((120, 160, 3), 240, dtype=np.uint8)  # white wall background
    mask = np.zeros((120, 160), dtype=bool)
    mask[30:80, 40:100] = True
    rgb[mask] = (140, 90, 50)  # brown wood table inside mask only
    mat = extract_appearance_and_materials(rgb, (20, 10, 140, 110), "table", mask=mask)
    assert mat["mask_cropped"] is True
    assert mat["material_verified"] is False
    assert mat["mean_color_rgb"] == [140.0, 90.0, 50.0]
    assert len(mat["material_hypotheses"]) >= 1


def test_observability_separates_missing_evidence_from_suggestions():
    rep = assess_scene_observability([], [], processed_views=["v0"])
    assert "insufficient_camera_viewpoints" in rep.missing_evidence
    assert "unseen_wall_surfaces" in rep.missing_evidence
    assert "incomplete_floor_visibility" in rep.missing_evidence
    assert all(s.missing_evidence_type in rep.missing_evidence for s in rep.suggestions)


def test_real_colmap_photos_full_multimodel_rtx4050_640_and_1024(tmp_path):
    photo_dir = Path("data/worldstore/colmap-sessions/wld_1a3fc79fe10f/current/images")
    if not photo_dir.exists():
        pytest.skip("COLMAP photo fixture directory not present")
    photos = sorted(str(p) for p in photo_dir.glob("*.jpg"))[:3]
    if len(photos) < 2:
        pytest.skip("Need at least 2 COLMAP photos")

    res_640 = run_sequential_perception_pipeline(
        photos,
        max_edge=640,
        use_owlvit=True,
        use_grounding_dino=True,
        use_florence2=True,
        use_sam2=True,
        use_dinov2=True,
        extract_materials=True,
        queries=FULL_ARCHITECTURAL_VOCABULARY[:20],
    )
    res_1024 = run_sequential_perception_pipeline(
        photos,
        max_edge=1024,
        use_owlvit=True,
        use_grounding_dino=True,
        use_florence2=True,
        use_sam2=True,
        use_dinov2=True,
        extract_materials=True,
        queries=FULL_ARCHITECTURAL_VOCABULARY[:20],
    )

    print("\n[MegaSprint Real Photos 640px] ", res_640["telemetry"])
    print("[MegaSprint Real Photos 1024px]", res_1024["telemetry"])

    assert res_640["telemetry"]["observations_total"] > 0
    assert res_1024["telemetry"]["observations_total"] > 0
    assert res_640["telemetry"]["valid_masks"] > 0
    assert res_1024["telemetry"]["valid_masks"] > 0

    # Persist into WorldIR + WorldStore round-trip
    world = WorldIR("wld_megasprint_real")
    sg: VisualSceneGraph = res_1024["scene_graph"]
    _, report = bridge_scene_to_world(world, sg.observations)
    # Relative-depth observations without camera poses must NOT fabricate metric ledger entries
    assert report.added_ledger_observations == 0

    store = WorldStore(str(tmp_path / "ws"))
    sv = store.save_version(world, parent=None)
    reloaded = store.load_version(sv.version_id)
    assert "scene_intelligence" in reloaded.metadata
    assert reloaded.metadata["scene_intelligence"]["observation_count"] == len(sg.observations)
