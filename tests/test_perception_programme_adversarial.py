"""Adversarial, orchestrator, video, dataset-adapter, and E2E tests for Programmes A-T."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from benchmarks.dataset_adapters import (
    compute_depth_metrics,
    compute_detection_metrics,
    compute_segmentation_metrics,
    probe_datasets,
)
from engine.math import Quat, Vec3
from perception.evidence_enrichment import (
    CrossViewAssociationCandidate,
    ObservationDepthEvidence,
    compute_observation_quality_score,
    evaluate_association_quality,
    process_video_file,
    validate_camera_bundle,
)
from perception.inventory import write_engine_inventory_json
from perception.orchestrator import InferencePlan, InferencePlanError, run_inference_plan
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore


def test_inference_plan_refuses_missing_required_model():
    bad_plan = InferencePlan(plan_name="bad", required_models=("nonexistent_model_xyz",))
    with pytest.raises(InferencePlanError, match="nonexistent_model_xyz"):
        run_inference_plan([], plan=bad_plan)


def test_camera_bundle_validation_catches_invalid_poses():
    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    good_cam = PinholeCamera(intr, CameraExtrinsics(Vec3(0.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))
    bad_quat = PinholeCamera(intr, CameraExtrinsics(Vec3(0.0, 0.0, 0.0), Quat(2.0, 2.0, 2.0, 2.0)))
    nan_pos = PinholeCamera(intr, CameraExtrinsics(Vec3(float("nan"), 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))

    rep = validate_camera_bundle({"c_good": good_cam, "c_badq": bad_quat, "c_nan": nan_pos})
    assert rep.total_cameras == 3
    assert rep.valid_cameras == 1
    assert rep.all_valid is False
    assert "unnormalized_quaternion" in rep.invalid_cameras["c_badq"]
    assert rep.invalid_cameras["c_nan"] == "non_finite_pose"


def test_association_quality_benchmark_and_failure_diagnostics():
    preds = [
        CrossViewAssociationCandidate("a1", "b1", "v0", "v1", "door", "door", 0.9, 0.8, "matched", 0.15, "ok"),
        CrossViewAssociationCandidate("a2", "b3", "v0", "v1", "door", "door", 0.88, 0.7, "matched", 0.2, "false_merge"),
        CrossViewAssociationCandidate("a3", "a4", "v0", "v0", "window", "window", 1.0, 0.0, "ambiguous_repeated_structure", 0.8, "same_view"),
    ]
    gt = [("a1", "b1"), ("a5", "b5")]
    rep = evaluate_association_quality(preds, gt)
    assert rep.precision == pytest.approx(0.5)
    assert rep.recall == pytest.approx(0.5)
    assert rep.false_merges == 1
    assert rep.false_splits == 1
    kinds = {d["failure_kind"] for d in rep.failure_diagnostics}
    assert "embedding_symmetry_confusion" in kinds
    assert "sfm_low_overlap_disconnection" in kinds


def test_documented_observation_quality_score():
    de = ObservationDepthEvidence("o1", 2.0, 0.1, 0.9, 0.5, True, 0, "relative_uncalibrated", "relative", "da2", "v1")
    good = compute_observation_quality_score(confidence=0.9, mask_validity="valid", depth_evidence=de)
    blurry = compute_observation_quality_score(confidence=0.9, mask_validity="valid", depth_evidence=de, image_is_blurry=True)
    assert good["quality_score"] > blurry["quality_score"]
    assert good["formula"] == "0.35*det + 0.25*mask + 0.20*depth + 0.20*img"


def test_real_mp4_video_decoding_deduplication_and_motion(tmp_path):
    mp4 = tmp_path / "walkthrough.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
    assert writer.isOpened()
    f_static = np.full((120, 160, 3), 40, dtype=np.uint8)
    cv2.rectangle(f_static, (20, 20), (60, 90), (200, 200, 200), -1)
    for _ in range(4):
        writer.write(f_static)  # duplicate static frames
    for shift in (20, 45, 70):
        f_move = np.full((120, 160, 3), 40, dtype=np.uint8)
        cv2.rectangle(f_move, (20 + shift, 20), (60 + shift, 90), (200, 200, 200), -1)
        writer.write(f_move)
    writer.release()

    res = process_video_file(str(mp4), min_l1_diff=2.0)
    assert res["decoded_frames"] == 7
    assert res["dropped_duplicate_count"] >= 2
    assert len(res["keyframes"]) >= 3
    assert res["keyframes"][0]["frame_index"] == 0
    assert res["keyframes"][0]["timestamp_s"] == 0.0


def test_dataset_adapters_and_metrics():
    catalog = {e.name: e for e in probe_datasets()}
    assert catalog["scannet"].availability_status == "blocked"
    assert catalog["local_colmap_sessions"].availability_status in ("available", "blocked")

    det_m = compute_detection_metrics([("door", (10, 10, 50, 50))], [("door", (12, 12, 50, 50))])
    assert det_m["precision"] == 1.0 and det_m["recall"] == 1.0

    m1 = np.zeros((20, 20), dtype=bool)
    m1[2:10, 2:10] = True
    seg_m = compute_segmentation_metrics(m1, m1)
    assert seg_m["iou"] == 1.0 and seg_m["dice"] == 1.0

    d_m = compute_depth_metrics(np.full((10, 10), 2.0), np.full((10, 10), 2.0))
    assert d_m == {"abs_rel": 0.0, "rmse": 0.0, "delta_1": 1.0}


def test_engine_inventory_generation(tmp_path):
    out = write_engine_inventory_json(tmp_path / "ENGINE_INVENTORY.json")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) >= 8
    assert any(item["name"] == "SpatialInputsV1Producer" for item in data)


def test_e2e_inference_plan_on_real_colmap_session_to_worldstore(tmp_path):
    photo_dir = Path("data/worldstore/colmap-sessions/wld_841c36c182db/current/images")
    if not photo_dir.exists():
        pytest.skip("COLMAP photo fixture wld_841c36c182db not present")
    photos = sorted(str(p) for p in photo_dir.glob("*.jpg"))[:2]
    plan = InferencePlan(
        plan_name="e2e_colmap_test",
        required_models=("depth_anything", "mask_rcnn"),
        optional_models=("owlvit", "grounding_dino", "sam2_hf", "dinov2"),
        max_edge=640,
    )
    res = run_inference_plan(photos, run_id="e2e_run_1", plan=plan)
    assert res.spatial_inputs.schema_version == "spatial_inputs.v1"
    assert len(res.spatial_inputs.observations) > 0
    assert res.telemetry["valid_masks"] > 0

    world = WorldIR("wld_e2e_plan")
    world.metadata["scene_intelligence"] = res.scene_graph.to_dict()
    store = WorldStore(str(tmp_path / "ws"))
    sv = store.save_version(world, parent=None)
    loaded = store.load_version(sv.version_id)
    assert loaded.metadata["scene_intelligence"]["observation_count"] == len(res.scene_graph.observations)


def test_cross_view_association_uses_real_epipolar_residual_px():
    from perception.detection.normalized import normalize_detection
    from perception.evidence_enrichment import associate_cross_view_detections

    intr = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    cam_a = PinholeCamera(intr, CameraExtrinsics(Vec3(0.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))
    cam_b = PinholeCamera(intr, CameraExtrinsics(Vec3(1.0, 0.0, 0.0), Quat(1.0, 0.0, 0.0, 0.0)))
    # Horizontal translation along X -> same Y row (v=240) is epipolar-consistent; shifted Y (v=400) is inconsistent
    da = normalize_detection({"label": "chair", "bbox_px": (300, 220, 340, 260), "score": 0.9}, image_width=640, image_height=480, image_id="va")
    db_good = normalize_detection({"label": "chair", "bbox_px": (220, 220, 260, 260), "score": 0.9}, image_width=640, image_height=480, image_id="vb")
    db_bad = normalize_detection({"label": "chair", "bbox_px": (220, 380, 260, 420), "score": 0.9}, image_width=640, image_height=480, image_id="vc")
    vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assocs = associate_cross_view_detections(
        [da, db_good, db_bad],
        embeddings_by_id={da.detection_id: vec, db_good.detection_id: vec, db_bad.detection_id: vec},
        cameras_by_view={"va": cam_a, "vb": cam_b, "vc": cam_b},
    )
    by_pair = {(a.view_a, a.view_b): a for a in assocs}
    assert by_pair[("va", "vb")].status == "matched"
    assert by_pair[("va", "vc")].status == "rejected"


def test_structural_regions_never_claim_reconstructed_wall_plane():
    from perception.scene_intelligence import build_visual_scene_graph, extract_structural_regions
    from perception.world_bridge import (
        assess_image_quality,
        build_visual_observations_from_detections,
    )
    from world_ir.observations import DepthSource

    rgb = np.full((240, 320, 3), 150, dtype=np.uint8)
    qual = assess_image_quality(rgb)
    depth = np.full((240, 320), 3.0, dtype=np.float32)
    obs = build_visual_observations_from_detections(
        source_evidence_id="ev0",
        source_image_id="ev0",
        model_id="grounding_dino",
        model_version="tiny",
        depth_map=depth,
        depth_source=DepthSource.RELATIVE,
        quality=qual,
        detections=[
            {"label": "wall", "bbox_px": [5.0, 5.0, 310.0, 220.0], "score": 0.7},
            {"label": "door", "bbox_px": [100.0, 40.0, 180.0, 220.0], "score": 0.65},
        ],
    )
    sg = build_visual_scene_graph(obs, processed_views=["ev0"])
    sr = extract_structural_regions(sg)
    assert sr["counts"]["walls"] == 1
    assert sr["counts"]["openings"] == 1
    assert sr["wall_regions"][0]["plane_reconstructed"] is False
    assert sr["wall_regions"][0]["partial_or_unresolved"] is True


def test_annotation_manifest_separates_pseudo_labels_and_compares_detectors():
    from benchmarks.annotation_manifest import (
        AnnotationManifest,
        BoxAnnotation,
        ImageAnnotation,
        InvalidEvaluationGroundTruthError,
        compare_detectors_on_manifest,
        evaluate_detector_on_manifest,
    )

    pseudo_manifest = AnnotationManifest(
        manifest_id="m_pseudo",
        dataset_name="colmap_local",
        split="unscored",
        images=(
            ImageAnnotation("im0", "im0.jpg", 640, 480, "model_pseudo_label", "owlvit", (BoxAnnotation("chair", (10, 10, 50, 50)),)),
        ),
    )
    with pytest.raises(InvalidEvaluationGroundTruthError):
        evaluate_detector_on_manifest(pseudo_manifest, "owlvit", lambda im: [])

    reviewed = AnnotationManifest(
        manifest_id="m_rev",
        dataset_name="colmap_local",
        split="test",
        images=(
            ImageAnnotation("im0", "im0.jpg", 640, 480, "human_reviewed", "reviewer_1", (BoxAnnotation("door", (10, 10, 100, 200)),)),
        ),
    )
    cmp = compare_detectors_on_manifest(
        reviewed,
        {
            "maskrcnn": lambda im: [],
            "grounding_dino": lambda im: [("door", (12, 12, 100, 200))],
        },
    )
    assert cmp["results"]["maskrcnn"]["overall"]["recall"] == 0.0
    assert cmp["results"]["grounding_dino"]["overall"]["precision"] == 1.0
    assert cmp["results"]["grounding_dino"]["overall"]["recall"] == 1.0


def test_video_mp4_to_evidence_world_job_emits_spatial_inputs_v1(tmp_path):
    from apps.api import evidence_world_stage
    from engine.multiengine.executor import TorchResourceProbe
    from perception.spatial_inputs import SPATIAL_INPUTS_FILE
    from tests.test_evidence_world_job import _setup

    mp4 = tmp_path / "room_scan.mp4"
    writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    assert writer.isOpened()
    for shift in (0, 30, 60):
        frame = np.full((240, 320, 3), 80, dtype=np.uint8)
        cv2.rectangle(frame, (20 + shift, 30), (100 + shift, 180), (220, 200, 180), -1)
        writer.write(frame)
    writer.release()

    reg, adapters = _setup()
    facts = evidence_world_stage.run_for_job(
        team_id="team_v",
        world_id="world_v",
        job_id="job_v",
        sources=[str(mp4)],
        base=tmp_path / "ws",
        flow_inputs=lambda: (reg, adapters, TorchResourceProbe()),
        detection="legacy",
    )
    assert "video_ingestion" in facts
    assert facts["video_ingestion"][0]["decoded_frames"] == 3
    assert len(facts["video_ingestion"][0]["keyframes"]) >= 2
    assert facts["spatial_inputs_v1"]["schema_version"] == "spatial_inputs.v1"
    si_file = evidence_world_stage.evidence_root("team_v", "world_v", "job_v", base=tmp_path / "ws") / SPATIAL_INPUTS_FILE
    assert si_file.is_file()


def test_nms_dedup_preserves_cross_model_disagreement_and_mask_refinement():
    from perception.detection.normalized import (
        normalize_detection,
        suppress_duplicate_normalized_detections,
    )
    from perception.segmentation.arch_segmenter import compare_masks_iou, refine_binary_mask

    # Same model + same label + high IoU -> suppressed; different model or label -> preserved
    d1 = normalize_detection({"label": "chair", "bbox_px": (10, 10, 100, 100), "score": 0.9}, image_width=320, image_height=240, model_name="gdino")
    d2 = normalize_detection({"label": "chair", "bbox_px": (12, 12, 100, 100), "score": 0.7}, image_width=320, image_height=240, model_name="gdino")
    d3 = normalize_detection({"label": "sofa", "bbox_px": (12, 12, 100, 100), "score": 0.8}, image_width=320, image_height=240, model_name="owlvit")
    kept, suppressed = suppress_duplicate_normalized_detections([d1, d2, d3])
    assert suppressed == 1
    assert len(kept) == 2

    raw_mask = np.zeros((100, 100), dtype=bool)
    raw_mask[20:60, 20:60] = True
    raw_mask[40, 40] = False  # 1px interior hole
    raw_mask[5:7, 5:7] = True  # 4px speckle island
    refined = refine_binary_mask(raw_mask)
    assert refined[40, 40] == True  # noqa: E712
    assert refined[5, 5] == False  # noqa: E712
    assert compare_masks_iou(raw_mask, refined) > 0.9


def test_depth_surface_orientation_and_video_occlusion_track_recovery():
    from perception.detection.normalized import normalize_detection
    from perception.evidence_enrichment import (
        estimate_surface_orientation_from_depth,
        link_video_object_tracks,
    )

    # Vertical gradient -> horizontal_floor_or_ceiling
    ys = np.linspace(1.0, 5.0, 100, dtype=np.float32)[:, None]
    floor_depth = np.repeat(ys, 100, axis=1)
    ori = estimate_surface_orientation_from_depth(floor_depth)
    assert ori["orientation_hypothesis"] == "horizontal_floor_or_ceiling"
    assert ori["relative_depth_basis"] is True

    # Track recovery across 1-frame occlusion
    f0 = [normalize_detection({"label": "chair", "bbox_px": (50, 50, 150, 150), "score": 0.9}, image_width=320, image_height=240, frame_index=0, timestamp_s=0.0)]
    f1: list = []  # occluded frame
    f2 = [normalize_detection({"label": "chair", "bbox_px": (55, 52, 155, 152), "score": 0.88}, image_width=320, image_height=240, frame_index=2, timestamp_s=0.2)]
    tr_res = link_video_object_tracks([f0, f1, f2], max_frame_gap=2)
    assert tr_res["track_count"] == 1
    assert tr_res["persistent_tracks"] == 1
    assert tr_res["occlusion_recovered_tracks"] == 1


def test_vram_aware_plan_selection_and_improvement_ledger(tmp_path):
    from perception.improvement_ledger import write_improvement_ledger
    from perception.orchestrator import (
        BALANCED_INTERIOR_PLAN,
        FAST_STRUCTURAL_PLAN,
        FULL_FORENSIC_PLAN,
        select_inference_plan_for_inputs,
    )

    assert select_inference_plan_for_inputs(["a.jpg"], vram_mb=3500.0) == FAST_STRUCTURAL_PLAN
    assert select_inference_plan_for_inputs(["a.jpg"] * 15, vram_mb=6140.0) == FAST_STRUCTURAL_PLAN
    assert select_inference_plan_for_inputs(["a.jpg", "b.jpg"], vram_mb=6140.0) == BALANCED_INTERIOR_PLAN
    assert select_inference_plan_for_inputs(["a.jpg"], forensic=True) == FULL_FORENSIC_PLAN

    ledger_path = write_improvement_ledger(tmp_path / "LEDGER.json")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["counts"]["implemented"] >= 11
    assert ledger["counts"]["blocked"] >= 2
    assert ledger["counts"]["proposed"] >= 1


def test_compare_local_feature_matchers_on_real_colmap_pair():
    from perception.evidence_enrichment import compare_local_feature_matchers

    photo_dir = Path("data/worldstore/colmap-sessions/wld_1a3fc79fe10f/current/images")
    if not photo_dir.exists():
        pytest.skip("COLMAP photo fixture not present")
    imgs = sorted(str(p) for p in photo_dir.glob("*.jpg"))
    if len(imgs) < 2:
        pytest.skip("Need at least 2 photos")
    res = compare_local_feature_matchers(imgs[0], imgs[1], max_side=640)
    assert res["lightglue_disk"]["inliers"] > res["sift"]["inliers"]
    assert res["lightglue_aliked"]["inliers"] > res["sift"]["inliers"]


def test_depth_anything_v2_metric_indoor_backend_executes_on_real_photo():
    from evidence.session import EvidenceItem, EvidenceKind
    from perception.availability import real_models_by_name
    from perception.depth.depth_anything_backend import DepthAnythingV2MetricBackend

    avail = real_models_by_name()
    if avail["depth_anything_metric"].status != "available":
        pytest.skip("depth_anything_metric checkpoint not cached")
    photo_dir = Path("data/worldstore/colmap-sessions/wld_1a3fc79fe10f/current/images")
    if not photo_dir.exists():
        pytest.skip("COLMAP photo fixture not present")
    img = sorted(str(p) for p in photo_dir.glob("*.jpg"))[0]
    backend = DepthAnythingV2MetricBackend()
    maps = backend.estimate_depth([EvidenceItem(id="ev_m0", kind=EvidenceKind.PHOTO, source_uri=img)])
    backend.unload()
    assert len(maps) == 1
    assert maps[0].unit == "meters"
    arr = np.asarray(maps[0].values, dtype=np.float32)
    assert float(arr.min()) > 0.1
    assert float(arr.max()) < 50.0
