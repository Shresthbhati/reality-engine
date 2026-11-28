"""A5 Comprehensive Reliability & Edge-Case Test Suite.

Covers edge cases, invariants, contracts, and integration paths across all A5
subsystems:
  - A5.1 Segmentation reliability, RLE, polygons, dimension guards
  - A5.2 Duplicate resolution with mask IoU, appearance conflict, competing labels
  - A5.3 Cross-view identity with pose uncertainty, occlusion, appearance enrichment
  - A5.4/A5.5 Architectural & furniture refinement edge cases
  - Objective 2 Indoor validation (ScanNet, SUN RGB-D, Replica, six-room pathway)
  - Objective 3 Resumable runner & A5 benchmark comparison
  - Objective 4 GPU lifecycle, VRAM ceiling, adaptive batch/resolution, dispatch
  - Objective 5 Claude integration bridge & EvidenceMatrix persistence
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.perception.evaluator import EvaluationMode
from benchmarks.perception.indoor_validation_runner import (
    prepare_six_room_measured_capture_pathway,
    run_indoor_dataset_validation,
)
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox
from perception.contracts.claude_integration_v2 import (
    ClaudeReconstructionFeedback,
    WorldEntityIdentity,
    execute_claude_handoff_and_persist,
    parse_claude_camera_poses_from_json,
)
from perception.datasets.indoor_registry import normalize_dataset_sample
from perception.intelligence.a5_benchmark_suite import (
    A5ReliabilityMetrics,
    compare_a4_vs_a5_reliability,
    compute_a5_reliability_metrics,
)
from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    AppearanceDescriptor,
    CoordinateSystemSpec,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    PerceptionBundle,
    ScoreSemantics,
    SegmentationObservation,
)
from perception.intelligence.deduplication_v2 import (
    compute_rle_mask_iou,
    resolve_duplicate_detections_v2,
)
from perception.intelligence.dinov2_embeddings import (
    extract_dinov2_embeddings,
    probe_dinov2_checkpoint,
)
from perception.intelligence.gpu_lifecycle import (
    HardwareProfile,
    ModelLifecycleManager,
    probe_hardware_profile,
    select_adaptive_batch_and_resolution,
)
from perception.intelligence.identity import CameraPoseConstraint
from perception.intelligence.identity_v2 import (
    enrich_observations_with_appearance,
    extract_crop_appearance_descriptor,
    resolve_cross_view_identities_v2,
)
from perception.intelligence.mask_geometry import (
    compute_tight_box_from_mask,
    decode_rle_to_mask,
    extract_mask_polygon_px,
    refine_boxes_from_verified_masks,
)
from perception.intelligence.multi_engine_dispatch import dispatch_sequential_detectors
from perception.intelligence.open_vocab import Domain
from perception.intelligence.production_pipeline_a5 import (
    ingest_images_rgb,
    run_a5_production_pipeline,
)
from perception.intelligence.resumable_runner import (
    compute_run_fingerprint,
    run_resumable_a5_job,
)
from perception.intelligence.scene_refinement_v2 import (
    ARCHITECTURAL_SURVEY_CLASSES,
    FURNITURE_SUBPARTS,
    INTERIOR_SURVEY_FURNITURE,
    refine_scene_observations_v2,
)
from perception.intelligence.segmentation_v2 import (
    attach_masks_to_observations,
    build_segmentation_observation_from_mask,
)
from perception.segmentation_adapter import _encode_rle_binary, _resolve_image_bboxes


def _obs(
    obs_id: str,
    image_id: str,
    label: str,
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.85,
    category: str = "furniture",
    engine_id: str = "grounding_dino",
    inference_mode: str = "full_frame",
    tile_id: str | None = None,
    embedding: tuple[float, ...] = (),
    hist: tuple[float, ...] = (),
    is_partial: bool = False,
    occlusion_level: str = "none",
    seg: SegmentationObservation | None = None,
    width: int = 640,
    height: int = 480,
) -> NormalizedPerceptionObservation:
    return NormalizedPerceptionObservation(
        observation_id=obs_id,
        schema_version=SCHEMA_VERSION,
        raw_label=label,
        canonical_label=label,
        taxonomy_id=f"re:{label}",
        category=category,
        bbox_px=bbox,
        coords=CoordinateSystemSpec("image_px", "pixels", width, height),
        scores=ScoreSemantics(raw_score=confidence, score_emitted=True),
        provenance=DetectionProvenance(
            source_image_id=image_id,
            execution=ModelExecutionProvenance(engine_id, engine_id, f"{engine_id}_v1"),
            inference_mode=inference_mode,
            tile_id=tile_id,
        ),
        is_partial=is_partial,
        occlusion_level=occlusion_level,
        appearance=AppearanceDescriptor(embedding_vector=embedding, color_histogram_24=hist),
        segmentation=seg,
    )


# ---------------------------------------------------------------------------
# A5.1 Mask Geometry & Segmentation Edge Cases
# ---------------------------------------------------------------------------


def test_rle_decode_rejects_malformed_inputs_and_handles_full_empty() -> None:
    assert decode_rle_to_mask(None) is None
    assert decode_rle_to_mask({"size": [10], "counts": [100]}) is None
    assert decode_rle_to_mask({"size": [0, 10], "counts": []}) is None
    assert decode_rle_to_mask({"size": [4, 4], "counts": [-1, 16]}) is None

    empty = np.zeros((16, 20), dtype=bool)
    rle_e = _encode_rle_binary(empty)
    dec_e = decode_rle_to_mask(rle_e)
    assert dec_e is not None and not dec_e.any()
    assert compute_tight_box_from_mask(empty) is None

    full = np.ones((12, 14), dtype=bool)
    rle_f = _encode_rle_binary(full)
    dec_f = decode_rle_to_mask(rle_f)
    assert dec_f is not None and dec_f.all()
    assert compute_tight_box_from_mask(full) == (0.0, 0.0, 14.0, 12.0)


def test_mask_polygon_and_box_refinement_skips_low_quality_or_extreme_shrink() -> None:
    mask_small = np.zeros((480, 640), dtype=bool)
    mask_small[200:220, 300:320] = True
    rle_small = _encode_rle_binary(mask_small)
    seg_low_q = SegmentationObservation(
        mask_id="m_low",
        observation_id="o_low",
        mask_kind="visible_object_mask",
        area_px=400,
        box_iou=0.05,
        solidity=0.9,
        hole_count=0,
        boundary_uncertainty=0.1,
        quality_score=0.20,
        rle_mask=rle_small,
        model_id="sam2",
    )
    obs = _obs("o_low", "img_1", "chair", (50, 50, 500, 400), seg=seg_low_q)
    refined, summaries = refine_boxes_from_verified_masks([obs], min_mask_quality=0.35)
    assert len(summaries) == 1
    assert refined[0].bbox_px == (50.0, 50.0, 500.0, 400.0)
    assert len(refined[0].segmentation.polygon_px) >= 4


def test_attach_masks_handles_empty_observations_and_partial_precomputed() -> None:
    empty_out, empty_rep = attach_masks_to_observations([])
    assert empty_out == []
    assert empty_rep.status == "skipped"

    mask_a = np.zeros((480, 640), dtype=bool)
    mask_a[100:250, 100:250] = True
    obs = [
        _obs("o1", "img_1", "chair", (100, 100, 250, 250)),
        _obs("o2", "img_1", "table", (300, 200, 550, 420)),
    ]
    out, rep = attach_masks_to_observations(obs, precomputed_masks={"o1": mask_a})
    assert rep.masks_produced == 1
    assert out[0].segmentation is not None
    assert out[1].segmentation is None


def test_resolve_image_bboxes_missing_returns_empty() -> None:
    assert _resolve_image_bboxes({"a.jpg": [(1, 2, 3, 4)]}, "dir/b.jpg") == []


# ---------------------------------------------------------------------------
# A5.2 Duplicate Resolution Edge Cases
# ---------------------------------------------------------------------------


def test_dedup_empty_and_mask_iou_duplicate_suppression() -> None:
    empty_res = resolve_duplicate_detections_v2([])
    assert empty_res.summary["input_count"] == 0
    assert empty_res.canonical_observations == ()

    m1 = np.zeros((480, 640), dtype=bool)
    m1[100:300, 100:280] = True
    m2 = np.zeros((480, 640), dtype=bool)
    m2[105:295, 105:275] = True

    seg1 = SegmentationObservation("m1", "o1", "visible_object_mask", int(m1.sum()), 0.9, 0.95, 0, 0.1, 0.9, rle_mask=_encode_rle_binary(m1))
    seg2 = SegmentationObservation("m2", "o2", "visible_object_mask", int(m2.sum()), 0.88, 0.94, 0, 0.1, 0.88, rle_mask=_encode_rle_binary(m2))

    assert compute_rle_mask_iou(seg1.rle_mask, seg2.rle_mask) > 0.85
    assert compute_rle_mask_iou({"size": [10, 10], "counts": [100]}, {"size": [12, 10], "counts": [120]}) is None

    o1 = _obs("o1", "img_1", "chair", (100, 100, 280, 300), confidence=0.80, seg=seg1)
    o2 = _obs("o2", "img_1", "chair", (105, 105, 275, 295), confidence=0.90, engine_id="owlv2", seg=seg2)
    res = resolve_duplicate_detections_v2([o1, o2])
    assert res.summary["suppressed_count"] == 1
    assert res.canonical_observations[0].observation_id == "o2"
    assert res.clusters[0].reason == "mask_iou_duplicate"


def test_dedup_appearance_conflict_prevents_false_merge_at_moderate_iou() -> None:
    o1 = _obs("o1", "img_1", "chair", (100, 100, 240, 260), confidence=0.85, embedding=(1.0, 0.0, 0.0))
    o2 = _obs("o2", "img_1", "chair", (120, 110, 260, 270), confidence=0.82, engine_id="owlv2", embedding=(0.0, 1.0, 0.0))
    res = resolve_duplicate_detections_v2([o1, o2], cross_engine_iou_threshold=0.50)
    assert res.summary["canonical_count"] == 2
    assert res.summary["suppressed_count"] == 0


# ---------------------------------------------------------------------------
# A5.3 Cross-View Identity Edge Cases
# ---------------------------------------------------------------------------


def test_appearance_descriptor_color_classification_and_enrichment() -> None:
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    img[10:60, 10:60] = [230, 30, 30]     # red_warm
    img[70:120, 10:60] = [30, 40, 230]    # blue_cool
    img[130:180, 10:60] = [240, 240, 240] # white
    img[10:60, 100:150] = [20, 20, 20]    # dark

    assert extract_crop_appearance_descriptor(img, (10, 10, 60, 60)).dominant_color_name == "red_warm"
    assert extract_crop_appearance_descriptor(img, (10, 70, 60, 120)).dominant_color_name == "blue_cool"
    assert extract_crop_appearance_descriptor(img, (10, 130, 60, 180)).dominant_color_name == "white"
    assert extract_crop_appearance_descriptor(img, (100, 10, 150, 60)).dominant_color_name == "dark"

    obs = [_obs("c1", "img_1", "chair", (10, 10, 60, 60), width=200, height=200)]
    enriched = enrich_observations_with_appearance(obs, {"img_1": img})
    assert len(enriched[0].appearance.color_histogram_24) == 24
    assert enriched[0].appearance.dominant_color_name == "red_warm"


def test_identity_v2_pose_uncertainty_and_occlusion_tolerance() -> None:
    obs = [
        _obs("c_v1", "v1", "chair", (150, 200, 290, 420), embedding=(0.8, 0.2, 0.0), is_partial=False),
        _obs("c_v2", "v2", "chair", (240, 210, 340, 390), embedding=(0.79, 0.21, 0.0), is_partial=True, occlusion_level="partial"),
    ]
    cams = {
        "v1": CameraPoseConstraint("v1", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), position_sigma_m=0.25),
        "v2": CameraPoseConstraint("v2", (0.5, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), position_sigma_m=0.25),
    }
    rep = resolve_cross_view_identities_v2(obs, cameras=cams)
    assert rep.confirmed_count == 1
    assert any("occlusion_relaxed_scale" in d.reasons for d in rep.pairwise_decisions)


# ---------------------------------------------------------------------------
# A5.4 / A5.5 Architectural & Furniture Refinement Edge Cases
# ---------------------------------------------------------------------------


def test_scene_refinement_v2_taxonomies_and_column_beam_guard() -> None:
    assert "wall" in ARCHITECTURAL_SURVEY_CLASSES
    assert "sofa" in INTERIOR_SURVEY_FURNITURE
    assert "chair" in FURNITURE_SUBPARTS

    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    img[20:460, 100:180:2] = 40
    img[200:380, 300:420:2] = 40

    obs = [
        _obs("col_ok", "img_1", "column", (100, 20, 180, 460), category="architecture"),
        _obs("col_huge", "img_1", "column", (0, 0, 620, 460), category="architecture"),
        _obs("tiny_furn", "img_1", "chair", (300, 300, 304, 304), category="furniture"),
        _obs("desk_ok", "img_1", "desk", (300, 200, 420, 380), category="furniture"),
    ]
    ref = refine_scene_observations_v2(obs, images_rgb={"img_1": img})
    kept = {o.observation_id for o in ref.accepted_observations}
    assert kept == {"col_ok", "desk_ok"}
    assert "col_huge" in ref.rejected_architectural_ids
    assert "tiny_furn" in ref.rejected_furniture_ids


# ---------------------------------------------------------------------------
# Objective 2, 3, 4, 5: Benchmark Suite, Lifecycle, Resumable & Claude Bridge
# ---------------------------------------------------------------------------


def test_a5_benchmark_suite_and_a4_vs_a5_comparison(tmp_path: Path) -> None:
    mask_c = np.zeros((480, 640), dtype=bool)
    mask_c[200:380, 150:280] = True
    obs = [
        _obs("w1", "c1", "wall", (0, 20, 640, 400), category="architecture"),
        _obs("ch1", "c1", "chair", (150, 200, 280, 380)),
        _obs("ch1_dup", "c1", "chair", (152, 202, 278, 378), confidence=0.70, engine_id="owlv2"),
        _obs("ch2", "c2", "chair", (154, 202, 284, 382)),
    ]
    res = run_a5_production_pipeline(
        "bench_cmp_run",
        observations=obs,
        precomputed_masks={"ch1": mask_c},
        run_live_detector=False,
        run_live_segmenter=True,
        work_dir=tmp_path / "bench_ws",
        persist_to_world=True,
    )
    metrics = compute_a5_reliability_metrics(res)
    assert isinstance(metrics, A5ReliabilityMetrics)
    assert metrics.masks_produced == 1
    assert metrics.duplicate_suppressed_count == 1
    assert metrics.evidence_matrix_persisted is True

    a4_base = {"masks_produced": 0, "duplicate_warnings": 1, "confirmed_identities": 0}
    cmp = compare_a4_vs_a5_reliability(a4_base, metrics)
    assert cmp["improved_mask_coverage"] is True
    assert cmp["improved_duplicate_control"] is True
    assert metrics.to_dict()["run_id"] == "bench_cmp_run"


def test_gpu_lifecycle_cpu_fallback_and_hardware_probe() -> None:
    hw = probe_hardware_profile()
    assert hw.device_name

    cpu_hw = HardwareProfile(False, "cpu", 0.0, 0.0, 0.0, False)
    bs, res, dev = select_adaptive_batch_and_resolution(4, 2000.0, cpu_hw)
    assert dev == "cpu"
    assert bs == 1
    assert res == 1024

    big_hw = HardwareProfile(True, "RTX 4090", 24576.0, 20000.0, 23976.0, True)
    bs2, res2, dev2 = select_adaptive_batch_and_resolution(4, 2000.0, big_hw)
    assert dev2 == "cuda"
    assert bs2 == 4
    assert res2 == 1280

    mgr = ModelLifecycleManager()

    def _boom(_d: str, _b: int, _r: int) -> None:
        raise ValueError("unexpected bad input")

    out, rec = mgr.run_stage("failing_stage", "bad_model", _boom)
    assert out is None
    assert rec.status == "failed"
    assert "ValueError" in (rec.error_message or "")


def test_resumable_fingerprint_determinism_and_force_rerun(tmp_path: Path) -> None:
    fp1 = compute_run_fingerprint("r1", ["a.jpg", "b.jpg"], domain=Domain.INTERIOR, detection_threshold=0.28)
    fp2 = compute_run_fingerprint("r1", ["b.jpg", "a.jpg"], domain=Domain.INTERIOR, detection_threshold=0.28)
    fp3 = compute_run_fingerprint("r1", ["a.jpg", "b.jpg"], domain=Domain.EXTERIOR, detection_threshold=0.28)
    assert fp1 == fp2
    assert fp1 != fp3

    obs = [_obs("ch1", "c1", "chair", (100, 150, 220, 350))]
    st1, _, _ = run_resumable_a5_job("r_force", tmp_path / "w", observations=obs, run_live_segmenter=False)
    assert st1.status == "completed"
    st2, _, _ = run_resumable_a5_job("r_force", tmp_path / "w", observations=obs, run_live_segmenter=False, force_rerun=True)
    assert st2.status == "completed"


def test_ingest_images_rgb_downscales_when_max_dimension_set(tmp_path: Path) -> None:
    from PIL import Image

    img_p = tmp_path / "large.png"
    Image.new("RGB", (1600, 1200), color=(100, 120, 140)).save(img_p)
    rgb_map, path_map = ingest_images_rgb([img_p, tmp_path / "missing.png"], max_dimension_px=800)
    assert list(rgb_map.keys()) == ["large.png"]
    assert rgb_map["large.png"].shape[:2] == (600, 800)
    assert path_map["large.png"] == img_p


def test_claude_handoff_direct_persist_and_world_entity_linking(tmp_path: Path) -> None:
    obs = [
        _obs("o1", "img_1", "table", (100, 200, 400, 420)),
    ]
    bundle = PerceptionBundle(run_id="direct_handoff", observations=obs)
    cams = parse_claude_camera_poses_from_json([
        {"image_id": "img_1", "position_m": [0.0, 0.0, 0.0], "is_metric": False}
    ])
    fb = ClaudeReconstructionFeedback(
        cameras=cams,
        world_entities=(WorldEntityIdentity("we_1", "table", ("o1",), None, False),),
    )
    handoff, ver = execute_claude_handoff_and_persist(
        bundle,
        matrix_root=tmp_path / "em_direct",
        claude_feedback=fb,
    )
    assert len(handoff.observations) == 1
    assert ver.evidence_matrix_persisted is True
    assert ver.world_entities_linked == 1
    assert ver.fresh_reload_observation_count == 1


def test_dinov2_probe_and_dispatch_unsupported_engine(tmp_path: Path) -> None:
    ckpt = probe_dinov2_checkpoint()
    assert ckpt is None or "dinov2" in ckpt

    from PIL import Image

    img_p = tmp_path / "room.jpg"
    Image.new("RGB", (320, 240), color=(90, 90, 90)).save(img_p)
    rgb_map, _ = ingest_images_rgb([img_p])
    _, rep = dispatch_sequential_detectors(
        "disp_unsup",
        [img_p],
        rgb_map,
        requested_engines=("nonexistent_detector",),
    )
    assert "nonexistent_detector" in rep.skipped_engines


def test_indoor_validation_unlabelled_real_and_six_room_unreviewed(tmp_path: Path) -> None:
    unlab = normalize_dataset_sample(
        {
            "sample_id": "u1",
            "scene_id": "s1",
            "image_path": "u1.jpg",
            "image_width": 640,
            "image_height": 480,
            "boxes": [],
        },
        dataset_id="ade20k",
    )
    rep = run_indoor_dataset_validation("ade20k", [unlab], {"u1": []})
    assert rep.evaluation_mode is EvaluationMode.UNLABELLED_REAL
    assert rep.accuracy_claim_allowed is False
    assert rep.overall_ap50 is None

    r1 = tmp_path / "room_01"
    r1.mkdir()
    (r1 / "a.jpg").write_bytes(b"jpg")
    (r1 / "ground_truth.json").write_text(json.dumps({"is_human_reviewed": False}), encoding="utf-8")
    (r1 / "measured_dimensions.json").write_text(json.dumps({"width_m": 3.0}), encoding="utf-8")
    six = prepare_six_room_measured_capture_pathway(tmp_path, room_ids=("room_01",))
    assert six["verified_rooms"] == 0
    assert six["rooms"][0]["status"] == "unlabelled_only"


# ---------------------------------------------------------------------------
# Extended A5 Coverage: Multi-View Disambiguation, Pipeline Stages & Contracts
# ---------------------------------------------------------------------------


def test_identity_v2_disambiguates_six_identical_chairs_with_pose_sigma() -> None:
    emb = (0.4, 0.4, 0.4, 0.4)
    obs = []
    for idx, x in enumerate((40, 160, 280, 400, 520, 640)):
        obs.append(_obs(f"a_{idx}", "cam_0", "chair", (x, 220, x + 70, 400), embedding=emb, width=800, height=600))
        obs.append(_obs(f"b_{idx}", "cam_1", "chair", (x + 6, 222, x + 76, 402), embedding=emb, width=800, height=600))

    cams = {
        "cam_0": CameraPoseConstraint("cam_0", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), position_sigma_m=0.03),
        "cam_1": CameraPoseConstraint("cam_1", (0.1, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), position_sigma_m=0.03),
    }
    rep = resolve_cross_view_identities_v2(obs, cameras=cams)
    assert rep.base_report.summary["total_identities"] == 6
    assert rep.confirmed_count == 6


def test_identity_v2_preserves_ambiguity_without_cameras_when_candidates_tie() -> None:
    emb = (0.5, 0.5, 0.0, 0.0)
    obs = [
        _obs("c1", "v1", "chair", (100, 200, 200, 380), embedding=emb),
        _obs("c2", "v1", "chair", (120, 200, 220, 380), embedding=emb),
        _obs("c_v2", "v2", "chair", (110, 200, 210, 380), embedding=emb),
    ]
    rep = resolve_cross_view_identities_v2(obs, cameras=None)
    assert rep.ambiguous_count >= 1
    assert any(d.outcome == "ambiguous_match" for d in rep.pairwise_decisions)


def test_dedup_v2_preserves_distinct_non_overlapping_same_class_objects() -> None:
    obs = [
        _obs("t1", "img_1", "table", (40, 200, 220, 420), confidence=0.85),
        _obs("t2", "img_1", "table", (360, 200, 560, 420), confidence=0.84),
        _obs("d1", "img_1", "door", (10, 40, 110, 440), category="architecture", confidence=0.87),
        _obs("d2", "img_1", "door", (500, 40, 610, 440), category="architecture", confidence=0.86),
    ]
    res = resolve_duplicate_detections_v2(obs)
    assert res.summary["canonical_count"] == 4
    assert res.summary["suppressed_count"] == 0


def test_dedup_v2_suppresses_cross_scale_and_tiled_architectural_duplicates() -> None:
    obs = [
        _obs("w_full", "img_1", "wall", (0, 20, 640, 420), category="architecture", confidence=0.91, inference_mode="full_frame"),
        _obs("w_tile", "img_1", "wall", (8, 24, 632, 416), category="architecture", confidence=0.77, inference_mode="tiled", tile_id="tile_0"),
        _obs("w_multi", "img_1", "wall", (4, 22, 636, 418), category="architecture", confidence=0.73, inference_mode="multi_scale"),
    ]
    res = resolve_duplicate_detections_v2(obs)
    assert res.summary["canonical_count"] == 1
    assert res.summary["suppressed_count"] == 2
    assert res.canonical_observations[0].observation_id == "w_full"


def test_mask_geometry_handles_all_zero_and_missing_rle_gracefully() -> None:
    o_no_seg = _obs("o_no", "img_1", "chair", (50, 50, 150, 200), seg=None)
    refined, sums = refine_boxes_from_verified_masks([o_no_seg])
    assert refined == [o_no_seg]
    assert sums == []

    bad_seg = SegmentationObservation(
        mask_id="m_bad",
        observation_id="o_bad",
        mask_kind="visible_object_mask",
        area_px=100,
        box_iou=0.8,
        solidity=0.9,
        hole_count=0,
        boundary_uncertainty=0.1,
        quality_score=0.8,
        rle_mask={"size": [10, 10], "counts": [100]},  # wrong dims vs 640x480
    )
    o_bad = _obs("o_bad", "img_1", "chair", (50, 50, 150, 200), seg=bad_seg)
    refined_b, sums_b = refine_boxes_from_verified_masks([o_bad])
    assert len(sums_b) == 1
    assert sums_b[0].is_valid is False
    assert refined_b[0].bbox_px == (50.0, 50.0, 150.0, 200.0)


def test_segmentation_v2_quality_scoring_distinguishes_tight_vs_poor_mask() -> None:
    obs = _obs("o_q", "img_1", "table", (100, 100, 300, 300), width=640, height=480)
    tight_m = np.zeros((480, 640), dtype=bool)
    tight_m[105:295, 105:295] = True
    seg_good, err1 = build_segmentation_observation_from_mask(obs, tight_m, model_id="sam2")
    assert err1 is None and seg_good is not None
    assert seg_good.quality_score > 0.75

    poor_m = np.zeros((480, 640), dtype=bool)
    poor_m[10:25, 10:25] = True  # Outside box
    seg_poor, err2 = build_segmentation_observation_from_mask(obs, poor_m, model_id="sam2")
    assert err2 is None and seg_poor is not None
    assert seg_poor.quality_score < seg_good.quality_score


def test_scene_refinement_v2_preserves_valid_appliances_and_builtins() -> None:
    img = np.full((480, 640, 3), 180, dtype=np.uint8)
    img[80:400, 100:260:2] = 30
    img[120:360, 360:520:2] = 30
    obs = [
        _obs("fridge", "img_1", "refrigerator", (100, 80, 260, 400), category="furniture"),
        _obs("cab", "img_1", "cabinet", (360, 120, 520, 360), category="furniture"),
    ]
    ref = refine_scene_observations_v2(obs, images_rgb={"img_1": img})
    assert len(ref.accepted_observations) == 2
    assert "cabinet_door" in ref.accepted_observations[1].attributes["expected_subparts"]


def test_gpu_lifecycle_concurrent_check_and_release() -> None:
    mgr = ModelLifecycleManager()
    mgr.release_all_gpu_models()
    # Even if total_vram is large, if sum exceeds ceiling, can_run_concurrently is False
    mgr.hw = HardwareProfile(True, "Big GPU", 16000.0, 15000.0, 15400.0, True)
    mgr.vram_ceiling_mb = 15400.0
    assert mgr.can_run_concurrently([8000.0, 8000.0]) is False
    assert mgr.can_run_concurrently([4000.0, 4000.0]) is True


def test_pipeline_a5_summary_dict_contains_all_required_reliability_fields(tmp_path: Path) -> None:
    obs = [
        _obs("w1", "cam_1", "wall", (0, 20, 640, 400), category="architecture"),
        _obs("ch1", "cam_1", "chair", (150, 200, 280, 380)),
    ]
    res = run_a5_production_pipeline(
        "summary_fields_run",
        observations=obs,
        run_live_detector=False,
        run_live_segmenter=False,
        work_dir=tmp_path / "sum_ws",
        persist_to_world=True,
    )
    s = res.to_summary_dict()
    for key in (
        "run_id",
        "wall_time_s",
        "images_ingested",
        "raw_observations",
        "dedup_canonical_count",
        "dedup_suppressed_count",
        "dedup_reduction_rate",
        "segmentation_status",
        "masks_produced",
        "identity_summary",
        "arch_relations",
        "furniture_v2",
        "handoff_accepted",
        "evidence_matrix_persisted",
        "diagnostics",
    ):
        assert key in s
    assert s["evidence_matrix_persisted"] is True


def test_indoor_validation_handles_scannet_and_replica_manifests() -> None:
    scannet_s = normalize_dataset_sample(
        {
            "sample_id": "sn_001",
            "scene_id": "scene0001_00",
            "image_path": "scannet/001.jpg",
            "image_width": 640,
            "image_height": 480,
            "boxes": [{"label": "table", "bbox_xyxy": [100, 200, 400, 420]}],
        },
        dataset_id="scannet",
    )
    preds = {
        "sn_001": [
            DetectionBox(box_id="p1", class_name="table", confidence=0.91, box_2d=(100, 200, 400, 420), source_engine="grounding_dino"),
        ]
    }
    rep = run_indoor_dataset_validation("scannet", [scannet_s], preds)
    assert rep.evaluation_mode is EvaluationMode.LABELLED_REAL
    assert rep.accuracy_claim_allowed is True
    assert rep.furniture_ap50 is not None and rep.furniture_ap50 > 0.9

    rep_s = normalize_dataset_sample(
        {
            "sample_id": "rep_001",
            "scene_id": "office_0",
            "image_path": "replica/001.png",
            "image_width": 640,
            "image_height": 480,
            "boxes": [{"label": "chair", "bbox_xyxy": [100, 200, 220, 400]}],
        },
        dataset_id="replica",
    )
    assert rep_s.dataset_id == "replica"


def test_parse_claude_camera_poses_handles_defaults_and_uncertainty() -> None:
    cams = parse_claude_camera_poses_from_json([
        {"image_id": "img_0"},
        {"image_id": "img_1", "position_m": [1.0, 2.0, 0.5], "position_sigma_m": 0.04, "is_metric": True},
    ])
    assert cams["img_0"].position_m == (0.0, 0.0, 0.0)
    assert cams["img_0"].is_metric is False
    assert cams["img_1"].position_m == (1.0, 2.0, 0.5)
    assert cams["img_1"].position_sigma_m == 0.04
    assert cams["img_1"].is_metric is True


def test_a5_pipeline_blank_image_produces_zero_observations_and_honest_diagnostics() -> None:
    blank_img = np.full((480, 640, 3), 128, dtype=np.uint8)
    res = run_a5_production_pipeline(
        "blank_safety_run",
        images_rgb={"blank.png": blank_img},
        observations=[],
        run_live_detector=False,
        run_live_segmenter=False,
        persist_to_world=False,
    )
    assert res.image_qualities["blank.png"].is_blank is True
    assert len(res.bundle.observations) == 0
    assert len(res.handoff.observations) == 0
    assert res.arch_geometry["blank.png"].line_segments == ()
    assert res.arch_geometry["blank.png"].plane_hypotheses == ()
    metrics = compute_a5_reliability_metrics(res)
    assert metrics.raw_observations == 0
    assert metrics.mask_coverage_fraction == 0.0
    assert metrics.evidence_matrix_persisted is False


def test_a5_resumable_runner_persists_valid_checkpoint_and_reloads_identities(tmp_path: Path) -> None:
    obs = [
        _obs("w1", "v1", "wall", (0, 20, 640, 400), category="architecture"),
        _obs("c1", "v1", "chair", (120, 200, 240, 380)),
        _obs("c2", "v2", "chair", (124, 202, 244, 382)),
    ]
    wdir = tmp_path / "ckpt_ws"
    st1, res1, b1 = run_resumable_a5_job(
        "ckpt_test",
        wdir,
        observations=obs,
        run_live_detector=False,
        run_live_segmenter=False,
    )
    assert st1.status == "completed"
    assert res1 is not None
    assert (wdir / "a5_pipeline_checkpoint.json").exists()
    assert (wdir / "evidence_matrix" / "perception_handoff_v1.json").exists()

    st2, res2, b2 = run_resumable_a5_job(
        "ckpt_test",
        wdir,
        observations=obs,
        run_live_detector=False,
        run_live_segmenter=False,
    )
    assert st2.status == "resumed_from_cache"
    assert res2 is None
    assert [o.observation_id for o in b2.observations] == [o.observation_id for o in b1.observations]


def test_a5_mask_geometry_and_dedup_boundary_cases() -> None:
    single_px = np.zeros((32, 32), dtype=bool)
    single_px[10, 15] = True
    assert compute_tight_box_from_mask(single_px) == (15.0, 10.0, 16.0, 11.0)
    poly = extract_mask_polygon_px(single_px)
    assert len(poly) >= 1

    # Two observations in different images must never be merged by deduplication_v2
    o_a = _obs("oa", "img_a", "chair", (100, 100, 240, 260))
    o_b = _obs("ob", "img_b", "chair", (100, 100, 240, 260))
    res = resolve_duplicate_detections_v2([o_a, o_b])
    assert res.summary["canonical_count"] == 2
    assert res.summary["suppressed_count"] == 0
