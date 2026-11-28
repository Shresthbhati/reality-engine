"""Scenario, stress, and invariance tests for A3 Perception Intelligence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.perception.evaluator import EvaluationMode, run_perception_benchmark
from benchmarks.perception.indoor_dataset_adapter import GroundTruthBox, evaluate_detections_against_ground_truth
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox
from perception.intelligence import (
    SCHEMA_VERSION,
    AppearanceDescriptor,
    CoordinateSystemSpec,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    ObservationLifecycle,
    ObservationRevision,
    PerceptionBundle,
    PerceptionEngineScheduler,
    ScoreSemantics,
    SegmentationObservation,
    aggregate_cross_view_materials,
    analyze_architecture,
    analyze_furniture,
    analyze_image_quality,
    build_intelligence_bundle,
    classify_furniture_hierarchy,
    evaluate_instance_mask,
    extract_appearance_and_materials,
    infer_scene_relationships,
    plan_active_refinement,
    plan_adaptive_crops,
    reconcile_crop_duplicates,
    resolve_cross_view_identities,
    run_perception_diagnostics,
    separate_overlapping_masks,
)


def _obs(
    oid: str,
    label: str,
    box: tuple[float, float, float, float],
    *,
    img: str = "s1.jpg",
    cat: str = "furniture",
    score: float | None = 0.80,
    eng: str = "grounding_dino",
    fam: str = "grounding-dino",
    emb: tuple[float, ...] = (),
    hist: tuple[float, ...] = (),
    partial: bool = False,
    occ: str = "none",
    seg: SegmentationObservation | None = None,
    comp: tuple[dict, ...] = (),
    w: int = 1280,
    h: int = 960,
) -> NormalizedPerceptionObservation:
    return NormalizedPerceptionObservation(
        observation_id=oid,
        schema_version=SCHEMA_VERSION,
        raw_label=label,
        canonical_label=label,
        taxonomy_id=f"interior.{cat}.{label}",
        category=cat,
        bbox_px=box,
        coords=CoordinateSystemSpec(f"image:{img}", "pixels", w, h),
        scores=ScoreSemantics(raw_score=score, score_emitted=(score is not None)),
        provenance=DetectionProvenance(
            source_image_id=img,
            execution=ModelExecutionProvenance(eng, fam, "tiny"),
        ),
        is_partial=partial,
        occlusion_level=occ,
        appearance=AppearanceDescriptor(color_histogram_24=hist, embedding_vector=emb),
        segmentation=seg,
        competing_labels=comp,
    )


def test_scenario_dining_room_eight_chairs_and_table():
    """Eight visually similar dining chairs around a table must remain distinct identities."""
    emb_c = (0.4, 0.4, 0.4, 0.4)
    emb_t = (0.1, 0.9, 0.1, 0.0)
    obs = [
        _obs("v1_wall", "wall", (0, 20, 1280, 760), img="v1.jpg", cat="architecture"),
        _obs("v1_floor", "floor", (0, 650, 1280, 960), img="v1.jpg", cat="architecture"),
        _obs("v1_tbl", "dining_table", (300, 420, 950, 740), img="v1.jpg", emb=emb_t),
    ]
    for i in range(4):
        obs.append(_obs(f"v1_c{i}", "chair", (220 + i * 190, 440, 340 + i * 190, 740), img="v1.jpg", emb=emb_c))
        obs.append(_obs(f"v2_c{i}", "chair", (230 + i * 190, 445, 350 + i * 190, 745), img="v2.jpg", emb=emb_c))
    obs.append(_obs("v2_tbl", "dining_table", (310, 425, 960, 745), img="v2.jpg", emb=emb_t))

    bundle = build_intelligence_bundle("dining_8", obs)
    assert bundle.diagnostics["furniture_summary"]["total_furniture_items"] >= 9
    assert len(bundle.identities) >= 5
    # Table uniquely matches across views
    tbl_ids = [i for i in bundle.identities if i.canonical_label == "dining_table"]
    assert len(tbl_ids) == 1
    assert tbl_ids[0].status == "confirmed_match"


def test_scenario_kitchen_and_bathroom_fixtures_classification():
    kitchen = [
        _obs("k1", "kitchen_furniture", (50, 350, 500, 820), cat="furniture"),
        _obs("k2", "refrigerator", (520, 120, 760, 840), cat="appliance"),
        _obs("k3", "oven", (780, 360, 980, 820), cat="appliance"),
        _obs("k4", "sink", (200, 320, 360, 450), cat="fixture"),
    ]
    bath = [
        _obs("b1", "toilet", (100, 420, 300, 800), img="bath.jpg", cat="fixture"),
        _obs("b2", "bathtub", (400, 480, 1100, 880), img="bath.jpg", cat="fixture"),
        _obs("b3", "bathroom_fixture", (320, 350, 550, 750), img="bath.jpg", cat="fixture"),
    ]
    res_k = analyze_furniture(kitchen)
    res_b = analyze_furniture(bath)
    assert res_k.summary["total_furniture_items"] == 4
    assert res_b.summary["total_furniture_items"] == 3
    assert classify_furniture_hierarchy("refrigerator")[0] == "Appliances"
    assert classify_furniture_hierarchy("toilet")[0] == "Fixtures"


def test_scenario_partially_occluded_object_cross_view_association():
    emb = (0.85, 0.15, 0.0, 0.0)
    full_sofa = _obs("s_full", "sofa", (100, 350, 700, 750), img="v1.jpg", emb=emb)
    part_sofa = _obs("s_part", "sofa", (50, 360, 320, 740), img="v2.jpg", emb=emb, partial=True, occ="partial")
    rep = resolve_cross_view_identities([full_sofa, part_sofa], match_threshold=0.55)
    assert rep.summary["confirmed_matches"] == 1


def test_scenario_multiple_openings_mixed_host_walls():
    w_left = _obs("wl", "wall", (0, 50, 500, 800), img="hall.jpg", cat="architecture")
    d1 = _obs("d1", "door", (120, 180, 280, 760), img="hall.jpg", cat="architecture")
    d1f = _obs("d1f", "door_frame", (110, 170, 290, 770), img="hall.jpg", cat="architecture")
    w_orphan = _obs("wo", "window", (800, 150, 1050, 450), img="hall.jpg", cat="architecture")

    arch = analyze_architecture([w_left, d1, d1f, w_orphan])
    by_id = {a.observation.observation_id: a for a in arch.observations}
    assert by_id["d1"].host_wall_id == "wl"
    assert by_id["d1"].is_verified_opening is True
    assert by_id["wo"].host_wall_id is None
    assert by_id["wo"].is_verified_opening is False


def test_scenario_overlapping_three_masks_and_depth_stats():
    m1 = np.zeros((100, 100), dtype=bool)
    m1[10:60, 10:60] = True
    m2 = np.zeros((100, 100), dtype=bool)
    m2[30:80, 30:80] = True
    m3 = np.zeros((100, 100), dtype=bool)
    m3[50:95, 50:95] = True

    sep = separate_overlapping_masks([m1, m2, m3], [0.92, 0.78, 0.65])
    assert not np.any(sep[0] & sep[1])
    assert not np.any(sep[1] & sep[2])
    assert not np.any(sep[0] & sep[2])

    depth = np.linspace(1.0, 5.0, 10000, dtype=np.float32).reshape(100, 100)
    ev = evaluate_instance_mask(sep[0], observation_id="m1", box_px=(10, 10, 60, 60), canonical_label="chair", depth_map=depth)
    assert ev.observation.depth_mean_relative is not None
    assert 1.0 <= ev.observation.depth_mean_relative <= 5.0


def test_scenario_material_consistency_across_three_views():
    img_brown = np.full((80, 80, 3), (125, 80, 45), dtype=np.uint8)
    _, h1 = extract_appearance_and_materials(img_brown, (10, 10, 70, 70), canonical_label="table")
    _, h2 = extract_appearance_and_materials(img_brown, (10, 10, 70, 70), canonical_label="table")
    img_gray = np.full((80, 80, 3), (130, 130, 130), dtype=np.uint8)
    _, h3 = extract_appearance_and_materials(img_gray, (10, 10, 70, 70), canonical_label="table")
    cons = aggregate_cross_view_materials("id_t", [h1, h2, h3])
    assert cons.is_physically_verified is False
    assert cons.consensus_material == "wood"
    assert sum(cons.view_votes.values()) == 2


def test_scenario_reconcile_duplicates_across_scales_and_crops():
    boxes = [
        DetectionBox("full_1", "window", (100, 100, 300, 300), 0.82),
        DetectionBox("tile_1", "window", (104, 102, 298, 296), 0.71),
        DetectionBox("crop_1", "window_frame", (95, 95, 305, 305), 0.65),
    ]
    kept = reconcile_crop_duplicates(boxes, iou_threshold=0.55)
    assert len(kept) == 2  # same-class duplicate suppressed, different class window_frame kept


def test_scenario_evaluator_calibration_and_per_class_ap():
    gt = {
        "a.jpg": [
            GroundTruthBox("a.jpg", "chair", "chair", (10, 10, 100, 100)),
            GroundTruthBox("a.jpg", "door", "door", (200, 20, 320, 400)),
        ]
    }
    preds = {
        "a.jpg": [
            DetectionBox("p1", "chair", (12, 12, 98, 98), 0.92),
            DetectionBox("p2", "door", (202, 22, 318, 398), 0.88),
            DetectionBox("p3", "chair", (400, 400, 500, 500), 0.40),  # FP
        ]
    }
    m = evaluate_detections_against_ground_truth(preds, gt)
    assert m["total_tp"] == 2
    assert m["total_fp"] == 1
    assert m["total_fn"] == 0

    rep = run_perception_benchmark(
        mode=EvaluationMode.PHYSICALLY_MEASURED,
        dataset_name="measured_room",
        predictions_by_image=preds,
        gt_by_image=gt,
    )
    assert rep.accuracy_claim_allowed is True
    assert rep.ece is not None and rep.brier_score is not None
    assert rep.per_class["door"]["ap"] == 1.0


def test_scenario_active_refinement_prioritizes_conflicts_over_low_gain():
    o_conf = _obs(
        "o_conf",
        "cabinet",
        (100, 100, 300, 400),
        comp=({"engine_id": "mask_rcnn", "label": "wardrobe", "raw_score": 0.65},),
    )
    o_small = _obs("o_small", "socket", (10, 10, 40, 40), score=0.25)
    plan = plan_active_refinement([o_conf, o_small], compute_budget=5.0)
    assert plan.should_continue is True
    assert len(plan.actions) >= 2


def test_scenario_diagnostics_with_empty_bundle_and_unscored_detections():
    unscored = _obs("u1", "window", (50, 50, 200, 200), score=None, eng="florence2", fam="florence2")
    bundle = PerceptionBundle(run_id="r_fl", observations=[unscored])
    diag = run_perception_diagnostics(bundle)
    assert diag.status in ("healthy", "degraded")
    assert json.loads(bundle.to_json())["run_id"] == "r_fl"
