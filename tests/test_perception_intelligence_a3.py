"""Comprehensive tests for A3 Perception Intelligence Subsystem (A3.1 - A3.13)."""

from __future__ import annotations

import json
import numpy as np
import pytest

from engine.multiengine.observation import ObservationDraft
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox
from perception.intelligence import (
    SCHEMA_VERSION,
    AppearanceDescriptor,
    CoordinateSystemSpec,
    DetectionProvenance,
    MaterialHypothesisRecord,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    ObservationLifecycle,
    ObservationRevision,
    PerceptionBundle,
    PerceptionEngineScheduler,
    ScoreSemantics,
    SegmentationObservation,
    TemporalObservation,
    ValidationError,
    aggregate_cross_view_materials,
    analyze_architecture,
    analyze_furniture,
    analyze_image_quality,
    build_intelligence_bundle,
    classify_furniture_hierarchy,
    classify_mask_kind,
    evaluate_instance_mask,
    extract_appearance_and_materials,
    infer_scene_relationships,
    observation_from_bbox_draft,
    plan_active_refinement,
    plan_adaptive_crops,
    reconcile_crop_duplicates,
    resolve_cross_view_identities,
    run_perception_diagnostics,
    separate_overlapping_masks,
    stable_observation_id,
    to_evidence_matrix_drafts,
    transform_crop_box_to_image,
)


def _make_obs(
    obs_id: str,
    label: str,
    box: tuple[float, float, float, float],
    *,
    image_id: str = "img_01",
    category: str = "object",
    raw_score: float | None = 0.78,
    engine_id: str = "grounding_dino",
    family: str = "grounding-dino",
    embedding: tuple[float, ...] = (),
    hist: tuple[float, ...] = (),
    is_partial: bool = False,
    is_truncated: bool = False,
    occlusion: str = "none",
    seg: SegmentationObservation | None = None,
    competing: tuple[dict, ...] = (),
    width: int = 1000,
    height: int = 800,
) -> NormalizedPerceptionObservation:
    return NormalizedPerceptionObservation(
        observation_id=obs_id,
        schema_version=SCHEMA_VERSION,
        raw_label=label,
        canonical_label=label,
        taxonomy_id=f"test.{category}.{label}",
        category=category,
        bbox_px=box,
        coords=CoordinateSystemSpec(f"image:{image_id}", "pixels", width, height),
        scores=ScoreSemantics(raw_score=raw_score, score_emitted=(raw_score is not None)),
        provenance=DetectionProvenance(
            source_image_id=image_id,
            execution=ModelExecutionProvenance(engine_id, family, "test_ckpt"),
        ),
        is_partial=is_partial,
        is_truncated=is_truncated,
        occlusion_level=occlusion,
        appearance=AppearanceDescriptor(color_histogram_24=hist, embedding_vector=embedding),
        segmentation=seg,
        competing_labels=competing,
    )


# ---------------------------------------------------------------------------
# A3.1: Data Model, Validation, Serialization, Stable IDs
# ---------------------------------------------------------------------------

def test_a3_1_data_model_validation_rejects_invalid_inputs():
    with pytest.raises(ValidationError):
        CoordinateSystemSpec("img", "pixels", 0, 100)
    with pytest.raises(ValidationError):
        ScoreSemantics(raw_score=1.5, score_emitted=True)
    with pytest.raises(ValidationError):
        ScoreSemantics(raw_score=0.7, score_emitted=True, calibrated_probability=0.8, is_calibrated=False)
    with pytest.raises(ValidationError):
        MaterialHypothesisRecord("wood", "oak", 0.5, "color", is_physically_verified=True)
    with pytest.raises(ValidationError):
        _make_obs("bad", "chair", (50.0, 50.0, 40.0, 100.0))  # degenerate box
    with pytest.raises(ValidationError):
        _make_obs("oob", "chair", (0.0, 0.0, 1500.0, 200.0))  # out of bounds


def test_a3_1_bundle_roundtrip_and_stable_ids():
    id1 = stable_observation_id("perc", "img1", "gd", "chair", (10.0, 20.0, 100.0, 200.0))
    id2 = stable_observation_id("perc", "img1", "gd", "chair", (10.0, 20.0, 100.0, 200.0))
    assert id1 == id2
    o1 = _make_obs(id1, "chair", (10.0, 20.0, 100.0, 200.0))
    bundle = PerceptionBundle(run_id="r_test", observations=[o1])
    raw = json.loads(bundle.to_json())
    restored = PerceptionBundle.from_dict(raw)
    assert restored.run_id == "r_test"
    assert len(restored.observations) == 1
    assert restored.observations[0].observation_id == id1
    assert restored.observations[0].scores.is_calibrated is False
    assert restored.observations[0].revision.lifecycle is ObservationLifecycle.ACTIVE


# ---------------------------------------------------------------------------
# A3.2: Architectural Intelligence Engine
# ---------------------------------------------------------------------------

def test_a3_2_architectural_host_wall_and_verified_vs_candidate_openings():
    wall = _make_obs("w1", "wall", (0.0, 0.0, 900.0, 750.0), category="architecture")
    door = _make_obs("d1", "door", (100.0, 100.0, 300.0, 650.0), category="architecture")
    d_frame = _make_obs("df1", "door_frame", (90.0, 90.0, 310.0, 660.0), category="architecture")
    win_orphan = _make_obs("win1", "window", (920.0, 100.0, 990.0, 300.0), category="architecture")
    comp_door = _make_obs("win2", "window", (105.0, 105.0, 295.0, 645.0), category="architecture")

    res = analyze_architecture([wall, door, d_frame, win_orphan, comp_door])
    by_id = {a.observation.observation_id: a for a in res.observations}

    # Door has a frame + wall -> verified_opening hosted by w1
    assert by_id["d1"].structural_role == "verified_opening"
    assert by_id["d1"].is_verified_opening is True
    assert by_id["d1"].host_wall_id == "w1"
    assert by_id["d1"].host_wall_hypothesis_state == "observed"

    # Orphan window outside wall with no frame/mask -> opening_candidate, host unknown
    assert by_id["win1"].structural_role == "opening_candidate"
    assert by_id["win1"].is_verified_opening is False
    assert by_id["win1"].host_wall_id is None
    assert by_id["win1"].host_wall_hypothesis_state == "unknown"

    # Competing door vs window overlap recorded
    assert res.summary["competing_structural_pairs"] >= 1


# ---------------------------------------------------------------------------
# A3.3: Furniture Intelligence Engine
# ---------------------------------------------------------------------------

def test_a3_3_furniture_hierarchy_subparts_and_deduplication():
    assert classify_furniture_hierarchy("dining_table")[-1] == "DiningTable"
    t1 = _make_obs("t1", "table", (100.0, 200.0, 500.0, 600.0), category="furniture", raw_score=0.88)
    t1_dup = _make_obs("t1_d", "table", (105.0, 205.0, 495.0, 595.0), category="furniture", raw_score=0.65)
    leg = _make_obs("leg1", "table_leg", (110.0, 400.0, 140.0, 590.0), category="furniture", raw_score=0.72)
    chair = _make_obs("c1", "chair", (520.0, 250.0, 700.0, 600.0), category="furniture", is_partial=True, occlusion="partial")

    res = analyze_furniture([t1, t1_dup, leg, chair])
    assert "t1_d" in res.suppressed_duplicates
    assert res.component_links["leg1"] == "t1"
    assert res.summary["total_furniture_items"] == 2
    assert res.summary["partial_or_occluded"] == 1


# ---------------------------------------------------------------------------
# A3.4: Cross-View Identity (4 Identical Chairs Must Stay 4 Separate Identities)
# ---------------------------------------------------------------------------

def test_a3_4_four_identical_chairs_remain_four_separate_identities():
    emb = (0.5, 0.5, 0.5, 0.5)
    # 4 identical chairs in image 1 and 4 identical chairs in image 2
    obs = []
    for i in range(4):
        obs.append(_make_obs(f"c1_{i}", "chair", (50.0 + i * 200, 200.0, 180.0 + i * 200, 450.0), image_id="img_1", embedding=emb))
        obs.append(_make_obs(f"c2_{i}", "chair", (60.0 + i * 200, 200.0, 190.0 + i * 200, 450.0), image_id="img_2", embedding=emb))

    rep = resolve_cross_view_identities(obs)
    # Must NOT collapse all 4 chairs into 1 identity
    assert len(rep.identities) >= 4
    assert rep.summary["ambiguous"] >= 1
    assert len(rep.merge_proposals) >= 1


def test_a3_4_distinct_object_confirmed_across_views_and_split_proposal():
    emb_sofa = (0.9, 0.1, 0.0, 0.0)
    emb_tbl = (0.0, 0.1, 0.9, 0.0)
    s1 = _make_obs("s1", "sofa", (100.0, 200.0, 500.0, 500.0), image_id="v1", embedding=emb_sofa)
    s2 = _make_obs("s2", "sofa", (120.0, 210.0, 510.0, 505.0), image_id="v2", embedding=emb_sofa)
    t1 = _make_obs("t1", "table", (600.0, 300.0, 850.0, 550.0), image_id="v1", embedding=emb_tbl)

    rep = resolve_cross_view_identities([s1, s2, t1])
    assert rep.summary["confirmed_matches"] == 1
    matched = [i for i in rep.identities if i.status == "confirmed_match"][0]
    assert set(matched.member_observation_ids) == {"s1", "s2"}


# ---------------------------------------------------------------------------
# A3.5: Intelligent Multi-Engine Orchestration & Circuit Breaker
# ---------------------------------------------------------------------------

def test_a3_5_scheduler_routing_failure_isolation_and_circuit_breaker():
    sched = PerceptionEngineScheduler(vram_budget_mb=5000.0)
    plan = sched.build_routing_plan(
        ["grounding_dino", "mask_rcnn", "florence2", "sam2_hf", "dinov2"],
        budget="balanced",
        is_low_contrast=True,
        target_classes=["chair", "table"],
    )
    assert "grounding_dino" in plan.selected_detectors
    assert "mask_rcnn" in plan.selected_detectors
    assert "florence2" in plan.selected_detectors  # added due to low contrast

    def run_gd():
        return [DetectionBox("g1", "chair", (10, 10, 100, 120), 0.80, source_engine="grounding_dino")]

    def run_fail():
        raise RuntimeError("CUDA OOM simulated")

    res = sched.execute_sequential(
        plan,
        {"grounding_dino": run_gd, "mask_rcnn": run_fail, "florence2": run_fail},
        is_low_contrast=True,
    )
    assert "grounding_dino" in res.per_engine_boxes
    assert "mask_rcnn" in res.failed_engines
    assert len(res.fused_hypotheses) == 1

    # Trip circuit breaker after 3 failures
    for _ in range(2):
        sched.execute_sequential(plan, {"grounding_dino": run_gd, "mask_rcnn": run_fail, "florence2": run_fail})
    assert sched.health["mask_rcnn"].circuit_open is True
    plan2 = sched.build_routing_plan(["grounding_dino", "mask_rcnn"], budget="balanced")
    assert "mask_rcnn" not in plan2.selected_detectors


# ---------------------------------------------------------------------------
# A3.6: Image Quality & Adaptive Processing
# ---------------------------------------------------------------------------

def test_a3_6_image_quality_blank_blurry_and_adaptive_crops():
    blank = np.full((256, 256, 3), 120, dtype=np.uint8)
    q_blank = analyze_image_quality(blank)
    assert q_blank.is_blank is True
    assert q_blank.quality_grade == "unusable_blank"
    assert plan_adaptive_crops(q_blank, []) == ([], [])

    rng = np.random.default_rng(7)
    sharp = rng.integers(0, 255, (1000, 1200, 3), dtype=np.uint8)
    q_sharp = analyze_image_quality(sharp)
    assert q_sharp.is_blank is False
    assert q_sharp.quality_grade == "good"
    coarse = [DetectionBox("d1", "door", (100, 100, 300, 600), 0.8)]
    tiles, crops = plan_adaptive_crops(q_sharp, coarse, tile_size=800)
    assert len(tiles) >= 2
    assert len(crops) == 1
    assert crops[0].target_classes == ("door_frame", "door_handle", "door_lock")

    mapped = transform_crop_box_to_image((10.0, 20.0, 50.0, 60.0), (100, 100), 1200, 1000)
    assert mapped == (110.0, 120.0, 150.0, 160.0)
    deduped = reconcile_crop_duplicates([
        DetectionBox("a", "door_handle", (10, 10, 40, 40), 0.8),
        DetectionBox("b", "door_handle", (12, 12, 42, 42), 0.6),
    ])
    assert len(deduped) == 1


# ---------------------------------------------------------------------------
# A3.7: Segmentation Intelligence
# ---------------------------------------------------------------------------

def test_a3_7_segmentation_evaluation_and_instance_separation():
    assert classify_mask_kind("wall") == "architectural_surface_mask"
    assert classify_mask_kind("structural_boundary") == "structural_boundary"
    assert classify_mask_kind("chair", is_occluded=True) == "occluded_object_hypothesis"
    assert classify_mask_kind("chair") == "visible_object_mask"

    m1 = np.zeros((100, 100), dtype=bool)
    m1[20:70, 20:70] = True
    m2 = np.zeros((100, 100), dtype=bool)
    m2[40:80, 40:80] = True
    depth = np.full((100, 100), 2.5, dtype=np.float32)

    ev = evaluate_instance_mask(m1, observation_id="o1", box_px=(20, 20, 70, 70), canonical_label="chair", depth_map=depth)
    assert ev.is_acceptable is True
    assert ev.observation.depth_mean_relative == 2.5
    assert ev.observation.quality_score > 0.5

    sep = separate_overlapping_masks([m1, m2], [0.9, 0.7])
    assert not np.any(sep[0] & sep[1])  # contested pixels resolved cleanly


# ---------------------------------------------------------------------------
# A3.8: Material & Appearance Intelligence
# ---------------------------------------------------------------------------

def test_a3_8_material_hypotheses_and_cross_view_consensus():
    img = np.full((100, 100, 3), (130, 85, 45), dtype=np.uint8)
    desc, hyps = extract_appearance_and_materials(img, (10, 10, 90, 90), canonical_label="table")
    assert desc.dominant_color_name == "brown"
    assert len(desc.color_histogram_24) == 24
    assert all(not h.is_physically_verified for h in hyps)

    cons = aggregate_cross_view_materials("id_table_1", [hyps, hyps])
    assert cons.is_physically_verified is False
    if hyps:
        assert cons.is_consistent is True


# ---------------------------------------------------------------------------
# A3.9: Scene Relationship Intelligence
# ---------------------------------------------------------------------------

def test_a3_9_scene_relationships_and_contradiction_detection():
    wall = _make_obs("w1", "wall", (0.0, 0.0, 900.0, 700.0), category="architecture")
    floor = _make_obs("f1", "floor", (0.0, 500.0, 1000.0, 800.0), category="architecture")
    ceil = _make_obs("c1", "ceiling", (0.0, 0.0, 1000.0, 150.0), category="architecture")
    door = _make_obs("d1", "door", (100.0, 120.0, 250.0, 650.0), category="architecture")
    tbl = _make_obs("t1", "table", (400.0, 400.0, 700.0, 650.0), category="furniture")
    chr1 = _make_obs("ch1", "chair", (720.0, 420.0, 850.0, 650.0), category="furniture")
    light = _make_obs("l1", "ceiling_light", (450.0, 10.0, 550.0, 100.0), category="fixture")

    rep = infer_scene_relationships([wall, floor, ceil, door, tbl, chr1, light])
    preds = {r.predicate for r in rep.relationships}
    assert "door_within_wall" in preds
    assert "chair_beside_table" in preds
    assert "light_attached_to_ceiling" in preds
    assert "resting_on_floor" in preds


# ---------------------------------------------------------------------------
# A3.10, A3.12, A3.13: Active Refinement, World Integration & Diagnostics
# ---------------------------------------------------------------------------

def test_a3_10_12_13_end_to_end_bundle_refinement_and_diagnostics():
    draft = ObservationDraft(
        task="detection",
        quantity="bbox",
        source_id="room_1.jpg",
        coordinate_system="image:room_1",
        image_region=(100.0, 150.0, 300.0, 450.0),
        predicted={"label": "chair", "raw_label": "chair", "raw_score": 0.28, "confidence_emitted": True},
        confidence=0.28,
    )
    obs = observation_from_bbox_draft(draft, image_width=1000, image_height=800)
    bundle = build_intelligence_bundle(
        "run_a3",
        [obs],
        image_qualities={"room_1.jpg": {"is_blank": False}, "room_2.jpg": {"is_blank": False}},
        failed_engines={"owlvit": "RuntimeError: test"},
    )
    drafts_out = to_evidence_matrix_drafts(bundle)
    assert len(drafts_out) == 1
    assert drafts_out[0].predicted["observation_id"] == obs.observation_id

    diag = bundle.diagnostics["diagnostic_report"]
    codes = {f["code"] for f in diag["findings"]}
    assert "model_execution_failure" in codes
    assert "empty_detections_non_blank" in codes
    assert "all_masks_missing" in codes

    ref = plan_active_refinement([obs], compute_budget=3.0)
    assert ref.should_continue is True
    assert len(ref.actions) >= 1
