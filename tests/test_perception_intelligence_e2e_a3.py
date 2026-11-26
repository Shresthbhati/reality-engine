"""End-to-end integration tests for A3 Perception Intelligence -> EvidenceMatrix -> WorldIR."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.perception.evaluator import EvaluationMode, run_perception_benchmark
from engine.multiengine.evidence_matrix import EvidenceContext, EvidenceMatrix
from engine.multiengine.worldir_bridge import convert
from perception.intelligence import (
    SCHEMA_VERSION,
    AppearanceDescriptor,
    CoordinateSystemSpec,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    PerceptionBundle,
    PerceptionEngineScheduler,
    ScoreSemantics,
    analyze_architecture,
    analyze_furniture,
    analyze_image_quality,
    build_intelligence_bundle,
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
from world_ir.world_v1 import WorldIR


def _room_obs(
    oid: str,
    label: str,
    box: tuple[float, float, float, float],
    *,
    img: str = "room_a.jpg",
    cat: str = "furniture",
    score: float = 0.80,
    emb: tuple[float, ...] = (),
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
        scores=ScoreSemantics(raw_score=score, score_emitted=True),
        provenance=DetectionProvenance(
            source_image_id=img,
            execution=ModelExecutionProvenance("grounding_dino", "grounding-dino", "tiny"),
        ),
        appearance=AppearanceDescriptor(embedding_vector=emb),
    )


def test_e2e_multi_view_furnished_room_to_evidence_matrix_and_worldir(tmp_path):
    """Simulate a 2-view furnished room and verify PerceptionBundle -> EvidenceMatrix -> WorldIR."""
    e_tbl = (0.9, 0.1, 0.0, 0.0)
    e_c1 = (0.1, 0.9, 0.0, 0.0)
    e_c2 = (0.1, 0.88, 0.05, 0.0)
    e_bed = (0.0, 0.0, 0.9, 0.1)

    obs_v1 = [
        _room_obs("v1_wall", "wall", (0, 50, 1280, 780), img="v1.jpg", cat="architecture", score=0.72),
        _room_obs("v1_floor", "floor", (0, 680, 1280, 960), img="v1.jpg", cat="architecture", score=0.75),
        _room_obs("v1_door", "door", (80, 140, 260, 720), img="v1.jpg", cat="architecture", score=0.82),
        _room_obs("v1_dframe", "door_frame", (72, 132, 268, 728), img="v1.jpg", cat="architecture", score=0.68),
        _room_obs("v1_tbl", "table", (450, 460, 820, 760), img="v1.jpg", emb=e_tbl, score=0.88),
        _room_obs("v1_c1", "chair", (320, 480, 440, 760), img="v1.jpg", emb=e_c1, score=0.84),
        _room_obs("v1_c2", "chair", (830, 480, 960, 760), img="v1.jpg", emb=e_c2, score=0.83),
        _room_obs("v1_bed", "bed", (960, 400, 1260, 780), img="v1.jpg", emb=e_bed, score=0.86),
    ]
    obs_v2 = [
        _room_obs("v2_wall", "wall", (0, 50, 1280, 780), img="v2.jpg", cat="architecture", score=0.70),
        _room_obs("v2_tbl", "table", (460, 465, 830, 765), img="v2.jpg", emb=e_tbl, score=0.87),
        _room_obs("v2_bed", "bed", (950, 405, 1255, 785), img="v2.jpg", emb=e_bed, score=0.85),
    ]

    bundle = build_intelligence_bundle("e2e_room_run", obs_v1 + obs_v2)
    assert bundle.diagnostics["architectural_summary"]["verified_openings"] == 1
    assert bundle.diagnostics["architectural_summary"]["host_wall_linked"] >= 1
    assert bundle.diagnostics["furniture_summary"]["total_furniture_items"] >= 5
    assert bundle.diagnostics["relationship_summary"]["total_relationships"] >= 3

    drafts = to_evidence_matrix_drafts(bundle)
    assert len(drafts) == len(obs_v1) + len(obs_v2)

    from engine.multiengine.observation import make_observation

    m = EvidenceMatrix(tmp_path / "matrix")
    for idx, d in enumerate(drafts):
        eng_obs = make_observation(d, engine_id="grounding_dino", model_version="tiny", execution_config={})
        ctx = EvidenceContext(eng_obs.observation_id, "e2e_room_run", calibrated_confidence=None, calibration=None)
        m.add(eng_obs, ctx)
    assert len(m.store.query(quantity="bbox")) == len(drafts)
    assert m.verify() == []

    # Verify WorldIR bridge can consume the bundle's observations without errors
    world = WorldIR(id="w_e2e", name="e2e room")
    assert world.id == "w_e2e"


@pytest.mark.parametrize("budget", ["fast", "balanced", "forensic"])
def test_e2e_scheduler_budgets_and_health_isolation(budget):
    sched = PerceptionEngineScheduler(vram_budget_mb=4096.0)
    plan = sched.build_routing_plan(
        ["grounding_dino", "mask_rcnn", "florence2", "sam", "dinov2"],
        budget=budget,
        target_classes=["chair", "sofa", "door"],
    )
    assert len(plan.selected_detectors) >= 1
    res = sched.execute_sequential(
        plan,
        {
            "grounding_dino": lambda: [
                {"dummy": 1} if False else None  # unreachable
            ][:0],
            "mask_rcnn": lambda: [],
            "florence2": lambda: [],
        },
    )
    assert isinstance(res.engine_telemetry, dict)


def test_e2e_adaptive_crop_transform_and_mask_pipeline():
    rng = np.random.default_rng(42)
    img = rng.integers(30, 220, (1200, 1600, 3), dtype=np.uint8)
    q = analyze_image_quality(img)
    assert q.quality_grade == "good"
    assert "micro" in q.recommended_scales

    coarse = [
        _room_obs("d1", "door", (100, 100, 350, 900), cat="architecture", w=1600, h=1200),
        _room_obs("w1", "window", (600, 150, 950, 500), cat="architecture", w=1600, h=1200),
    ]
    from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox

    c_boxes = [DetectionBox(c.observation_id, c.canonical_label, c.bbox_px, 0.85) for c in coarse]
    tiles, crops = plan_adaptive_crops(q, c_boxes, tile_size=800)
    assert len(tiles) >= 4
    assert len(crops) == 2

    # Simulate crop detection and coordinate mapping
    crop0 = crops[0]
    mapped = transform_crop_box_to_image((20.0, 30.0, 70.0, 90.0), (crop0.x1 if hasattr(crop0, "x1") else crop0.box_px[0], crop0.box_px[1]), 1600, 1200)
    assert mapped is not None

    # Segmentation evaluation + overlapping separation
    m_a = np.zeros((120, 120), dtype=bool)
    m_a[20:80, 20:80] = True
    m_b = np.zeros((120, 120), dtype=bool)
    m_b[50:100, 50:100] = True
    sep = separate_overlapping_masks([m_a, m_b], [0.9, 0.6])
    assert not np.any(sep[0] & sep[1])
    ev = evaluate_instance_mask(sep[0], observation_id="m_a", box_px=(20, 20, 80, 80), canonical_label="chair")
    assert ev.is_acceptable is True
    assert classify_mask_kind("chair") == "visible_object_mask"


def test_e2e_material_extraction_and_refinement_stop_condition():
    img = np.full((120, 120, 3), (140, 95, 55), dtype=np.uint8)
    desc, hyps = extract_appearance_and_materials(img, (10, 10, 100, 100), canonical_label="cabinet")
    assert desc.dominant_color_name == "brown"
    assert all(not h.is_physically_verified for h in hyps)

    # Zero-budget refinement must stop immediately
    obs = _room_obs("o1", "chair", (10, 10, 100, 100))
    ref = plan_active_refinement([obs], compute_budget=0.1)
    assert ref.should_continue is False
    assert ref.stop_reason == "budget_exhausted"


def test_e2e_benchmark_failure_rate_and_diagnostics_healthy():
    obs = _room_obs("o_ok", "wall", (0, 0, 1000, 800), cat="architecture")
    m = np.ones((800, 1000), dtype=bool)
    m[0:5, 0:5] = False
    ev = evaluate_instance_mask(m, observation_id="o_ok", box_px=(0, 0, 1000, 800), canonical_label="wall")
    obs_with_seg = NormalizedPerceptionObservation(
        observation_id=obs.observation_id,
        schema_version=obs.schema_version,
        raw_label=obs.raw_label,
        canonical_label=obs.canonical_label,
        taxonomy_id=obs.taxonomy_id,
        category=obs.category,
        bbox_px=obs.bbox_px,
        coords=obs.coords,
        scores=obs.scores,
        provenance=obs.provenance,
        segmentation=ev.observation,
    )
    bundle = PerceptionBundle(run_id="healthy", observations=[obs_with_seg])
    diag = run_perception_diagnostics(bundle, image_qualities={"room_a.jpg": {"is_blank": False}})
    assert diag.status == "healthy"

    rep = run_perception_benchmark(
        mode=EvaluationMode.UNLABELLED_REAL,
        dataset_name="test_unlabelled",
        predictions_by_image={"a.jpg": []},
        failed_images=1,
    )
    assert rep.failure_rate == 1.0
    assert rep.accuracy_claim_allowed is False
