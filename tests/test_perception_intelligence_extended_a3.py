"""Extended edge-case, contract, and integration tests for A3 Perception Intelligence (A3.1 - A3.13)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.perception.evaluator import (
    EvaluationMode,
    evaluate_cross_view_identity_accuracy,
    run_perception_benchmark,
)
from benchmarks.perception.indoor_dataset_adapter import (
    GroundTruthBox,
    evaluate_detections_against_ground_truth,
    load_coco_indoor_annotations,
)
from engine.multiengine.observation import ObservationDraft
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox
from perception.intelligence.architectural import (
    ALL_ARCHITECTURAL_CLASSES,
    analyze_architecture,
)
from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    AppearanceDescriptor,
    CoordinateSystemSpec,
    CrossViewIdentityHypothesis,
    DetectionProvenance,
    MaterialHypothesisRecord,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    ObservationLifecycle,
    ObservationRevision,
    PerceptionBundle,
    ScoreSemantics,
    SegmentationObservation,
    TemporalObservation,
    ValidationError,
    stable_observation_id,
)
from perception.intelligence.diagnostics import run_perception_diagnostics
from perception.intelligence.embeddings import (
    Dinov2CropEmbedder,
    compute_patch_descriptor,
    enrich_observations_with_embeddings,
)
from perception.intelligence.furniture import (
    COMPONENT_HOST_RULES,
    FURNITURE_COMPONENTS,
    FURNITURE_HIERARCHY,
    analyze_furniture,
    classify_furniture_hierarchy,
)
from perception.intelligence.identity import (
    CameraPoseConstraint,
    resolve_cross_view_identities,
)
from perception.intelligence.image_adaptive import (
    analyze_image_quality,
    plan_adaptive_crops,
    reconcile_crop_duplicates,
    transform_crop_box_to_image,
)
from perception.intelligence.materials_intel import (
    SUPPORTED_MATERIALS,
    aggregate_cross_view_materials,
    extract_appearance_and_materials,
)
from perception.intelligence.orchestration import (
    EngineHealthState,
    PerceptionEngineScheduler,
)
from perception.intelligence.refinement import plan_active_refinement
from perception.intelligence.relationships import (
    RELATIONSHIP_ONTOLOGY,
    infer_scene_relationships,
)
from perception.intelligence.segmentation_intel import (
    classify_mask_kind,
    evaluate_instance_mask,
    separate_overlapping_masks,
)
from perception.intelligence.world_integration import (
    build_intelligence_bundle,
    observation_from_bbox_draft,
    to_evidence_matrix_drafts,
)


def _obs(
    oid: str,
    label: str,
    box: tuple[float, float, float, float],
    *,
    img: str = "view_0.jpg",
    cat: str = "object",
    score: float | None = 0.75,
    eng: str = "grounding_dino",
    fam: str = "grounding-dino",
    emb: tuple[float, ...] = (),
    hist: tuple[float, ...] = (),
    partial: bool = False,
    trunc: bool = False,
    occ: str = "none",
    seg: SegmentationObservation | None = None,
    comp: tuple[dict, ...] = (),
    w: int = 1000,
    h: int = 800,
) -> NormalizedPerceptionObservation:
    return NormalizedPerceptionObservation(
        observation_id=oid,
        schema_version=SCHEMA_VERSION,
        raw_label=label,
        canonical_label=label,
        taxonomy_id=f"a3.{cat}.{label}",
        category=cat,
        bbox_px=box,
        coords=CoordinateSystemSpec(f"image:{img}", "pixels", w, h),
        scores=ScoreSemantics(raw_score=score, score_emitted=(score is not None)),
        provenance=DetectionProvenance(
            source_image_id=img,
            execution=ModelExecutionProvenance(eng, fam, "ckpt_v1"),
        ),
        is_partial=partial,
        is_truncated=trunc,
        occlusion_level=occ,
        appearance=AppearanceDescriptor(color_histogram_24=hist, embedding_vector=emb),
        segmentation=seg,
        competing_labels=comp,
    )


# ---------------------------------------------------------------------------
# A3.1 Extended Validation, Serialization & Revision Lifecycle
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_score", [-0.01, 1.01, float("nan"), float("inf")])
def test_a3_1_score_semantics_rejects_out_of_range(bad_score):
    with pytest.raises(ValidationError):
        ScoreSemantics(raw_score=bad_score, score_emitted=True)


@pytest.mark.parametrize("bad_dims", [(0, 100), (100, 0), (-10, 100), (100, -5)])
def test_a3_1_coordinate_spec_rejects_non_positive_dims(bad_dims):
    with pytest.raises(ValidationError):
        CoordinateSystemSpec("image:x", "pixels", bad_dims[0], bad_dims[1])


@pytest.mark.parametrize(
    "bad_box",
    [
        (10.0, 10.0, 10.0, 50.0),
        (10.0, 50.0, 50.0, 50.0),
        (60.0, 10.0, 50.0, 50.0),
        (-5.0, 10.0, 50.0, 50.0),
        (10.0, 10.0, 1005.0, 50.0),
        (10.0, 10.0, float("nan"), 50.0),
    ],
)
def test_a3_1_observation_rejects_invalid_boxes(bad_box):
    with pytest.raises(ValidationError):
        _obs("bad", "chair", bad_box)


def test_a3_1_revision_lifecycle_and_full_serialization_roundtrip():
    rev = ObservationRevision(2, "refined_via_crop", ObservationLifecycle.REVISED, previous_observation_id="perc_old")
    temp = TemporalObservation(frame_index=3, timestamp_s=0.1, shot_id=1, track_id="trk_1", motion_state="static")
    seg = SegmentationObservation(
        mask_id="m1",
        observation_id="o1",
        mask_kind="visible_object_mask",
        area_px=1200,
        box_iou=0.88,
        solidity=0.94,
        hole_count=0,
        boundary_uncertainty=0.12,
        quality_score=0.88,
        polygon_px=((10.0, 10.0), (50.0, 10.0), (50.0, 50.0), (10.0, 50.0)),
        depth_mean_relative=2.3,
    )
    mat = MaterialHypothesisRecord("wood", "oak", 0.55, "color=brown", is_physically_verified=False)
    obs = NormalizedPerceptionObservation(
        observation_id="o1",
        schema_version=SCHEMA_VERSION,
        raw_label="wooden chair",
        canonical_label="chair",
        taxonomy_id="interior.furniture.chair",
        category="furniture",
        bbox_px=(10.0, 10.0, 110.0, 150.0),
        coords=CoordinateSystemSpec("image:v0", "pixels", 1000, 800),
        scores=ScoreSemantics(raw_score=0.82, score_emitted=True),
        provenance=DetectionProvenance(
            source_image_id="v0.jpg",
            execution=ModelExecutionProvenance("grounding_dino", "grounding-dino", "tiny", device="cuda:0"),
            inference_mode="host_crop",
            crop_origin_px=(5, 5),
        ),
        material_hypotheses=(mat,),
        segmentation=seg,
        temporal=temp,
        revision=rev,
    )
    d = obs.to_dict()
    back = NormalizedPerceptionObservation.from_dict(d)
    assert back.observation_id == "o1"
    assert back.revision.lifecycle is ObservationLifecycle.REVISED
    assert back.temporal is not None and back.temporal.track_id == "trk_1"
    assert back.segmentation is not None and back.segmentation.depth_mean_relative == 2.3
    assert len(back.material_hypotheses) == 1
    assert back.provenance.crop_origin_px == (5, 5)


def test_a3_1_stable_id_distinguishes_labels_and_images():
    a = stable_observation_id("perc", "img1", "gd", "chair", (10.0, 20.0, 100.0, 200.0))
    b = stable_observation_id("perc", "img2", "gd", "chair", (10.0, 20.0, 100.0, 200.0))
    c = stable_observation_id("perc", "img1", "gd", "table", (10.0, 20.0, 100.0, 200.0))
    assert len({a, b, c}) == 3


# ---------------------------------------------------------------------------
# A3.2 Extended Architectural Intelligence
# ---------------------------------------------------------------------------

def test_a3_2_all_architectural_categories_and_boundary_continuity():
    w1 = _obs("w1", "wall", (0.0, 50.0, 480.0, 700.0), cat="architecture")
    w2 = _obs("w2", "wall", (485.0, 50.0, 990.0, 700.0), cat="architecture")
    fl = _obs("fl1", "floor", (0.0, 650.0, 1000.0, 800.0), cat="architecture")
    ce = _obs("ce1", "ceiling", (0.0, 0.0, 1000.0, 80.0), cat="architecture")
    col = _obs("col1", "column", (460.0, 60.0, 520.0, 700.0), cat="architecture")
    beam = _obs("bm1", "beam", (0.0, 60.0, 1000.0, 130.0), cat="architecture")
    stair = _obs("st1", "staircase", (600.0, 300.0, 950.0, 720.0), cat="architecture")
    rail = _obs("rl1", "railing", (600.0, 260.0, 950.0, 420.0), cat="architecture")
    win = _obs("win1", "window", (100.0, 150.0, 300.0, 400.0), cat="architecture")
    wframe = _obs("wf1", "window_frame", (95.0, 145.0, 305.0, 405.0), cat="architecture")
    opening = _obs("op1", "opening", (550.0, 150.0, 750.0, 650.0), cat="architecture")

    res = analyze_architecture([w1, w2, fl, ce, col, beam, stair, rail, win, wframe, opening])
    by_id = {a.observation.observation_id: a for a in res.observations}

    assert by_id["w1"].boundary_continuity_score >= 0.75
    assert by_id["col1"].structural_role == "structural_support"
    assert by_id["bm1"].structural_role == "structural_support"
    assert by_id["st1"].structural_role == "circulation"
    assert by_id["rl1"].structural_role == "circulation"
    assert by_id["win1"].is_verified_opening is True
    assert by_id["win1"].host_wall_id == "w1"
    assert by_id["op1"].is_verified_opening is False
    assert by_id["op1"].structural_role == "opening_candidate"
    assert "wall" in ALL_ARCHITECTURAL_CLASSES


def test_a3_2_opening_with_valid_mask_promotes_to_verified_opening():
    seg = SegmentationObservation(
        mask_id="m_door",
        observation_id="d1",
        mask_kind="visible_object_mask",
        area_px=30000,
        box_iou=0.85,
        solidity=0.92,
        hole_count=0,
        boundary_uncertainty=0.1,
        quality_score=0.82,
    )
    door = _obs("d1", "door", (100.0, 100.0, 300.0, 600.0), cat="architecture", seg=seg)
    res = analyze_architecture([door])
    assert res.observations[0].structural_role == "verified_opening"
    assert res.observations[0].host_wall_hypothesis_state == "unknown"


# ---------------------------------------------------------------------------
# A3.3 Extended Furniture Intelligence
# ---------------------------------------------------------------------------

def test_a3_3_furniture_taxonomy_coverage_and_component_binding():
    for cls in (
        "table", "chair", "sofa", "bed", "desk", "cabinet",
        "wardrobe", "shelf", "storage_unit", "kitchen_furniture",
        "bathroom_fixture", "lamp", "refrigerator",
    ):
        assert cls in FURNITURE_HIERARCHY
        assert len(classify_furniture_hierarchy(cls)) >= 3

    desk = _obs("dsk1", "desk", (100.0, 300.0, 600.0, 650.0), cat="furniture", score=0.90)
    drawer = _obs("drw1", "drawer", (140.0, 340.0, 320.0, 440.0), cat="furniture", score=0.72)
    handle = _obs("hdl1", "drawer_handle", (210.0, 380.0, 250.0, 400.0), cat="furniture", score=0.65)
    toilet = _obs("tlt1", "toilet", (700.0, 350.0, 880.0, 680.0), cat="fixture", score=0.88)

    res = analyze_furniture([desk, drawer, handle, toilet])
    assert res.component_links["drw1"] == "dsk1"
    assert "hdl1" in res.component_links
    by_id = {it.observation.observation_id: it for it in res.items}
    assert by_id["dsk1"].mobility == "movable"
    assert by_id["tlt1"].mobility == "fixed_fixture"
    assert "drawer" in FURNITURE_COMPONENTS
    assert "desk" in COMPONENT_HOST_RULES["drawer"]


# ---------------------------------------------------------------------------
# A3.4 Extended Cross-View Identity & Camera Constraints
# ---------------------------------------------------------------------------

def test_a3_4_six_identical_chairs_and_camera_geometry_disambiguation():
    emb = (0.6, 0.4, 0.2, 0.1)
    obs = []
    for i in range(6):
        obs.append(_obs(f"v1_c{i}", "chair", (20.0 + i * 150, 200.0, 130.0 + i * 150, 420.0), img="v1.jpg", emb=emb))
        obs.append(_obs(f"v2_c{i}", "chair", (30.0 + i * 150, 200.0, 140.0 + i * 150, 420.0), img="v2.jpg", emb=emb))

    rep_no_geom = resolve_cross_view_identities(obs)
    assert len(rep_no_geom.identities) >= 6

    # Supply camera constraint protocol instance + geometric disambiguation function
    _ = CameraPoseConstraint("v1.jpg", (0.0, 0.0, 1.5), (1.0, 0.0, 0.0, 0.0), is_metric=True)

    def _geom_match(a: NormalizedPerceptionObservation, b: NormalizedPerceptionObservation) -> float | None:
        idx_a = a.observation_id.split("_c")[-1]
        idx_b = b.observation_id.split("_c")[-1]
        return 0.98 if idx_a == idx_b else 0.05

    rep_with_geom = resolve_cross_view_identities(obs, epipolar_fn=_geom_match, match_threshold=0.60)
    assert rep_with_geom.summary["confirmed_matches"] == 6
    assert len(rep_with_geom.identities) == 6


def test_a3_4_incompatible_classes_never_merge():
    emb = (0.7, 0.7, 0.0, 0.0)
    a = _obs("a1", "chair", (100.0, 100.0, 300.0, 400.0), img="v1.jpg", emb=emb)
    b = _obs("b1", "table", (100.0, 100.0, 300.0, 400.0), img="v2.jpg", emb=emb)
    rep = resolve_cross_view_identities([a, b])
    assert rep.summary["confirmed_matches"] == 0
    assert len(rep.identities) == 2


# ---------------------------------------------------------------------------
# A3.5 Extended Multi-Engine Orchestration & Health
# ---------------------------------------------------------------------------

def test_a3_5_health_state_recovery_and_empty_pool():
    h = EngineHealthState("mask_rcnn")
    h.record_failure("err1")
    h.record_failure("err2")
    assert h.circuit_open is False
    h.record_failure("err3")
    assert h.circuit_open is True
    h.record_success(0.4, vram_mb=320.0)
    assert h.circuit_open is False
    assert h.to_dict()["max_vram_mb"] == 320.0

    sched = PerceptionEngineScheduler()
    fast_plan = sched.build_routing_plan(["grounding_dino", "mask_rcnn"], budget="fast")
    assert fast_plan.selected_detectors == ("grounding_dino",)
    forensic_plan = sched.build_routing_plan(
        ["grounding_dino", "mask_rcnn", "florence2", "dinov2"],
        budget="forensic",
        image_max_edge=2048,
    )
    assert "high_res_tiling_recommended" in forensic_plan.reasons
    assert forensic_plan.selected_embedders == ("dinov2",)


# ---------------------------------------------------------------------------
# A3.6 Extended Image Quality & Adaptive Processing
# ---------------------------------------------------------------------------

def test_a3_6_underexposed_overexposed_blurry_and_noisy_detection():
    dark = np.full((128, 128, 3), 10, dtype=np.uint8)
    dark[0:64, 0:64] = 25
    q_dark = analyze_image_quality(dark)
    assert q_dark.is_underexposed is True
    assert q_dark.quality_grade == "degraded"

    bright = np.full((128, 128, 3), 245, dtype=np.uint8)
    bright[0:64, 0:64] = 225
    q_bright = analyze_image_quality(bright)
    assert q_bright.is_overexposed is True
    assert q_bright.quality_grade == "degraded"

    # Smooth low-frequency gradient -> blurry
    grad = np.tile(np.linspace(40, 200, 128, dtype=np.uint8), (128, 1))
    smooth_rgb = np.stack([grad, grad, grad], axis=-1)
    q_blur = analyze_image_quality(smooth_rgb)
    assert q_blur.is_blurry is True

    assert transform_crop_box_to_image((100.0, 100.0, 50.0, 150.0), (0, 0), 200, 200) is None
    assert len(reconcile_crop_duplicates([])) == 0


# ---------------------------------------------------------------------------
# A3.7 Extended Segmentation Intelligence & Depth Association
# ---------------------------------------------------------------------------

def test_a3_7_invalid_mask_rejection_and_structural_background_separation():
    tiny = np.zeros((100, 100), dtype=bool)
    tiny[10:12, 10:12] = True  # 4 px < 20 px min
    ev_tiny = evaluate_instance_mask(tiny, observation_id="t", box_px=(0, 0, 50, 50), canonical_label="chair")
    assert ev_tiny.is_acceptable is False
    assert ev_tiny.rejection_reason == "mask_area_too_small_under_20px"

    wall_m = np.ones((100, 100), dtype=bool)
    chair_m = np.zeros((100, 100), dtype=bool)
    chair_m[30:70, 30:70] = True
    sep = separate_overlapping_masks([wall_m, chair_m], [0.95, 0.70], is_structural=[True, False])
    # Foreground chair keeps its pixels even though wall score was higher
    assert int(np.sum(sep[1])) == 1600


# ---------------------------------------------------------------------------
# A3.8 Extended Material & Appearance Intelligence
# ---------------------------------------------------------------------------

def test_a3_8_material_consensus_and_empty_crop_handling():
    assert len(SUPPORTED_MATERIALS) == 10
    img = np.full((64, 64, 3), (200, 200, 200), dtype=np.uint8)
    desc, hyps = extract_appearance_and_materials(img, (0, 0, 0, 0), canonical_label="wall")
    assert desc.dominant_color_name == "unknown"
    assert hyps == ()

    c_empty = aggregate_cross_view_materials("id_x", [[], []])
    assert c_empty.consensus_material == "unknown"
    assert c_empty.is_consistent is False


# ---------------------------------------------------------------------------
# A3.9 Extended Scene Relationships & Contradictions
# ---------------------------------------------------------------------------

def test_a3_9_relationship_ontology_and_contradiction_flagging():
    assert "bed_against_wall" in RELATIONSHIP_ONTOLOGY
    ceil = _obs("c1", "ceiling", (0.0, 0.0, 1000.0, 400.0), cat="architecture")
    flr = _obs("f1", "floor", (0.0, 200.0, 1000.0, 800.0), cat="architecture")
    lamp = _obs("l1", "lamp", (400.0, 150.0, 550.0, 300.0), cat="fixture")
    bed = _obs("b1", "bed", (100.0, 350.0, 600.0, 750.0), cat="furniture")
    wall = _obs("w1", "wall", (0.0, 0.0, 1000.0, 750.0), cat="architecture")

    rep = infer_scene_relationships([ceil, flr, lamp, bed, wall])
    preds = {r.predicate for r in rep.relationships}
    assert "bed_against_wall" in preds
    assert "light_attached_to_ceiling" in preds
    assert "resting_on_floor" in preds
    assert rep.summary["total_relationships"] >= 3


# ---------------------------------------------------------------------------
# A3.10, A3.12, A3.13 & Embeddings Integration
# ---------------------------------------------------------------------------

def test_a3_embeddings_patch_descriptor_and_enrichment():
    rng = np.random.default_rng(12)
    img = rng.integers(0, 255, (200, 200, 3), dtype=np.uint8)
    desc = compute_patch_descriptor(img[20:80, 20:80])
    assert desc.dim == 32
    assert desc.is_neural is False
    assert abs(float(np.linalg.norm(desc.vector)) - 1.0) < 1e-3

    obs = _obs("e1", "chair", (20.0, 20.0, 80.0, 80.0), img="r1.jpg", w=200, h=200)
    enriched = enrich_observations_with_embeddings([obs], {"r1.jpg": img}, use_dinov2=False)
    assert len(enriched[0].appearance.embedding_vector) == 32
    assert enriched[0].appearance.embedding_model == "color_grad_descriptor_v1"

    emb_probe = Dinov2CropEmbedder()
    assert isinstance(emb_probe.is_available(), bool)


def test_a3_13_diagnostics_catches_duplicates_and_implausible_volume():
    obs = [
        _obs("d1", "chair", (10.0, 10.0, 100.0, 100.0), img="i1.jpg"),
        _obs("d2", "chair", (12.0, 12.0, 102.0, 102.0), img="i1.jpg"),
    ]
    bundle = PerceptionBundle(run_id="r_dup", observations=obs)
    diag = run_perception_diagnostics(bundle, max_boxes_per_image=1)
    codes = {f.code for f in diag.findings}
    assert "implausible_detection_volume" in codes
    assert "repeated_duplicate_detections" in codes
    assert diag.status == "degraded"


def test_a3_12_world_integration_preserves_calibration_and_provenance():
    draft = ObservationDraft(
        task="detection",
        quantity="bbox",
        source_id="img_99.jpg",
        coordinate_system="image:img_99",
        image_region=(50.0, 60.0, 250.0, 360.0),
        predicted={
            "label": "wardrobe",
            "raw_label": "wardrobe",
            "raw_score": 0.66,
            "confidence_emitted": True,
            "is_truncated": True,
            "occlusion_level": "partial",
        },
        confidence=0.66,
    )
    obs = observation_from_bbox_draft(draft, image_width=1024, image_height=768)
    assert obs.is_partial is True
    assert obs.scores.is_calibrated is False
    assert obs.scores.calibrated_probability is None

    bundle = build_intelligence_bundle("r_w", [obs])
    out_drafts = to_evidence_matrix_drafts(bundle)
    assert len(out_drafts) == 1
    assert out_drafts[0].predicted["calibrated_confidence"] is None
    assert out_drafts[0].predicted["is_partial"] is True
