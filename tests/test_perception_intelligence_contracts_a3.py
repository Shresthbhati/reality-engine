"""Contract, invariance, and schema compatibility tests for A3 Perception Intelligence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.perception.evaluator import EvaluationMode, run_perception_benchmark
from engine.multiengine.observation import ObservationDraft
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox
from perception.intelligence import (
    SCHEMA_VERSION,
    AppearanceDescriptor,
    ArchitecturalObservation,
    CoordinateSystemSpec,
    CrossViewIdentityHypothesis,
    DetectionProvenance,
    FurnitureObservation,
    MaterialHypothesisRecord,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    ObservationLifecycle,
    ObservationRevision,
    PerceptionBundle,
    PerceptionEngineScheduler,
    ScoreSemantics,
    SegmentationObservation,
    SpatialRelationshipObservation,
    TemporalObservation,
    ValidationError,
    aggregate_cross_view_materials,
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
    resolve_cross_view_identities,
    run_perception_diagnostics,
    separate_overlapping_masks,
    to_evidence_matrix_drafts,
)


def _make_obs(
    oid: str,
    label: str,
    box: tuple[float, float, float, float],
    *,
    img: str = "c_view.jpg",
    cat: str = "furniture",
    score: float | None = 0.77,
    emb: tuple[float, ...] = (),
    w: int = 1024,
    h: int = 768,
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
            execution=ModelExecutionProvenance("grounding_dino", "grounding-dino", "tiny"),
        ),
        appearance=AppearanceDescriptor(embedding_vector=emb),
    )


def test_contract_schema_version_and_missing_confidence_semantics():
    obs = _make_obs("fl_1", "window", (100, 100, 300, 350), cat="architecture", score=None)
    d = obs.to_dict()
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["scores"]["raw_score"] is None
    assert d["scores"]["score_emitted"] is False
    assert d["scores"]["is_calibrated"] is False
    assert d["scores"]["calibrated_probability"] is None
    restored = NormalizedPerceptionObservation.from_dict(d)
    assert restored.scores.raw_score is None
    assert restored.scores.score_emitted is False


def test_contract_calibrated_score_requires_is_calibrated_true():
    with pytest.raises(ValidationError):
        ScoreSemantics(raw_score=0.8, score_emitted=True, calibrated_probability=0.75, is_calibrated=False)
    valid = ScoreSemantics(
        raw_score=0.8,
        score_emitted=True,
        calibrated_probability=0.75,
        is_calibrated=True,
        calibration_dataset="indoor_dev_v1",
        calibration_model_version="platt_v1",
        calibration_note="per_class_platt",
    )
    assert valid.to_dict()["calibration_dataset"] == "indoor_dev_v1"


def test_contract_empty_observation_id_or_label_rejected():
    with pytest.raises(ValidationError):
        _make_obs("", "chair", (10, 10, 50, 50))
    with pytest.raises(ValidationError):
        _make_obs("id1", "   ", (10, 10, 50, 50))


def test_contract_architectural_host_wall_never_invented_for_distant_opening():
    wall = _make_obs("w1", "wall", (0, 0, 300, 700), cat="architecture")
    door = _make_obs("d1", "door", (600, 100, 800, 700), cat="architecture")
    res = analyze_architecture([wall, door])
    by_id = {a.observation.observation_id: a for a in res.observations}
    assert by_id["d1"].host_wall_id is None
    assert by_id["d1"].host_wall_hypothesis_state == "unknown"


def test_contract_furniture_never_merges_distinct_classes_at_high_iou():
    tbl = _make_obs("t1", "table", (100, 200, 500, 600), cat="furniture", score=0.85)
    dsk = _make_obs("d1", "desk", (105, 205, 495, 595), cat="furniture", score=0.80)
    res = analyze_furniture([tbl, dsk])
    assert len(res.items) == 2
    assert res.suppressed_duplicates == ()


def test_contract_cross_view_identity_never_merges_same_image_observations():
    emb = (1.0, 0.0, 0.0, 0.0)
    c1 = _make_obs("c1", "chair", (50, 100, 200, 400), img="same.jpg", emb=emb)
    c2 = _make_obs("c2", "chair", (300, 100, 450, 400), img="same.jpg", emb=emb)
    rep = resolve_cross_view_identities([c1, c2])
    assert len(rep.identities) == 2
    assert rep.summary["confirmed_matches"] == 0


def test_contract_image_quality_rejects_invalid_array_shape():
    with pytest.raises(ValueError):
        analyze_image_quality(np.zeros((64, 64), dtype=np.uint8))


def test_contract_segmentation_rejects_full_frame_mask():
    full = np.ones((100, 100), dtype=bool)
    ev = evaluate_instance_mask(full, observation_id="f1", box_px=(0, 0, 100, 100), canonical_label="wall")
    assert ev.is_acceptable is False
    assert ev.rejection_reason == "mask_area_exceeds_98_percent_frame"


def test_contract_material_hypothesis_forbids_physical_verification_flag():
    with pytest.raises(ValidationError):
        MaterialHypothesisRecord("metal", "steel", 0.7, "specular", is_physically_verified=True)


def test_contract_relationship_observation_to_dict():
    rel = SpatialRelationshipObservation(
        relationship_id="r1",
        subject_id="c1",
        predicate="chair_beside_table",
        object_id="t1",
        coordinate_frame="image_space",
        support_state="visually_observed",
        confidence_score=0.8,
        supporting_images=("img1.jpg",),
    )
    d = rel.to_dict()
    assert d["predicate"] == "chair_beside_table"
    assert d["supporting_images"] == ["img1.jpg"]


def test_contract_draft_to_observation_preserves_tiling_and_subpart_mode():
    draft_tiled = ObservationDraft(
        task="detection",
        quantity="bbox",
        source_id="img1.jpg",
        coordinate_system="image:img1",
        image_region=(10.0, 10.0, 50.0, 50.0),
        predicted={"label": "socket", "inner_engine": "grounding_dino_tiled", "raw_score": 0.44},
        confidence=0.44,
    )
    obs_t = observation_from_bbox_draft(draft_tiled, image_width=1024, image_height=768)
    assert obs_t.provenance.inference_mode == "tiled"

    draft_sub = ObservationDraft(
        task="detection",
        quantity="bbox",
        source_id="img1.jpg",
        coordinate_system="image:img1",
        image_region=(10.0, 10.0, 50.0, 50.0),
        predicted={"label": "door_handle", "inner_engine": "grounding_dino_subpart", "raw_score": 0.51},
        confidence=0.51,
    )
    obs_s = observation_from_bbox_draft(draft_sub, image_width=1024, image_height=768)
    assert obs_s.provenance.inference_mode == "host_crop"
    assert len(to_evidence_matrix_drafts(PerceptionBundle(run_id="r", observations=[obs_t, obs_s]))) == 2


def test_contract_benchmark_report_to_dict_has_all_fields():
    rep = run_perception_benchmark(
        mode=EvaluationMode.UNLABELLED_REAL,
        dataset_name="demo",
        predictions_by_image={"a.jpg": [DetectionBox("b1", "door", (10, 10, 100, 200), 0.7)]},
    ).to_dict()
    for key in (
        "mode", "dataset_name", "images_evaluated", "precision", "recall", "f1",
        "mean_ap", "mean_iou", "duplicate_rate", "mask_quality_mean",
        "identity_accuracy", "classification_accuracy", "ece", "brier_score",
        "wall_time_s", "peak_vram_mb", "failure_rate", "accuracy_claim_allowed", "caveat",
    ):
        assert key in rep
