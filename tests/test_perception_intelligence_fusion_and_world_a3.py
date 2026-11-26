"""Integration tests for A3 Multi-Engine Fusion, Revision Chains, and EvidenceMatrix Persistence."""

from __future__ import annotations

import json

import numpy as np
import pytest

from engine.multiengine.evidence_matrix import EvidenceContext, EvidenceMatrix
from engine.multiengine.observation import ObservationDraft, make_observation
from engine.multiengine.spatial_intelligence.detection_pipeline import (
    DetectionBox,
    LearnedConfidenceCalibrator,
)
from perception.detection.multi_engine_fusion import (
    associate_cross_view_detections,
    engine_family,
    fuse_multi_engine_detections,
    select_detection_engines,
)
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
    ScoreSemantics,
    build_intelligence_bundle,
    observation_from_bbox_draft,
    to_evidence_matrix_drafts,
)


def test_engine_family_grouping_prevents_double_counting():
    assert engine_family("grounding_dino") == "grounding-dino"
    assert engine_family("grounding_dino_tiled") == "grounding-dino"
    assert engine_family("grounding_dino_subpart") == "grounding-dino"
    assert engine_family("mask_rcnn") == "maskrcnn"
    assert engine_family("florence2") == "florence2"


def test_fusion_with_fitted_calibrator_vs_uncalibrated():
    b1 = DetectionBox("b1", "chair", (10, 10, 100, 120), 0.78, source_engine="grounding_dino", source_image_id="v1.jpg")
    b2 = DetectionBox("b2", "chair", (12, 12, 102, 122), 0.74, source_engine="mask_rcnn", source_image_id="v1.jpg")

    uncal = fuse_multi_engine_detections({"grounding_dino": [b1], "mask_rcnn": [b2]})
    assert len(uncal) == 1
    assert uncal[0].calibrated_confidence is None
    assert uncal[0].calibration_status == "uncalibrated"
    assert uncal[0].independent_family_count == 2

    cal = LearnedConfidenceCalibrator(
        temperature=1.1,
        bias=0.05,
        provenance={"is_fitted": True, "dataset_split": "dev_v1"},
    )
    cal_fused = fuse_multi_engine_detections({"grounding_dino": [b1], "mask_rcnn": [b2]}, calibrator=cal)
    assert cal_fused[0].calibrated_confidence is not None
    assert cal_fused[0].calibration_status == "global_platt"


def test_fusion_preserves_florence2_unscored_semantics():
    b_gd = DetectionBox("g1", "window", (50, 50, 200, 220), 0.71, source_engine="grounding_dino", source_image_id="v1.jpg")
    b_fl = DetectionBox(
        "f1", "window", (52, 52, 202, 222), 0.5,
        source_engine="florence2", source_image_id="v1.jpg",
        attributes={"confidence_emitted": False},
    )
    fused = fuse_multi_engine_detections({"grounding_dino": [b_gd], "florence2": [b_fl]})
    assert len(fused) == 1
    assert fused[0].raw_scores["florence2"] is None
    assert fused[0].raw_scores["grounding_dino"] == 0.71
    assert fused[0].independent_family_count == 2


def test_cross_view_detections_with_geometric_scorer():
    a = DetectionBox(
        "a1", "table", (100, 200, 400, 500), 0.85,
        source_image_id="img1.jpg", appearance_features=(0.6, 0.4, 0.0, 0.0),
    )
    b = DetectionBox(
        "b1", "table", (120, 210, 420, 510), 0.82,
        source_image_id="img2.jpg", appearance_features=(0.6, 0.4, 0.0, 0.0),
    )
    res = associate_cross_view_detections(
        [a], [b],
        geometric_score_fn=lambda x, y: 0.92,
    )
    assert len(res) == 1
    assert res[0].status == "matched"
    assert res[0].target_box_id == "b1"


def test_routing_falls_back_cleanly_when_primary_missing():
    assert select_detection_engines(["mask_rcnn"], budget="fast") == ["mask_rcnn"]
    assert select_detection_engines(["owlvit"], budget="fast") == ["owlvit"]
    assert select_detection_engines([], budget="balanced") == []


@pytest.mark.parametrize("lifecycle", list(ObservationLifecycle))
def test_observation_revision_lifecycle_states_roundtrip(lifecycle):
    rev = ObservationRevision(3, f"transition_{lifecycle.value}", lifecycle, previous_observation_id="prev_1")
    d = rev.to_dict()
    back = ObservationRevision.from_dict(d)
    assert back.lifecycle is lifecycle
    assert back.revision_number == 3


def test_evidence_matrix_full_roundtrip_with_architectural_and_furniture_bundle(tmp_path):
    obs_list = [
        NormalizedPerceptionObservation(
            observation_id="w_1",
            schema_version=SCHEMA_VERSION,
            raw_label="wall",
            canonical_label="wall",
            taxonomy_id="interior.surface.wall",
            category="architecture",
            bbox_px=(0.0, 0.0, 1000.0, 750.0),
            coords=CoordinateSystemSpec("image:r1", "pixels", 1000, 800),
            scores=ScoreSemantics(raw_score=0.72, score_emitted=True),
            provenance=DetectionProvenance("r1.jpg", ModelExecutionProvenance("grounding_dino", "grounding-dino", "tiny")),
        ),
        NormalizedPerceptionObservation(
            observation_id="d_1",
            schema_version=SCHEMA_VERSION,
            raw_label="door",
            canonical_label="door",
            taxonomy_id="interior.opening.door",
            category="architecture",
            bbox_px=(150.0, 100.0, 350.0, 720.0),
            coords=CoordinateSystemSpec("image:r1", "pixels", 1000, 800),
            scores=ScoreSemantics(raw_score=0.81, score_emitted=True),
            provenance=DetectionProvenance("r1.jpg", ModelExecutionProvenance("grounding_dino", "grounding-dino", "tiny")),
        ),
        NormalizedPerceptionObservation(
            observation_id="c_1",
            schema_version=SCHEMA_VERSION,
            raw_label="chair",
            canonical_label="chair",
            taxonomy_id="interior.furniture.chair",
            category="furniture",
            bbox_px=(500.0, 400.0, 700.0, 750.0),
            coords=CoordinateSystemSpec("image:r1", "pixels", 1000, 800),
            scores=ScoreSemantics(raw_score=0.88, score_emitted=True),
            provenance=DetectionProvenance("r1.jpg", ModelExecutionProvenance("mask_rcnn", "maskrcnn", "coco")),
        ),
    ]
    bundle = build_intelligence_bundle("run_matrix_rt", obs_list)
    drafts = to_evidence_matrix_drafts(bundle)
    m = EvidenceMatrix(tmp_path / "em")
    for d in drafts:
        eo = make_observation(d, engine_id=d.predicted["inner_engine"], model_version="v1", execution_config={})
        m.add(eo, EvidenceContext(eo.observation_id, "run_matrix_rt", calibrated_confidence=None, calibration=None))
    assert m.verify() == []
    assert len(m.store.query(quantity="bbox")) == 3


def test_observation_from_bbox_draft_handles_unscored_draft():
    draft = ObservationDraft(
        task="detection",
        quantity="bbox",
        source_id="img_unscored.jpg",
        coordinate_system="image:img_unscored",
        image_region=(20.0, 30.0, 120.0, 180.0),
        predicted={"label": "sofa", "confidence_emitted": False, "inner_engine": "florence2"},
        confidence=None,
    )
    obs = observation_from_bbox_draft(draft, image_width=800, image_height=600, architecture_family="florence2")
    assert obs.scores.raw_score is None
    assert obs.scores.score_emitted is False
    assert obs.scores.calibrated_probability is None
    assert obs.canonical_label == "sofa"


def test_bundle_json_serialization_is_deterministic():
    obs = NormalizedPerceptionObservation(
        observation_id="det_1",
        schema_version=SCHEMA_VERSION,
        raw_label="desk",
        canonical_label="desk",
        taxonomy_id="interior.furniture.desk",
        category="furniture",
        bbox_px=(100.0, 200.0, 500.0, 600.0),
        coords=CoordinateSystemSpec("image:x", "pixels", 800, 600),
        scores=ScoreSemantics(raw_score=0.75, score_emitted=True),
        provenance=DetectionProvenance("x.jpg", ModelExecutionProvenance("grounding_dino", "grounding-dino", "tiny")),
    )
    b1 = PerceptionBundle(run_id="det_run", observations=[obs])
    b2 = PerceptionBundle(run_id="det_run", observations=[obs])
    assert b1.to_json() == b2.to_json()
    assert json.loads(b1.to_json())["observations"][0]["canonical_label"] == "desk"
