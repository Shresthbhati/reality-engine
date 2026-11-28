"""Tests for expanded 64-annotation GT dataset, per-class metrics, OCR eval, and production wiring."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
)
from perception.evaluation.gt_benchmark import (
    PredBoxAnnotation,
    PredRelationship,
    evaluate_multiview_identity,
    evaluate_scene_graph,
)
from perception.evaluation.production_eval import (
    connect_attributes_and_scene_graph,
    evaluate_ocr_candidates,
    evaluate_per_class_metrics,
    load_expanded_gt_with_ocr,
)


def test_expanded_gt_dataset_and_per_class_metrics() -> None:
    gt_boxes, gt_rels, ocr_regions = load_expanded_gt_with_ocr(
        Path("datasets/gt_perception_benchmark/annotations.json")
    )
    assert len(gt_boxes) >= 60
    assert len(gt_rels) >= 10
    assert len(ocr_regions) >= 4

    # Simulate partial predictions (drop hinges & drains to verify per-class honesty)
    preds = [
        PredBoxAnnotation(
            g.image_id,
            g.category,
            (g.bbox_xyxy[0] + 2.0, g.bbox_xyxy[1] + 2.0, g.bbox_xyxy[2] + 2.0, g.bbox_xyxy[3] + 2.0),
            score=0.90,
            track_id=g.track_id,
        )
        for g in gt_boxes
        if g.category not in {"hinge", "drain"}
    ]
    per_cls = {m.category: m for m in evaluate_per_class_metrics(gt_boxes, preds)}
    assert per_cls["switch"].recall == 1.0
    assert per_cls["switch"].mean_center_err_px is not None
    assert per_cls["hinge"].recall == 0.0
    assert per_cls["drain"].recall == 0.0

    id_rep = evaluate_multiview_identity(gt_boxes, preds)
    assert id_rep.f1 == 1.0
    assert id_rep.id_switches == 0

    sg_rep = evaluate_scene_graph(
        gt_rels,
        [PredRelationship(r.source_track_id, r.target_track_id, r.kind) for r in gt_rels],
    )
    assert sg_rep.f1 == 1.0


def test_ocr_candidate_evaluation_and_unavailable_handling() -> None:
    _, _, ocr_regions = load_expanded_gt_with_ocr(
        Path("datasets/gt_perception_benchmark/annotations.json")
    )
    imgs = {r["image_file"]: np.zeros((1200, 1600, 3), dtype=np.uint8) for r in ocr_regions}

    res_ok = evaluate_ocr_candidates(
        ocr_regions,
        imgs,
        lambda _im, _box: "KWH 04821",
        candidate_id="mock_ocr",
    )
    assert res_ok.status == "succeeded"
    assert res_ok.exact_accuracy == 0.5  # 2 of 4 regions are KWH 04821

    def broken_ocr(_im, _box):
        raise RuntimeError("Florence-2 checkpoint not cached")

    res_fail = evaluate_ocr_candidates(ocr_regions, imgs, broken_ocr, candidate_id="florence2_ocr")
    assert res_fail.status == "unavailable"
    assert res_fail.exact_accuracy is None


def test_production_observations_connect_to_attributes_and_scene_graph() -> None:
    rgb = np.full((200, 200, 3), 180, dtype=np.uint8)
    depth = np.full((200, 200), 2.0, dtype=np.float32)

    wall = ObservationRecord(
        evidence_id="gdn:img0:0",
        source_image_id="img0.jpg",
        coordinate_frame=CoordinateFrame(frame_id="image", scale_state="image", units="pixels"),
        model_id=ModelIdentity("detection", "grounding_dino", "tiny", "Apache-2.0", hardware_class="cuda"),
        label="wall",
        category="architecture",
        bbox_px=(0.0, 0.0, 200.0, 200.0),
        confidence=0.9,
        calibration_quality="unverified",
        metadata={"ontology_path": "architecture/wall"},
    )
    switch = ObservationRecord(
        evidence_id="gdn:img0:1",
        source_image_id="img0.jpg",
        coordinate_frame=CoordinateFrame(frame_id="image", scale_state="image", units="pixels"),
        model_id=ModelIdentity("detection", "grounding_dino", "tiny", "Apache-2.0", hardware_class="cuda"),
        label="switch",
        category="electrical",
        bbox_px=(60.0, 80.0, 80.0, 105.0),
        confidence=0.82,
        calibration_quality="unverified",
        metadata={"ontology_path": "fixtures/electrical/switch"},
    )
    bundles, sg = connect_attributes_and_scene_graph(
        [wall, switch],
        image_arrays={"img0.jpg": rgb},
        depth_maps={"img0.jpg": depth},
    )
    assert len(bundles) == 2
    assert any(a.attribute_name == "dominant_color" and a.provenance == "observed" for a in bundles[1].attributes)
    assert sg.observed_count == 1
    assert sg.edges[0].source_id == "gdn:img0:1"
    assert sg.edges[0].target_id == "gdn:img0:0"
