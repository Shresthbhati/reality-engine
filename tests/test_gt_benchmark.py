"""Tests for COCO + multi-view identity + scene-graph GT benchmark evaluator (PR-C)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from perception.evaluation.gt_benchmark import (
    GTBoxAnnotation,
    PredBoxAnnotation,
    PredRelationship,
    evaluate_detection_coco,
    evaluate_multiview_identity,
    evaluate_scene_graph,
    load_coco_gt_bundle,
)


def test_coco_ap_small_recall_and_mask_iou() -> None:
    m_gt = np.zeros((32, 32), dtype=bool)
    m_gt[4:16, 4:16] = True
    m_pred = np.zeros((32, 32), dtype=bool)
    m_pred[4:16, 4:16] = True

    gt = [
        GTBoxAnnotation("img0", "chair", (10.0, 10.0, 110.0, 110.0), track_id="t1", mask=m_gt, is_small=False),
        GTBoxAnnotation("img0", "switch", (200.0, 200.0, 220.0, 224.0), track_id="t2", is_small=True),
    ]
    preds = [
        PredBoxAnnotation("img0", "chair", (10.0, 10.0, 110.0, 110.0), score=0.95, track_id="p1", mask=m_pred),
        # Missed the small switch!
    ]
    rep = evaluate_detection_coco(gt, preds)
    assert rep.per_class_ap50["chair"] == 1.0
    assert rep.per_class_ap50["switch"] == 0.0
    assert rep.ap50 == 0.5
    assert rep.small_object_recall50 == 0.0
    assert rep.mean_mask_iou == 1.0


def test_multiview_identity_detects_id_switch() -> None:
    gt = [
        GTBoxAnnotation("v0", "door", (0.0, 0.0, 100.0, 200.0), track_id="door_A"),
        GTBoxAnnotation("v1", "door", (10.0, 0.0, 110.0, 200.0), track_id="door_A"),
    ]
    preds = [
        PredBoxAnnotation("v0", "door", (0.0, 0.0, 100.0, 200.0), score=0.9, track_id="pred_1"),
        PredBoxAnnotation("v1", "door", (10.0, 0.0, 110.0, 200.0), score=0.9, track_id="pred_2"),
    ]
    rep = evaluate_multiview_identity(gt, preds)
    assert rep.id_switches == 1
    assert rep.recall == 0.0


def test_gt_fixture_bundle_loads_and_evaluates() -> None:
    gt_boxes, gt_rels = load_coco_gt_bundle(Path("datasets/gt_perception_benchmark/annotations.json"))
    assert len(gt_boxes) == 11
    assert len(gt_rels) == 3

    preds = [
        PredBoxAnnotation(g.image_id, g.category, g.bbox_xyxy, score=0.9, track_id=g.track_id)
        for g in gt_boxes
    ]
    det_rep = evaluate_detection_coco(gt_boxes, preds)
    id_rep = evaluate_multiview_identity(gt_boxes, preds)
    sg_rep = evaluate_scene_graph(
        gt_rels,
        [PredRelationship(r.source_track_id, r.target_track_id, r.kind) for r in gt_rels],
    )
    assert det_rep.ap50 == 1.0
    assert det_rep.ap50_95 == 1.0
    assert det_rep.small_object_recall50 == 1.0
    assert id_rep.f1 == 1.0
    assert id_rep.id_switches == 0
    assert sg_rep.f1 == 1.0
