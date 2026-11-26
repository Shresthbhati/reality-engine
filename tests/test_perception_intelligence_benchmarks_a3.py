"""Tests for A3.11 Perception Benchmark & Evaluation Framework."""

from __future__ import annotations

import json

from benchmarks.perception.evaluator import (
    EvaluationMode,
    evaluate_cross_view_identity_accuracy,
    run_perception_benchmark,
)
from benchmarks.perception.indoor_dataset_adapter import GroundTruthBox, load_coco_indoor_annotations
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox
from perception.intelligence.data_model import CrossViewIdentityHypothesis


def test_unlabelled_real_never_claims_accuracy():
    preds = {
        "img1.jpg": [
            DetectionBox("p1", "window", (10, 10, 100, 120), 0.75),
            DetectionBox("p2", "window", (12, 12, 102, 122), 0.55),  # duplicate
        ]
    }
    rep = run_perception_benchmark(
        mode=EvaluationMode.UNLABELLED_REAL,
        dataset_name="south_building",
        predictions_by_image=preds,
        wall_time_s=2.5,
        peak_vram_mb=400.0,
    )
    assert rep.accuracy_claim_allowed is False
    assert rep.precision is None
    assert rep.recall is None
    assert rep.mean_ap is None
    assert rep.ece is None
    assert rep.duplicate_rate == 0.5


def test_synthetic_vs_labelled_real_benchmark_metrics(tmp_path):
    coco = {
        "images": [
            {"id": 1, "file_name": "r1.jpg", "width": 800, "height": 600},
            {"id": 2, "file_name": "r2.jpg", "width": 800, "height": 600},
        ],
        "categories": [
            {"id": 1, "name": "chair"},
            {"id": 2, "name": "door"},
            {"id": 3, "name": "table"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 100, 100, 150]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [300, 50, 120, 350]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [60, 110, 100, 150]},
            {"id": 4, "image_id": 2, "category_id": 3, "bbox": [400, 250, 200, 150]},
        ],
    }
    p = tmp_path / "indoor.json"
    p.write_text(json.dumps(coco), encoding="utf-8")
    _, gt = load_coco_indoor_annotations(p, is_physically_surveyed=False)

    preds = {
        "r1.jpg": [
            DetectionBox("p1", "chair", (52, 102, 148, 248), 0.90),
            DetectionBox("p2", "door", (302, 52, 418, 398), 0.85),
        ],
        "r2.jpg": [
            DetectionBox("p3", "chair", (62, 112, 158, 258), 0.88),
            DetectionBox("p4", "sofa", (400, 250, 600, 400), 0.70),  # FP for table
        ],
    }

    syn = run_perception_benchmark(
        mode=EvaluationMode.SYNTHETIC,
        dataset_name="synth_room",
        predictions_by_image=preds,
        gt_by_image=gt,
        mask_quality_scores=[0.82, 0.78],
    )
    assert syn.accuracy_claim_allowed is False
    assert syn.precision == 0.75
    assert syn.recall == 0.75
    assert syn.mean_ap > 0.5
    assert syn.mask_quality_mean == 0.80
    assert syn.ece is not None

    real_rep = run_perception_benchmark(
        mode=EvaluationMode.LABELLED_REAL,
        dataset_name="labelled_indoor",
        predictions_by_image=preds,
        gt_by_image=gt,
    )
    assert real_rep.accuracy_claim_allowed is True
    assert real_rep.per_class["chair"]["ap"] == 1.0


def test_cross_view_identity_accuracy_metric():
    idents = [
        CrossViewIdentityHypothesis("id1", ("o1", "o2"), "chair", "confirmed_match", 0.9),
        CrossViewIdentityHypothesis("id2", ("o3", "o4"), "chair", "confirmed_match", 0.85),
    ]
    gt = {"o1": "phys_chair_A", "o2": "phys_chair_A", "o3": "phys_chair_A", "o4": "phys_chair_B"}
    acc = evaluate_cross_view_identity_accuracy(idents, gt)
    assert acc == 0.5


def test_sunrgbd_and_scannet_adapters(tmp_path):
    from benchmarks.perception.scannet_sunrgbd_adapter import (
        load_scannet_2d_annotations,
        load_sunrgbd_2d_annotations,
        normalize_indoor_dataset_label,
    )

    assert normalize_indoor_dataset_label("sofa_chair", dataset_kind="sunrgbd") == "chair"
    assert normalize_indoor_dataset_label("shelves", dataset_kind="scannet") == "shelf"

    sun_p = tmp_path / "sun.json"
    sun_p.write_text(json.dumps([
        {"image_id": "sun_01.jpg", "objects": [{"label": "dresser", "bbox_xyxy": [10, 20, 110, 220]}]}
    ]), encoding="utf-8")
    m_sun, gt_sun = load_sunrgbd_2d_annotations(sun_p)
    assert m_sun.is_physically_surveyed is True
    assert gt_sun["sun_01.jpg"][0].canonical_label == "cabinet"

    scan_p = tmp_path / "scan.json"
    scan_p.write_text(json.dumps([
        {"image_id": "scan_01.jpg", "instances": [{"nyu40_label": "bookshelf", "bbox_xywh": [30, 40, 100, 150]}]}
    ]), encoding="utf-8")
    m_scan, gt_scan = load_scannet_2d_annotations(scan_p)
    assert m_scan.dataset_name == "scannetv2"
    assert gt_scan["scan_01.jpg"][0].canonical_label == "shelf"
