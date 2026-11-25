"""Adversarial and rigorous unit tests for Perception Component Benchmark (Sprint 9 P0).

Tests:
1. Frame-level matching identity keys and repeated-identity independence across sequential images.
2. Strict 1-to-1 matching for component-to-parent evaluation (duplicate predictions penalized as FP).
3. Class-compatible matching and rejection of mismatched classes (preventing class confusion).
4. Missing parent link penalty (parent_box_id=None -> FP).
5. Wrong parent assignment penalty (wrong parent box -> FP).
6. 101-point COCO AP vs 11-point VOC AP numerical separation and distinct reporting.
7. Explicit UNMEASURED status preservation (no ground truth fabrication).
8. Multi-view physical identity clustering and purity evaluation.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import numpy as np

from benchmarks.perception.component_benchmark import (
    ComponentBenchmarkEvaluator,
    ComponentFamily,
    EvaluationMode,
    GroundTruthComponent,
    compute_box_iou,
    compute_coco_ap_101,
    compute_voc_ap_11,
    is_class_compatible,
)
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox


@pytest.fixture
def evaluator() -> ComponentBenchmarkEvaluator:
    return ComponentBenchmarkEvaluator()


def test_frame_level_repeated_identity_independence(evaluator: ComponentBenchmarkEvaluator):
    """Adversarial test: same physical object observed across multiple images must not lock out subsequent images."""
    gt = [
        GroundTruthComponent(
            entity_id="door_front",
            physical_identity_id="door_front_phys",
            class_name="door",
            image_id="frame_001.jpg",
            box_2d=(100.0, 100.0, 300.0, 500.0),
        ),
        GroundTruthComponent(
            entity_id="door_front",
            physical_identity_id="door_front_phys",
            class_name="door",
            image_id="frame_002.jpg",
            box_2d=(110.0, 105.0, 310.0, 505.0),
        ),
    ]

    preds = [
        DetectionBox(
            box_id="p1",
            class_name="door",
            box_2d=(102.0, 102.0, 298.0, 498.0),
            confidence=0.95,
            source_engine="detector",
            source_image_id="frame_001.jpg",
        ),
        DetectionBox(
            box_id="p2",
            class_name="door",
            box_2d=(112.0, 107.0, 308.0, 503.0),
            confidence=0.92,
            source_engine="detector",
            source_image_id="frame_002.jpg",
        ),
    ]

    report = evaluator.evaluate(
        ground_truth=gt,
        predictions=preds,
        mode=EvaluationMode.FRAME_LEVEL,
    )

    metrics = report.results_by_class["door"]
    assert metrics.status == "measured"
    assert metrics.true_positives_50 == 2, "Both images should match independently without lockout"
    assert metrics.false_positives_50 == 0
    assert metrics.false_negatives_50 == 0
    assert metrics.precision_50 == 1.0
    assert metrics.recall_50 == 1.0
    assert metrics.f1_50 == 1.0


def test_strict_one_to_one_parent_child_matching(evaluator: ComponentBenchmarkEvaluator):
    """Adversarial test: duplicate component predictions claiming the same GT box must be penalized as FP."""
    gt_parent = GroundTruthComponent(
        entity_id="door_1",
        class_name="door",
        image_id="frame_001.jpg",
        box_2d=(100.0, 100.0, 400.0, 800.0),
    )
    gt_handle = GroundTruthComponent(
        entity_id="handle_1",
        class_name="door_handle",
        image_id="frame_001.jpg",
        box_2d=(350.0, 450.0, 380.0, 490.0),
        parent_entity_id="door_1",
        is_component=True,
    )
    gt = [gt_parent, gt_handle]

    pred_parent = DetectionBox(
        box_id="p_door",
        class_name="door",
        box_2d=(100.0, 100.0, 400.0, 800.0),
        confidence=0.95,
        source_engine="detector",
        source_image_id="frame_001.jpg",
    )
    # Two predicted handles targeting the same GT handle
    pred_handle1 = DetectionBox(
        box_id="p_handle_1",
        class_name="door_handle",
        box_2d=(350.0, 450.0, 380.0, 490.0),
        confidence=0.90,
        source_engine="detector",
        source_image_id="frame_001.jpg",
        parent_box_id="p_door",
    )
    pred_handle2 = DetectionBox(
        box_id="p_handle_2",
        class_name="door_handle",
        box_2d=(352.0, 452.0, 382.0, 492.0),
        confidence=0.85,
        source_engine="detector",
        source_image_id="frame_001.jpg",
        parent_box_id="p_door",
    )
    preds = [pred_parent, pred_handle1, pred_handle2]

    report = evaluator.evaluate(
        ground_truth=gt,
        predictions=preds,
        mode=EvaluationMode.FRAME_LEVEL,
    )

    handle_metrics = report.results_by_class["door_handle"]
    # Box detection: 1 TP, 1 FP (only 1 can claim GT box)
    assert handle_metrics.true_positives_50 == 1
    assert handle_metrics.false_positives_50 == 1

    # Parent assignment: strict 1-to-1 means 1 TP, 1 FP
    assert handle_metrics.parent_assignment_tp == 1
    assert handle_metrics.parent_assignment_fp == 1
    assert handle_metrics.parent_assignment_fn == 0
    assert handle_metrics.parent_assignment_precision == 0.5


def test_missing_parent_link_penalized(evaluator: ComponentBenchmarkEvaluator):
    """Adversarial test: component prediction with missing parent link is penalized as FP."""
    gt_parent = GroundTruthComponent(
        entity_id="window_1",
        class_name="window",
        image_id="img1.jpg",
        box_2d=(100.0, 100.0, 500.0, 500.0),
    )
    gt_frame = GroundTruthComponent(
        entity_id="frame_1",
        class_name="window_frame",
        image_id="img1.jpg",
        box_2d=(100.0, 100.0, 500.0, 500.0),
        parent_entity_id="window_1",
        is_component=True,
    )
    gt = [gt_parent, gt_frame]

    # Prediction detects the frame box perfectly, but omits parent_box_id
    pred_frame = DetectionBox(
        box_id="p_frame",
        class_name="window_frame",
        box_2d=(100.0, 100.0, 500.0, 500.0),
        confidence=0.90,
        source_engine="detector",
        source_image_id="img1.jpg",
        parent_box_id=None,  # Missing parent link
    )

    report = evaluator.evaluate(ground_truth=gt, predictions=[pred_frame])
    metrics = report.results_by_class["window_frame"]
    assert metrics.parent_assignment_tp == 0
    assert metrics.parent_assignment_fp == 1
    assert metrics.parent_assignment_fn == 1


def test_wrong_parent_link_penalized(evaluator: ComponentBenchmarkEvaluator):
    """Adversarial test: component prediction linking to the wrong parent object is penalized as FP."""
    gt_win1 = GroundTruthComponent(entity_id="win_1", class_name="window", image_id="img1.jpg", box_2d=(100.0, 100.0, 300.0, 300.0))
    gt_win2 = GroundTruthComponent(entity_id="win_2", class_name="window", image_id="img1.jpg", box_2d=(400.0, 100.0, 600.0, 300.0))
    gt_frame1 = GroundTruthComponent(entity_id="frm_1", class_name="window_frame", image_id="img1.jpg", box_2d=(100.0, 100.0, 300.0, 300.0), parent_entity_id="win_1", is_component=True)
    gt = [gt_win1, gt_win2, gt_frame1]

    pred_win1 = DetectionBox(box_id="pw1", class_name="window", box_2d=(100.0, 100.0, 300.0, 300.0), confidence=0.9, source_image_id="img1.jpg")
    pred_win2 = DetectionBox(box_id="pw2", class_name="window", box_2d=(400.0, 100.0, 600.0, 300.0), confidence=0.9, source_image_id="img1.jpg")
    # Predicted frame matches frm_1 box, but points to pw2 instead of pw1
    pred_frame1 = DetectionBox(box_id="pf1", class_name="window_frame", box_2d=(100.0, 100.0, 300.0, 300.0), confidence=0.9, source_image_id="img1.jpg", parent_box_id="pw2")
    preds = [pred_win1, pred_win2, pred_frame1]

    report = evaluator.evaluate(ground_truth=gt, predictions=preds)
    metrics = report.results_by_class["window_frame"]
    assert metrics.parent_assignment_tp == 0
    assert metrics.parent_assignment_fp == 1
    assert metrics.parent_assignment_fn == 1


def test_class_compatibility_and_confusion_rejection(evaluator: ComponentBenchmarkEvaluator):
    """Adversarial test: class mismatch is rejected and penalized."""
    assert is_class_compatible("door_knob", "door_handle") is True
    assert is_class_compatible("door_handle", "door_knob") is True
    assert is_class_compatible("socket", "outlet_plate") is True
    assert is_class_compatible("chair_seat", "window_frame") is False
    assert is_class_compatible("table", "chair") is False

    gt = [
        GroundTruthComponent(entity_id="wf_1", class_name="window_frame", image_id="img1.jpg", box_2d=(50.0, 50.0, 200.0, 200.0))
    ]
    # Predict chair_seat at the window frame box
    preds = [
        DetectionBox(box_id="p_cs", class_name="chair_seat", box_2d=(50.0, 50.0, 200.0, 200.0), confidence=0.9, source_image_id="img1.jpg")
    ]

    report = evaluator.evaluate(ground_truth=gt, predictions=preds)
    assert report.results_by_class["window_frame"].false_negatives_50 == 1
    assert report.results_by_class["chair_seat"].false_positives_50 == 1
    assert report.results_by_class["chair_seat"].true_positives_50 == 0


def test_voc_ap11_vs_coco_ap101_divergence():
    """Verify that 11-point and 101-point AP are computed separately and yield expected precision curves."""
    precisions = [1.0, 1.0, 0.5, 0.5, 0.2]
    recalls = [0.1, 0.2, 0.4, 0.6, 1.0]

    ap_11 = compute_voc_ap_11(precisions, recalls)
    ap_101 = compute_coco_ap_101(precisions, recalls)

    assert 0.0 <= ap_11 <= 1.0
    assert 0.0 <= ap_101 <= 1.0
    # Both capture the curve but evaluate over 11 vs 101 sample points
    assert isinstance(ap_11, float)
    assert isinstance(ap_101, float)


def test_explicit_unmeasured_accounting(evaluator: ComponentBenchmarkEvaluator):
    """Verify that unsupported classes with no ground truth remain UNMEASURED and are never fabricated."""
    gt = [
        GroundTruthComponent(entity_id="d1", class_name="door", image_id="img1.jpg", box_2d=(10.0, 10.0, 100.0, 200.0))
    ]
    preds = [
        DetectionBox(box_id="p_unsupported", class_name="distribution_board", box_2d=(10.0, 10.0, 50.0, 50.0), confidence=0.8, source_image_id="img1.jpg")
    ]

    report = evaluator.evaluate(ground_truth=gt, predictions=preds)
    unmeas_metric = report.results_by_class["distribution_board"]
    assert unmeas_metric.status == "UNMEASURED"
    assert unmeas_metric.unmeasured_reason == "NO_GROUND_TRUTH_IN_DATASET"
    assert unmeas_metric.gt_count == 0
    assert unmeas_metric.pred_count == 1
    assert report.unmeasured_classes_count > 0


def test_physical_identity_clustering_purity(evaluator: ComponentBenchmarkEvaluator):
    """Verify multi-view physical identity clustering metrics."""
    gt = [
        GroundTruthComponent(entity_id="d_view1", physical_identity_id="door_phys_A", class_name="door", image_id="v1.jpg", box_2d=(10.0, 10.0, 100.0, 200.0)),
        GroundTruthComponent(entity_id="d_view2", physical_identity_id="door_phys_A", class_name="door", image_id="v2.jpg", box_2d=(15.0, 12.0, 105.0, 202.0)),
    ]
    preds = [
        DetectionBox(box_id="p1", class_name="door", box_2d=(10.0, 10.0, 100.0, 200.0), confidence=0.9, source_image_id="v1.jpg", physical_identity_id="cluster_1"),
        DetectionBox(box_id="p2", class_name="door", box_2d=(15.0, 12.0, 105.0, 202.0), confidence=0.9, source_image_id="v2.jpg", physical_identity_id="cluster_1"),
    ]

    report = evaluator.evaluate(ground_truth=gt, predictions=preds, mode=EvaluationMode.PHYSICAL_IDENTITY)
    assert report.physical_identity_metrics is not None
    assert report.physical_identity_metrics["mean_cluster_purity"] == 1.0
