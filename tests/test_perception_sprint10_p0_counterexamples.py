"""Sprint 10 P0 Counterexamples and Audit Tests for PR #379.

These tests independently reproduce the exact defects diagnosed during the PR #379 audit:
1. is_class_compatible incorrectly treated parent-child classes (e.g. door and door_handle) as compatible.
2. _evaluate_parent_child_links penalized unparented components as false positives.
3. _evaluate_parent_child_links selected the best IoU box before checking claimed status, blinding it to alternative valid GT targets.
4. _evaluate_physical_identity_clustering reported 1.0 purity for 100% fragmented singleton tracks.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
from benchmarks.perception.component_benchmark import (
    ComponentBenchmarkEvaluator,
    EvaluationMode,
    GroundTruthComponent,
    is_class_compatible,
)
from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox


def test_audit_defect_1_class_compatibility_must_reject_parent_part():
    """Defect 1: Parent-part classes must NOT be compatible for detection or identity."""
    # True synonyms must match
    assert is_class_compatible("door_handle", "door_knob") is True
    assert is_class_compatible("socket", "outlet_plate") is True
    assert is_class_compatible("tap", "faucet") is True

    # Parent-part relationships MUST be rejected
    assert is_class_compatible("door", "door_frame") is False
    assert is_class_compatible("door_frame", "door") is False
    assert is_class_compatible("door", "door_handle") is False
    assert is_class_compatible("door_handle", "door") is False
    assert is_class_compatible("window", "window_frame") is False
    assert is_class_compatible("sink", "faucet") is False


def test_audit_defect_2_unparented_component_not_penalized_when_unlinked():
    """Defect 2: If GT component has no parent, predicting no parent must NOT be a link FP."""
    evaluator = ComponentBenchmarkEvaluator()

    gt_unparented = GroundTruthComponent(
        entity_id="frame_solo",
        class_name="window_frame",
        image_id="img1.jpg",
        box_2d=(100.0, 100.0, 300.0, 300.0),
        parent_entity_id=None,  # No parent in GT
        is_component=True,
    )

    pred_unparented = DetectionBox(
        box_id="p_frame",
        class_name="window_frame",
        box_2d=(100.0, 100.0, 300.0, 300.0),
        confidence=0.9,
        source_image_id="img1.jpg",
        parent_box_id=None,  # Correctly predicted no parent
    )

    # In Defect 2, this produced parent_assignment_fp = 1
    report = evaluator.evaluate(
        ground_truth=[gt_unparented],
        predictions=[pred_unparented],
    )
    metrics = report.results_by_class["window_frame"]
    assert metrics.parent_assignment_fp == 0, "Correctly predicting no parent should not be penalized as FP"


def test_audit_defect_3_greedy_link_assignment_alternative_gt_recovery():
    """Defect 3: Second prediction should be able to claim a secondary unclaimed GT box."""
    evaluator = ComponentBenchmarkEvaluator()

    gt_win1 = GroundTruthComponent(entity_id="win_1", class_name="window", image_id="img1.jpg", box_2d=(100.0, 100.0, 300.0, 300.0))
    gt_win2 = GroundTruthComponent(entity_id="win_2", class_name="window", image_id="img1.jpg", box_2d=(250.0, 100.0, 450.0, 300.0))

    # Two adjacent frames
    gt_frm1 = GroundTruthComponent(entity_id="frm_1", class_name="window_frame", image_id="img1.jpg", box_2d=(100.0, 100.0, 300.0, 300.0), parent_entity_id="win_1", is_component=True)
    gt_frm2 = GroundTruthComponent(entity_id="frm_2", class_name="window_frame", image_id="img1.jpg", box_2d=(250.0, 100.0, 450.0, 300.0), parent_entity_id="win_2", is_component=True)

    pred_win1 = DetectionBox(box_id="pw1", class_name="window", box_2d=(100.0, 100.0, 300.0, 300.0), confidence=0.9, source_image_id="img1.jpg")
    pred_win2 = DetectionBox(box_id="pw2", class_name="window", box_2d=(250.0, 100.0, 450.0, 300.0), confidence=0.9, source_image_id="img1.jpg")

    # Pred frame 1 matches frm_1 best
    pred_frm1 = DetectionBox(box_id="pf1", class_name="window_frame", box_2d=(100.0, 100.0, 300.0, 300.0), confidence=0.95, source_image_id="img1.jpg", parent_box_id="pw1")
    # Pred frame 2 has overlap with frm_1 (e.g. IoU 0.40) and frm_2 (IoU 0.85)
    pred_frm2 = DetectionBox(box_id="pf2", class_name="window_frame", box_2d=(245.0, 100.0, 445.0, 300.0), confidence=0.90, source_image_id="img1.jpg", parent_box_id="pw2")

    report = evaluator.evaluate(
        ground_truth=[gt_win1, gt_win2, gt_frm1, gt_frm2],
        predictions=[pred_win1, pred_win2, pred_frm1, pred_frm2],
    )
    metrics = report.results_by_class["window_frame"]
    assert metrics.parent_assignment_tp == 2, "Both frames should link to their respective parents"
    assert metrics.parent_assignment_fp == 0


def test_audit_defect_4_physical_identity_must_measure_completeness():
    """Defect 4: Evaluating multi-view physical identity must penalize total track fragmentation."""
    evaluator = ComponentBenchmarkEvaluator()

    # One physical door observed in 4 different views
    gt = [
        GroundTruthComponent(entity_id="d_v1", physical_identity_id="door_physical_1", class_name="door", image_id="v1.jpg", box_2d=(10.0, 10.0, 100.0, 200.0)),
        GroundTruthComponent(entity_id="d_v2", physical_identity_id="door_physical_1", class_name="door", image_id="v2.jpg", box_2d=(12.0, 10.0, 102.0, 200.0)),
        GroundTruthComponent(entity_id="d_v3", physical_identity_id="door_physical_1", class_name="door", image_id="v3.jpg", box_2d=(14.0, 10.0, 104.0, 200.0)),
        GroundTruthComponent(entity_id="d_v4", physical_identity_id="door_physical_1", class_name="door", image_id="v4.jpg", box_2d=(16.0, 10.0, 106.0, 200.0)),
    ]

    # Model fails completely to track across views: creates 4 separate singleton tracks
    preds_fragmented = [
        DetectionBox(box_id="p1", class_name="door", box_2d=(10.0, 10.0, 100.0, 200.0), confidence=0.9, source_image_id="v1.jpg", physical_identity_id="cluster_1"),
        DetectionBox(box_id="p2", class_name="door", box_2d=(12.0, 10.0, 102.0, 200.0), confidence=0.9, source_image_id="v2.jpg", physical_identity_id="cluster_2"),
        DetectionBox(box_id="p3", class_name="door", box_2d=(14.0, 10.0, 104.0, 200.0), confidence=0.9, source_image_id="v3.jpg", physical_identity_id="cluster_3"),
        DetectionBox(box_id="p4", class_name="door", box_2d=(16.0, 10.0, 106.0, 200.0), confidence=0.9, source_image_id="v4.jpg", physical_identity_id="cluster_4"),
    ]

    report = evaluator.evaluate(
        ground_truth=gt,
        predictions=preds_fragmented,
        mode=EvaluationMode.PHYSICAL_IDENTITY,
    )

    phys = report.physical_identity_metrics
    assert phys is not None
    # Cluster purity alone was 1.0 (blind to fragmentation)
    # The benchmark must report completeness and F1 to expose the 100% fragmentation failure
    assert "mean_cluster_completeness" in phys
    assert phys["mean_cluster_completeness"] == 0.25, "1 GT entity split into 4 clusters should have completeness 0.25"
    assert "physical_identity_f1" in phys
    assert phys["physical_identity_f1"] < 0.50, "Harmonic mean F1 must be penalized by fragmentation"
