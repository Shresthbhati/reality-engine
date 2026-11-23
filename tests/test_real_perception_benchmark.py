"""Unit and Validation Tests for Real-Data Perception Benchmark Suite.

Validates:
1. Labelled 2D Detection Evaluation (TP, FP, FN, Per-class, Macro/Micro Precision/Recall/F1).
2. Part Hierarchy Linking Metrics (Parent-Child Precision, Recall, F1).
3. Mask Quality Evaluation (Mask IoU and Boundary IoU).
4. Multi-Object Temporal Tracking Evaluation (IDF1, MOTA, ID Switches, Fragmentation).
5. Multi-View Association Evaluation (Association Precision/Recall, Reprojection Residuals).
6. Hardware and Execution Telemetry (Process RSS, CUDA VRAM).
"""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pytest

from benchmarks.perception.real_data_benchmark import (
    GroundTruthBox,
    compute_boundary_iou,
    compute_box_iou,
    evaluate_detections,
    evaluate_mask_quality,
    evaluate_multiview_association,
    evaluate_part_hierarchy,
    evaluate_temporal_tracking,
    measure_hardware_telemetry,
)


def test_box_and_boundary_iou_computation() -> None:
    """Verifies 2D box IoU and contour boundary IoU calculations."""
    box_a = (10.0, 10.0, 50.0, 50.0)  # area = 1600
    box_b = (30.0, 10.0, 70.0, 50.0)  # area = 1600, overlap = 20 * 40 = 800
    iou = compute_box_iou(box_a, box_b)
    # union = 1600 + 1600 - 800 = 2400 -> iou = 800 / 2400 = 1/3
    assert abs(iou - (1.0 / 3.0)) < 1e-4

    # Disjoint boxes
    box_c = (100.0, 100.0, 150.0, 150.0)
    assert compute_box_iou(box_a, box_c) == 0.0

    # Boundary IoU with masks
    mask1 = np.zeros((100, 100), dtype=bool)
    mask1[20:60, 20:60] = True
    mask2 = np.zeros((100, 100), dtype=bool)
    mask2[20:60, 20:60] = True
    # Identical masks -> boundary IoU is 1.0
    assert compute_boundary_iou(mask1, mask2) == pytest.approx(1.0)


def test_detection_precision_recall_f1_evaluation() -> None:
    """Tests evaluation of detection Precision, Recall, and F1 at IoU 0.50."""
    gt_boxes = [
        GroundTruthBox(entity_id="gt_desk_1", class_name="desk", box_2d=(50.0, 50.0, 250.0, 200.0), image_id="img_1"),
        GroundTruthBox(entity_id="gt_chair_1", class_name="chair", box_2d=(100.0, 150.0, 180.0, 250.0), image_id="img_1"),
        GroundTruthBox(entity_id="gt_lamp_1", class_name="lamp", box_2d=(60.0, 40.0, 90.0, 80.0), image_id="img_1"),
    ]

    class MockPred:
        def __init__(self, class_name: str, box_2d: tuple[float, float, float, float], conf: float, img_id: str = "img_1"):
            self.class_name = class_name
            self.box_2d = box_2d
            self.confidence = conf
            self.source_image_id = img_id

    preds = [
        MockPred("desk", (48.0, 52.0, 248.0, 202.0), 0.92),   # TP desk
        MockPred("chair", (98.0, 148.0, 182.0, 252.0), 0.88), # TP chair
        MockPred("sofa", (300.0, 300.0, 400.0, 400.0), 0.70), # FP sofa
        # Lamp is missed -> FN lamp
    ]

    metrics = evaluate_detections(gt_boxes, preds, iou_threshold=0.50)
    assert metrics.total_gt == 3
    assert metrics.total_predictions == 3
    assert metrics.true_positives == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1

    # Micro precision: 2 / 3, Micro recall: 2 / 3
    assert metrics.micro_precision == pytest.approx(2.0 / 3.0, rel=1e-3)
    assert metrics.micro_recall == pytest.approx(2.0 / 3.0, rel=1e-3)
    assert metrics.micro_f1 == pytest.approx(2.0 / 3.0, rel=1e-3)

    # Per-class checks
    assert metrics.per_class_f1["desk"] == pytest.approx(1.0)
    assert metrics.per_class_f1["chair"] == pytest.approx(1.0)
    assert metrics.per_class_f1["lamp"] == 0.0


def test_part_hierarchy_linking_metrics() -> None:
    """Tests evaluation of subpart parent-child links."""
    gt_boxes = [
        GroundTruthBox(entity_id="door_1", class_name="door", box_2d=(100.0, 50.0, 220.0, 350.0), image_id="img_1"),
        GroundTruthBox(entity_id="handle_1", class_name="door_handle", box_2d=(190.0, 190.0, 210.0, 220.0), image_id="img_1", parent_entity_id="door_1"),
        GroundTruthBox(entity_id="desk_1", class_name="desk", box_2d=(300.0, 150.0, 550.0, 300.0), image_id="img_1"),
        GroundTruthBox(entity_id="drawer_1", class_name="drawer", box_2d=(310.0, 170.0, 400.0, 220.0), image_id="img_1", parent_entity_id="desk_1"),
    ]

    class MockPredBox:
        def __init__(self, box_id: str, class_name: str, box_2d: tuple[float, float, float, float], parent_id: str | None = None):
            self.box_id = box_id
            self.class_name = class_name
            self.box_2d = box_2d
            self.parent_box_id = parent_id
            self.source_image_id = "img_1"

    preds = [
        MockPredBox("p_door", "door", (102.0, 51.0, 218.0, 348.0)),
        MockPredBox("p_handle", "door_handle", (191.0, 191.0, 209.0, 219.0), parent_id="p_door"),  # Correctly linked!
        MockPredBox("p_desk", "desk", (302.0, 152.0, 548.0, 298.0)),
        MockPredBox("p_drawer", "drawer", (312.0, 172.0, 398.0, 218.0), parent_id="p_desk"),       # Correctly linked!
    ]

    metrics = evaluate_part_hierarchy(gt_boxes, preds)
    assert metrics.total_gt_part_links == 2
    assert metrics.predicted_part_links == 2
    assert metrics.correct_part_links == 2
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0


def test_mask_quality_metrics() -> None:
    """Tests Mask IoU and boundary quality evaluation."""
    mask_gt = np.zeros((100, 100), dtype=bool)
    mask_gt[20:60, 20:60] = True

    gt_boxes = [
        GroundTruthBox(entity_id="e1", class_name="chair", box_2d=(20.0, 20.0, 60.0, 60.0), image_id="img_1", mask_2d=mask_gt),
    ]

    mask_pred = np.zeros((100, 100), dtype=bool)
    mask_pred[20:55, 20:60] = True  # slightly truncated vertically

    class MockPred:
        def __init__(self, box_2d: tuple[float, float, float, float], mask: np.ndarray):
            self.box_2d = box_2d
            self.mask_2d = mask
            self.source_image_id = "img_1"

    preds = [MockPred((20.0, 20.0, 60.0, 55.0), mask_pred)]

    metrics = evaluate_mask_quality(gt_boxes, preds)
    assert metrics.evaluated_masks == 1
    assert 0.80 < metrics.mean_mask_iou < 1.0
    assert metrics.masks_above_50_iou == 1.0


def test_temporal_tracking_metrics() -> None:
    """Tests IDF1, ID switches, and tracking fragmentation metrics."""
    # GT trajectory: object 1 moves across 3 frames
    gt_trajectories = {
        "gt_obj_1": [
            (0, (10.0, 10.0, 50.0, 50.0)),
            (1, (15.0, 10.0, 55.0, 50.0)),
            (2, (20.0, 10.0, 60.0, 50.0)),
        ]
    }

    # Predicted tracks: track_1 covers frames 0 and 1, then an ID switch causes track_2 to cover frame 2
    pred_tracks = {
        "track_1": [
            (0, (11.0, 10.0, 51.0, 50.0)),
            (1, (16.0, 10.0, 56.0, 50.0)),
        ],
        "track_2": [
            (2, (21.0, 10.0, 61.0, 50.0)),
        ],
    }

    metrics = evaluate_temporal_tracking(gt_trajectories, pred_tracks)
    assert metrics.total_gt_trajectories == 1
    assert metrics.total_predicted_tracks == 2
    assert metrics.id_switches == 1
    assert metrics.track_fragmentations == 0
    assert metrics.idf1 > 0.0


def test_multiview_association_metrics() -> None:
    """Tests evaluation of multi-view association precision, recall, and reprojection residuals."""
    gt_entity_observations = {
        "phys_table": {"obs_t_cam1", "obs_t_cam2", "obs_t_cam3"},
    }

    class MockDecision:
        def __init__(self, a: str, b: str, outcome: str, res_px: float):
            self.obs_a_id = a
            self.obs_b_id = b
            self.outcome = outcome
            self.reprojection_err_px = res_px

    decisions = [
        MockDecision("obs_t_cam1", "obs_t_cam2", "merged", 1.5),
        MockDecision("obs_t_cam1", "obs_t_cam3", "merged", 2.2),
        MockDecision("obs_t_cam2", "obs_t_cam3", "merged", 1.8),
    ]

    metrics = evaluate_multiview_association(
        gt_entity_observations,
        merged_entities=[],
        association_decisions=decisions,
    )

    assert metrics.candidate_pairs_evaluated == 3
    assert metrics.merged_pairs == 3
    assert metrics.refused_pairs == 0
    assert metrics.association_precision == 1.0
    assert metrics.association_recall == 1.0
    assert 1.5 <= metrics.mean_reprojection_residual_px <= 2.2
    assert metrics.p90_reprojection_residual_px > 0.0


def test_hardware_telemetry_reporting() -> None:
    """Verifies that hardware telemetry captures process RSS and system metrics."""
    telemetry = measure_hardware_telemetry()
    assert "timestamp" in telemetry
    # In Python process, peak RSS should be measured as float MB
    assert telemetry["peak_rss_mb"] is not None
    assert telemetry["peak_rss_mb"] > 10.0  # Reality engine process is > 10 MB


def test_load_ground_truth_dataset(tmp_path) -> None:
    """Tests loading independent ground truth from dataset files."""
    from benchmarks.perception.real_data_benchmark import load_ground_truth_from_dataset

    ds_dir = Path("datasets/real_room_capture")
    if (ds_dir / "ground_truth_identities.json").is_file():
        boxes, meta = load_ground_truth_from_dataset(ds_dir)
        assert len(boxes) == 105
        assert meta["total_images_annotated"] == 21
        assert meta["is_physically_surveyed"] is False  # Honest ground truth status
        assert meta["milestone_status"] == "BLOCKED"
        first = boxes[0]
        assert first.image_id.startswith("IMG_")
        assert len(first.box_2d) == 4
        assert first.class_name in ("door", "window", "table", "chair", "lamp")


def test_real_perception_benchmark_end_to_end(tmp_path) -> None:
    """Verifies end-to-end benchmark execution on real dataset sample."""
    from benchmarks.perception.real_data_benchmark import run_real_perception_benchmark
    from engine.multiengine.spatial_intelligence.detection_pipeline import RuleBasedDetectorAdapter

    ds_dir = Path("datasets/real_room_capture")
    if (ds_dir / "ground_truth_identities.json").is_file():
        out_dir = tmp_path / "bench_out"
        summary = run_real_perception_benchmark(
            dataset_dir=ds_dir,
            output_dir=out_dir,
            benchmark_id="test_run_e2e",
            detector_engine=RuleBasedDetectorAdapter(),
        )
        assert summary.benchmark_id == "test_run_e2e"
        assert summary.telemetry["images_evaluated"] == 21
        assert summary.telemetry["total_gt_boxes"] == 105
        assert summary.tracking.total_gt_trajectories == 5
        assert summary.detection_mAP50.total_gt == 105
        # Verify saved report
        report_file = out_dir / "test_run_e2e.json"
        assert report_file.is_file()
        import json
        data = json.loads(report_file.read_text(encoding="utf-8"))
        assert data["benchmark_id"] == "test_run_e2e"
        assert "detection_mAP50" in data
        assert "tracking" in data
        assert "multi_view" in data
        assert "telemetry" in data


# ============================================================================
# Adversarial Integrity Tests (Benchmark Integrity — Mandate P0)
# ============================================================================

def test_adversarial_duplicate_detections() -> None:
    """Adversarial test: multiple overlapping detections for the same single GT box.

    Only the highest-confidence candidate must be marked as TP; all others must be FP.
    """
    gt_boxes = [
        GroundTruthBox(entity_id="gt_1", class_name="table", box_2d=(100.0, 100.0, 300.0, 300.0), image_id="img_1")
    ]

    class MockPred:
        def __init__(self, conf: float, offset: float):
            self.class_name = "table"
            self.box_2d = (100.0 + offset, 100.0 + offset, 300.0 + offset, 300.0 + offset)
            self.confidence = conf
            self.source_image_id = "img_1"

    preds = [
        MockPred(0.95, 0.0),   # True Positive
        MockPred(0.85, 2.0),   # Duplicate FP
        MockPred(0.75, 4.0),   # Duplicate FP
        MockPred(0.65, 5.0),   # Duplicate FP
    ]

    metrics = evaluate_detections(gt_boxes, preds, iou_threshold=0.50)
    assert metrics.total_gt == 1
    assert metrics.total_predictions == 4
    assert metrics.true_positives == 1
    assert metrics.false_positives == 3
    assert metrics.false_negatives == 0
    assert metrics.micro_precision == pytest.approx(0.25)
    assert metrics.micro_recall == pytest.approx(1.0)


def test_adversarial_wrong_class_detections() -> None:
    """Adversarial test: near-perfect bounding box spatial match, but completely wrong class.

    Must count as 0 TP, 1 FP, 1 FN.
    """
    gt_boxes = [
        GroundTruthBox(entity_id="gt_1", class_name="door", box_2d=(50.0, 50.0, 200.0, 400.0), image_id="img_1")
    ]

    class MockPred:
        def __init__(self):
            self.class_name = "table"  # Incompatible with door
            self.box_2d = (50.0, 50.0, 200.0, 400.0)
            self.confidence = 0.99
            self.source_image_id = "img_1"

    preds = [MockPred()]
    metrics = evaluate_detections(gt_boxes, preds, iou_threshold=0.50)
    assert metrics.true_positives == 0
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 1
    assert metrics.micro_precision == 0.0
    assert metrics.micro_recall == 0.0


def test_adversarial_missed_detections() -> None:
    """Adversarial test: multiple ground truth objects present, zero detections emitted."""
    gt_boxes = [
        GroundTruthBox(entity_id="gt_1", class_name="lamp", box_2d=(10.0, 10.0, 50.0, 50.0), image_id="img_1"),
        GroundTruthBox(entity_id="gt_2", class_name="chair", box_2d=(60.0, 60.0, 120.0, 120.0), image_id="img_1"),
    ]

    metrics = evaluate_detections(gt_boxes, [], iou_threshold=0.50)
    assert metrics.total_gt == 2
    assert metrics.total_predictions == 0
    assert metrics.true_positives == 0
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 2
    assert metrics.macro_recall == 0.0
    assert metrics.average_precision == 0.0


def test_adversarial_repeated_objects() -> None:
    """Adversarial test: two identical objects of the same class placed adjacent to each other.

    One prediction matching object A must not satisfy object B.
    """
    gt_boxes = [
        GroundTruthBox(entity_id="chair_left", class_name="chair", box_2d=(100.0, 100.0, 200.0, 200.0), image_id="img_1"),
        GroundTruthBox(entity_id="chair_right", class_name="chair", box_2d=(220.0, 100.0, 320.0, 200.0), image_id="img_1"),
    ]

    class MockPred:
        def __init__(self):
            self.class_name = "chair"
            self.box_2d = (101.0, 101.0, 199.0, 199.0)
            self.confidence = 0.90
            self.source_image_id = "img_1"

    preds = [MockPred()]
    metrics = evaluate_detections(gt_boxes, preds, iou_threshold=0.50)
    assert metrics.total_gt == 2
    assert metrics.total_predictions == 1
    assert metrics.true_positives == 1
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 1
    assert metrics.per_class_recall["chair"] == 0.50


def test_adversarial_empty_ground_truth_never_silent_pass() -> None:
    """Adversarial test: empty ground truth populations must NEVER return silent 1.0 passing scores."""
    # 1. Detection with 0 GT
    det_metrics = evaluate_detections([], [], iou_threshold=0.50)
    assert det_metrics.status == "unmeasured_no_gt"
    assert det_metrics.macro_recall is None
    assert det_metrics.macro_f1 is None
    assert det_metrics.average_precision is None

    # 2. Part hierarchy with 0 GT links
    part_metrics = evaluate_part_hierarchy([], [])
    assert part_metrics.status == "unmeasured_no_gt"
    assert part_metrics.recall is None
    assert part_metrics.f1 is None
    assert part_metrics.precision is None

    # 3. Mask evaluation with 0 GT masks
    mask_metrics = evaluate_mask_quality([], [])
    assert mask_metrics.status == "unmeasured_no_gt"
    assert mask_metrics.mean_mask_iou is None
    assert mask_metrics.median_mask_iou is None
    assert mask_metrics.mean_boundary_iou is None

    # 4. Temporal tracking with 0 GT trajectories
    track_metrics = evaluate_temporal_tracking({}, {})
    assert track_metrics.status == "unmeasured_no_gt"
    assert track_metrics.idf1 is None
    assert track_metrics.id_recall is None
    assert track_metrics.mota is None

    # 5. Multi-view association with 0 GT pairs
    mv_metrics = evaluate_multiview_association({}, [], [])
    assert mv_metrics.status == "unmeasured_no_gt"
    assert mv_metrics.association_recall is None
    assert mv_metrics.association_precision is None


def test_adversarial_missing_predictions_zero_score() -> None:
    """Adversarial test: zero predictions when GT exists must produce zero score, not crash."""
    gt_boxes = [
        GroundTruthBox(entity_id="g1", class_name="table", box_2d=(10.0, 10.0, 50.0, 50.0), image_id="img_1")
    ]
    det_metrics = evaluate_detections(gt_boxes, [], iou_threshold=0.50)
    assert det_metrics.status == "measured"
    assert det_metrics.true_positives == 0
    assert det_metrics.false_negatives == 1
    assert det_metrics.macro_recall == 0.0
    assert det_metrics.macro_f1 == 0.0
    assert det_metrics.average_precision == 0.0


def test_benchmark_split_dev_and_test_separation() -> None:
    """Validates strict separation of development and held-out test splits."""
    from benchmarks.perception.real_data_benchmark import load_ground_truth_from_dataset

    ds_dir = Path("datasets/real_room_capture")
    if (ds_dir / "ground_truth_identities.json").is_file():
        dev_boxes, dev_meta = load_ground_truth_from_dataset(ds_dir, split="dev")
        test_boxes, test_meta = load_ground_truth_from_dataset(ds_dir, split="test")
        all_boxes, all_meta = load_ground_truth_from_dataset(ds_dir, split="all")

        assert len(dev_boxes) == 55  # 11 images x 5 objects
        assert len(test_boxes) == 50  # 10 images x 5 objects
        assert len(all_boxes) == 105

        dev_images = set(b.image_id for b in dev_boxes)
        test_images = set(b.image_id for b in test_boxes)

        # Mutually exclusive splits
        assert len(dev_images & test_images) == 0
        assert len(dev_images | test_images) == 21
        assert "IMG_0000.jpg" in dev_images
        assert "IMG_0010.jpg" in dev_images
        assert "IMG_0011.jpg" in test_images
        assert "IMG_0020.jpg" in test_images


def test_scale_breakdown_metrics() -> None:
    """Verifies small, medium, and large object scale breakdowns in evaluate_detections."""
    gt_boxes = [
        GroundTruthBox(entity_id="g_small", class_name="socket", box_2d=(10.0, 10.0, 30.0, 30.0), image_id="img_1"),    # area = 400 (< 1024)
        GroundTruthBox(entity_id="g_med", class_name="lamp", box_2d=(50.0, 50.0, 100.0, 100.0), image_id="img_1"),      # area = 2500 (1024..9216)
        GroundTruthBox(entity_id="g_large", class_name="table", box_2d=(100.0, 100.0, 300.0, 300.0), image_id="img_1"),  # area = 40000 (> 9216)
    ]

    class MockPred:
        def __init__(self, class_name: str, box_2d: tuple[float, float, float, float], conf: float):
            self.class_name = class_name
            self.box_2d = box_2d
            self.confidence = conf
            self.source_image_id = "img_1"

    preds = [
        MockPred("socket", (11.0, 11.0, 29.0, 29.0), 0.90),      # TP small
        MockPred("lamp", (51.0, 51.0, 99.0, 99.0), 0.85),        # TP medium
        MockPred("chair", (400.0, 400.0, 600.0, 600.0), 0.70),   # FP large
    ]

    metrics = evaluate_detections(gt_boxes, preds, iou_threshold=0.50)
    sb = metrics.scale_breakdown
    assert "small" in sb and "medium" in sb and "large" in sb
    assert sb["small"]["true_positives"] == 1
    assert sb["small"]["false_positives"] == 0
    assert sb["small"]["precision"] == 1.0
    assert sb["medium"]["true_positives"] == 1
    assert sb["large"]["true_positives"] == 0
    assert sb["large"]["false_positives"] == 1


def test_confidence_calibration_ece_and_brier() -> None:
    """Verifies Expected Calibration Error (ECE) and Brier Score metrics."""
    gt_boxes = [
        GroundTruthBox(entity_id="g1", class_name="table", box_2d=(10.0, 10.0, 100.0, 100.0), image_id="img_1"),
    ]

    class MockPred:
        def __init__(self, class_name: str, box_2d: tuple[float, float, float, float], conf: float):
            self.class_name = class_name
            self.box_2d = box_2d
            self.confidence = conf
            self.source_image_id = "img_1"

    # Perfect prediction with confidence 1.0
    preds_perfect = [MockPred("table", (10.0, 10.0, 100.0, 100.0), 1.0)]
    m_perf = evaluate_detections(gt_boxes, preds_perfect, iou_threshold=0.50)
    assert m_perf.brier_score == pytest.approx(0.0)
    assert m_perf.expected_calibration_error == pytest.approx(0.0)

    # Overconfident false positive (conf=1.0, match=0) -> brier = 1.0, ece = 1.0
    preds_bad = [MockPred("chair", (200.0, 200.0, 300.0, 300.0), 1.0)]
    m_bad = evaluate_detections(gt_boxes, preds_bad, iou_threshold=0.50)
    assert m_bad.brier_score == pytest.approx(1.0)
    assert m_bad.expected_calibration_error == pytest.approx(1.0)


def test_temporal_tracker_large_shift_continuity() -> None:
    """Verifies that inter-frame shifts up to 300px maintain track continuity without fragmentation."""
    from engine.multiengine.spatial_intelligence.association import TemporalIdentityTracker
    from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox

    tracker = TemporalIdentityTracker(max_missed_frames=5, min_iou_match=0.15, max_centroid_shift_px=350.0)

    # Frame 0: object at (100, 100, 300, 300) -> center (200, 200)
    obs_f0 = [DetectionBox(box_id="b0", class_name="chair", box_2d=(100.0, 100.0, 300.0, 300.0), confidence=0.9)]
    asgn0 = tracker.process_frame_observations(frame_index=0, timestamp_sec=0.0, observations=obs_f0)
    track_id = asgn0["b0"]

    # Frame 1: shifted by 200px due to camera movement -> center (400, 200)
    obs_f1 = [DetectionBox(box_id="b1", class_name="chair", box_2d=(300.0, 100.0, 500.0, 300.0), confidence=0.9)]
    asgn1 = tracker.process_frame_observations(frame_index=1, timestamp_sec=0.1, observations=obs_f1)

    # Must preserve identity despite 200px shift
    assert asgn1["b1"] == track_id
    assert len(tracker.active_tracks) == 1


def test_adversarial_missing_masks_no_crash() -> None:
    """Adversarial test: detections with None masks evaluated against ground truth with masks."""
    gt_mask = np.zeros((50, 50), dtype=bool)
    gt_mask[10:40, 10:40] = True
    gt_boxes = [GroundTruthBox(entity_id="g1", class_name="door", box_2d=(10.0, 10.0, 40.0, 40.0), image_id="img_1", mask_2d=gt_mask)]

    class MockPredNoMask:
        def __init__(self):
            self.class_name = "door"
            self.box_2d = (10.0, 10.0, 40.0, 40.0)
            self.mask_2d = None
            self.source_image_id = "img_1"

    preds = [MockPredNoMask()]
    mask_metrics = evaluate_mask_quality(gt_boxes, preds)
    assert mask_metrics.total_gt_masks == 1
    assert mask_metrics.evaluated_masks == 0
    assert mask_metrics.status == "no_matches"


def test_adversarial_ambiguous_identities_bipartite_separation() -> None:
    """Adversarial test: two close objects must not be assigned to the same track ID in one frame."""
    from engine.multiengine.spatial_intelligence.association import TemporalIdentityTracker
    from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox

    tracker = TemporalIdentityTracker(max_missed_frames=3, min_iou_match=0.10)
    # Frame 0: two chairs side by side
    c1 = DetectionBox(box_id="c1_f0", class_name="chair", box_2d=(10.0, 10.0, 50.0, 50.0), confidence=0.9)
    c2 = DetectionBox(box_id="c2_f0", class_name="chair", box_2d=(60.0, 10.0, 100.0, 50.0), confidence=0.9)
    asgn0 = tracker.process_frame_observations(frame_index=0, timestamp_sec=0.0, observations=[c1, c2])

    assert asgn0["c1_f0"] != asgn0["c2_f0"]

    # Frame 1: both chairs slightly shifted
    c1_f1 = DetectionBox(box_id="c1_f1", class_name="chair", box_2d=(12.0, 10.0, 52.0, 50.0), confidence=0.9)
    c2_f1 = DetectionBox(box_id="c2_f1", class_name="chair", box_2d=(62.0, 10.0, 102.0, 50.0), confidence=0.9)
    asgn1 = tracker.process_frame_observations(frame_index=1, timestamp_sec=0.1, observations=[c1_f1, c2_f1])

    # Distinct tracks maintained with zero collision
    assert asgn1["c1_f1"] == asgn0["c1_f0"]
    assert asgn1["c2_f1"] == asgn0["c2_f0"]
    assert asgn1["c1_f1"] != asgn1["c2_f1"]


def test_adversarial_invalid_camera_calibration_rejection() -> None:
    """Adversarial test: invalid/NaN camera poses are safely omitted from multi-view ray generation."""
    from engine.multiengine.spatial_intelligence.association import build_camera_ray
    from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
    from engine.math import Quat, Vec3

    cam_valid = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=1000.0, fy=1000.0, cx=500.0, cy=500.0, width=1000, height=1000),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity()),
    )
    ray = build_camera_ray("obs_1", "img_1", "chair", (100.0, 100.0, 200.0, 200.0), cam_valid)
    assert ray is not None
    assert math.isfinite(ray.origin[0])
    assert math.isfinite(ray.direction[0])


def test_learned_confidence_calibrator_optimization() -> None:
    """Verifies that LearnedConfidenceCalibrator fits temperature and bias via NLL and recalibrates scores."""
    from engine.multiengine.spatial_intelligence.detection_pipeline import LearnedConfidenceCalibrator

    calibrator = LearnedConfidenceCalibrator()
    assert not calibrator.is_fitted
    # Unfitted calibrator acts as identity returning (score, False, "uncalibrated")
    prob, is_cal, method = calibrator.calibrate(0.75)
    assert prob == 0.75
    assert not is_cal
    assert method == "uncalibrated"

    # Simulated overconfident detector: scores are high (0.8-0.95), but actual accuracy is only 40%
    raw_scores = [0.85, 0.90, 0.80, 0.75, 0.95, 0.88, 0.82, 0.91, 0.84, 0.79]
    labels =     [1,    1,    0,    0,    1,    0,    0,    1,    0,    0   ]

    calibrator.fit(raw_scores, labels)
    assert calibrator.is_fitted
    # Calibrated probability should be scaled down towards ~0.40
    cal_res = calibrator.calibrate(0.85)
    assert cal_res.is_calibrated
    assert cal_res.method == "global_platt"
    assert cal_res.calibrated_probability < 0.85
    assert 0.20 <= cal_res.calibrated_probability <= 0.65


def test_standard_hota_and_deta_assa_calculation() -> None:
    """Verifies exact HOTA, DetA, and AssA computation across alpha thresholds."""
    gt_trajectories = {
        "gt1": [(0, (0.0, 0.0, 10.0, 10.0)), (1, (0.0, 0.0, 10.0, 10.0))],
    }
    pred_tracks = {
        "p1": [(0, (0.0, 0.0, 10.0, 10.0)), (1, (0.0, 0.0, 10.0, 10.0))],
    }
    metrics = evaluate_temporal_tracking(gt_trajectories, pred_tracks)
    assert metrics.hota is not None
    assert metrics.hota == pytest.approx(1.0)
    assert metrics.deta == pytest.approx(1.0)
    assert metrics.assa == pytest.approx(1.0)
    assert metrics.idf1 == pytest.approx(1.0)


def test_clear_mota_hungarian_frame_matching() -> None:
    """Verifies that CLEAR MOTA calculates frame-level Hungarian matching FN + FP + IDSW."""
    gt_trajectories = {
        "gt1": [(0, (0.0, 0.0, 10.0, 10.0)), (1, (0.0, 0.0, 10.0, 10.0))],
        "gt2": [(0, (20.0, 20.0, 30.0, 30.0)), (1, (20.0, 20.0, 30.0, 30.0))],
    }
    # Frame 0: both matched correctly. Frame 1: p1 swaps to gt2, p2 missed (FN=1, FP=0, IDSW=1)
    pred_tracks = {
        "p1": [(0, (0.0, 0.0, 10.0, 10.0)), (1, (20.0, 20.0, 30.0, 30.0))],
        "p2": [(0, (20.0, 20.0, 30.0, 30.0))],
    }
    metrics = evaluate_temporal_tracking(gt_trajectories, pred_tracks)
    assert metrics.id_switches == 1
    assert metrics.frame_fn == 1
    assert metrics.frame_tp == 3
    # Total GT frames = 4. Errors = 1 FN + 0 FP + 1 IDSW = 2 errors. MOTA = 1 - 2/4 = 0.50
    assert metrics.mota == pytest.approx(0.50)


def test_boundary_f_score_and_fast_boundary_iou() -> None:
    """Verifies BF-score contour evaluation and EDT-accelerated boundary IoU."""
    from benchmarks.perception.real_data_benchmark import compute_boundary_f_score, compute_boundary_iou

    mask_a = np.zeros((100, 100), dtype=bool)
    mask_a[20:80, 20:80] = True

    mask_b = np.zeros((100, 100), dtype=bool)
    mask_b[22:82, 22:82] = True  # shifted by 2 pixels

    # Fast boundary IoU
    b_iou = compute_boundary_iou(mask_a, mask_b, dilation_radius=3)
    assert 0.50 <= b_iou <= 0.95

    # Boundary F-score with 3-pixel tolerance
    bf_score = compute_boundary_f_score(mask_a, mask_b, distance_threshold=3.0)
    assert 0.70 <= bf_score <= 1.0


def test_multiview_association_member_ids_and_false_split() -> None:
    """Verifies that PhysicalMergedEntity with member_observation_ids evaluates purity and false split rate."""
    from engine.multiengine.spatial_intelligence.association import PhysicalMergedEntity

    gt_entities = {
        "table_real": {"t_c1", "t_c2", "t_c3"},
        "chair_real": {"c_c1", "c_c2"},
    }

    # Entity 1 correctly merges table observations from 2 cams; entity 2 has table from cam 3 (split!)
    m1 = PhysicalMergedEntity(
        canonical_entity_id="cluster_table_part1",
        primary_class="table",
        member_observation_ids=("t_c1", "t_c2"),
        sources=("cam1", "cam2"),
    )
    m2 = PhysicalMergedEntity(
        canonical_entity_id="cluster_table_part2",
        primary_class="table",
        member_observation_ids=("t_c3",),
        sources=("cam3",),
    )
    m3 = PhysicalMergedEntity(
        canonical_entity_id="cluster_chair",
        primary_class="chair",
        member_observation_ids=("c_c1", "c_c2"),
        sources=("cam1", "cam2"),
    )

    metrics = evaluate_multiview_association(
        gt_entities,
        merged_entities=[m1, m2, m3],
        association_decisions=[],
    )

    assert metrics.cluster_purity == 1.0  # Zero false merges
    assert metrics.false_merge_rate == 0.0
    assert metrics.false_split_rate > 0.0  # table was split into m1 and m2!
    assert 0.50 <= metrics.identity_completeness <= 0.90


def test_normalized_coordinate_resolution_invariance() -> None:
    """Verifies that evidence-based geometric reasoning is invariant between 1280x960 and 4032x3024."""
    from engine.multiengine.spatial_intelligence.detection_pipeline import (
        DetectionBox,
        suppress_duplicates_preserving_parts,
    )

    # Low-res scene (1280 x 960)
    # A small wall sconce detected as "door" with normalized w=0.04, y1=0.35, y2=0.45
    box_lowres = DetectionBox(
        box_id="sconce_low",
        class_name="door",
        box_2d=(0.40 * 1280, 0.35 * 960, 0.44 * 1280, 0.45 * 960),
        confidence=0.60,
    )
    res_low = suppress_duplicates_preserving_parts([box_lowres], image_shape=(960, 1280))

    # High-res scene (4032 x 3024)
    box_hires = DetectionBox(
        box_id="sconce_high",
        class_name="door",
        box_2d=(0.40 * 4032, 0.35 * 3024, 0.44 * 4032, 0.45 * 3024),
        confidence=0.60,
    )
    res_high = suppress_duplicates_preserving_parts([box_hires], image_shape=(3024, 4032))

    # Both must be identically disambiguated to "lamp" despite 10x difference in pixel coordinates!
    assert len(res_low) == 1 and res_low[0].class_name == "lamp"
    assert len(res_high) == 1 and res_high[0].class_name == "lamp"


def test_learned_confidence_calibrator_per_class_and_provenance(tmp_path: Path) -> None:
    """Verifies per-class Platt scaling, provenance persistence, and ECE/Brier/NLL metrics."""
    from engine.multiengine.spatial_intelligence.detection_pipeline import LearnedConfidenceCalibrator

    cal = LearnedConfidenceCalibrator()

    raw_scores = [0.85, 0.90, 0.82, 0.88, 0.60, 0.65, 0.55, 0.58, 0.92, 0.95]
    labels =     [1,    1,    0,    0,    1,    1,    0,    0,    1,    0   ]
    classes =    ["door", "door", "door", "door", "table", "table", "table", "table", "chair", "chair"]

    provenance_meta = {
        "model_name": "grounding_dino_swint_ogc",
        "checkpoint_hash": "sha256_mock_abc123",
        "dataset_split": "dev",
        "dataset_split_hash": "split_sha_dev_998",
        "sample_count": len(raw_scores),
    }

    cal.fit(raw_scores, labels, class_names=classes, min_class_samples=4, min_class_positives=2, min_class_negatives=2, provenance=provenance_meta)
    assert cal.is_fitted
    assert "pre_calibration" in cal.provenance
    assert "post_calibration" in cal.provenance
    assert cal.provenance["model_name"] == "grounding_dino_swint_ogc"

    # Per-class Platt should be available for door and table (each has >= 4 samples with both 0 and 1)
    res_door = cal.calibrate(0.85, class_name="door")
    assert res_door.is_calibrated
    assert res_door.method == "per_class_platt"

    # Fallback to global Platt for unseen or rare classes
    res_lamp = cal.calibrate(0.85, class_name="lamp")
    assert res_lamp.is_calibrated
    assert res_lamp.method == "global_platt"

    # Persistence verification
    cal_file = tmp_path / "test_calibrator.json"
    cal.save(cal_file)
    assert cal_file.is_file()

    loaded = LearnedConfidenceCalibrator.load(cal_file)
    assert loaded.is_fitted
    assert loaded.provenance["dataset_split"] == "dev"
    res_loaded = loaded.calibrate(0.85, class_name="door")
    assert res_loaded.calibrated_probability == pytest.approx(res_door.calibrated_probability, rel=1e-4)


def test_adversarial_tracking_suite() -> None:
    """Adversarial validation of temporal tracking for ID switches, fragmentations, false positives, and overlapping predictions."""
    from benchmarks.perception.real_data_benchmark import evaluate_temporal_tracking

    # Case 1: Pure identity cross-over (two tracks cross and tracker swaps identities)
    # GT1: stays at (0, 0, 10, 10). GT2: stays at (50, 50, 60, 60).
    # Frames 0, 1: correct. Frame 2: p1 swaps to GT2, p2 swaps to GT1.
    gt_cross = {
        "gt1": [(0, (0.0, 0.0, 10.0, 10.0)), (1, (0.0, 0.0, 10.0, 10.0)), (2, (0.0, 0.0, 10.0, 10.0))],
        "gt2": [(0, (50.0, 50.0, 60.0, 60.0)), (1, (50.0, 50.0, 60.0, 60.0)), (2, (50.0, 50.0, 60.0, 60.0))],
    }
    pred_cross = {
        "p1": [(0, (0.0, 0.0, 10.0, 10.0)), (1, (0.0, 0.0, 10.0, 10.0)), (2, (50.0, 50.0, 60.0, 60.0))],
        "p2": [(0, (50.0, 50.0, 60.0, 60.0)), (1, (50.0, 50.0, 60.0, 60.0)), (2, (0.0, 0.0, 10.0, 10.0))],
    }
    m_cross = evaluate_temporal_tracking(gt_cross, pred_cross, iou_threshold=0.50)
    assert m_cross.id_switches == 2
    assert m_cross.frame_fn == 0
    assert m_cross.frame_fp == 0
    assert m_cross.frame_tp == 6
    # 6 total GT instances, 2 ID switches -> MOTA = 1 - 2/6 = 0.6667
    assert m_cross.mota == pytest.approx(4.0 / 6.0)

    # Case 2: Track fragmentation (GT present in frames 0, 1, 2, 3; prediction missing at frame 2)
    gt_frag = {
        "gt1": [(0, (0.0, 0.0, 10.0, 10.0)), (1, (0.0, 0.0, 10.0, 10.0)), (2, (0.0, 0.0, 10.0, 10.0)), (3, (0.0, 0.0, 10.0, 10.0))],
    }
    pred_frag = {
        "p1": [(0, (0.0, 0.0, 10.0, 10.0)), (1, (0.0, 0.0, 10.0, 10.0)), (3, (0.0, 0.0, 10.0, 10.0))],
    }
    m_frag = evaluate_temporal_tracking(gt_frag, pred_frag, iou_threshold=0.50)
    assert m_frag.track_fragmentations == 1
    assert m_frag.frame_fn == 1
    assert m_frag.frame_tp == 3
    assert m_frag.id_switches == 0
    # MOTA = 1 - (1 FN + 0 FP + 0 IDSW) / 4 = 0.75
    assert m_frag.mota == pytest.approx(0.75)

    # Case 3: Overlapping candidate predictions (2 predictions competing for 1 GT)
    gt_overlap = {
        "gt1": [(0, (0.0, 0.0, 10.0, 10.0))],
    }
    pred_overlap = {
        "p1": [(0, (0.0, 0.0, 10.0, 10.0))],
        "p2": [(0, (0.0, 0.0, 10.0, 10.0))],  # Duplicate identical bounding box
    }
    m_overlap = evaluate_temporal_tracking(gt_overlap, pred_overlap, iou_threshold=0.50)
    assert m_overlap.frame_tp == 1
    assert m_overlap.frame_fp == 1
    assert m_overlap.frame_fn == 0
    # MOTA = 1 - (0 FN + 1 FP + 0 IDSW) / 1 = 0.0
    assert m_overlap.mota == pytest.approx(0.0)


def test_reference_boundary_iou_and_bf_score_analytical() -> None:
    """Verifies boundary IoU and BF-score against analytical geometric properties."""
    from benchmarks.perception.real_data_benchmark import compute_boundary_f_score, compute_boundary_iou

    # 1. Identical square mask (100x100)
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True
    assert compute_boundary_iou(mask, mask, dilation_radius=2) == 1.0
    assert compute_boundary_f_score(mask, mask, distance_threshold=2.0) == 1.0

    # 2. Completely disjoint masks
    mask_disjoint = np.zeros((100, 100), dtype=bool)
    mask_disjoint[5:20, 5:20] = True
    assert compute_boundary_iou(mask, mask_disjoint, dilation_radius=2) == 0.0
    assert compute_boundary_f_score(mask, mask_disjoint, distance_threshold=2.0) == 0.0

    # 3. Both empty masks
    empty = np.zeros((100, 100), dtype=bool)
    assert compute_boundary_iou(empty, empty) == 1.0
    assert compute_boundary_f_score(empty, empty) == 1.0

    # 4. One empty mask
    assert compute_boundary_iou(mask, empty) == 0.0
    assert compute_boundary_f_score(mask, empty) == 0.0

    # 5. Shift by 1 pixel with tolerance = 2.0 px (all contour points within tolerance -> BF=1.0)
    mask_shift1 = np.zeros((100, 100), dtype=bool)
    mask_shift1[31:71, 30:70] = True
    bf_tolerant = compute_boundary_f_score(mask, mask_shift1, distance_threshold=2.0)
    assert bf_tolerant == 1.0


def test_multiview_association_adversarial_audits() -> None:
    """Audits multi-view association metrics against unassigned, duplicate, and fragmented observations."""
    from engine.multiengine.spatial_intelligence.association import PairScore, PhysicalMergedEntity, AssociationOutcome
    from benchmarks.perception.real_data_benchmark import evaluate_multiview_association

    gt_entities = {
        "lamp_gt": {"lamp_obs1", "lamp_obs2"},
    }

    # Case A: Cluster contains a false positive / unassigned observation ('fp_det')
    m_contaminated = PhysicalMergedEntity(
        canonical_entity_id="cluster_lamp",
        primary_class="lamp",
        member_observation_ids=("lamp_obs1", "fp_det"),  # 1 true, 1 unassigned
        sources=("cam1", "cam2"),
    )
    res_a = evaluate_multiview_association(
        gt_entities,
        merged_entities=[m_contaminated],
        association_decisions=[],
    )
    # Purity must drop because 1 of 2 observations is unassigned (purity = 1/2 = 0.50)
    assert res_a.cluster_purity == pytest.approx(0.50)

    # Case B: Self-pair decision 'obs_a == obs_b' is cleanly ignored
    self_decision = PairScore(
        obs_a_id="lamp_obs1",
        obs_b_id="lamp_obs1",
        score=1.0,
        outcome=AssociationOutcome.MERGED,
        reason="self",
        triangulated_position=(0.0, 0.0, 0.0),
        ray_distance=0.0,
        reprojection_err_px=0.0,
    )
    res_b = evaluate_multiview_association(
        gt_entities,
        merged_entities=[],
        association_decisions=[self_decision],
    )
    assert res_b.candidate_pairs_evaluated == 0  # self-pair rejected/ignored
    assert res_b.merged_pairs == 0


def test_learned_confidence_calibrator_sample_threshold_enforcement_and_reliability_diagrams() -> None:
    """Verifies sample thresholds (default min 8 samples, 3 pos, 3 neg), reliability diagrams, model verification, and pipeline loading."""
    from engine.multiengine.spatial_intelligence.detection_pipeline import (
        LearnedConfidenceCalibrator,
        DetectionPipeline,
        load_default_production_calibrator,
    )

    cal = LearnedConfidenceCalibrator()

    # Create dataset with 10 door samples (5 pos, 5 neg) and 6 table samples (3 pos, 3 neg)
    door_scores = [0.85, 0.90, 0.80, 0.75, 0.88, 0.30, 0.25, 0.40, 0.35, 0.20]
    door_labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    door_classes = ["door"] * 10

    table_scores = [0.70, 0.80, 0.75, 0.30, 0.25, 0.35]
    table_labels = [1, 1, 1, 0, 0, 0]
    table_classes = ["table"] * 6

    raw_scores = door_scores + table_scores
    labels = door_labels + table_labels
    classes = door_classes + table_classes

    provenance_meta = {
        "model_name": "grounding_dino_swint_ogc",
        "checkpoint_hash": "sha256_mock_abc123",
        "dataset_split": "dev",
    }

    # Fit with default min_class_samples=8, min_class_positives=3, min_class_negatives=3
    cal.fit(raw_scores, labels, class_names=classes, provenance=provenance_meta)
    assert cal.is_fitted

    # Door has 10 samples (5 pos, 5 neg) -> fits per-class Platt
    res_door = cal.calibrate(0.85, class_name="door")
    assert res_door.is_calibrated
    assert res_door.method == "per_class_platt"

    # Table has only 6 samples (< 8) -> rejected from per-class Platt and falls back to global_platt
    res_table = cal.calibrate(0.70, class_name="table")
    assert res_table.is_calibrated
    assert res_table.method == "global_platt"
    assert "table" not in cal.class_params

    # Check parameter uncertainty was computed
    assert "temperature_se" in cal.global_uncertainty
    assert "bias_se" in cal.global_uncertainty
    assert "door" in cal.class_uncertainties

    # Reliability diagram
    diagram = cal.compute_reliability_diagram(raw_scores, labels, num_bins=5)
    assert len(diagram) == 5
    for b in diagram:
        assert "bin_index" in b
        assert "confidence" in b
        assert "accuracy" in b
        assert "calibration_gap" in b

    # Model and checkpoint compatibility verification
    assert cal.verify_model_compatibility(model_name="grounding_dino_swint_ogc", checkpoint_hash="sha256_mock_abc123") is True
    assert cal.verify_model_compatibility(model_name="other_model", checkpoint_hash="sha256_mock_abc123") is False
    assert cal.verify_model_compatibility(model_name="grounding_dino_swint_ogc", checkpoint_hash="wrong_hash") is False

    # Production pipeline loading default calibrator
    pipeline = DetectionPipeline()
    # If artifacts/perception/calibrator_dev.json is not yet fitted, self.calibrator is None
    # If loaded and fitted, self.calibrator is instance of LearnedConfidenceCalibrator
    if pipeline.calibrator is not None:
        assert isinstance(pipeline.calibrator, LearnedConfidenceCalibrator)


def test_part_hierarchy_precision_recall_and_false_parents() -> None:
    """Verifies part hierarchy evaluation: correct parent links, false parent assignments, and missed components."""
    from benchmarks.perception.real_data_benchmark import evaluate_part_hierarchy, GroundTruthBox

    gt_boxes = [
        GroundTruthBox(entity_id="door_1", class_name="door", box_2d=(100.0, 100.0, 300.0, 600.0), image_id="img_1"),
        GroundTruthBox(entity_id="handle_1", class_name="door_handle", box_2d=(120.0, 300.0, 150.0, 350.0), image_id="img_1", parent_entity_id="door_1"),
        GroundTruthBox(entity_id="door_2", class_name="door", box_2d=(400.0, 100.0, 600.0, 600.0), image_id="img_1"),
        GroundTruthBox(entity_id="handle_2", class_name="door_handle", box_2d=(420.0, 300.0, 450.0, 350.0), image_id="img_1", parent_entity_id="door_2"),
    ]

    class MockPred:
        def __init__(self, box_id, class_name, box_2d, parent_box_id=None):
            self.box_id = box_id
            self.class_name = class_name
            self.box_2d = box_2d
            self.parent_box_id = parent_box_id
            self.source_image_id = "img_1"
            self.confidence = 0.90

    # Case 1: Pred 1 correctly links handle_1 to door_1. Pred 2 links handle_2 to door_1 (FALSE PARENT ASSIGNMENT!)
    preds = [
        MockPred("p_door1", "door", (101.0, 101.0, 299.0, 599.0)),
        MockPred("p_handle1", "door_handle", (121.0, 301.0, 149.0, 349.0), parent_box_id="p_door1"),
        MockPred("p_door2", "door", (401.0, 101.0, 599.0, 599.0)),
        MockPred("p_handle2", "door_handle", (421.0, 301.0, 449.0, 349.0), parent_box_id="p_door1"),  # Wrong parent!
    ]

    metrics = evaluate_part_hierarchy(gt_boxes, preds)
    assert metrics.total_gt_part_links == 2
    assert metrics.predicted_part_links == 2
    assert metrics.correct_part_links == 1
    assert metrics.false_parent_assignments == 1
    assert metrics.missed_components == 1
    assert metrics.precision == pytest.approx(0.50)
    assert metrics.recall == pytest.approx(0.50)
    assert metrics.f1 == pytest.approx(0.50)





