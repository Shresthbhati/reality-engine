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


