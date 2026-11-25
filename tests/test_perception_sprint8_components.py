"""Perception Sprint 8 Component and Specificity Tests.

Tests:
1. Multi-scale component ROI generation for host entities (doors, windows, tables, chairs).
2. Specificity-based component ownership resolution (preventing door handle -> wall).
3. Ambiguous parent handling without guessing.
4. Duplicate suppression preserving subparts.
5. Component benchmark evaluator accounting for sample counts and UNMEASURED classes.
"""

from __future__ import annotations

import pytest

from engine.multiengine.spatial_intelligence.detection_pipeline import (
    DetectionBox,
    DetectionScale,
    generate_component_rois_for_host,
    resolve_component_ownership,
    suppress_duplicates_preserving_parts,
)
from benchmarks.perception.component_benchmark import (
    ComponentBenchmarkEvaluator,
    ComponentFamily,
    GroundTruthComponent,
)


def test_generate_component_rois_for_host():
    """Verifies that high-resolution sub-regions are targeted for host classes."""
    # 1. Door host
    door_box = DetectionBox("d1", "door", (100.0, 100.0, 500.0, 900.0), 0.90)
    door_rois = generate_component_rois_for_host(door_box, 1280, 960)
    roi_names = [r[4] for r in door_rois]
    assert "full_host" in roi_names
    assert "door_latch_left" in roi_names
    assert "door_latch_right" in roi_names

    # Check boundaries of door latch ROI
    latch_left = next(r for r in door_rois if r[4] == "door_latch_left")
    assert latch_left[0] >= 0
    assert latch_left[1] >= door_box.y1
    assert latch_left[3] <= door_box.y2

    # 2. Table host
    table_box = DetectionBox("t1", "table", (200.0, 400.0, 800.0, 800.0), 0.88)
    table_rois = generate_component_rois_for_host(table_box, 1280, 960)
    table_roi_names = [r[4] for r in table_rois]
    assert "full_host" in table_roi_names
    assert "table_top_band" in table_roi_names

    # 3. Window host
    window_box = DetectionBox("w1", "window", (50.0, 100.0, 350.0, 500.0), 0.92)
    win_rois = generate_component_rois_for_host(window_box, 1280, 960)
    win_roi_names = [r[4] for r in win_rois]
    assert "full_host" in win_roi_names
    assert "window_lower_sill" in win_roi_names

    # 4. Chair host
    chair_box = DetectionBox("c1", "chair", (600.0, 500.0, 750.0, 900.0), 0.85)
    chair_rois = generate_component_rois_for_host(chair_box, 1280, 960)
    chair_roi_names = [r[4] for r in chair_rois]
    assert "full_host" in chair_roi_names
    assert "chair_seat_zone" in chair_roi_names


def test_resolve_component_ownership_specificity():
    """Verifies that direct structural parents take precedence over generic background surfaces."""
    wall = DetectionBox("wall_01", "wall", (0.0, 0.0, 1280.0, 960.0), 0.98)
    door = DetectionBox("door_01", "door", (200.0, 150.0, 600.0, 900.0), 0.91)
    handle_box = (500.0, 500.0, 550.0, 560.0)

    # Both wall and door physically contain handle_box
    best_id, is_ambig, candidates = resolve_component_ownership(
        handle_box, "door_handle", [wall, door]
    )
    # Door handle must bind to door (specificity 3), NOT wall (specificity 1)
    assert best_id == "door_01"
    assert is_ambig is False
    assert candidates == ["door_01"]


def test_resolve_component_ownership_ambiguity():
    """Verifies that ambiguous parents are marked ambiguous rather than guessed."""
    door_a = DetectionBox("door_left", "door", (100.0, 100.0, 500.0, 900.0), 0.90)
    door_b = DetectionBox("door_right", "door", (200.0, 100.0, 600.0, 900.0), 0.89)
    overlap_handle = (250.0, 480.0, 300.0, 540.0)

    best_id, is_ambig, candidates = resolve_component_ownership(
        overlap_handle, "door_handle", [door_a, door_b]
    )
    assert is_ambig is True
    assert set(candidates) == {"door_left", "door_right"}


def test_resolve_component_ownership_subpart_cannot_host():
    """Verifies that a subpart cannot host another subpart."""
    handle_parent = DetectionBox("h1", "door_handle", (100.0, 100.0, 200.0, 200.0), 0.80, parent_box_id="door_1")
    tiny_part = (120.0, 120.0, 140.0, 140.0)

    best_id, is_ambig, candidates = resolve_component_ownership(
        tiny_part, "door_knob", [handle_parent]
    )
    assert best_id is None
    assert len(candidates) == 0


def test_suppress_duplicates_preserving_parts():
    """Verifies that micro components overlapping parent objects are NOT suppressed by NMS."""
    door = DetectionBox("d1", "door", (100.0, 100.0, 500.0, 900.0), 0.92)
    handle = DetectionBox("h1", "door_handle", (420.0, 480.0, 460.0, 540.0), 0.85, scale=DetectionScale.MICRO)

    rejections = []
    retained = suppress_duplicates_preserving_parts(
        [door, handle],
        iou_threshold=0.50,
        image_shape=(960, 1280),
        rejection_records=rejections,
    )
    retained_ids = {d.box_id for d in retained}
    assert "d1" in retained_ids
    assert "h1" in retained_ids
    assert len(retained) == 2


def test_component_benchmark_evaluator_unmeasured_accounting():
    """Verifies that classes without GT are reported as UNMEASURED and do not skew macro metrics."""
    gt_list = [
        GroundTruthComponent("gt_d1", "door", "img_01", (100.0, 100.0, 500.0, 900.0), is_component=False),
        GroundTruthComponent("gt_h1", "door_handle", "img_01", (420.0, 480.0, 460.0, 540.0), parent_entity_id="gt_d1", is_component=True),
    ]
    pred_list = [
        DetectionBox("p_d1", "door", (105.0, 102.0, 498.0, 895.0), 0.90, source_image_id="img_01"),
        DetectionBox("p_h1", "door_handle", (422.0, 481.0, 458.0, 538.0), 0.86, parent_box_id="p_d1", source_image_id="img_01"),
    ]

    evaluator = ComponentBenchmarkEvaluator()
    report = evaluator.evaluate(
        ground_truth=gt_list,
        predictions=pred_list,
        eval_split="test",
        dataset_name="synthetic_test",
    )

    assert report.evaluated_classes_count >= 80
    assert report.measured_classes_count == 2
    assert report.unmeasured_classes_count == report.evaluated_classes_count - 2

    # Check unmeasured class
    outlet_metrics = report.results_by_class["socket"]
    assert outlet_metrics.status == "UNMEASURED"
    assert outlet_metrics.gt_count == 0
    assert outlet_metrics.precision_50 is None
    assert outlet_metrics.f1_50 is None

    # Check measured class
    handle_metrics = report.results_by_class["door_handle"]
    assert handle_metrics.status == "measured"
    assert handle_metrics.gt_count == 1
    assert handle_metrics.precision_50 == 1.0
    assert handle_metrics.recall_50 == 1.0
    assert handle_metrics.f1_50 == 1.0

    # Macro metrics are computed over measured classes only
    assert report.component_detection_macro_f1 == 1.0
    assert report.component_parent_link_macro_f1 == 1.0
