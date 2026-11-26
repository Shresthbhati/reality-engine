"""Tests for Workstream A (December 2026 Perception Directive):
A1-A8 + Priority 2 multi-engine routing, correlation-aware fusion, calibration integrity,
label cleaning, taxonomy completeness, and cross-view ambiguity preservation.
"""

from __future__ import annotations

import numpy as np

from engine.multiengine.adapters.perception_detectors import (
    clean_and_split_detection_label,
    probe_florence2,
    probe_grounding_dino,
)
from engine.multiengine.adapters.perception_segmenters import probe_sam2
from engine.multiengine.spatial_intelligence.detection_pipeline import (
    DetectionBox,
    DetectionScale,
    generate_domain_prompts,
)
from engine.multiengine.spatial_intelligence.taxonomy import Domain, resolve_taxonomy_class
from perception.detection.multi_engine_fusion import (
    associate_cross_view_detections,
    fuse_multi_engine_detections,
    select_detection_engines,
)


def test_clean_and_split_detection_label_rejects_bert_subwords_and_underscores():
    assert clean_and_split_detection_label("##rest", ["chair_armrest"]) == []
    assert clean_and_split_detection_label("_", ["door_handle"]) == []
    assert clean_and_split_detection_label("##vable", ["movable_frame"]) == []
    assert clean_and_split_detection_label("door handle", ["door_handle", "door_hinge"]) == ["door_handle"]
    assert clean_and_split_detection_label("window frame", ["window_frame", "window_sill"]) == ["window_frame"]


def test_macro_scale_included_in_default_domain_prompts():
    prompts = generate_domain_prompts(Domain.INTERIOR)
    assert DetectionScale.MACRO in prompts
    macro = set(prompts[DetectionScale.MACRO])
    assert "wall" in macro
    assert "floor" in macro
    assert "ceiling" in macro
    assert "column" in macro
    assert "beam" in macro
    assert "staircase" in macro


def test_a2_and_a3_taxonomy_classes_resolve_distinctly():
    assert resolve_taxonomy_class("opening").name == "opening"
    assert resolve_taxonomy_class("door").name == "door"
    assert resolve_taxonomy_class("door_frame").name == "door_frame"
    assert resolve_taxonomy_class("window").name == "window"
    assert resolve_taxonomy_class("window_frame").name == "window_frame"
    assert resolve_taxonomy_class("window_sill").name == "window_sill"
    assert resolve_taxonomy_class("railing").name == "railing"
    assert resolve_taxonomy_class("railing").domain == Domain.UNIVERSAL
    assert resolve_taxonomy_class("structural_boundary").name == "structural_boundary"
    assert resolve_taxonomy_class("storage_unit").name == "storage_unit"
    assert resolve_taxonomy_class("kitchen_furniture").name == "kitchen_furniture"
    assert resolve_taxonomy_class("bathroom_fixture").name == "bathroom_fixture"


def test_task_specific_detector_routing():
    avail = ["owlvit", "grounding_dino", "mask_rcnn", "florence2"]
    assert select_detection_engines(avail, budget="fast") == ["grounding_dino"]
    assert select_detection_engines(avail, budget="balanced") == ["grounding_dino", "mask_rcnn"]
    assert select_detection_engines(avail, budget="forensic") == ["grounding_dino", "mask_rcnn", "florence2"]
    assert select_detection_engines(avail, budget="balanced", is_low_contrast=True) == [
        "grounding_dino", "mask_rcnn", "florence2"
    ]


def test_correlation_aware_fusion_does_not_double_count_same_family_and_preserves_conflicts():
    box_gd1 = DetectionBox("gd1", "cabinet", (10.0, 10.0, 110.0, 110.0), 0.72, source_engine="grounding_dino")
    box_gd2 = DetectionBox("gd2", "cabinet", (12.0, 12.0, 112.0, 112.0), 0.68, source_engine="grounding_dino_tiled")
    box_mr = DetectionBox("mr1", "wardrobe", (11.0, 11.0, 111.0, 111.0), 0.65, source_engine="mask_rcnn")

    fused = fuse_multi_engine_detections({
        "grounding_dino": [box_gd1],
        "grounding_dino_tiled": [box_gd2],
        "mask_rcnn": [box_mr],
    })
    assert len(fused) == 1
    h = fused[0]
    # grounding_dino + grounding_dino_tiled share family "grounding-dino" -> 2 independent families, not 3!
    assert h.independent_family_count == 2
    assert h.supporting_families == ("grounding-dino", "maskrcnn")
    assert h.calibrated_confidence is None
    assert h.calibration_status == "uncalibrated"
    # Semantically distinct competing label ("wardrobe" vs "cabinet") is preserved
    assert any(c["label"] == "wardrobe" and c["engine_id"] == "mask_rcnn" for c in h.competing_hypotheses)


def test_cross_view_association_preserves_ambiguity_for_identical_chairs():
    c_src = DetectionBox(
        "src_chair", "chair", (10, 10, 100, 120), 0.85,
        source_image_id="img1", appearance_features=(0.5, 0.5, 0.0, 0.0),
    )
    c_t1 = DetectionBox(
        "tgt_chair_1", "chair", (20, 10, 110, 120), 0.82,
        source_image_id="img2", appearance_features=(0.5, 0.5, 0.0, 0.0),
    )
    c_t2 = DetectionBox(
        "tgt_chair_2", "chair", (220, 10, 310, 120), 0.80,
        source_image_id="img2", appearance_features=(0.49, 0.51, 0.0, 0.0),
    )
    res = associate_cross_view_detections([c_src], [c_t1, c_t2])
    assert len(res) == 1
    assert res[0].status == "ambiguous"
    assert res[0].target_box_id is None
    assert set(res[0].candidate_target_ids) == {"tgt_chair_1", "tgt_chair_2"}


def test_available_probes_for_real_cached_models():
    # On this workstation all three are cached; probes must return Availability cleanly
    assert probe_grounding_dino().ok
    assert probe_florence2().ok
    assert probe_sam2().ok
