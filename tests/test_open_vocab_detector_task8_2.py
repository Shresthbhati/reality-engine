"""Tests for Blueprint Task 8.2: Open-vocab detector via the registry."""

import numpy as np
import pytest

from perception import model_registry
from perception.detection.backends import FakeDetector
from perception.detection.interface import ImageDetection
from perception.detection.open_vocab import (
    OpenVocabDetector,
    box_iou,
    evaluate_detection_metrics,
)
from perception.ontology import get_default_ontology
from world_ir.observations import Observation


def test_open_vocab_models_registered_in_registry():
    spec = model_registry.get("owlvit_base_patch32")
    assert spec is not None
    assert spec.task == "open_vocabulary_detection"
    assert spec.license == "Apache-2.0"
    assert spec.commercial_ok is True

    owlv2 = model_registry.get("owlv2_base_patch16")
    assert owlv2 is not None
    assert owlv2.task == "open_vocabulary_detection"

    gdino = model_registry.get("grounding_dino_tiny")
    assert gdino is not None
    assert gdino.task == "open_vocabulary_detection"


def test_prompts_originate_from_ontology():
    onto = get_default_ontology()
    detector = OpenVocabDetector(model_name="owlvit_base_patch32", ontology=onto)
    prompts = detector.get_ontology_prompts()
    assert len(prompts) >= 30
    assert "sectional sofa" in prompts
    assert "dining table" in prompts
    assert "coffee table" in prompts
    assert "wall socket" in prompts


def test_detect_produces_observations_only():
    onto = get_default_ontology()

    # Scripted detector returning detections in image plane
    script = {
        "sectional sofa": [((100, 100, 600, 500), 0.92)],
        "coffee table": [((200, 300, 450, 450), 0.88)],
        "unknown widget": [((50, 50, 120, 120), 0.75)],
    }
    fake_backend = FakeDetector(script)

    detector = OpenVocabDetector(
        model_name="owlvit_base_patch32",
        backend=fake_backend,
        ontology=onto,
        confidence_threshold=0.5,
    )

    image_rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth_map = np.ones((720, 1280), dtype=np.float32) * 2.5  # 2.5m depth

    obs_list = detector.detect_observations(
        image_rgb=image_rgb,
        image_id="img-living-room-01",
        depth_map=depth_map,
        scale_state="metric",
    )

    # 1. Output must be strictly Observation instances
    assert all(isinstance(o, Observation) for o in obs_list)
    assert len(obs_list) >= 2

    # 2. Sectional sofa resolved to canonical ontology path
    sofa_obs = next(o for o in obs_list if "sofa" in o.label)
    assert sofa_obs.label == "furniture/seating/sofa/sectional"
    assert sofa_obs.confidence == 0.92
    assert sofa_obs.model.startswith("owlvit_base_patch32")
    assert sofa_obs.license == "Apache-2.0"
    assert sofa_obs.bbox_px == (100.0, 100.0, 600.0, 500.0)
    assert sofa_obs.world_pos[2] == 2.5

    # 3. Unknown widget resolved to canonical fallback
    fallback_obs = next(o for o in obs_list if "unknown" in o.label)
    assert fallback_obs.label == "fallback/unknown_object"


def test_min_pixels_resolution_aware_gating():
    onto = get_default_ontology()

    # A sofa requires min_pixels: 1500 in ontology.
    # A tiny bounding box with area 400 pixels (20x20) must be rejected!
    # A realistic bounding box (200x200 = 40,000 px) must be kept.
    script = {
        "sofa": [
            ((10, 10, 30, 30), 0.95),      # 20x20 = 400 px -> REJECTED by min_pixels
            ((100, 100, 300, 300), 0.91),  # 200x200 = 40,000 px -> ACCEPTED
        ]
    }
    detector = OpenVocabDetector(
        model_name="owlvit_base_patch32",
        backend=FakeDetector(script),
        ontology=onto,
        confidence_threshold=0.3,
    )

    image_rgb = np.zeros((600, 800, 3), dtype=np.uint8)
    obs = detector.detect_observations(image_rgb, "img-test")

    assert len(obs) == 1
    assert obs[0].bbox_px == (100.0, 100.0, 300.0, 300.0)


def test_box_iou_computation():
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(box1, box2) == 1.0

    box3 = (10.0, 10.0, 20.0, 20.0)
    assert box_iou(box1, box3) == 0.0

    box4 = (5.0, 0.0, 15.0, 10.0)
    # intersection: 5x10 = 50, union: 100 + 100 - 50 = 150 -> 50/150 = 1/3
    assert abs(box_iou(box1, box4) - (1.0 / 3.0)) < 1e-5


def test_precision_recall_evaluation_on_living_room_benchmark():
    # Synthetic living room detections vs ground truth
    onto = get_default_ontology()

    observations = [
        Observation(
            observation_id="obs-1",
            image_id="img-living-room",
            label="furniture/seating/sofa",
            world_pos=(0, 0, 2),
            sigma_m=0.1,
            depth_source="metric",
            confidence=0.95,
            bbox_px=(50, 100, 400, 350),
        ),
        Observation(
            observation_id="obs-2",
            image_id="img-living-room",
            label="furniture/tables/coffee_table",
            world_pos=(0, 0, 1.5),
            sigma_m=0.1,
            depth_source="metric",
            confidence=0.89,
            bbox_px=(150, 250, 300, 350),
        ),
        # False positive armchair
        Observation(
            observation_id="obs-3",
            image_id="img-living-room",
            label="furniture/seating/armchair",
            world_pos=(1, 0, 2),
            sigma_m=0.1,
            depth_source="metric",
            confidence=0.65,
            bbox_px=(500, 100, 600, 200),
        ),
    ]

    ground_truth = [
        {"image_id": "img-living-room", "label": "furniture/seating/sofa", "box_xyxy": (55, 95, 395, 345)},
        {"image_id": "img-living-room", "label": "furniture/tables/coffee_table", "box_xyxy": (152, 248, 298, 352)},
        # Missed television (False negative)
        {"image_id": "img-living-room", "label": "electronics/display/tv", "box_xyxy": (300, 50, 500, 180)},
    ]

    metrics = evaluate_detection_metrics(observations, ground_truth, iou_threshold=0.5)

    # Sofa: 1 TP, 0 FP, 0 FN -> Precision 1.0, Recall 1.0
    sofa_eval = metrics["furniture/seating/sofa"]
    assert sofa_eval.true_positives == 1
    assert sofa_eval.false_positives == 0
    assert sofa_eval.false_negatives == 0
    assert sofa_eval.precision == 1.0
    assert sofa_eval.recall == 1.0

    # Armchair: 0 TP, 1 FP, 0 FN -> Precision 0.0
    armchair_eval = metrics["furniture/seating/armchair"]
    assert armchair_eval.true_positives == 0
    assert armchair_eval.false_positives == 1
    assert armchair_eval.precision == 0.0

    # TV: 0 TP, 0 FP, 1 FN -> Recall 0.0
    tv_eval = metrics["electronics/display/tv"]
    assert tv_eval.true_positives == 0
    assert tv_eval.false_negatives == 1
    assert tv_eval.recall == 0.0
