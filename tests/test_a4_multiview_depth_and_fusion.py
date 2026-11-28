"""A4.6, A4.7, A4.8, A4.9 & A4.10 tests: multi-view perception, depth association, open-vocab/CLIP, fusion 2.0, active 2.0."""
from __future__ import annotations

import numpy as np
import pytest

from perception.intelligence.active_v2 import run_active_perception_v2
from perception.intelligence.clip_semantic import (
    ClipSemanticVerifier,
    rank_attribute_hypotheses_by_text,
)
from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    CoordinateSystemSpec,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    ScoreSemantics,
)
from perception.intelligence.depth_aware import (
    DepthMapContract,
    associate_observations_with_depth,
)
from perception.intelligence.fusion_v2 import fuse_observations_v2
from perception.intelligence.identity import CameraPoseConstraint
from perception.intelligence.multiview import run_multiview_perception
from perception.intelligence.open_vocab import (
    Domain,
    classify_open_vocabulary_label,
    generate_survey_prompts,
    track_class_confusions,
)


def _obs(
    obs_id: str,
    image_id: str,
    label: str,
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.80,
    engine_id: str = "owlv2",
    category: str = "furniture",
) -> NormalizedPerceptionObservation:
    return NormalizedPerceptionObservation(
        observation_id=obs_id,
        schema_version=SCHEMA_VERSION,
        raw_label=label,
        canonical_label=label,
        taxonomy_id=f"re:{label}",
        category=category,
        bbox_px=bbox,
        coords=CoordinateSystemSpec(
            frame_id="image_px",
            units="pixels",
            image_width=800,
            image_height=600,
        ),
        scores=ScoreSemantics(raw_score=confidence, score_emitted=True),
        provenance=DetectionProvenance(
            source_image_id=image_id,
            execution=ModelExecutionProvenance(
                engine_id=engine_id,
                architecture_family="vlm",
                checkpoint_id=f"{engine_id}_model",
            ),
        ),
    )


def test_multiview_perception_disambiguates_identical_chairs_across_views() -> None:
    cameras = {
        "cam_a": CameraPoseConstraint(image_id="cam_a", position_m=(0.0, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0)),
        "cam_b": CameraPoseConstraint(image_id="cam_b", position_m=(0.25, 0.0, 0.0), orientation_wxyz=(1.0, 0.0, 0.0, 0.0)),
    }
    observations = [
        _obs("a_c1", "cam_a", "chair", (40, 280, 140, 460)),
        _obs("a_c2", "cam_a", "chair", (220, 280, 320, 460)),
        _obs("a_c3", "cam_a", "chair", (440, 280, 540, 460)),
        _obs("a_c4", "cam_a", "chair", (640, 280, 740, 460)),
        _obs("b_c1", "cam_b", "chair", (45, 282, 145, 462)),
        _obs("b_c2", "cam_b", "chair", (225, 282, 325, 462)),
        _obs("b_c3", "cam_b", "chair", (445, 282, 545, 462)),
        _obs("b_c4", "cam_b", "chair", (645, 282, 745, 462)),
        _obs("a_w1", "cam_a", "wall", (0, 20, 800, 480), category="architecture"),
        _obs("b_w1", "cam_b", "wall", (0, 25, 800, 485), category="architecture"),
    ]
    report = run_multiview_perception(observations, cameras=cameras, match_threshold=0.55)
    chair_identities = [i for i in report.identity_report.identities if i.canonical_label == "chair"]
    assert len(chair_identities) == 4
    assert len(report.visibility_tracks) >= 4
    assert len(report.architectural_continuity) >= 1


def test_depth_map_contract_validates_units_and_computes_front_to_back_ranking() -> None:
    with pytest.raises(ValueError):
        DepthMapContract(
            image_id="img_1",
            depth_array=np.ones((60, 80), dtype=np.float32),
            scale_state="relative",
            units="meters",
            model_id="depth_anything_v2",
        )

    depth_arr = np.full((600, 800), 5.0, dtype=np.float32)
    depth_arr[300:480, 100:240] = 1.5
    depth_arr[260:440, 420:680] = 3.2

    contract = DepthMapContract(
        image_id="img_1",
        depth_array=depth_arr,
        scale_state="metric",
        units="meters",
        model_id="metric3d_v2",
    )
    obs = [
        _obs("chair_near", "img_1", "chair", (100, 300, 240, 480)),
        _obs("table_far", "img_1", "table", (420, 260, 680, 440)),
    ]
    res = associate_observations_with_depth(obs, contract)
    assert len(res.object_depths) == 2
    by_id = {a.observation_id: a for a in res.object_depths}
    assert by_id["chair_near"].median_depth == pytest.approx(1.5, abs=0.05)
    assert by_id["table_far"].median_depth == pytest.approx(3.2, abs=0.05)
    assert res.front_to_back_order[0] == "chair_near"


def test_open_vocab_prompts_confusion_and_clip_verifier_contract() -> None:
    cls_res = classify_open_vocabulary_label("sliding glass patio door")
    assert cls_res.canonical_label == "door"
    assert cls_res.is_known_class is True

    unknown_res = classify_open_vocabulary_label("xyzzyspoon_unrecognized_artifact")
    assert unknown_res.is_known_class is False

    prompts = generate_survey_prompts(domain=Domain.INTERIOR, include_unknown_candidates=True, max_prompts=40)
    assert len(prompts) >= 10
    assert len(prompts) <= 40

    confusions = track_class_confusions([("door", "window"), ("chair", "armchair"), ("door", "window")])
    assert confusions["total_confusion_events"] == 3

    verifier = ClipSemanticVerifier()
    crop = np.full((64, 64, 3), 128, dtype=np.uint8)
    ranked = rank_attribute_hypotheses_by_text(
        crop,
        ["wooden chair", "metal table", "glass door"],
        verifier=verifier,
        observation_id="obs_crop_1",
    )
    assert isinstance(ranked, dict)
    assert ranked["status"] in ("succeeded", "unavailable")
    if ranked["status"] == "succeeded":
        assert ranked["is_calibrated"] is False


def test_fusion_v2_handles_correlated_engines_and_disagreement() -> None:
    obs = [
        _obs("o1", "img_1", "chair", (100, 200, 250, 420), confidence=0.82, engine_id="owlv2"),
        _obs("o2", "img_1", "chair", (104, 202, 248, 418), confidence=0.79, engine_id="grounding_dino"),
        _obs("o3", "img_1", "sofa", (102, 205, 252, 422), confidence=0.65, engine_id="yolo_world"),
    ]
    decisions = fuse_observations_v2(obs, image_quality_grade="good", iou_threshold=0.50)
    assert len(decisions) == 1
    dec = decisions[0]
    assert dec.selected_label == "chair"
    assert len(dec.detector_disagreements) >= 1
    assert dec.quality_weight == 1.0
    assert 0.0 <= dec.propagated_uncertainty <= 1.0


def test_active_perception_v2_respects_budget_and_stopping_criterion() -> None:
    obs = [
        _obs("w_low", "img_low", "wall", (0, 20, 800, 480), confidence=0.35, category="architecture"),
        _obs("d_uncertain", "img_good", "door", (120, 100, 240, 440), confidence=0.42, category="architecture"),
        _obs("c_high", "img_good", "chair", (300, 300, 420, 460), confidence=0.92),
    ]
    report = run_active_perception_v2(
        obs,
        image_qualities={"img_low": {"quality_grade": "degraded"}, "img_good": {"quality_grade": "good"}},
        compute_budget=3.0,
    )
    assert report.base_plan.summary["estimated_cost_total"] <= 3.0 + 1e-6
    assert "img_low" in report.low_quality_images
    assert report.recommended_next_step
