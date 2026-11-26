"""A5 Tests (4, 5, 6, 7, 8, 9, 10): duplicates, separate identical objects, cross-view identity, and arch/furniture FPs."""
from __future__ import annotations

import numpy as np

from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    AppearanceDescriptor,
    CoordinateSystemSpec,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    ScoreSemantics,
    SegmentationObservation,
)
from perception.intelligence.deduplication_v2 import resolve_duplicate_detections_v2
from perception.intelligence.identity import CameraPoseConstraint
from perception.intelligence.identity_v2 import resolve_cross_view_identities_v2
from perception.intelligence.scene_refinement_v2 import refine_scene_observations_v2


def _obs(
    obs_id: str,
    image_id: str,
    label: str,
    bbox: tuple[float, float, float, float],
    *,
    confidence: float = 0.85,
    category: str = "furniture",
    engine_id: str = "grounding_dino",
    inference_mode: str = "full_frame",
    tile_id: str | None = None,
    embedding: tuple[float, ...] = (),
    seg: SegmentationObservation | None = None,
    width: int = 800,
    height: int = 600,
) -> NormalizedPerceptionObservation:
    return NormalizedPerceptionObservation(
        observation_id=obs_id,
        schema_version=SCHEMA_VERSION,
        raw_label=label,
        canonical_label=label,
        taxonomy_id=f"re:{label}",
        category=category,
        bbox_px=bbox,
        coords=CoordinateSystemSpec("image_px", "pixels", width, height),
        scores=ScoreSemantics(raw_score=confidence, score_emitted=True),
        provenance=DetectionProvenance(
            source_image_id=image_id,
            execution=ModelExecutionProvenance(engine_id, engine_id, f"{engine_id}_v1"),
            inference_mode=inference_mode,
            tile_id=tile_id,
        ),
        appearance=AppearanceDescriptor(embedding_vector=embedding),
        segmentation=seg,
    )


def test_4_and_5_duplicate_detection_across_engines_crops_and_architecture() -> None:
    obs = [
        # Cross-engine duplicate chair
        _obs("c_gd", "img_1", "chair", (100, 200, 240, 420), confidence=0.88, engine_id="grounding_dino"),
        _obs("c_owl", "img_1", "chair", (104, 202, 238, 418), confidence=0.81, engine_id="owlv2"),
        # Crop / tiled duplicate of same chair
        _obs("c_tile", "img_1", "chair", (102, 204, 236, 416), confidence=0.79, inference_mode="tiled", tile_id="t_0"),
        # Repeated architectural window_frame
        _obs("wf_1", "img_1", "window_frame", (400, 100, 560, 280), confidence=0.86, category="architecture"),
        _obs("wf_2", "img_1", "window_frame", (402, 102, 558, 278), confidence=0.72, category="architecture"),
    ]
    res = resolve_duplicate_detections_v2(obs)
    assert res.summary["input_count"] == 5
    assert res.summary["canonical_count"] == 2
    assert res.summary["suppressed_count"] == 3
    assert res.summary["duplicate_reduction_rate"] == 0.6
    kept_ids = {o.observation_id for o in res.canonical_observations}
    assert kept_ids == {"c_gd", "wf_1"}
    assert len(res.clusters) == 2


def test_6_separate_identical_looking_chairs_never_merge_within_or_across_views() -> None:
    identical_emb = (0.5, 0.5, 0.5, 0.5)
    # 4 identical chairs at distinct positions in cam_a and cam_b
    obs = [
        _obs("a1", "cam_a", "chair", (40, 280, 140, 460), embedding=identical_emb),
        _obs("a2", "cam_a", "chair", (220, 280, 320, 460), embedding=identical_emb),
        _obs("a3", "cam_a", "chair", (440, 280, 540, 460), embedding=identical_emb),
        _obs("a4", "cam_a", "chair", (640, 280, 740, 460), embedding=identical_emb),
        _obs("b1", "cam_b", "chair", (44, 282, 144, 462), embedding=identical_emb),
        _obs("b2", "cam_b", "chair", (224, 282, 324, 462), embedding=identical_emb),
        _obs("b3", "cam_b", "chair", (444, 282, 544, 462), embedding=identical_emb),
        _obs("b4", "cam_b", "chair", (644, 282, 744, 462), embedding=identical_emb),
    ]
    dedup = resolve_duplicate_detections_v2(obs)
    assert len(dedup.canonical_observations) == 8

    cameras = {
        "cam_a": CameraPoseConstraint("cam_a", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), position_sigma_m=0.05),
        "cam_b": CameraPoseConstraint("cam_b", (0.2, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), position_sigma_m=0.05),
    }
    id_rep = resolve_cross_view_identities_v2(dedup.canonical_observations, cameras=cameras)
    assert id_rep.base_report.summary["total_identities"] == 4
    assert id_rep.confirmed_count == 4


def test_7_and_8_cross_view_identity_outcomes_confirmed_ambiguous_and_rejected() -> None:
    # Distinct appearance + close pose -> confirmed_match
    # Competing similar candidates -> ambiguous_match (never forced into a match)
    # Far position + different appearance -> rejected_match
    obs = [
        _obs("a_sofa", "v1", "sofa", (100, 250, 350, 480), embedding=(0.9, 0.1, 0.0, 0.0)),
        _obs("b_sofa", "v2", "sofa", (105, 252, 355, 482), embedding=(0.89, 0.12, 0.0, 0.0)),
        _obs("a_c1", "v1", "chair", (420, 300, 500, 460), embedding=(0.5, 0.5, 0.0, 0.0)),
        _obs("a_c2", "v1", "chair", (460, 300, 540, 460), embedding=(0.5, 0.5, 0.0, 0.0)),
        _obs("b_c_amb", "v2", "chair", (440, 300, 520, 460), embedding=(0.5, 0.5, 0.0, 0.0)),
        _obs("a_far", "v1", "table", (20, 300, 120, 420), embedding=(1.0, 0.0, 0.0, 0.0)),
        _obs("b_far", "v2", "table", (660, 300, 760, 420), embedding=(0.0, 0.0, 1.0, 0.0)),
    ]
    cameras = {
        "v1": CameraPoseConstraint("v1", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), position_sigma_m=0.02),
        "v2": CameraPoseConstraint("v2", (0.1, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), position_sigma_m=0.02),
    }
    rep = resolve_cross_view_identities_v2(obs, cameras=cameras)
    outcomes = {d.outcome for d in rep.pairwise_decisions}
    assert "confirmed_match" in outcomes
    assert "ambiguous_match" in outcomes
    assert "rejected_match" in outcomes


def test_9_and_10_architectural_and_furniture_false_positive_filtering() -> None:
    # Textured image vs blank regions
    img = np.full((600, 800, 3), 220, dtype=np.uint8)
    # Textured door region on left
    img[120:480, 80:220:2] = 40
    # Textured chair region on right
    img[300:480, 500:640:2] = 50

    frag_seg = SegmentationObservation(
        mask_id="m_frag",
        observation_id="arch_frag",
        mask_kind="architectural_surface_mask",
        area_px=200,
        box_iou=0.1,
        solidity=0.05,
        hole_count=5,
        boundary_uncertainty=0.8,
        quality_score=0.1,
    )
    obs = [
        # Valid textured door
        _obs("door_ok", "img_1", "door", (80, 120, 220, 480), category="architecture"),
        # Tiny spurious wall (<0.3% area)
        _obs("wall_tiny", "img_1", "wall", (10, 10, 30, 30), category="architecture"),
        # Featureless uniform door
        _obs("door_blank", "img_1", "door", (300, 100, 420, 400), category="architecture"),
        # Fragmented architectural mask
        _obs("arch_frag", "img_1", "wall", (0, 0, 800, 500), category="architecture", seg=frag_seg),
        # Valid textured chair
        _obs("chair_ok", "img_1", "chair", (500, 300, 640, 480), category="furniture"),
        # Implausible full-frame furniture (>88% area)
        _obs("chair_huge", "img_1", "chair", (0, 0, 790, 590), category="furniture"),
        # Low-confidence furniture noise
        _obs("sofa_low", "img_1", "sofa", (500, 300, 640, 480), confidence=0.10, category="furniture"),
    ]
    ref = refine_scene_observations_v2(obs, images_rgb={"img_1": img}, min_confidence=0.22)
    kept = {o.observation_id for o in ref.accepted_observations}
    assert kept == {"door_ok", "chair_ok"}
    assert set(ref.rejected_architectural_ids) == {"wall_tiny", "door_blank", "arch_frag"}
    assert set(ref.rejected_furniture_ids) == {"chair_huge", "sofa_low"}
    assert ref.accepted_observations[1].attributes.get("is_survey_priority_furniture") is True
