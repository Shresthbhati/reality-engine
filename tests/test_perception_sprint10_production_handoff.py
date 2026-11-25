"""Unit tests for Perception Production Output Contract (Sprint 10 P6)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest
import numpy as np

from engine.multiengine.spatial_intelligence.perception_contract import (
    CoordinateConvention,
    MaskProvenance,
    ParentCandidate,
    PerceptionEntityOutput,
    PerceptionHandoffManifest,
    PhysicalIdentityHypothesis,
    RejectionReason,
    RejectionRecord,
)


def test_perception_handoff_contract_valid():
    """Verify that a well-formed handoff manifest passes validation cleanly."""
    parent_cand = ParentCandidate(
        parent_entity_id="window_01",
        relationship_kind="part_of",
        confidence=0.92,
        geometric_containment_ratio=0.88,
    )

    phys_hyp = PhysicalIdentityHypothesis(
        cluster_id="phys_cluster_win1",
        confidence=0.89,
        cross_view_observations=(("img_0.jpg", "window_01"), ("img_1.jpg", "window_02")),
        appearance_similarity=0.91,
        epipolar_consistency=0.87,
    )

    ent_parent = PerceptionEntityOutput(
        entity_id="window_01",
        source_image_id="img_0.jpg",
        source_engine="grounding_dino_tiny",
        class_name="window",
        confidence=0.94,
        calibrated_confidence=0.91,
        box_2d=(100.0, 100.0, 400.0, 400.0),
        coordinate_convention=CoordinateConvention.PIXEL_XYXY_TOPLEFT,
        mask_2d=np.ones((300, 300), dtype=bool),
        mask_provenance=MaskProvenance.SAM_VIT_B,
        physical_identity=phys_hyp,
    )

    ent_child = PerceptionEntityOutput(
        entity_id="frame_01",
        source_image_id="img_0.jpg",
        source_engine="grounding_dino_tiny",
        class_name="window_frame",
        confidence=0.88,
        calibrated_confidence=0.85,
        box_2d=(95.0, 95.0, 405.0, 405.0),
        coordinate_convention=CoordinateConvention.PIXEL_XYXY_TOPLEFT,
        parent_candidates=[parent_cand],
        selected_parent_id="window_01",
    )

    rejection = RejectionRecord(
        candidate_id="cand_bad_box",
        source_image_id="img_0.jpg",
        class_name="socket",
        rejection_reason=RejectionReason.CONFIDENCE_BELOW_THRESHOLD,
        details={"score": 0.12, "threshold": 0.35},
    )

    manifest = PerceptionHandoffManifest(
        session_id="sess_sprint10_001",
        images_processed=["img_0.jpg"],
        entities=[ent_parent, ent_child],
        rejections=[rejection],
    )

    valid, errors = manifest.validate_contract()
    assert valid is True
    assert len(errors) == 0

    m_dict = manifest.to_dict()
    assert m_dict["session_id"] == "sess_sprint10_001"
    assert m_dict["total_entities"] == 2
    assert m_dict["total_rejections"] == 1
    assert m_dict["entities"][0]["class_name"] == "window"
    assert m_dict["entities"][1]["selected_parent_id"] == "window_01"


def test_perception_handoff_contract_catches_violations():
    """Verify that contract validation flags degenerate bboxes, duplicate IDs, and self-parenting."""
    ent1 = PerceptionEntityOutput(
        entity_id="ent_duplicate",
        source_image_id="img_0.jpg",
        source_engine="test",
        class_name="door",
        confidence=0.9,
        calibrated_confidence=0.85,
        box_2d=(200.0, 200.0, 100.0, 100.0),  # Degenerate: x2 < x1, y2 < y1
    )

    ent2 = PerceptionEntityOutput(
        entity_id="ent_duplicate",  # Duplicate ID
        source_image_id="img_0.jpg",
        source_engine="test",
        class_name="door",
        confidence=1.5,  # Out of bounds
        calibrated_confidence=0.85,
        box_2d=(100.0, 100.0, 200.0, 200.0),
        selected_parent_id="ent_duplicate",  # Self-parenting
    )

    manifest = PerceptionHandoffManifest(
        session_id="sess_bad",
        entities=[ent1, ent2],
    )

    valid, errors = manifest.validate_contract()
    assert valid is False
    assert any("Duplicate entity_id" in e for e in errors)
    assert any("degenerate box_2d" in e for e in errors)
    assert any("confidence out of bounds" in e for e in errors)
    assert any("cannot be its own parent" in e for e in errors)
