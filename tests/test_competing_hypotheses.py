"""Unit tests for competing hypotheses and perception reconciliation (Master Development Plan Tasks 7.3 & 7.6)."""

import pytest

from provenance import Provenance
from world_ir import (
    CompetingHypothesis,
    Entity,
    EntityType,
    HypothesisSet,
    ReconciliationStrategy,
    WorldIR,
    attach_competing_hypothesis,
    get_active_hypothesis,
    get_competing_hypotheses,
    reconcile_hypotheses,
)


def test_competing_hypotheses_preservation():
    entity = Entity(id="ent-room-1", type=EntityType.ROOM, name="Room 1")

    # Model A: PolyLayout predicts bedroom (conf 0.60)
    attach_competing_hypothesis(
        entity=entity,
        domain="room_role",
        statement={"role": "bedroom"},
        producer="polylayout_model",
        confidence=0.60,
        supporting_evidence_ids=["img-01", "img-02"],
        rationale="Rectangular geometry with bed prior",
    )

    # Model B: Object classifier sees sink & toilet, predicts bathroom (conf 0.85)
    attach_competing_hypothesis(
        entity=entity,
        domain="room_role",
        statement={"role": "bathroom"},
        producer="fixture_detector",
        confidence=0.85,
        supporting_evidence_ids=["img-03"],
        rationale="Presence of porcelain sanitary fixtures",
    )

    hypotheses = get_competing_hypotheses(entity, "room_role")
    assert len(hypotheses) == 2
    # Verify both models are preserved (never averaged or overwritten)
    producers = {h.producer for h in hypotheses}
    assert "polylayout_model" in producers
    assert "fixture_detector" in producers

    # Active hypothesis should be the higher confidence one (bathroom)
    active = get_active_hypothesis(entity, "room_role")
    assert active is not None
    assert active.producer == "fixture_detector"
    assert active.statement["role"] == "bathroom"


def test_evidence_weighted_reconciliation():
    hyp_set = HypothesisSet(target_id="ent-wall-1", domain="classification")

    # Hypothesis 1: high confidence (0.75) but 0 supporting views
    hyp1 = CompetingHypothesis(
        hypothesis_id="h1",
        producer="single_view_detector",
        statement={"class": "partition_wall"},
        confidence=0.75,
        supporting_evidence_ids=(),
    )

    # Hypothesis 2: moderate confidence (0.65) with 5 multi-view cross-references
    hyp2 = CompetingHypothesis(
        hypothesis_id="h2",
        producer="multiview_plane_sweep",
        statement={"class": "load_bearing_wall"},
        confidence=0.65,
        supporting_evidence_ids=("img-1", "img-2", "img-3", "img-4", "img-5"),
    )

    hyp_set.hypotheses = [hyp1, hyp2]

    # Under EVIDENCE_WEIGHTED, hyp2 wins because of multi-view support
    winner = reconcile_hypotheses(hyp_set, strategy=ReconciliationStrategy.EVIDENCE_WEIGHTED)
    assert winner.hypothesis_id == "h2"
    assert winner.statement["class"] == "load_bearing_wall"


def test_human_correction_strict_override():
    hyp_set = HypothesisSet(target_id="ent-chair-1", domain="classification")

    # Machine model claims dining chair with 0.95 confidence
    h_ai = CompetingHypothesis(
        hypothesis_id="h-ai",
        producer="learned_detector_v2",
        statement={"entity_class": "furniture.chair"},
        confidence=0.95,
        provenance=Provenance.INFERRED,
    )

    # User manually tags it as an armchair / recliner
    h_user = CompetingHypothesis(
        hypothesis_id="h-user",
        producer="user_correction",
        statement={"entity_class": "furniture.sofa"},
        confidence=0.50,
        provenance=Provenance.ESTIMATED,
        rationale="User marked as recliner sofa in Studio",
    )

    hyp_set.hypotheses = [h_ai, h_user]

    # Human correction must strictly supersede AI
    winner = reconcile_hypotheses(hyp_set, strategy=ReconciliationStrategy.HIGHEST_CONFIDENCE)
    assert winner.hypothesis_id == "h-user"
    assert winner.statement["entity_class"] == "furniture.sofa"
    assert hyp_set.strategy_used == ReconciliationStrategy.USER_OVERRIDE.value


def test_conflict_detection_flagging():
    hyp_set = HypothesisSet(target_id="ent-slab-1", domain="material")

    # Appearance model sees wood (conf 0.82)
    h_wood = CompetingHypothesis(
        hypothesis_id="h-wood",
        producer="color_appearance_model",
        statement={"material": "wood_floor"},
        confidence=0.82,
        supporting_evidence_ids=("img-1",),
    )

    # Specular model sees tile (conf 0.80)
    h_tile = CompetingHypothesis(
        hypothesis_id="h-tile",
        producer="reflectance_sheen_model",
        statement={"material": "tile"},
        confidence=0.80,
        supporting_evidence_ids=("img-2",),
    )

    hyp_set.hypotheses = [h_wood, h_tile]

    winner = reconcile_hypotheses(hyp_set, conflict_threshold=0.70)
    # Both are >= 0.70 with diff < 0.15, so conflict must be flagged
    assert hyp_set.conflict_detected is True
    assert hyp_set.conflict_note is not None
    assert "Unresolved conflict" in hyp_set.conflict_note


def test_worldir_serialization_with_hypotheses():
    world = WorldIR(id="world-hyp-test")
    entity = Entity(id="ent-1", type=EntityType.ROOM, name="Kitchen or Living")

    attach_competing_hypothesis(
        entity=entity,
        domain="role",
        statement={"role": "kitchen"},
        producer="detector_a",
        confidence=0.70,
    )
    attach_competing_hypothesis(
        entity=entity,
        domain="role",
        statement={"role": "living"},
        producer="detector_b",
        confidence=0.60,
    )

    world.entities[entity.id] = entity

    # Serialize to dict and reload
    w_dict = world.to_dict()
    assert "hypotheses" in w_dict["entities"]["ent-1"]["custom_properties"]
    assert "role" in w_dict["entities"]["ent-1"]["custom_properties"]["hypotheses"]

    reloaded_world = WorldIR.from_dict(w_dict)
    reloaded_ent = reloaded_world.entities["ent-1"]

    candidates = get_competing_hypotheses(reloaded_ent, "role")
    assert len(candidates) == 2
    active = get_active_hypothesis(reloaded_ent, "role")
    assert active is not None
    assert active.statement["role"] == "kitchen"
