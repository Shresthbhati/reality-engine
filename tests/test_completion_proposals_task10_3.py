"""Unit tests for Completion Proposals with PRIOR_GENERATED (Blueprint Task 10.3 & Phase 10 Gate).

Verifies:
1. Completion proposals carry strictly `provenance = Provenance.GENERATED`,
   `statement_state = StatementState.PREDICTED`, and `evidence_state = "PRIOR_GENERATED"`.
2. Proposals are visibly marked (`visibly_marked = True`, display pattern dashed).
3. Invariant 3: NEVER labelled `OBSERVED` or `PARTIALLY_OBSERVED`, observations are empty.
4. Attached proposals pass `validate_world_ir` with 0 errors.
5. Acceptance gate checks continuity, collision, and evidence honesty.
6. Accepting a proposal preserves its PRIOR_GENERATED provenance and never auto-promotes to OBSERVED.
"""

from __future__ import annotations

import pytest

from provenance import Provenance
from reconstruction.completion import (
    CompletionGateVerdict,
    CompletionMethod,
    CompletionProposal,
    accept_completion_proposal,
    attach_completion_proposals,
    evaluate_completion_acceptance_gate,
    generate_completion_proposals_for_world,
    propose_backside_completion,
    propose_planar_extrapolation,
    propose_symmetry_completion,
)
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Observation,
    Vector3,
)
from world_ir.statement_state import StatementState
from world_ir.validation import validate_world_ir
from world_ir.world_v1 import WorldIR


def _create_test_world() -> WorldIR:
    world = WorldIR(id="world-test-completion")

    # Floor plane
    floor_geom = Geometry(
        id="geom-floor",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(6.0, 0.1, 6.0),
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.95,
    )
    floor_ent = Entity(
        id="ent-floor",
        type=EntityType.FLOOR,
        name="Main Floor",
        geometry_ids=["geom-floor"],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.95,
        observations=[Observation(id="obs-1", sensor_type="camera")],
    )
    floor_ent.custom_properties["evidence_state"] = "OBSERVED"

    # Wall segment (partially observed)
    wall_geom = Geometry(
        id="geom-wall-1",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(3.0, 2.5, 0.2),
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.85,
    )
    wall_ent = Entity(
        id="ent-wall-1",
        type=EntityType.WALL,
        name="North Wall",
        geometry_ids=["geom-wall-1"],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.85,
        observations=[Observation(id="obs-2", sensor_type="camera")],
    )
    wall_ent.custom_properties["evidence_state"] = "PARTIALLY_OBSERVED"

    # Desk / Furniture placed near the wall
    desk_geom = Geometry(
        id="geom-desk",
        type=GeometryType.BOX,
        bounds_min=Vector3(1.0, 0.1, 0.5),
        bounds_max=Vector3(2.5, 0.85, 1.2),
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.80,
    )
    desk_ent = Entity(
        id="ent-desk",
        type=EntityType.STRUCTURE,
        name="Study Desk",
        geometry_ids=["geom-desk"],
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.80,
        observations=[Observation(id="obs-3", sensor_type="camera")],
    )
    desk_ent.custom_properties["evidence_state"] = "OBSERVED"

    world.geometries["geom-floor"] = floor_geom
    world.entities["ent-floor"] = floor_ent
    world.geometries["geom-wall-1"] = wall_geom
    world.entities["ent-wall-1"] = wall_ent
    world.geometries["geom-desk"] = desk_geom
    world.entities["ent-desk"] = desk_ent

    return world


def test_propose_backside_completion_invariants():
    world = _create_test_world()
    proposal = propose_backside_completion(world, "ent-desk", depth_m=0.35, unobserved_axis="-z")

    assert proposal is not None
    assert proposal.target_entity_id == "ent-desk"
    assert proposal.completion_method == CompletionMethod.BACKSIDE_CLOSURE
    assert "rear_face" in proposal.unobserved_facets

    cand_ent = proposal.candidate_entity
    cand_geom = proposal.candidate_geometry

    # Strict provenance and invariant checks
    assert cand_ent.provenance == Provenance.GENERATED
    assert cand_ent.statement_state == StatementState.PREDICTED
    assert cand_geom.provenance == Provenance.GENERATED
    assert cand_ent.custom_properties["evidence_state"] == "PRIOR_GENERATED"
    assert cand_ent.custom_properties["is_completion_proposal"] is True
    assert cand_ent.custom_properties["visibly_marked"] is True
    assert cand_ent.custom_properties["display_style"]["pattern"] == "dashed"

    # CRITICAL: Empty observations (never claim sensor evidence)
    assert len(cand_ent.observations) == 0
    assert len(cand_geom.observations) == 0
    assert cand_ent.provenance != Provenance.OBSERVED
    assert cand_ent.custom_properties["evidence_state"] != "OBSERVED"


def test_propose_symmetry_and_planar_extrapolation():
    world = _create_test_world()

    # Symmetry proposal on desk
    sym_prop = propose_symmetry_completion(world, "ent-desk", symmetry_axis="x")
    assert sym_prop is not None
    assert sym_prop.completion_method == CompletionMethod.SYMMETRY_PRIOR
    assert sym_prop.candidate_entity.provenance == Provenance.GENERATED
    assert sym_prop.candidate_entity.custom_properties["evidence_state"] == "PRIOR_GENERATED"

    # Planar extrapolation on wall
    wall_prop = propose_planar_extrapolation(world, "ent-wall-1", extend_dist_m=1.0)
    assert wall_prop is not None
    assert wall_prop.completion_method == CompletionMethod.PLANAR_EXTRAPOLATION
    assert wall_prop.candidate_entity.provenance == Provenance.GENERATED
    assert wall_prop.candidate_entity.custom_properties["evidence_state"] == "PRIOR_GENERATED"
    assert wall_prop.candidate_geometry.bounds_max.x == pytest.approx(4.0)


def test_acceptance_gate_and_evidence_honesty():
    world = _create_test_world()
    proposal = propose_backside_completion(world, "ent-desk", depth_m=0.35, unobserved_axis="-z")
    assert proposal is not None

    verdict = evaluate_completion_acceptance_gate(world, proposal)
    assert verdict.passed is True
    assert verdict.evidence_honesty is True
    assert verdict.continuity_score >= 0.5
    assert verdict.collision_free is True
    assert len(verdict.rejection_reasons) == 0

    # Failure Injection 1: Illegally claim OBSERVED provenance
    dishonest_prop = propose_backside_completion(world, "ent-desk", depth_m=0.35)
    assert dishonest_prop is not None
    dishonest_prop.candidate_entity.provenance = Provenance.OBSERVED
    dishonest_verdict = evaluate_completion_acceptance_gate(world, dishonest_prop)
    assert dishonest_verdict.passed is False
    assert dishonest_verdict.evidence_honesty is False
    assert any("forbidden provenance" in r for r in dishonest_verdict.rejection_reasons)

    # Failure Injection 2: Claiming sensor observations
    dishonest_prop2 = propose_backside_completion(world, "ent-desk", depth_m=0.35)
    assert dishonest_prop2 is not None
    dishonest_prop2.candidate_entity.observations = [Observation(id="fake-obs", sensor_type="camera")]
    dishonest_verdict2 = evaluate_completion_acceptance_gate(world, dishonest_prop2)
    assert dishonest_verdict2.passed is False
    assert dishonest_verdict2.evidence_honesty is False
    assert any("claims sensor observations" in r for r in dishonest_verdict2.rejection_reasons)


def test_attach_proposals_to_world_and_world_validation():
    world = _create_test_world()
    proposals = generate_completion_proposals_for_world(world)
    assert len(proposals) >= 1

    # Attach to world
    updated_world = attach_completion_proposals(world, proposals)

    # Check that candidate entities and geometries are present in WorldIR
    for prop in proposals:
        assert prop.candidate_entity.id in updated_world.entities
        assert prop.candidate_geometry.id in updated_world.geometries

    assert "completion_proposals" in updated_world.metadata
    assert len(updated_world.metadata["completion_proposals"]) == len(proposals)

    # Check WorldIR Validation Invariants (Invariant 3 & Invariant 1)
    report = validate_world_ir(updated_world)
    # Filter out known pre-existing architectural warnings if any
    errors = [issue for issue in report.issues if issue.severity.value == "error"]
    assert len(errors) == 0, f"World validation failed: {[str(e) for e in errors]}"


def test_accept_completion_proposal_preserves_generated_provenance():
    world = _create_test_world()
    proposal = propose_backside_completion(world, "ent-desk", depth_m=0.35, unobserved_axis="-z")
    assert proposal is not None

    world_with_props = attach_completion_proposals(world, [proposal])
    prop_id = proposal.proposal_id

    # Accept the proposal
    accepted_world = accept_completion_proposal(world_with_props, prop_id)

    # Verify acceptance metadata
    accepted_prop = next(
        p for p in accepted_world.metadata["completion_proposals"] if p["proposal_id"] == prop_id
    )
    assert accepted_prop["is_accepted"] is True

    # CRITICAL INVARIANT: The accepted entity MUST STILL HAVE GENERATED / PRIOR_GENERATED provenance
    cand_ent = accepted_world.entities[proposal.candidate_entity.id]
    assert cand_ent.provenance == Provenance.GENERATED
    assert cand_ent.statement_state == StatementState.PREDICTED
    assert cand_ent.custom_properties["evidence_state"] == "PRIOR_GENERATED"
    assert cand_ent.custom_properties["is_accepted_completion"] is True
    assert cand_ent.provenance != Provenance.OBSERVED


def test_proposal_roundtrip_serialization():
    world = _create_test_world()
    proposal = propose_backside_completion(world, "ent-desk", depth_m=0.35, unobserved_axis="-z")
    assert proposal is not None
    evaluate_completion_acceptance_gate(world, proposal)

    d = proposal.to_dict()
    assert d["proposal_id"] == proposal.proposal_id
    assert d["completion_method"] == "backside_closure"
    assert d["candidate_entity"]["custom_properties"]["evidence_state"] == "PRIOR_GENERATED"

    restored = CompletionProposal.from_dict(d)
    assert restored.proposal_id == proposal.proposal_id
    assert restored.completion_method == proposal.completion_method
    assert restored.confidence == pytest.approx(proposal.confidence)
    assert restored.candidate_entity.provenance == Provenance.GENERATED
    assert restored.gate_verdict is not None
    assert restored.gate_verdict.passed is True
