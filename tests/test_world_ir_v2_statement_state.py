"""Tests for P9-01 WorldIR 2.0: statement-state classification + additive
schema migration (v1 worlds still load).
"""

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType
from world_ir.statement_state import StatementState, classify_statement_state
from world_ir.world_v1 import WorldIR


# ===== Statement-state classification =====

def test_observed_provenance_classifies_observed():
    assert classify_statement_state(Provenance.OBSERVED) == StatementState.OBSERVED


def test_reconstructed_provenance_classifies_derived():
    # RECONSTRUCTED = built from fusing other observed/derived statements.
    assert classify_statement_state(Provenance.RECONSTRUCTED) == StatementState.DERIVED


def test_estimated_and_inferred_provenance_classify_inferred():
    assert classify_statement_state(Provenance.ESTIMATED) == StatementState.INFERRED
    assert classify_statement_state(Provenance.INFERRED) == StatementState.INFERRED


def test_generated_provenance_with_simulation_hint_classifies_simulated():
    assert (
        classify_statement_state(Provenance.GENERATED, is_simulation=True)
        == StatementState.SIMULATED
    )


def test_generated_provenance_with_procedural_hint_classifies_procedural():
    assert (
        classify_statement_state(Provenance.GENERATED, is_procedural=True)
        == StatementState.PROCEDURAL
    )


def test_prediction_hint_always_classifies_predicted():
    assert (
        classify_statement_state(Provenance.ESTIMATED, is_prediction=True)
        == StatementState.PREDICTED
    )


def test_generated_provenance_without_hint_is_unclassified_not_guessed():
    # Ambiguous: GENERATED alone doesn't tell us simulation vs procedural.
    # Honest answer is UNCLASSIFIED, never a fabricated guess.
    assert classify_statement_state(Provenance.GENERATED) == StatementState.UNCLASSIFIED


def test_unknown_provenance_is_unclassified():
    assert classify_statement_state(Provenance.UNKNOWN) == StatementState.UNCLASSIFIED


def test_conflict_provenance_is_unclassified():
    assert classify_statement_state(Provenance.CONFLICT) == StatementState.UNCLASSIFIED


# ===== Entity statement_state round-trip =====

def test_entity_statement_state_defaults_to_none():
    entity = Entity(id="ent-1", provenance=Provenance.UNKNOWN)
    assert entity.statement_state is None
    d = entity.to_dict()
    assert d["statement_state"] is None
    restored = Entity.from_dict(d)
    assert restored.statement_state is None


def test_entity_statement_state_round_trips():
    entity = Entity(id="ent-1", provenance=Provenance.OBSERVED, statement_state=StatementState.OBSERVED)
    d = entity.to_dict()
    assert d["statement_state"] == "OBSERVED"
    restored = Entity.from_dict(d)
    assert restored.statement_state == StatementState.OBSERVED


# ===== Additive schema migration: v1-only worlds still load =====

def test_v1_only_entity_dict_loads_without_statement_state_key():
    """A hand-built v1 entity dict, using ONLY fields that existed before
    this change (no statement_state key at all), must still load -- and
    must come back with statement_state=None, never a fabricated value."""
    v1_entity_dict = {
        "id": "ent-legacy-1",
        "type": "building",
        "name": "Legacy Building",
        "transform": None,
        "geometry_ids": [],
        "material_ids": [],
        "surface_ids": [],
        "component_ids": [],
        "relationships": [],
        "semantic_labels": ["legacy"],
        "observations": [],
        "temporal_events": [],
        "provenance": "OBSERVED",
        "confidence": 0.9,
        "uncertainty": {"confidence": 0.9},
        "custom_properties": {},
        # deliberately no "statement_state" key
    }
    entity = Entity.from_dict(v1_entity_dict)
    assert entity.id == "ent-legacy-1"
    assert entity.type == EntityType.BUILDING
    assert entity.statement_state is None


def test_v1_only_world_dict_loads_without_error():
    """A hand-built v1 WorldIR dict (schema_version 1, no v2 fields
    anywhere) must load cleanly through the updated schema/validation
    path, proving the migration is additive and non-breaking."""
    v1_world_dict = {
        "schema_version": 1,
        "id": "world-legacy-1",
        "name": "Legacy World",
        "version": 1,
        "created_at": 0.0,
        "modified_at": 0.0,
        "entities": {
            "ent-legacy-1": {
                "id": "ent-legacy-1",
                "type": "building",
                "name": "Legacy Building",
                "transform": None,
                "geometry_ids": [],
                "material_ids": [],
                "surface_ids": [],
                "component_ids": [],
                "relationships": [],
                "semantic_labels": [],
                "observations": [],
                "temporal_events": [],
                "provenance": "OBSERVED",
                "confidence": 0.9,
                "uncertainty": {"confidence": 0.9},
                "custom_properties": {},
            }
        },
        "geometries": {},
        "materials": {},
        "surfaces": {},
        "components": {},
        "temporal_state": {},
        "temporal_events": {},
        "causal_relations": [],
        "main_branch_id": "branch-main-legacy",
        "branches": {},
        "scenarios": {},
        "coordinate_frame": "world",
        "transforms": {},
        "observations": {},
        "global_provenance": "UNKNOWN",
        "global_confidence": 0.5,
        "global_uncertainty": {"confidence": 0.5},
        "metadata": {},
    }

    world = WorldIR.from_dict(v1_world_dict)

    assert world.id == "world-legacy-1"
    assert len(world.entities) == 1
    entity = world.entities["ent-legacy-1"]
    assert entity.statement_state is None

    issues = world.validate()
    assert issues == []
