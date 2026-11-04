"""Unit tests for Blueprint Task 8.5: User Corrections.

Verifies:
- reclassify_entity (updates class, name, semantic_labels, writes user-basis decision)
- split_entity (subdivides bounds, creates 2 entities, writes user-basis decision)
- merge_entities (unions bounds, combines observations, rewires relationships, writes user-basis decision)
- API commit and validation guards.
"""

import pytest

from world_ir.corrections import (
    CorrectionError,
    merge_entities,
    reclassify_entity,
    split_entity,
)
from world_ir.schema_v1 import Entity, EntityType, Observation, Relationship, RelationshipKind
from world_ir.world_v1 import WorldIR


def test_reclassify_entity():
    world = WorldIR("test_reclass_world")
    chair = Entity(
        id="ent_chair_1",
        type=EntityType.STRUCTURE,
        name="Unknown Object",
        semantic_labels=["fallback/unknown_object"],
        custom_properties={"entity_class": "fallback/unknown_object"},
    )
    world.entities = {chair.id: chair}

    decision = reclassify_entity(
        world,
        "ent_chair_1",
        "furniture/seating/armchair",
        note="Identified as velvet armchair from photos",
        user_id="annotator_alice",
    )

    assert decision["kind"] == "reclassify"
    assert decision["basis"] == "user"
    assert decision["entity_id"] == "ent_chair_1"
    assert decision["old_class"] == "fallback/unknown_object"
    assert decision["new_class"] == "furniture/seating/armchair"

    # Verify entity state
    assert chair.name == "Armchair"
    assert "furniture/seating/armchair" in chair.semantic_labels
    assert chair.custom_properties["entity_class"] == "furniture/seating/armchair"
    assert chair.custom_properties["classification_basis"] == "user"
    assert chair.custom_properties["user_decision"]["kind"] == "reclassify"

    # Verify world decision ledger
    assert len(world.metadata["user_decisions"]) == 1
    assert world.metadata["user_decisions"][0]["basis"] == "user"


def test_reclassify_missing_entity_raises():
    world = WorldIR("test_empty_world")
    with pytest.raises(CorrectionError, match="not found"):
        reclassify_entity(world, "nonexistent", "furniture/table")


def test_split_entity():
    world = WorldIR("test_split_world")
    sofa_table = Entity(
        id="ent_merged_blob",
        type=EntityType.STRUCTURE,
        name="Merged Furniture",
        semantic_labels=["furniture/sofa"],
        custom_properties={"bounding_box": [(0.0, 0.0, 0.0), (4.0, 2.0, 0.9)]},
        observations=[Observation(id="obs_1"), Observation(id="obs_2")],
    )
    # Another entity targeting the original entity
    lamp = Entity(
        id="ent_lamp",
        type=EntityType.STRUCTURE,
        name="Floor Lamp",
        relationships=[Relationship(kind=RelationshipKind.ADJACENT_TO, target_id="ent_merged_blob")],
    )
    world.entities = {sofa_table.id: sofa_table, lamp.id: lamp}

    ent1, ent2, decision = split_entity(
        world,
        "ent_merged_blob",
        split_axis="x",
        split_ratio=0.5,
        note="Separating sofa from adjacent side table",
        user_id="user_bob",
    )

    assert decision["kind"] == "split"
    assert decision["basis"] == "user"
    assert decision["source_entity_id"] == "ent_merged_blob"
    assert len(decision["created_entity_ids"]) == 2

    # Original entity removed, replaced by two parts
    assert "ent_merged_blob" not in world.entities
    assert ent1.id in world.entities
    assert ent2.id in world.entities

    # Bounds split along X: [0.0, 2.0] and [2.0, 4.0]
    b1 = ent1.custom_properties["bounding_box"]
    b2 = ent2.custom_properties["bounding_box"]
    assert b1[0][0] == 0.0 and b1[1][0] == 2.0
    assert b2[0][0] == 2.0 and b2[1][0] == 4.0

    # Relationship rewired
    assert lamp.relationships[0].target_id == ent1.id

    # Decision recorded
    assert len(world.metadata["user_decisions"]) == 1
    assert world.metadata["user_decisions"][0]["kind"] == "split"


def test_split_invalid_params_raises():
    world = WorldIR("test_split_err_world")
    ent = Entity(id="ent_1", type=EntityType.STRUCTURE, name="Box")
    world.entities = {"ent_1": ent}

    with pytest.raises(CorrectionError, match="invalid split_axis"):
        split_entity(world, "ent_1", split_axis="w")

    with pytest.raises(CorrectionError, match="split_ratio"):
        split_entity(world, "ent_1", split_ratio=0.99)


def test_merge_entities():
    world = WorldIR("test_merge_world")
    left_half = Entity(
        id="sofa_left",
        type=EntityType.STRUCTURE,
        name="Sofa Left",
        semantic_labels=["furniture/sofa"],
        custom_properties={"bounding_box": [(0.0, 0.0, 0.0), (1.5, 1.0, 0.85)]},
        observations=[Observation(id="obs_left_1")],
    )
    right_half = Entity(
        id="sofa_right",
        type=EntityType.STRUCTURE,
        name="Sofa Right",
        semantic_labels=["furniture/sofa"],
        custom_properties={"bounding_box": [(1.4, 0.0, 0.0), (3.0, 1.0, 0.85)]},
        observations=[Observation(id="obs_right_1")],
    )
    cushion = Entity(
        id="cushion_1",
        type=EntityType.STRUCTURE,
        relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="sofa_right")],
    )
    world.entities = {
        left_half.id: left_half,
        right_half.id: right_half,
        cushion.id: cushion,
    }

    primary, decision = merge_entities(
        world,
        primary_entity_id="sofa_left",
        secondary_entity_ids=["sofa_right"],
        new_name="Sectional Sofa",
        note="Merged two detected halves of the same sectional",
        user_id="user_carol",
    )

    assert decision["kind"] == "merge"
    assert decision["basis"] == "user"
    assert decision["primary_entity_id"] == "sofa_left"
    assert "sofa_right" in decision["merged_entity_ids"]

    # Secondary entity removed
    assert "sofa_right" not in world.entities
    assert "sofa_left" in world.entities

    # Name and bounds updated to union
    assert primary.name == "Sectional Sofa"
    bbox = primary.custom_properties["bounding_box"]
    assert bbox[0][0] == 0.0 and bbox[1][0] == 3.0

    # Observations combined
    obs_ids = {o.id for o in primary.observations}
    assert "obs_left_1" in obs_ids
    assert "obs_right_1" in obs_ids

    # Relationships targeting sofa_right rewired to sofa_left
    assert cushion.relationships[0].target_id == "sofa_left"

    # User decision recorded in world metadata
    assert len(world.metadata["user_decisions"]) == 1
    assert world.metadata["user_decisions"][0]["kind"] == "merge"


def test_api_validate_commit_changes_corrections():
    """Verify that _validate_commit_changes accepts reclassify, split, merge."""
    from apps.api.routes_worlds import _validate_commit_changes

    payload = {
        "reclassify": {"new_class": "furniture/table", "note": "verified"},
        "split": {"split_axis": "y", "split_ratio": 0.4},
        "merge": {"secondary_entity_ids": ["ent_2"], "new_name": "Merged"},
    }
    validated = _validate_commit_changes(payload)
    assert "reclassify" in validated
    assert "split" in validated
    assert "merge" in validated
