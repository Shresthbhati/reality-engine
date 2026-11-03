"""Tests for Fine-Grained Entity Attributes and Subtype Intelligence (Wave 1, Blocks 1 & 2)."""

from __future__ import annotations

import pytest

from perception.entities.attributes import (
    EntityAttributes,
    MechanismType,
    MountingType,
    OperationalState,
    SUBTYPES_BY_CLASS,
    infer_entity_attributes,
)
from world_ir.schema_v1 import Entity, EntityType, Provenance
from world_ir.world_v1 import WorldIR


def test_fine_grained_subtypes_registered():
    """Verify standard architectural and furniture classes have registered granular subtypes."""
    assert "furniture.chair" in SUBTYPES_BY_CLASS
    assert "dining" in SUBTYPES_BY_CLASS["furniture.chair"]
    assert "office_swivel" in SUBTYPES_BY_CLASS["furniture.chair"]

    assert "opening.door" in SUBTYPES_BY_CLASS
    assert "sliding" in SUBTYPES_BY_CLASS["opening.door"]
    assert "pocket" in SUBTYPES_BY_CLASS["opening.door"]

    assert "electrical.switchboard.module.switch" in SUBTYPES_BY_CLASS
    assert "rocker" in SUBTYPES_BY_CLASS["electrical.switchboard.module.switch"]
    assert "dimmer_rotary" in SUBTYPES_BY_CLASS["electrical.switchboard.module.switch"]


def test_door_attribute_inference():
    """Test door attribute inference from dimensions and visual tags."""
    attr = infer_entity_attributes(
        class_id="opening.door",
        dimensions_m=(0.90, 2.10, 0.045),
        visual_tags=["sliding", "closed", "wood"],
        host_type="envelope.wall",
    )

    assert attr.class_id == "opening.door"
    assert attr.subtype == "sliding"
    assert attr.mechanism == MechanismType.SLIDING
    assert attr.mounting == MountingType.WALL_MOUNTED
    assert attr.state == OperationalState.CLOSED
    assert attr.dimensions_m == (0.90, 2.10, 0.045)
    assert attr.clearance_m == pytest.approx(0.10)
    assert attr.confidence >= 0.80

    # Serialization roundtrip
    d = attr.to_dict()
    assert d["mechanism"] == "sliding"
    assert d["state"] == "closed"
    assert d["dimensions_m"] == [0.9, 2.1, 0.045]

    restored = EntityAttributes.from_dict(d)
    assert restored.mechanism == MechanismType.SLIDING
    assert restored.mounting == MountingType.WALL_MOUNTED


def test_switchboard_attribute_attachment():
    """Test inferring switchboard attributes and attaching to a WorldIR entity."""
    world = WorldIR()
    entity = Entity(
        id="ent-switch-1",
        name="Bedside Switchboard",
        type=EntityType.STRUCTURE,
        provenance=Provenance.OBSERVED,
        confidence=0.92,
    )
    world.entities[entity.id] = entity

    attr = infer_entity_attributes(
        class_id="electrical.switchboard.module.switch",
        dimensions_m=(0.04, 0.04, 0.02),
        visual_tags=["rocker", "on"],
        host_type="envelope.wall",
    )
    assert attr.mechanism == MechanismType.ROCKER
    assert attr.state == OperationalState.ON

    attr.attach_to_entity(entity)
    assert "attributes" in entity.custom_properties
    assert entity.custom_properties["mechanism"] == "rocker"
    assert entity.custom_properties["mounting"] == "wall_mounted"


def test_uncertainty_preservation_on_unobserved_attributes():
    """Ensure system does not fabricate attributes when evidence is missing."""
    attr = infer_entity_attributes(
        class_id="furniture.chair",
        dimensions_m=(0.5, 0.9, 0.5),
        visual_tags=[],
    )
    # Subtype and mechanism remain unknown without visual evidence
    assert attr.subtype is None
    assert attr.mechanism == MechanismType.UNKNOWN
    # Mounting is grounded by physical ground-support prior with documented uncertainty note
    assert attr.mounting == MountingType.FLOOR_MOUNTED
    assert any("inferred from furniture ground support prior" in note for note in attr.uncertainty_notes)
