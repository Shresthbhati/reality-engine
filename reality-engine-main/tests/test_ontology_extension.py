"""Tests for the EntityType ontology extension (architectural/infrastructure granularity).

Verifies only that the new enum values exist, round-trip through Entity
serialization, and that no original value was removed or renamed. Does NOT
test any inference/classification logic — none was added.
"""

from world_ir.schema_v1 import Entity, EntityType

ORIGINAL_VALUES = {
    "BUILDING": "building",
    "STRUCTURE": "structure",
    "VEHICLE": "vehicle",
    "TERRAIN": "terrain",
    "VEGETATION": "vegetation",
    "WATER": "water",
    "NATURAL_HAZARD": "natural_hazard",
    "DEBRIS": "debris",
    "SENSOR": "sensor",
    "UNKNOWN": "unknown",
}

NEW_VALUES = [
    "room", "wall", "floor", "ceiling", "roof", "column", "beam",
    "door", "window", "stairs", "road", "curb", "sidewalk", "infrastructure",
]


def test_original_values_unchanged():
    for name, value in ORIGINAL_VALUES.items():
        member = getattr(EntityType, name)
        assert member.value == value


def test_new_values_are_valid_entitytype_constructions():
    for value in NEW_VALUES:
        assert EntityType(value).value == value


def test_new_values_round_trip_through_entity_serialization():
    for value in NEW_VALUES:
        entity = Entity(type=EntityType(value))
        restored = Entity.from_dict(entity.to_dict())
        assert restored.type == EntityType(value)
        assert restored.type.value == value
