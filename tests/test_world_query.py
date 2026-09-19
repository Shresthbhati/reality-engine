"""Tests for the WorldQuery facade (engine/scene_graph/query.py)."""

from __future__ import annotations

from world_ir.schema_v1 import (
    Entity, EntityType, Geometry, GeometryType, Observation,
    Relationship, RelationshipKind, TemporalEvent, TemporalEventType, Vector3,
)
from world_ir.world_v1 import WorldIR

from engine.scene_graph.query import WorldQuery


def _world_with(*entities, geometries=()) -> WorldIR:
    world = WorldIR()
    for e in entities:
        world.entities[e.id] = e
    for g in geometries:
        world.geometries[g.id] = g
    return world


def test_contents_of_and_container_of_delegate_to_scene_graph():
    room = Entity(id="room-1", type=EntityType.ROOM)
    chair = Entity(
        id="chair-1", type=EntityType.DEBRIS,
        relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="room-1")],
    )
    world = _world_with(room, chair)
    q = WorldQuery(world)

    assert [e.id for e in q.contents_of("room-1")] == ["chair-1"]
    assert q.container_of("chair-1").id == "room-1"


def test_query_by_kind_filters_by_entity_type():
    world = _world_with(
        Entity(id="city-1", type=EntityType.CITY),
        Entity(id="district-1", type=EntityType.DISTRICT),
        Entity(id="room-1", type=EntityType.ROOM),
        Entity(id="room-2", type=EntityType.ROOM),
    )
    q = WorldQuery(world)
    assert [e.id for e in q.query_by_kind(EntityType.ROOM)] == ["room-1", "room-2"]
    assert [e.id for e in q.query_by_kind(EntityType.CITY)] == ["city-1"]


def test_intersects_requires_real_geometry_bounds():
    geom = Geometry(id="geom-1", type=GeometryType.BOX, bounds_min=Vector3(0, 0, 0), bounds_max=Vector3(2, 2, 2))
    overlapping = Entity(id="ent-1", geometry_ids=[geom.id])
    # Point placement only, no geometry bounds -- must NOT satisfy intersects().
    point_only = Entity(id="ent-2", transform={"position": {"x": 1.0, "y": 1.0, "z": 1.0}})
    world = _world_with(overlapping, point_only, geometries=[geom])
    q = WorldQuery(world)

    hits = q.intersects((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert [e.id for e in hits] == ["ent-1"]


def test_by_confidence_filters():
    world = _world_with(
        Entity(id="hi", confidence=0.9),
        Entity(id="lo", confidence=0.2),
    )
    q = WorldQuery(world)
    assert [e.id for e in q.by_confidence(0.5)] == ["hi"]


def test_by_evidence_matches_observation_id_or_uri():
    world = _world_with(
        Entity(id="ent-1", observations=[Observation(id="obs-1", data_uri="file:///scan-42.jpg")]),
        Entity(id="ent-2", observations=[Observation(id="obs-2", data_uri="file:///scan-99.jpg")]),
    )
    q = WorldQuery(world)
    assert [e.id for e in q.by_evidence("obs-1")] == ["ent-1"]
    assert [e.id for e in q.by_evidence("scan-99")] == ["ent-2"]


def test_as_of_excludes_not_yet_created_and_already_destroyed():
    world = _world_with(
        Entity(id="always"),
        Entity(id="created-late", temporal_events=[
            TemporalEvent(type=TemporalEventType.CREATION, timestamp=100.0),
        ]),
        Entity(id="destroyed-early", temporal_events=[
            TemporalEvent(type=TemporalEventType.CREATION, timestamp=0.0),
            TemporalEvent(type=TemporalEventType.DESTRUCTION, timestamp=5.0),
        ]),
    )
    q = WorldQuery(world)
    present_at_10 = {e.id for e in q.as_of(10.0)}
    assert present_at_10 == {"always"}


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
