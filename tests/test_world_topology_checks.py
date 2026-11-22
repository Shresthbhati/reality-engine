"""Graph-level topology checks in validate_world_ir: containment cycles / self-relations are errors; disconnected rooms are a warning."""
from __future__ import annotations

from world_ir import WorldIR
from world_ir.schema_v1 import Entity, EntityType, Relationship, RelationshipKind
from world_ir.validation import ValidationSeverity, validate_world_ir

R = RelationshipKind


def _world(*ents: Entity) -> WorldIR:
    w = WorldIR()
    for e in ents:
        w.entities[e.id] = e
    return w


def _codes(w: WorldIR, severity=None) -> list[str]:
    return [i.code for i in validate_world_ir(w).issues if severity is None or i.severity is severity]


def _room(i: str, *rels: Relationship) -> Entity:
    return Entity(id=i, type=EntityType.ROOM, relationships=list(rels))


def test_containment_cycle_is_an_error():
    a = Entity(id="a", type=EntityType.WALL, relationships=[Relationship(R.PART_OF, "b")])
    b = Entity(id="b", type=EntityType.WALL, relationships=[Relationship(R.PART_OF, "a")])
    assert "topology_containment_cycle" in _codes(_world(a, b), ValidationSeverity.ERROR)


def test_contains_and_part_of_are_the_same_direction_so_a_consistent_pair_is_not_a_cycle():
    room = Entity(id="room", type=EntityType.ROOM, relationships=[Relationship(R.CONTAINS, "wall")])
    wall = Entity(id="wall", type=EntityType.WALL, relationships=[Relationship(R.PART_OF, "room")])
    assert "topology_containment_cycle" not in _codes(_world(room, wall))


def test_self_relationship_is_an_error():
    a = Entity(id="a", type=EntityType.WALL, relationships=[Relationship(R.INSIDE, "a")])
    assert "topology_self_relationship" in _codes(_world(a), ValidationSeverity.ERROR)


def test_rooms_without_a_connection_warn_and_are_never_an_error():
    w = _world(_room("r1"), _room("r2"))
    assert "topology_disconnected_rooms" in _codes(w, ValidationSeverity.WARNING)
    assert "topology_disconnected_rooms" not in _codes(w, ValidationSeverity.ERROR)


def test_a_door_connecting_two_rooms_joins_them():
    door = Entity(id="d", type=EntityType.DOOR, relationships=[Relationship(R.CONNECTS, "r1"), Relationship(R.CONNECTS, "r2")])
    assert "topology_disconnected_rooms" not in _codes(_world(_room("r1"), _room("r2"), door))


def test_direct_room_adjacency_joins_rooms_and_a_single_room_never_warns():
    assert "topology_disconnected_rooms" not in _codes(_world(_room("r1", Relationship(R.ADJOINS, "r2")), _room("r2")))
    assert "topology_disconnected_rooms" not in _codes(_world(_room("r1")))
