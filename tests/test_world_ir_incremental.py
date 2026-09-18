"""Tests for world_ir/incremental.py (P4.21 regional recompilation).

Distinct from tests/test_incremental.py, which covers the unrelated
generic stage-graph cache in engine/incremental.py.
"""

from __future__ import annotations

from world_ir.incremental import affected_closure
from world_ir.schema_v1 import Entity, Relationship, RelationshipKind
from world_ir.world_v1 import WorldIR


def _world_with(*entities) -> WorldIR:
    world = WorldIR()
    for e in entities:
        world.entities[e.id] = e
    return world


def test_changed_entity_always_included():
    world = _world_with(Entity(id="wall-1"))
    assert affected_closure(world, ["wall-1"]) == {"wall-1"}


def test_propagates_through_part_of_edge():
    room = Entity(id="room-1")
    wall = Entity(
        id="wall-1",
        relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="room-1")],
    )
    world = _world_with(room, wall)
    # wall-1 changed -> room-1 (its container) is affected too.
    assert affected_closure(world, ["wall-1"]) == {"wall-1", "room-1"}


def test_propagates_through_supports_edge_transitively():
    foundation = Entity(id="foundation-1")
    column = Entity(
        id="column-1",
        relationships=[Relationship(kind=RelationshipKind.SUPPORTS, target_id="beam-1")],
    )
    beam = Entity(id="beam-1")
    world = _world_with(foundation, column, beam)
    # column-1 changed -> beam-1 (via SUPPORTS) is affected; foundation-1 is not.
    result = affected_closure(world, ["column-1"])
    assert result == {"column-1", "beam-1"}


def test_does_not_propagate_through_adjacent_to_by_default():
    wall_a = Entity(
        id="wall-a",
        relationships=[Relationship(kind=RelationshipKind.ADJACENT_TO, target_id="wall-b")],
    )
    wall_b = Entity(id="wall-b")
    world = _world_with(wall_a, wall_b)
    assert affected_closure(world, ["wall-a"]) == {"wall-a"}


def test_max_hops_zero_returns_only_changed():
    room = Entity(id="room-1")
    wall = Entity(
        id="wall-1",
        relationships=[Relationship(kind=RelationshipKind.PART_OF, target_id="room-1")],
    )
    world = _world_with(room, wall)
    assert affected_closure(world, ["wall-1"], max_hops=0) == {"wall-1"}


def test_unknown_changed_id_is_dropped():
    world = _world_with(Entity(id="wall-1"))
    assert affected_closure(world, ["wall-1", "ghost-entity"]) == {"wall-1"}


def test_custom_relationship_kinds():
    a = Entity(
        id="a",
        relationships=[Relationship(kind=RelationshipKind.ADJACENT_TO, target_id="b")],
    )
    b = Entity(id="b")
    world = _world_with(a, b)
    result = affected_closure(world, ["a"], relationship_kinds={RelationshipKind.ADJACENT_TO})
    assert result == {"a", "b"}


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
