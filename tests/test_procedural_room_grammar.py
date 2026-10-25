"""Tests for procedural/room_grammar.py (P15-01: procedural world
generation -- "WorldIR -> urban/room grammar -> constraints ->
generated worlds").

Spec rules under test:

  - A RoomGrammarSpec is DATA: dimensions, wall openings, seed. The
    SAME generator produces rooms, and a floor layout from N rooms --
    no per-shape special cases.
  - Generation is DETERMINISTIC: same spec + seed -> byte-identical
    world (derivable, reproducible). PROCEDURAL statement state: every
    generated entity records its provenance as
    StatementState.PROCEDURAL (Provenance.GENERATED) -- generated
    geometry never masquerades as observed (constitution).
  - The grammar's constraints hold by construction: walls of a room
    bound its floor extent (wall AABBs meet at corners), the ceiling
    spans the floor, openings (doors/windows) are carved as gaps in
    the wall bounds they belong to -- openings are represented as
    separate entities with measured bounds, not decorative labels.
  - A floor layout places rooms on a deterministic grid with a real
    shared-wall structure; the WorldIR output validates
    (world_ir validation) and round-trips through to_dict/from_dict.
"""

from __future__ import annotations

import json

import pytest

from procedural.room_grammar import (
    RoomGrammarSpec,
    WallOpening,
    generate_floor_layout,
    generate_room,
)
from world_ir.schema_v1 import EntityType
from world_ir.statement_state import StatementState


class TestRoomGeneration:
    def test_generates_bounded_entities(self):
        spec = RoomGrammarSpec(
            name="office", width=4.0, depth=3.0, height=2.5)
        world = generate_room(spec)
        types = {e.type for e in world.entities.values()}
        assert {"wall", "floor", "ceiling"} <= types
        # Every geometric entity carries real bounds (the semantic
        # room entity is a relationship node, not geometry).
        for entity in world.entities.values():
            if entity.type == EntityType.ROOM:
                continue
            assert entity.geometry_ids
        for geom in world.geometries.values():
            assert geom.bounds_min is not None
            assert geom.bounds_max is not None
            assert geom.bounds_max.x > geom.bounds_min.x
            assert geom.bounds_max.y > geom.bounds_min.y
            assert geom.bounds_max.z > geom.bounds_min.z

    def test_procedural_provenance_everywhere(self):
        spec = RoomGrammarSpec(name="office", width=4.0, depth=3.0,
                               height=2.5)
        world = generate_room(spec)
        for entity in world.entities.values():
            assert entity.statement_state == StatementState.PROCEDURAL
            assert entity.provenance.value == "GENERATED"

    def test_deterministic_same_spec(self):
        spec = RoomGrammarSpec(name="office", width=4.0, depth=3.0,
                               height=2.5, seed=7)
        a = generate_room(spec)
        b = generate_room(spec)
        assert json.dumps(a.to_dict(), sort_keys=True) == \
            json.dumps(b.to_dict(), sort_keys=True)

    def test_openings_are_real_entities_with_bounds(self):
        spec = RoomGrammarSpec(
            name="office", width=4.0, depth=3.0, height=2.5,
            openings=[
                WallOpening(wall="south", kind="door",
                            lateral_offset=0.5, width=0.9,
                            bottom=0.0, top=2.1),
                WallOpening(wall="north", kind="window",
                            lateral_offset=1.0, width=1.2,
                            bottom=0.9, top=2.0),
            ])
        world = generate_room(spec)
        doors = [e for e in world.entities.values()
                 if e.type == "door"]
        windows = [e for e in world.entities.values()
                   if e.type == "window"]
        assert len(doors) == 1 and len(windows) == 1
        door_geom = world.geometries[doors[0].geometry_ids[0]]
        # The door's measured bounds: 0.9 wide, 2.1 high, at bottom 0.
        dx = door_geom.bounds_max.x - door_geom.bounds_min.x
        dy = door_geom.bounds_max.y - door_geom.bounds_min.y
        assert dx == pytest.approx(0.9)
        assert dy == pytest.approx(2.1)


class TestFloorLayout:
    def test_grid_layout_with_shared_walls(self):
        specs = [
            RoomGrammarSpec(name=f"room{i}", width=3.0, depth=3.0,
                            height=2.5, seed=i)
            for i in range(4)
        ]
        world = generate_floor_layout(specs, columns=2)
        rooms = [e for e in world.entities.values() if e.type == "room"]
        walls = [e for e in world.entities.values() if e.type == "wall"]
        assert len(rooms) == 4
        # 4 rooms in a 2x2 grid: 3 wall lines x 2 segments + ... more
        # than one wall per room boundary proves sharing happened
        # (a naive per-room generator would emit 16 walls; sharing
        # yields fewer).
        assert len(walls) < 16
        assert len(walls) >= 4

    def test_layout_is_deterministic(self):
        specs = [RoomGrammarSpec(name=f"r{i}", width=3.0, depth=3.0,
                                 height=2.5, seed=i) for i in range(3)]
        a = generate_floor_layout(specs, columns=2)
        b = generate_floor_layout(specs, columns=2)
        assert json.dumps(a.to_dict(), sort_keys=True) == \
            json.dumps(b.to_dict(), sort_keys=True)

    def test_layout_roundtrips(self):
        specs = [RoomGrammarSpec(name="r0", width=3.0, depth=3.0,
                                 height=2.5)]
        world = generate_floor_layout(specs, columns=1)
        d = json.loads(json.dumps(world.to_dict()))
        from world_ir.world_v1 import WorldIR
        restored = WorldIR.from_dict(d)
        assert len(restored.entities) == len(world.entities)


class TestYUpFrame:
    """Grammars author Z-up locally but emit the project's Y-up frame."""

    def test_room_height_is_world_y_footprint_is_x_and_neg_z(self):
        world = generate_room(RoomGrammarSpec("r", 4.0, 3.0, 2.4))
        b = lambda eid: world.geometries[f"geom-{eid}"]
        t = 0.2
        ceil, floor = b("r-ceiling"), b("r-floor")
        assert ceil.bounds_min.y == pytest.approx(2.4)
        assert ceil.bounds_max.y == pytest.approx(2.4 + t)
        assert floor.bounds_min.y == pytest.approx(-t)
        assert floor.bounds_max.y == pytest.approx(0.0)
        # Footprint: x in [0, 4], local y in [0, 3] -> world z in [-3, 0].
        assert (floor.bounds_min.x, floor.bounds_max.x) == pytest.approx((0.0, 4.0))
        assert (floor.bounds_min.z, floor.bounds_max.z) == pytest.approx((-3.0, 0.0))
        for side in ("north", "south", "east", "west"):
            w = b(f"r-wall-{side}")
            assert w.bounds_min.y == pytest.approx(0.0)
            assert w.bounds_max.y == pytest.approx(2.4)

    def test_generate_room_is_byte_identical_twice(self):
        spec = RoomGrammarSpec("r", 4.0, 3.0, 2.4, openings=[
            WallOpening("south", "door", 0.5, 0.9, 0.0, 2.1)])
        a = json.dumps(generate_room(spec).to_dict(), sort_keys=True)
        b = json.dumps(generate_room(spec).to_dict(), sort_keys=True)
        assert a == b
