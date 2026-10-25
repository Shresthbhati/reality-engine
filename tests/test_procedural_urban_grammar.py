"""Tests for procedural/urban_grammar.py (corridor + city builders)."""

from __future__ import annotations

import json

import pytest

from procedural.room_grammar import RoomGrammarSpec, generate_room
from procedural.urban_grammar import (
    CityGrammarSpec,
    CorridorGrammarSpec,
    generate_city,
    generate_corridor,
)
from world_ir.schema_v1 import EntityType, RelationshipKind
from world_ir.statement_state import StatementState
from world_ir.world_v1 import WorldIR


def _types(world, etype):
    return [e for e in world.entities.values() if e.type == etype]


def _dump(world):
    return json.dumps(world.to_dict(), sort_keys=True)


def test_corridor_has_two_long_walls_and_is_procedural():
    w = generate_corridor(CorridorGrammarSpec("hall", length=10, width=2, height=2.6))
    assert len(_types(w, EntityType.CORRIDOR)) == 1
    assert not _types(w, EntityType.ROOM)
    assert sorted(e.id for e in _types(w, EntityType.WALL)) == ["hall-wall-north", "hall-wall-south"]
    assert all(e.statement_state == StatementState.PROCEDURAL for e in w.entities.values())
    # Long walls span the corridor length.
    g = w.geometries["geom-hall-wall-north"]
    assert g.bounds_max.x - g.bounds_min.x == pytest.approx(10)


def test_corridor_y_axis_uses_side_walls_on_x():
    w = generate_corridor(CorridorGrammarSpec("c", length=8, axis="y"))
    assert sorted(e.id for e in _types(w, EntityType.WALL)) == ["c-wall-east", "c-wall-west"]


def test_corridor_connects_existing_rooms_and_rejects_unknown():
    w = generate_room(RoomGrammarSpec("a", 4, 3, 2.4))
    generate_corridor(CorridorGrammarSpec("c", 6, connects=["a-room"]), world=w, origin=(4, 0, 0))
    (c,) = _types(w, EntityType.CORRIDOR)
    assert [(r.kind, r.target_id) for r in c.relationships] == [(RelationshipKind.ADJACENT_TO, "a-room")]
    with pytest.raises(ValueError):
        generate_corridor(CorridorGrammarSpec("d", 6, connects=["nope"]), world=w)


@pytest.mark.parametrize("bad", [dict(length=0), dict(length=1, width=-1), dict(length=1, axis="z")])
def test_corridor_rejects_bad_spec(bad):
    with pytest.raises(ValueError):
        CorridorGrammarSpec("c", **bad)


def test_city_structure_counts():
    w = generate_city(CityGrammarSpec("town", blocks_x=3, blocks_y=2, seed=1))
    assert len(_types(w, EntityType.BUILDING)) == 6
    assert len(_types(w, EntityType.STREET)) == (2 + 1) + (3 + 1)
    (city,) = _types(w, EntityType.CITY)
    assert len(city.relationships) == 6 + 7
    assert all(e.statement_state == StatementState.PROCEDURAL for e in w.entities.values())


def test_city_is_deterministic_and_seed_changes_heights():
    a = generate_city(CityGrammarSpec("t", 2, 2, seed=5))
    b = generate_city(CityGrammarSpec("t", 2, 2, seed=5))
    c = generate_city(CityGrammarSpec("t", 2, 2, seed=6))
    assert _dump(a) == _dump(b)
    assert _dump(a) != _dump(c)


def test_city_buildings_stay_off_streets():
    w = generate_city(CityGrammarSpec("t", 2, 2, lot_size=20, street_width=8))
    g = w.geometries["geom-t-b0-0-floor"]
    # Local footprint starts at (8, 8); world z = -local y, so z spans [-28, -8].
    assert g.bounds_min.x == pytest.approx(8) and g.bounds_max.z == pytest.approx(-8)


def test_city_street_slab_is_thin_along_y_not_z():
    w = generate_city(CityGrammarSpec("t", 2, 2, lot_size=20, street_width=8))
    g = w.geometries["geom-t-street-x0"]
    assert g.bounds_max.y - g.bounds_min.y == pytest.approx(0.2)
    assert g.bounds_max.z - g.bounds_min.z == pytest.approx(8)


@pytest.mark.parametrize("bad", [dict(blocks_x=0), dict(blocks_x=11, blocks_y=10),
                                  dict(lot_size=0), dict(min_height=10, max_height=5)])
def test_city_rejects_bad_spec(bad):
    with pytest.raises(ValueError):
        CityGrammarSpec("t", **bad)


def test_round_trip():
    w = generate_city(CityGrammarSpec("t", 2, 1))
    restored = WorldIR.from_dict(json.loads(json.dumps(w.to_dict())))
    assert set(restored.entities) == set(w.entities)
