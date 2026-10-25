"""C1: stored generator specs + scoped regeneration (spec_store)."""

from __future__ import annotations

import json

import pytest

from procedural import spec_store as ss
from world_ir.world_v1 import WorldIR


def _room(name, width=4.0, depth=5.0, height=2.8, origin=(0.0, 0.0, 0.0)):
    return ss.GeneratedSpec(
        "room", name,
        params={"width": width, "depth": depth, "height": height,
                "openings": [], "seed": 3},
        seed=3, origin=list(origin))


def _corridor(name, to, side="east", length=6.0):
    return ss.GeneratedSpec(
        "corridor", name,
        params={"length": length, "width": 2.0, "height": 2.6, "axis": "x",
                "connects": [to]},
        attach={"to": to, "side": side})


def _fp(world, eid):
    return json.dumps(world.entities[eid].to_dict(), sort_keys=True), json.dumps(
        [world.geometries[g].to_dict() for g in world.entities[eid].geometry_ids],
        sort_keys=True)


@pytest.fixture()
def two_rooms_and_corridor():
    world = ss.generate(None, _room("A"))
    world = ss.generate(world, _room("B", origin=(30.0, 0.0, 0.0)))
    return ss.generate(world, _corridor("Hall", "A-room"))


def test_stamping_covers_every_created_entity(two_rooms_and_corridor):
    w = two_rooms_and_corridor
    for eid, e in w.entities.items():
        owner = e.custom_properties["generated_by"]
        assert owner in ("A-room", "B-room", "Hall-corridor"), eid
    assert set(ss.load_specs(w)) == {"A-room", "B-room", "Hall-corridor"}
    assert ss.load_specs(w)["A-room"].generator == "room_grammar@1"


def test_resize_is_scoped_and_dependent_follows(two_rooms_and_corridor):
    w = two_rooms_and_corridor
    b_before = {e: _fp(w, e) for e in ss.owned_ids(w, "B-room")}
    new, changed, removed, unchanged = ss.regenerate(w, "A-room", {"width": 9.0})

    # (a) A's entities changed, B's are byte-identical
    assert "A-floor" in changed and "A-wall-east" in changed
    for eid, fp in b_before.items():
        assert _fp(new, eid) == fp and eid in unchanged and eid not in changed
    # (b) corridor x-min == A's new x-max in world coords
    a_xmax = new.geometries["geom-A-floor"].bounds_max.x
    assert a_xmax == 9.0
    assert new.geometries["geom-Hall-floor"].bounds_min.x == a_xmax
    assert "Hall-floor" in changed
    # input untouched
    assert w.geometries["geom-A-floor"].bounds_max.x == 4.0
    assert removed == []


def test_unchanged_params_regenerate_to_zero_changes(two_rooms_and_corridor):
    w = two_rooms_and_corridor
    new, changed, removed, unchanged = ss.regenerate(w, "A-room", {})
    assert changed == [] and removed == []
    assert json.dumps(new.to_dict(), sort_keys=True) == json.dumps(w.to_dict(), sort_keys=True)
    _, changed, _, _ = ss.regenerate(w, "Hall-corridor", {"length": 6.0})
    assert changed == []


def test_removed_ids_reported_when_openings_dropped():
    spec = _room("A")
    spec.params["openings"] = [{"wall": "north", "kind": "door", "lateral_offset": 1.0,
                                "width": 0.9, "bottom": 0.0, "top": 2.1}]
    w = ss.generate(None, spec)
    assert "A-opening-0" in w.entities
    new, changed, removed, _ = ss.regenerate(w, "A-room", {"openings": []})
    assert removed == ["A-opening-0"] and "geom-A-opening-0" not in new.geometries


def test_attach_sides_and_self_attach_rejected(two_rooms_and_corridor):
    w = two_rooms_and_corridor
    a = w.geometries["geom-A-floor"]
    west, *_ = ss.regenerate(w, "Hall-corridor", {"attach": {"to": "A-room", "side": "west"}})
    assert west.geometries["geom-Hall-floor"].bounds_max.x == a.bounds_min.x
    south, *_ = ss.regenerate(w, "Hall-corridor", {"attach": {"to": "A-room", "side": "south"}})
    # local south = +y = world -z: corridor floor's max-z touches A's min-z
    assert south.geometries["geom-Hall-floor"].bounds_max.z == a.bounds_min.z
    with pytest.raises(ValueError, match="cycle"):
        ss.regenerate(w, "Hall-corridor", {"attach": {"to": "Hall-corridor", "side": "east"}})


def test_cycle_via_dependents():
    w = ss.generate(None, _room("A"))
    w = ss.generate(w, _corridor("H1", "A-room"))
    w = ss.generate(w, _corridor("H2", "H1-corridor"))
    with pytest.raises(ValueError, match="cycle"):
        ss.regenerate(w, "H1-corridor", {"attach": {"to": "H2-corridor", "side": "east"}})
    # chain follows: resizing A moves H1 then H2
    new, changed, _, _ = ss.regenerate(w, "A-room", {"width": 10.0})
    h1 = new.geometries["geom-H1-floor"]
    assert h1.bounds_min.x == 10.0
    assert new.geometries["geom-H2-floor"].bounds_min.x == h1.bounds_max.x


def test_unknown_spec_and_legacy_world():
    from procedural.room_grammar import RoomGrammarSpec, generate_room

    legacy = generate_room(RoomGrammarSpec("Old", 3, 3, 2.5))
    assert ss.load_specs(legacy) == {}
    with pytest.raises(ss.UnknownSpec):
        ss.regenerate(legacy, "Old-room", {"width": 5.0})


def test_delete_specs_removes_geometries(two_rooms_and_corridor):
    new, removed = ss.delete_specs(two_rooms_and_corridor, ["Hall-corridor"])
    assert "Hall-corridor" not in new.entities and "geom-Hall-floor" not in new.geometries
    assert "Hall-floor" in removed and "A-floor" in new.entities
    assert isinstance(new, WorldIR)
