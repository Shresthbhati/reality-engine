"""F7: rooms joined only through a shared opening (not by merely touching)."""

from __future__ import annotations

from perception.architecture.room_join import join_rooms_through_openings
from world_ir import Entity, EntityType, RelationshipKind, WorldIR


def _wall(wid, room_id, start, end, *, opening=None, thickness=0.1, height=2.6):
    return Entity(id=wid, type=EntityType.WALL, custom_properties={
        "room_id": room_id, "thickness_m": thickness, "height_m": height,
        "wall_slab": {"start_xyz_m": list(start), "end_xyz_m": list(end), "up": [0, 0, 1],
                      "height_m": height, "thickness_m": thickness,
                      "openings": [opening] if opening else []},
    })


def _world(*, opening_on_b=True):
    w = WorldIR()
    w.entities["room-a"] = Entity(id="room-a", type=EntityType.ROOM, name="A", custom_properties={
        "footprint_polygon_xyz_m": [[0, 0, 0], [0, 3, 0], [4, 3, 0], [4, 0, 0]]})
    w.entities["room-b"] = Entity(id="room-b", type=EntityType.ROOM, name="B", custom_properties={
        "footprint_polygon_xyz_m": [[4, 0, 0], [4, 3, 0], [8, 3, 0], [8, 0, 0]]})
    oa = {"entity_id": "door-a", "kind": "door", "offset_m": 1.0, "sill_m": 0.0, "width_m": 0.9, "height_m": 2.05}
    ob = {"entity_id": "door-b", "kind": "door", "offset_m": 1.1, "sill_m": 0.0, "width_m": 0.9, "height_m": 2.05}
    w.entities["door-a"] = Entity(id="door-a", type=EntityType.DOOR)
    w.entities["door-b"] = Entity(id="door-b", type=EntityType.DOOR)
    w.entities["wall-a"] = _wall("wall-a", "room-a", (4, 0, 0), (4, 3, 0), opening=oa)
    w.entities["wall-b"] = _wall("wall-b", "room-b", (4, 3, 0), (4, 0, 0), opening=ob if opening_on_b else None)
    return w


def test_shared_door_connects_the_two_rooms_with_evidence():
    w = _world()
    rep = join_rooms_through_openings(w)
    assert len(rep["joins"]) == 1 and not rep["warnings"]
    join = rep["joins"][0]
    assert {join["room_a"], join["room_b"]} == {"room-a", "room-b"}
    assert join["shared_openings"] and join["shared_openings"][0]["a"] == "door-a"
    a = w.entities["room-a"]
    assert any(r.kind == RelationshipKind.CONNECTS and r.target_id == "room-b" for r in a.relationships)
    assert w.entities["room-b"].custom_properties["adjacent_room_ids"] == ["room-a"]
    assert w.entities["door-a"].custom_properties["shared_doorway"] is True
    assert w.entities["door-a"].custom_properties["connects_room_ids"] == ["room-a", "room-b"]


def test_one_opening_on_a_shared_partition_is_still_a_passage():
    w = _world(opening_on_b=False)
    rep = join_rooms_through_openings(w)
    assert len(rep["joins"]) == 1
    assert rep["joins"][0]["shared_openings"][0]["a"] == "door-a"


def test_touching_rooms_without_an_opening_are_not_joined():
    w = _world(opening_on_b=False)
    w.entities["wall-a"].custom_properties["wall_slab"]["openings"] = []
    rep = join_rooms_through_openings(w)
    assert rep["joins"] == []
    assert not any(r.kind == RelationshipKind.CONNECTS for r in w.entities["room-a"].relationships)


def test_far_apart_rooms_are_not_joined_even_with_doors():
    w = _world()
    w.entities["wall-b"] = _wall("wall-b", "room-b", (8, 3, 0), (8, 0, 0),
                                 opening={"entity_id": "door-b", "kind": "door", "offset_m": 1.1, "sill_m": 0.0,
                                          "width_m": 0.9, "height_m": 2.05})
    rep = join_rooms_through_openings(w)
    assert rep["joins"] == []
