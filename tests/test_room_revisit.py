"""M4: accepting a second visit replaces the old room with the new capture and keeps the old room's door joins.

World: room 1 (anchor) joined to room 2 through their doors; room 3 is a second capture of room 2 (same synthetic room, own frame).
"""
import copy

import numpy as np
import pytest

from engine.pipeline.room_revisit import RoomRevisitError, accept_revisit
from engine.pipeline.rooms import connect_rooms, room_prefix
from engine.pipeline.single_image import fuse_single_views
from tests.test_room_connect import _centre, _door, _outward, _room


@pytest.fixture()
def world(tmp_path, monkeypatch):
    w = fuse_single_views([_room(tmp_path, monkeypatch, "1"), _room(tmp_path, monkeypatch, "2"), _room(tmp_path, monkeypatch, "3")]).world
    connect_rooms(w, _door(w, "boot1").id, _door(w, "boot2").id)
    return w


def _ids(w, prefix):
    return {e for e in w.entities if room_prefix(e) == prefix}


def test_the_old_room_is_replaced_and_its_join_is_remade_through_the_new_rooms_door(world):
    res = accept_revisit(world, "boot2", "boot3")
    d1, d3 = _door(world, "boot1"), _door(world, "boot3")   # fetched after: accepting swaps in a consistent copy of the world
    assert _ids(world, "boot2") == set() and _ids(world, "boot3")
    assert [r["state"] for r in res["rejoined"]] == ["placed"] and res["lost_connections"] == []
    assert np.linalg.norm(_centre(world, d1) - _centre(world, d3)) < 0.05
    assert float(_outward(world, d1) @ _outward(world, d3)) == pytest.approx(-1.0, abs=0.02)
    assert d1.custom_properties["connects_to"] == d3.id and d3.custom_properties["connects_to"] == d1.id


def test_the_apartment_is_consistent_after_the_swap_and_history_is_recorded(world):
    from world_ir.validation import validate_world_ir

    res = accept_revisit(world, "boot2", "boot3")
    errors = [i for i in validate_world_ir(world).issues if str(i.severity.value).lower() == "error"]
    assert errors == [], [str(i) for i in errors][:3]
    rec = world.metadata["room_revisits"][0]
    assert rec["old_room"] == "boot2" and rec["new_room"] == "boot3" and rec["state"] == "accepted" and rec["entities_removed"] == res["entities_removed"]
    assert all(c["doors"] and "boot2" not in " ".join(c["doors"]) for c in world.metadata["room_connections"])


def test_a_connection_with_no_matching_door_in_the_new_room_is_reported_lost_not_kept(tmp_path, monkeypatch):
    w = fuse_single_views([_room(tmp_path, monkeypatch, "1"), _room(tmp_path, monkeypatch, "2"),
                           _room(tmp_path, monkeypatch, "3", door=(-1.5, 1.5, 0.0, 2.1))]).world   # the 3 m opening is not the old 1 m door
    connect_rooms(w, _door(w, "boot1").id, _door(w, "boot2").id)
    res = accept_revisit(w, "boot2", "boot3")
    assert res["rejoined"] == [] and len(res["lost_connections"]) == 1
    assert "connects_to" not in _door(w, "boot1").custom_properties         # no dangling pointer to the removed door


def test_a_room_with_no_joins_is_simply_replaced(tmp_path, monkeypatch):
    w = fuse_single_views([_room(tmp_path, monkeypatch, "1"), _room(tmp_path, monkeypatch, "2")]).world
    res = accept_revisit(w, "boot1", "boot2")
    assert res["rejoined"] == [] and res["lost_connections"] == [] and res["entities_removed"] > 0 and _ids(w, "boot1") == set()


def test_bad_requests_change_nothing(world):
    before = copy.deepcopy(world.to_dict())
    for old, new, msg in (("boot2", "boot2", "same room"), ("boot2", "boot9", "no room"), ("boot9", "boot3", "no room")):
        with pytest.raises(RoomRevisitError, match=msg):
            accept_revisit(world, old, new)
    assert world.to_dict() == before
