"""M4: two rooms captured in separate sessions are joined through a shared door (rigid move of the second room, no re-measurement).

Each room is the textured synthetic box room (tests/layout_synth.py) with a door on its +z wall, bootstrapped through the learned-layout
path under its own id prefix, then put in one world. Joining makes the doors coincide and face each other.
"""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from engine.pipeline.rooms import RoomConnectionError, connect_rooms, room_prefix
from engine.pipeline.single_image import bootstrap_single_image, fuse_single_views
from evidence.image_check import inspect_image
from evidence.session import EvidenceItem, EvidenceKind
from tests.layout_synth import render


def _room(tmp_path, monkeypatch, tag, door=(-0.5, 0.5, 0.0, 2.1)):
    import engine.pipeline.single_image as si

    monkeypatch.setattr(si, "ASSUMED_HFOV_DEG", 80.0)
    view = render(0.0, 0.0, 0.0, 0.0, hfov=80.0, window=door)
    view.masks["door"], view.masks["window"] = view.masks["window"], np.zeros_like(view.masks["window"])  # the cut-out is a door
    p = tmp_path / f"room{tag}.png"
    Image.fromarray(view.rgb).save(p)
    item = EvidenceItem(id=f"ev-{tag}", kind=EvidenceKind.PHOTO, source_uri=p.as_uri())
    an = SimpleNamespace(depth_m=view.depth, masks=view.masks, rgb=view.rgb)
    return item, bootstrap_single_image(item, inspect_image(p), layout_analysis=an, tag=tag)


@pytest.fixture()
def world(tmp_path, monkeypatch):
    a, b = _room(tmp_path, monkeypatch, "1"), _room(tmp_path, monkeypatch, "2")
    return fuse_single_views([a, b]).world


def _door(world, prefix):
    return next(e for e in world.entities.values() if e.type.value == "door" and room_prefix(e.id) == prefix)


def _centre(world, door):
    b = world.geometries[door.geometry_ids[0]]
    return np.array([(b.bounds_min.x + b.bounds_max.x) / 2, (b.bounds_min.z + b.bounds_max.z) / 2])


def _outward(world, door):
    return np.array(world.entities[door.custom_properties["host_wall_id"]].custom_properties["wall_slab"]["outward"])[[0, 2]]


def test_the_two_doors_coincide_and_face_each_other_after_joining(world):
    d1, d2 = _door(world, "boot1"), _door(world, "boot2")
    res = connect_rooms(world, d1.id, d2.id)
    assert abs(res["rotation_deg"]) == pytest.approx(180.0, abs=1.0)  # both doors sat on the +z wall: the second room turns round
    assert np.linalg.norm(_centre(world, d1) - _centre(world, d2)) < 0.05
    assert float(_outward(world, d1) @ _outward(world, d2)) == pytest.approx(-1.0, abs=0.02)


def test_joining_is_rigid_the_second_rooms_walls_keep_their_lengths_and_distance_to_the_door(world):
    d2 = _door(world, "boot2")
    walls = [e for e in world.entities.values() if e.type.value == "wall" and room_prefix(e.id) == "boot2"]
    before = sorted(round(w.custom_properties["wall_slab"]["length_m"], 3) for w in walls)
    start0 = np.array(walls[0].custom_properties["wall_slab"]["start_xyz_m"])
    gap0 = np.linalg.norm(start0[[0, 2]] - _centre(world, d2))
    connect_rooms(world, _door(world, "boot1").id, d2.id)
    after = sorted(round(w.custom_properties["wall_slab"]["length_m"], 3) for w in walls)
    start1 = np.array(walls[0].custom_properties["wall_slab"]["start_xyz_m"])
    assert before == after
    assert np.linalg.norm(start1[[0, 2]] - _centre(world, d2)) == pytest.approx(gap0, abs=1e-6)


def test_the_connection_is_recorded_on_both_doors_and_in_the_world(world):
    d1, d2 = _door(world, "boot1"), _door(world, "boot2")
    connect_rooms(world, d1.id, d2.id)
    assert d1.custom_properties["connects_to"] == d2.id and d2.custom_properties["connects_to"] == d1.id
    conn = world.metadata["room_connections"][0]
    assert conn["doors"] == [d1.id, d2.id] and conn["state"] == "user_confirmed_doors_coincide"


def test_doors_of_different_width_are_refused_and_nothing_moves(tmp_path, monkeypatch):
    a = _room(tmp_path, monkeypatch, "1")
    b = _room(tmp_path, monkeypatch, "2", door=(-1.5, 1.5, 0.0, 2.1))   # a 3 m wide opening: not the same door
    world = fuse_single_views([a, b]).world
    d2 = _door(world, "boot2")
    pos = dict(d2.transform["position"])
    with pytest.raises(RoomConnectionError, match="width"):
        connect_rooms(world, _door(world, "boot1").id, d2.id)
    assert d2.transform["position"] == pos


def test_a_door_cannot_be_joined_to_a_door_of_the_same_room(world):
    with pytest.raises(RoomConnectionError, match="same room"):
        connect_rooms(world, _door(world, "boot1").id, _door(world, "boot1").id)


def test_the_joined_apartment_still_passes_the_world_validator(world):
    from world_ir.validation import validate_world_ir

    connect_rooms(world, _door(world, "boot1").id, _door(world, "boot2").id)
    errors = [i for i in validate_world_ir(world).issues if str(i.severity.value).lower() == "error"]
    assert errors == [], [str(i) for i in errors][:3]
