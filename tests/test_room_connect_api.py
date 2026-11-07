"""M4 through the API: POST /rooms/connect joins two rooms by their doors and persists a new world version."""
from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from tests.test_commit_validation_contract import _head, world  # noqa: E402,F401  (fixture reuse)
from tests.test_room_connect import _door, _room  # noqa: E402


def test_connecting_two_rooms_persists_a_version_and_a_bad_pair_is_a_422(world, tmp_path, monkeypatch):
    from engine.pipeline.single_image import fuse_single_views
    import apps.api.routes_worlds as rw

    c, wid, _ = world
    two_rooms = fuse_single_views([_room(tmp_path, monkeypatch, "1"), _room(tmp_path, monkeypatch, "2")]).world
    d1, d2 = _door(two_rooms, "boot1").id, _door(two_rooms, "boot2").id

    async def load(w, version_id=None):
        import copy

        return copy.deepcopy(two_rooms)

    monkeypatch.setattr(rw, "_load_worldir_from_store", load)
    before = _head(c, wid)

    same_room = c.post(f"/api/worlds/{wid}/rooms/connect", json={"door_id": d1, "other_door_id": d1})
    assert same_room.status_code == 422 and "same room" in same_room.text
    assert _head(c, wid) == before                      # a refused connection moves nothing

    ok = c.post(f"/api/worlds/{wid}/rooms/connect", json={"door_id": d1, "other_door_id": d2})
    assert ok.status_code == 200, ok.text
    assert ok.json()["operation"] == "connect_rooms" and _head(c, wid) != before
    stored = c.get(f"/api/worlds/{wid}/worldir").json()
    assert stored["entities"][d1]["custom_properties"]["connects_to"] == d2
    assert stored["metadata"]["room_connections"][0]["state"] == "user_confirmed_doors_coincide"
