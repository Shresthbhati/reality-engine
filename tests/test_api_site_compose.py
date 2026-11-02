"""POST /api/sites/{id}/compose and GET/PUT /api/sites/{id}/anchor."""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("greenlet")
from fastapi.testclient import TestClient

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR


def y_room(w=4.0, d=3.0) -> WorldIR:
    poly = [[0, 0, 0], [w, 0, 0], [w, 0, d], [0, 0, d]]
    room = Entity(id="room-1", type=EntityType.ROOM, provenance=Provenance.INFERRED, confidence=0.9,
                  custom_properties={"footprint_polygon_xyz_m": poly,
                                     "footprint_basis": {"e1": [1, 0, 0], "e2": [0, 0, 1], "up": [0, 1, 0]}})
    wd = WorldIR(id="tmp", name="room", entities={"room-1": room})
    wd.metadata = {"compiled_from": {"up": [0.0, 1.0, 0.0]}}
    return wd


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield SimpleNamespace(c=c, db=db_mod)


def _world(env, name="room", world: WorldIR | None = None, commit=True) -> str:
    wid = env.c.post("/api/worlds", json={"name": name}).json()["id"]
    if commit:
        from apps.api import worldstore_service as svc

        async def go():
            async with env.db.get_sessionmaker()() as db:
                await svc.commit_version(db, world_id=wid, world=world or y_room(), parent=None)

        asyncio.run(go())
    return wid


def _site(env, **kw) -> str:
    return env.c.post("/api/sites", json={"name": "S", **kw}).json()["id"]


def _place(env, sid, wid, **kw) -> dict:
    r = env.c.post(f"/api/sites/{sid}/placements", json={"world_id": wid, "cell_x": 0, "cell_z": 0, **kw})
    assert r.status_code == 201
    return r.json()


def _compose(env, sid, **kw):
    return env.c.post(f"/api/sites/{sid}/compose", json={"name": "Street", "kind": "neighbourhood", **kw})


def test_compose_creates_world_then_new_version_of_the_same_world(env):
    sid = _site(env, grid_cell_m=50)
    a, b = _world(env, "A"), _world(env, "B")
    pa = _place(env, sid, a)
    pb = _place(env, sid, b, cell_x=1)
    conn = [{"kind": "street", "from_placement_id": pa["id"], "to_placement_id": pb["id"]}]
    r = _compose(env, sid, connectors=conn)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["kind"] == "neighbourhood" and j["source_count"] == 2
    # the street runs edge to edge between the two buildings, so nothing captured is overlapped: carriageway, STREET
    # and both sidewalks all stay (a centre-to-centre line used to cut through the rooms and lose the carriageway)
    assert j["generated_count"] == 4
    assert not any("dropped generated" in w for w in j["warnings"])
    ir = env.c.get(f"/api/worlds/{j['world_id']}/worldir").json()
    assert len(ir["entities"]) == j["entity_count"]
    assert f"{a}/room-1" in ir["entities"] and f"{b}/room-1" in ir["entities"]
    cx = [p["center_m"]["x"] for p in (pa, pb)]
    xs = [pt[0] for pt in ir["entities"][f"{b}/room-1"]["custom_properties"]["footprint_polygon_xyz_m"]]
    assert (min(xs) + max(xs)) / 2 == pytest.approx(cx[1])
    assert [c["world_id"] for c in ir["metadata"]["composed_from"]] == [a, b]
    # same inputs: de-duplicated to the same version; moving a placement: a NEW version of the SAME world
    again = _compose(env, sid, connectors=conn).json()
    assert again["world_id"] == j["world_id"] and again["version_id"] == j["version_id"]
    env.c.patch(f"/api/sites/{sid}/placements/{pb['id']}", json={"offset_x_m": 5.0})
    j2 = _compose(env, sid, connectors=conn).json()
    assert j2["world_id"] == j["world_id"] and j2["version_id"] != j["version_id"]
    vers = {v["id"]: v for v in env.c.get(f"/api/worlds/{j['world_id']}/versions").json()["items"]}
    assert set(vers) == {j["version_id"], j2["version_id"]}
    assert vers[j2["version_id"]]["parent_version_id"] == j["version_id"]
    # a different name is a different composite world
    assert env.c.post(f"/api/sites/{sid}/compose", json={"name": "Other", "kind": "house"}).json()["world_id"] != j["world_id"]


def test_compose_errors(env):
    assert _compose(env, "nope").status_code == 404
    sid = _site(env)
    assert _compose(env, sid).status_code == 422  # no placements
    assert env.c.post(f"/api/sites/{sid}/compose", json={"name": "x", "kind": "city"}).status_code == 422
    empty = _world(env, "no-version", commit=False)
    _place(env, sid, empty)
    r = _compose(env, sid)
    assert r.status_code == 422 and "no head version" in r.json()["detail"]


def test_compose_409_for_mixed_up_axes_and_scale(env):
    sid = _site(env)
    bad = y_room()
    bad.entities["room-1"].custom_properties["footprint_basis"]["up"] = [0, 0, 1]
    p = _place(env, sid, _world(env, "bad", bad))
    r = _compose(env, sid)
    assert r.status_code == 409 and "mixed up axes" in r.json()["detail"]
    sid2 = _site(env)
    _place(env, sid2, _world(env, "ok"), scale=2.0)
    r = _compose(env, sid2)
    assert r.status_code == 409 and "scale" in r.json()["detail"]
    assert p["id"]


def test_compose_validates_connectors_and_double_placement(env):
    sid = _site(env)
    w = _world(env)
    p = _place(env, sid, w)
    bad_width = _compose(env, sid, connectors=[{"kind": "street", "from_placement_id": p["id"],
                                                "to_placement_id": p["id"], "width_m": 0.1}])
    assert bad_width.status_code == 422
    same = _compose(env, sid, connectors=[{"kind": "driveway", "from_placement_id": p["id"],
                                           "to_placement_id": p["id"]}])
    assert same.status_code == 422 and "coincide" in same.json()["detail"]
    foreign = _compose(env, sid, connectors=[{"kind": "driveway", "from_placement_id": p["id"],
                                              "to_placement_id": "plc_other"}])
    assert foreign.status_code == 422
    _place(env, sid, w, cell_x=3)
    twice = _compose(env, sid)
    assert twice.status_code == 422 and "placed twice" in twice.json()["detail"]


def test_anchor_roundtrip_and_validation(env):
    sid = _site(env)
    assert env.c.get(f"/api/sites/{sid}/anchor").json() == {"lat": None, "lon": None, "rotation_deg": None}
    assert env.c.get(f"/api/sites/{sid}").json()["anchor"] is None
    r = env.c.put(f"/api/sites/{sid}/anchor", json={"lat": 12.97, "lon": 77.59, "rotation_deg": -30})
    assert r.status_code == 200 and r.json() == {"lat": 12.97, "lon": 77.59, "rotation_deg": 330.0}
    assert env.c.get(f"/api/sites/{sid}/anchor").json() == r.json()
    site = env.c.get(f"/api/sites/{sid}").json()
    assert site["anchor"] == r.json() and {"id", "name", "grid_cell_m", "placements", "overlaps"} <= set(site)
    assert env.c.put(f"/api/sites/{sid}/anchor", json={"lat": 1, "lon": 2}).json()["rotation_deg"] == 0.0  # update in place
    for body in ({"lat": 91, "lon": 0}, {"lat": -91, "lon": 0}, {"lat": 0, "lon": 181}, {"lat": 0, "lon": -181}):
        assert env.c.put(f"/api/sites/{sid}/anchor", json=body).status_code == 422
    assert env.c.get("/api/sites/nope/anchor").status_code == 404
    assert env.c.put("/api/sites/nope/anchor", json={"lat": 0, "lon": 0}).status_code == 404
    assert env.c.delete(f"/api/sites/{sid}").status_code == 204


def test_edge_point_leaves_the_footprint_toward_the_other_building():
    from apps.api.routes_sites import _edge_point
    # 10 x 4 m building at the origin, other building 100 m east: connector starts at its east edge (+ margin)
    x, z = _edge_point((0.0, 0.0), (100.0, 0.0), (5.0, 2.0))
    assert abs(x - 5.5) < 1e-9 and z == 0.0
    # diagonal: leaves through the nearer side (depth side here), never beyond the other centre
    x, z = _edge_point((0.0, 0.0), (10.0, 10.0), (5.0, 2.0))
    assert 2.0 < z < 3.0 and x == z
    # unknown footprint falls back to the centre; coincident centres do not divide by zero
    assert _edge_point((1.0, 2.0), (9.0, 9.0), None) == (1.0, 2.0)
    assert _edge_point((1.0, 2.0), (1.0, 2.0), (5.0, 2.0)) == (1.0, 2.0)
