"""/api/sites: user-directed layout of worlds on a metric grid (not geo-registration)."""

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


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    import apps.api.routes_sites as rs
    fps: dict = {}

    async def fake_fp(db, world_id):
        return fps.get(world_id)

    monkeypatch.setattr(rs, "_world_footprint", fake_fp)
    with TestClient(main_mod.app) as c:
        yield SimpleNamespace(c=c, fps=fps, db=db_mod)


def _world(env, name="w", fp=None) -> str:
    wid = env.c.post("/api/worlds", json={"name": name}).json()["id"]
    if fp:
        env.fps[wid] = fp
    return wid


def _site(env, **kw) -> str:
    r = env.c.post("/api/sites", json={"name": "S", **kw})
    assert r.status_code == 201
    return r.json()["id"]


def _place(env, sid, wid, **kw):
    return env.c.post(f"/api/sites/{sid}/placements", json={"world_id": wid, "cell_x": 0, "cell_z": 0, **kw})


def test_site_crud(env):
    sid = _site(env, grid_cell_m=25)
    assert env.c.get("/api/sites").json() == [{"id": sid, "name": "S", "grid_cell_m": 25.0, "placement_count": 0}]
    got = env.c.get(f"/api/sites/{sid}").json()
    assert got["placements"] == [] and got["overlaps"] == [] and got["grid_cell_m"] == 25.0
    assert env.c.delete(f"/api/sites/{sid}").status_code == 204
    assert env.c.get(f"/api/sites/{sid}").status_code == 404
    assert env.c.get("/api/sites").json() == []


def test_default_grid_cell(env):
    assert env.c.get(f"/api/sites/{_site(env)}").json()["grid_cell_m"] == 50.0


def test_placement_add_patch_delete(env):
    sid, wid = _site(env), _world(env, "Taj")
    r = _place(env, sid, wid, label="back", cell_x=2, cell_z=-1, rotation_deg=-90)
    assert r.status_code == 201
    p = r.json()
    assert p["world_name"] == "Taj" and p["label"] == "back" and p["rotation_deg"] == 270.0
    assert env.c.get("/api/sites").json()[0]["placement_count"] == 1
    r = env.c.patch(f"/api/sites/{sid}/placements/{p['id']}",
                    json={"cell_x": 3, "scale": 2, "rotation_deg": 725, "label": None})
    assert r.status_code == 200
    q = r.json()
    assert (q["cell_x"], q["cell_z"], q["scale"], q["rotation_deg"], q["label"]) == (3, -1, 2.0, 5.0, None)
    assert env.c.delete(f"/api/sites/{sid}/placements/{p['id']}").status_code == 204
    assert env.c.get(f"/api/sites/{sid}").json()["placements"] == []
    assert env.c.delete(f"/api/sites/{sid}/placements/{p['id']}").status_code == 404


def test_validation_422(env):
    sid, wid = _site(env), _world(env)
    assert env.c.post("/api/sites", json={"name": ""}).status_code == 422
    assert env.c.post("/api/sites", json={"name": "x", "grid_cell_m": 0}).status_code == 422
    assert env.c.post("/api/sites", json={"name": "x", "grid_cell_m": 5001}).status_code == 422
    assert _place(env, sid, wid, scale=0).status_code == 422
    assert _place(env, sid, wid, scale=101).status_code == 422
    assert _place(env, sid, wid, cell_x=501).status_code == 422
    assert _place(env, sid, wid, cell_z=-501).status_code == 422
    p = _place(env, sid, wid).json()
    url = f"/api/sites/{sid}/placements/{p['id']}"
    assert env.c.patch(url, json={"scale": 0}).status_code == 422
    assert env.c.patch(url, json={"cell_x": 9999}).status_code == 422
    assert env.c.patch(url, json={"cell_z": None}).status_code == 422


def test_404s(env):
    sid, wid = _site(env), _world(env)
    assert env.c.get("/api/sites/nope").status_code == 404
    assert env.c.delete("/api/sites/nope").status_code == 404
    assert _place(env, "nope", wid).status_code == 404
    assert _place(env, sid, "no-such-world").status_code == 404
    assert env.c.patch(f"/api/sites/{sid}/placements/nope", json={"scale": 1}).status_code == 404
    other = _site(env)
    p = _place(env, other, wid).json()
    assert env.c.patch(f"/api/sites/{sid}/placements/{p['id']}", json={"scale": 1}).status_code == 404


def test_center_m_arithmetic(env):
    sid, wid = _site(env, grid_cell_m=100), _world(env)
    p = _place(env, sid, wid, cell_x=2, cell_z=-3, offset_x_m=7.5, offset_z_m=-2).json()
    assert p["center_m"] == {"x": 257.5, "z": -252.0}


def test_footprint_scaled_and_unknown(env):
    sid = _site(env)
    known, unknown = _world(env, fp=(10.0, 4.0)), _world(env)
    a = _place(env, sid, known, scale=2.5).json()["footprint"]
    assert a == {"width_m": 25.0, "depth_m": 10.0, "source": "worldir"}
    b = _place(env, sid, unknown).json()["footprint"]
    assert b == {"width_m": None, "depth_m": None, "source": "unknown"}


def test_overlap_detection(env):
    sid = _site(env, grid_cell_m=50)
    w1, w2, w3 = (_world(env, fp=(40.0, 40.0)) for _ in range(3))
    a = _place(env, sid, w1, cell_x=0).json()["id"]
    b = _place(env, sid, w2, cell_x=0, offset_x_m=30).json()["id"]  # centres 30 m apart, half-widths sum 40
    _place(env, sid, w3, cell_x=5)  # far away
    ov = env.c.get(f"/api/sites/{sid}").json()["overlaps"]
    assert len(ov) == 1 and set(ov[0]) == {a, b}
    # moving b so edges just touch (40 m apart) is not an overlap
    env.c.patch(f"/api/sites/{sid}/placements/{b}", json={"offset_x_m": 40})
    assert env.c.get(f"/api/sites/{sid}").json()["overlaps"] == []


def test_overlap_allowed_not_rejected(env):
    sid, w = _site(env), _world(env, fp=(40.0, 40.0))
    assert _place(env, sid, w).status_code == 201
    assert _place(env, sid, w).status_code == 201


def test_unknown_footprint_never_overlaps(env):
    sid = _site(env)
    _place(env, sid, _world(env))
    _place(env, sid, _world(env))
    _place(env, sid, _world(env, fp=(1000.0, 1000.0)))
    assert env.c.get(f"/api/sites/{sid}").json()["overlaps"] == []


def test_delete_site_removes_placements(env):
    from sqlalchemy import func, select
    from apps.api.models import SitePlacement
    sid, wid = _site(env), _world(env)
    _place(env, sid, wid)
    _place(env, sid, wid, cell_x=1)
    assert env.c.delete(f"/api/sites/{sid}").status_code == 204

    async def count():
        async with env.db.get_sessionmaker()() as s:
            return (await s.execute(select(func.count()).select_from(SitePlacement))).scalar_one()
    assert asyncio.run(count()) == 0


def _fake_world(polygon):
    from types import SimpleNamespace
    from world_ir.schema_v1 import EntityType
    room = SimpleNamespace(type=EntityType.ROOM, custom_properties={"footprint_polygon_xyz_m": polygon})
    return SimpleNamespace(entities={"r": room})


def test_extent_reads_the_room_polygon_in_either_up_axis():
    from apps.api.routes_sites import _extent
    z_up = [[0, 0, 0], [5, 0, 0], [5, 4, 0], [0, 4, 0]]      # compiler interior frame: height on the 3rd axis
    y_up = [[0, 0, 0], [5, 0, 0], [5, 0, 4], [0, 0, 4]]      # canonical frame: height on the 2nd axis
    assert _extent(_fake_world(z_up)) == (5, 4)
    assert _extent(_fake_world(y_up)) == (5, 4)


def test_extent_is_none_without_a_footprint():
    from apps.api.routes_sites import _extent
    assert _extent(_fake_world([])) is None
    assert _extent(_fake_world([[0, 0, 0], [1, 0, 0]])) is None
