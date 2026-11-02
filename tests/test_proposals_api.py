"""/api/worlds/{id}/proposals against a real DB + WorldStore: propose, review, accept (DB-mirrored commit), reject."""

from __future__ import annotations

import asyncio
import copy
import importlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from fastapi.testclient import TestClient

from provenance import Provenance
from tests.test_world_proposals import base_world, ent, geom

WID = "wld_prop01"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod
    import apps.api.worldstore_service as ws

    importlib.reload(db_mod)
    ws._stores.clear()

    async def boot():
        from apps.api.models import World

        await db_mod.init_db()
        async with db_mod.get_sessionmaker()() as db:
            db.add(World(id=WID, name="proposals"))
            await db.commit()
            w = base_world()
            row = await ws.commit_version(db, world_id=WID, world=w, parent=None)
            return row.id

    base_id = asyncio.run(boot())
    asyncio.run(db_mod.dispose_db())
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as client:
        yield client, base_id


def cand(mutate):
    w = copy.deepcopy(base_world())
    mutate(w)
    return w.to_dict()


def add_window(w):
    w.geometries["g-new"] = geom("g-new", (9, 9, 0), (10, 10, 1))
    w.entities["window"] = ent("window", "g-new", obs=("obs-w",))


def weaker_wall_b(w):
    w.geometries["g-wall-b"] = geom("g-wall-b", hi=(0.2, 5.5, 3), prov=Provenance.INFERRED)
    w.entities["wall-b"] = ent("wall-b", "g-wall-b", Provenance.INFERRED, 0.4, obs=("obs-x",))


def propose(client, mutate, **extra):
    r = client.post(f"/api/worlds/{WID}/proposals", json={"candidate": cand(mutate), **extra})
    assert r.status_code == 201, r.text
    return r.json()


def test_route_is_mounted(env):
    client, _ = env
    assert "/api/worlds/{world_id}/proposals" in client.get("/openapi.json").json()["paths"]


def test_propose_review_accept_advances_head_and_records_the_decision(env):
    client, base_id = env
    p = propose(client, add_window, source_session_ids=["sess-2"])
    assert p["base_version_id"] == base_id and p["summary"]["supported"] == 1 and p["status"] == "pending"
    listed = client.get(f"/api/worlds/{WID}/proposals").json()["proposals"]
    assert [x["proposal_id"] for x in listed] == [p["proposal_id"]]

    r = client.post(f"/api/worlds/{WID}/proposals/{p['proposal_id']}/accept", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version_id"] != base_id and body["proposal"]["status"] == "accepted"
    assert body["proposal"]["decision"]["applied"] == ["window"]
    world = client.get(f"/api/worlds/{WID}/worldir").json()
    assert "window" in str(world)                                                  # HEAD now serves the new version
    versions = client.get(f"/api/worlds/{WID}/versions").json()
    assert base_id in str(versions) and body["version_id"] in str(versions)        # base is still in history


def test_conflict_is_422_until_explicitly_overridden_with_a_reason(env):
    client, _ = env
    p = propose(client, weaker_wall_b)
    url = f"/api/worlds/{WID}/proposals/{p['proposal_id']}/accept"
    r = client.post(url, json={"entity_ids": ["wall-b"]})
    assert r.status_code == 422 and "override_conflicts" in r.json()["detail"]
    r = client.post(url, json={"entity_ids": ["wall-b"], "override_conflicts": ["wall-b"]})
    assert r.status_code == 422 and "reason" in r.json()["detail"]
    r = client.post(url, json={"entity_ids": ["wall-b"], "override_conflicts": ["wall-b"], "reason": "site survey"})
    assert r.status_code == 200 and r.json()["proposal"]["decision"]["overridden_conflicts"] == ["wall-b"]


def test_stale_proposal_is_409_not_a_silent_rebase(env):
    client, _ = env
    first, second = propose(client, add_window), propose(client, lambda w: w.entities.pop("door"))
    assert client.post(f"/api/worlds/{WID}/proposals/{first['proposal_id']}/accept", json={}).status_code == 200
    r = client.post(f"/api/worlds/{WID}/proposals/{second['proposal_id']}/accept", json={"entity_ids": ["door"]})
    assert r.status_code == 409 and "stale" in r.json()["detail"]


def test_reject_is_final_and_errors_are_precise(env):
    client, _ = env
    p = propose(client, lambda w: w.entities.pop("door"))
    base = f"/api/worlds/{WID}/proposals/{p['proposal_id']}"
    assert client.post(base + "/reject", json={"reason": "  "}).status_code == 422
    assert client.post(base + "/reject", json={"reason": "partial scan"}).status_code == 200
    assert client.post(base + "/accept", json={"entity_ids": ["door"]}).status_code == 409
    assert client.get(f"/api/worlds/{WID}/proposals/p-nope").status_code == 404
    assert client.post("/api/worlds/wld_missing/proposals", json={"candidate": {}}).status_code == 404
    assert client.post(f"/api/worlds/{WID}/proposals", json={"candidate": {"entities": 5}}).status_code == 422
