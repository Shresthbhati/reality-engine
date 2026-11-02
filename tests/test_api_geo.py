"""/api/geo/context: offline (httpx MockTransport injected through the route's client seam)."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("greenlet")
from fastapi.testclient import TestClient

FX = Path(__file__).parent / "fixtures" / "osm"


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    import apps.api.routes_geo as rg
    state = SimpleNamespace(calls=0, response=None)

    def handler(req):
        state.calls += 1
        return state.response

    monkeypatch.setattr(rg, "_make_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    state.response = httpx.Response(200, json=json.loads((FX / "synthetic_street_furniture.json").read_text(encoding="utf-8")))
    with TestClient(main_mod.app) as c:
        state.c = c
        yield state


BODY = {"lat": 22.5757, "lon": 88.4312, "radius_m": 300}


def test_import_creates_world_then_refresh_adds_version(env):
    r = env.c.post("/api/geo/context", json=BODY)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["source"] == "overpass" and j["attribution"] == "(c) OpenStreetMap contributors"
    assert j["element_count"] == 22 and j["counts_by_kind"]["building"] == 3 and j["counts_by_kind"]["jersey_barrier"] == 1
    assert j["estimated_count"] > 0
    # second call: served from cache (no extra network call), same world; unchanged content dedups to same version
    r2 = env.c.post("/api/geo/context", json=BODY)
    assert r2.status_code == 200 and r2.json()["source"] == "cache" and env.calls == 1
    assert r2.json()["world_id"] == j["world_id"]
    assert [w["id"] for w in env.c.get("/api/worlds").json()["items"]] == [j["world_id"]]


def test_refresh_with_changed_data_commits_new_version(env):
    j = env.c.post("/api/geo/context", json={**BODY, "name": "Site A"}).json()
    data = json.loads((FX / "synthetic_street_furniture.json").read_text(encoding="utf-8"))
    data["elements"] = data["elements"][:5]
    env.response = httpx.Response(200, json=data)
    j2 = env.c.post("/api/geo/context", json={**BODY, "radius_m": 301, "name": "Site A"}).json()
    assert j2["world_id"] == j["world_id"] and j2["version_id"] != j["version_id"]


@pytest.mark.parametrize("patch", [{"lat": 91}, {"lon": -181}, {"radius_m": 49}, {"radius_m": 1001}])
def test_bad_input_is_422(env, patch):
    assert env.c.post("/api/geo/context", json={**BODY, **patch}).status_code == 422
    assert env.calls == 0


def test_unreachable_is_502_and_never_fake(env):
    env.response = httpx.Response(504)
    r = env.c.post("/api/geo/context", json=BODY)
    assert r.status_code == 502 and "detail" in r.json()
    assert env.c.get("/api/worlds").json()["items"] == []


def test_rate_limit_is_429_passthrough(env):
    env.response = httpx.Response(429)
    r = env.c.post("/api/geo/context", json=BODY)
    assert r.status_code == 429 and "rate limit" in r.json()["detail"].lower()
