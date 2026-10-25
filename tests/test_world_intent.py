"""World Intent: persisted per-world intent + an honest resource estimate (never invented numbers)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def _world(client) -> str:
    return client.post("/api/worlds", json={"name": "Intent site"}).json()["id"]


def test_intent_404_for_unknown_world(client):
    assert client.put("/api/worlds/wld_nope/intent", json={"scale": "city", "capabilities": []}).status_code == 404
    assert client.get("/api/worlds/wld_nope/intent").status_code == 404


def test_intent_get_404_until_saved_then_roundtrips_and_updates(client):
    wid = _world(client)
    assert client.get(f"/api/worlds/{wid}/intent").status_code == 404
    r = client.put(f"/api/worlds/{wid}/intent", json={"scale": "building", "capabilities": ["geometry", "measurement"]})
    assert r.status_code == 200
    assert client.get(f"/api/worlds/{wid}/intent").json()["capabilities"] == ["geometry", "measurement"]
    client.put(f"/api/worlds/{wid}/intent", json={"scale": "street", "capabilities": ["geometry"]})
    got = client.get(f"/api/worlds/{wid}/intent").json()
    assert got["scale"] == "street" and got["capabilities"] == ["geometry"] and got["world_id"] == wid


@pytest.mark.parametrize("body", [
    {"scale": "galaxy", "capabilities": []},
    {"scale": "city", "capabilities": ["teleportation"]},
    {"scale": "city"},
])
def test_intent_rejects_unknown_values(client, body):
    wid = _world(client)
    assert client.put(f"/api/worlds/{wid}/intent", json=body).status_code == 422


def test_intent_dedupes_capabilities(client):
    wid = _world(client)
    client.put(f"/api/worlds/{wid}/intent", json={"scale": "city", "capabilities": ["geometry", "geometry"]})
    assert client.get(f"/api/worlds/{wid}/intent").json()["capabilities"] == ["geometry"]


def _est(client, **over):
    body = {"scale": "building", "capabilities": ["geometry"], "file_count": 40, "total_bytes": 10_000_000,
            "video_count": 0, **over}
    return client.post("/api/intent/estimate", json=body)


def test_estimate_never_invents_time_without_history(client):
    r = _est(client)
    assert r.status_code == 200
    t = r.json()["time"]
    assert t["value"] is None and t["kind"] == "unknown" and "no previous runs" in t["basis"]


def test_estimate_fields_are_labelled(client):
    prof = _est(client).json()
    for k in ("compute", "memory", "storage", "time"):
        assert set(prof[k]) >= {"value", "basis", "kind"}
        assert prof[k]["kind"] in ("detected", "derived", "unknown")
        if prof[k]["value"] is None:
            assert prof[k]["kind"] == "unknown"
    assert prof["compute"]["kind"] == "detected"
    assert isinstance(prof["compute"]["value"]["cuda_gpu"], bool)
    assert isinstance(prof["compute"]["value"]["colmap"], bool)
    assert prof["storage"]["value"] == 10_000_000 and prof["storage"]["kind"] == "derived"


def test_estimate_warns_city_scale_with_little_evidence(client):
    w = _est(client, scale="city", file_count=20).json()["warnings"]
    assert any("city is not a bigger photograph" in x for x in w)
    assert not _est(client, scale="interior_space", file_count=20).json()["warnings"]


def test_estimate_validation(client):
    assert _est(client, scale="galaxy").status_code == 422
    assert _est(client, capabilities=["nope"]).status_code == 422
    assert _est(client, file_count=-1).status_code == 422


def test_build_estimate_time_derived_from_samples():
    from apps.api.routes_intent import build_estimate

    hw = {"cuda_gpu": False, "cuda_basis": "test", "colmap": True, "ram_bytes": None}
    prof = build_estimate("building", ["geometry"], 10, 1000, 0, hw, [2.0, 4.0, 6.0])
    assert prof["time"]["value"] == 40.0 and prof["time"]["kind"] == "derived"
    assert prof["time"]["sample_size"] == 3
    assert prof["memory"]["value"] is None and prof["memory"]["kind"] == "unknown"
