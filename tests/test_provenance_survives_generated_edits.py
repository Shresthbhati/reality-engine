"""Regression (found by the Phase G acceptance journey): a generated-object commit used to record NO source
sessions, so every reconstructed entity lost its evidence trace the moment a room was built on top of it, while the
provenance route would have traced GENERATED entities to photos had the sessions been inherited naively."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
from fastapi.testclient import TestClient

from provenance import Provenance
from world_ir import Entity, EntityType
from world_ir.world_v1 import WorldIR


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    import apps.api.worldstore_service as ws

    ws._store = None
    with TestClient(main_mod.app) as c:
        yield c


def test_reconstructed_entity_keeps_evidence_trace_across_generated_commits(client):
    from apps.api import worldstore_service as svc
    import apps.api.db as db_mod

    wid = client.post("/api/worlds", json={"name": "W"}).json()["id"]
    sid = client.post("/api/sessions", json={"name": "S"}).json()["id"]
    up = client.post(f"/api/uploads?session_id={sid}", files={"file": ("a.jpg", b"\xff\xd8\xff\xe0abc", "image/jpeg")}).json()

    wir = WorldIR(id=wid)
    wir.entities["wall-1"] = Entity(id="wall-1", type=EntityType.WALL, name="wall", provenance=Provenance.INFERRED,
                                    confidence=0.3)

    async def seed():
        async with db_mod.get_sessionmaker()() as db:
            await svc.commit_version(db, world_id=wid, world=wir, parent=None, source_session_ids=[sid])

    asyncio.run(seed())
    before = client.get(f"/api/worlds/{wid}/entities/wall-1/provenance").json()
    assert before["trace_level"] == "session" and before["evidence"][0]["evidence_id"] == up["evidence_id"]

    r = client.post(f"/api/worlds/{wid}/rooms", json={"name": "A", "width": 4, "depth": 4, "height": 2.6})
    assert r.status_code == 201, r.text

    after = client.get(f"/api/worlds/{wid}/entities/wall-1/provenance").json()
    assert after["trace_level"] == "session" and after["evidence"] == before["evidence"]
    assert after["version_id"] != before["version_id"]
    gen = client.get(f"/api/worlds/{wid}/entities/A-floor/provenance").json()
    assert gen["trace_level"] == "none" and gen["evidence"] == []
