"""Phase 2: the whole persistence lifecycle survives a backup, a restore into a clean environment, and an application restart.

Create world -> upload evidence -> reconstruct -> versions + artifacts -> backup -> restore into a CLEAN directory ->
start a second application instance on it -> reload world -> compare -> edit -> new version -> export.

SYNTHETIC verification: the reconstruction runs on the deterministic ``_TwoViewBackend`` test backend over stand-in
image bytes, not on photographs of a physical room. What is verified is that persistence is lossless, not geometry accuracy.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps import backup as bk
from tests.test_reconstruction_lifecycle_semantics import _wait


@contextmanager
def _application(base: Path, monkeypatch):
    """One application instance (fresh process state) over the data directory ``base``."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(base / 'app.db').as_posix()}")
    monkeypatch.setenv("WORLDSTORE_ROOT", str(base / "worldstore"))
    monkeypatch.setenv("STORAGE_ROOT", str(base / "artifacts"))
    monkeypatch.setenv("REALITY_TEST_BACKEND", "tests.test_cli_compile:_TwoViewBackend")
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.worldstore_service as ws_svc
    ws_svc._store = None
    ws_svc._stores.clear()
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    with TestClient(main_mod.app) as client:
        yield client


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _snapshot(client: TestClient, wid: str) -> dict:
    """Everything a user could observe about the world: versions, the world itself, evidence bytes."""
    versions = client.get(f"/api/worlds/{wid}/versions").json()["items"]
    worldir = client.get(f"/api/worlds/{wid}/worldir").json()
    evidence = client.get("/api/evidence").json()["items"]
    return {
        "versions": [(v["id"], v.get("artifact_hash"), v.get("parent_version_id")) for v in versions],
        "worldir": _canonical(worldir),
        "entity_ids": sorted(worldir["entities"]),
        "geometry": _canonical(worldir.get("geometries", {})),
        "relationships": _canonical({k: e.get("relationships", []) for k, e in sorted(worldir["entities"].items())}),
        "provenance": _canonical({k: e.get("provenance") for k, e in sorted(worldir["entities"].items())}),
        "evidence": sorted((e["id"], hashlib.sha256(client.get(f"/api/evidence/{e['id']}/artifact").content).hexdigest())
                           for e in evidence),
    }


def test_backup_restore_restart_edit_export_lifecycle(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()

    # ---- original deployment: build a world through the API ------------------------------------------------------
    with _application(a, monkeypatch) as client:
        sid = client.post("/api/sessions", json={"name": "Lifecycle"}).json()["id"]
        wid = client.post("/api/worlds", json={"name": "Lifecycle world"}).json()["id"]
        assert client.post(f"/api/worlds/{wid}/attach/{sid}").status_code == 200
        for i in range(3):
            r = client.post(f"/api/uploads?session_id={sid}",
                            files={"file": (f"frame_{i:02d}.jpg", b"jpeg-" + f"{i}".encode() * 8, "image/jpeg")})
            assert r.status_code == 201, r.text
        job = client.post(f"/api/sessions/{sid}/reconstruct")
        assert job.status_code == 200, job.text
        _wait(client, job.json()["job_id"])
        before = _snapshot(client, wid)
        assert before["versions"] and before["entity_ids"] and before["evidence"], "the lifecycle needs real content to compare"
        export_before = client.post(f"/api/worlds/{wid}/export", json={"format": "glb"})
        assert export_before.status_code == 200, export_before.text

    # ---- backup the stopped deployment, restore into a clean directory -------------------------------------------
    archive = tmp_path / "backup" / "a.tar.gz"
    manifest = bk.create_backup(archive, db_path=a / "app.db", worldstore_root=a / "worldstore", storage_root=a / "artifacts")
    assert all(c["status"] == "included" for c in manifest["components"].values())
    assert bk.verify_backup(archive)["files"] == manifest["files"]
    result = bk.restore_backup(archive, db_path=b / "app.db", worldstore_root=b / "worldstore", storage_root=b / "artifacts")
    assert result["consistency"]["problems"] == []
    assert result["consistency"]["checked"]["world_versions"] == len(before["versions"])

    # ---- restart: a second application instance on the restored data ----------------------------------------------
    with _application(b, monkeypatch) as client:
        assert client.get(f"/api/worlds/{wid}").status_code == 200
        after = _snapshot(client, wid)
        for key in ("versions", "entity_ids", "geometry", "relationships", "provenance", "evidence", "worldir"):
            assert after[key] == before[key], f"{key} changed across backup -> restore -> restart"

        # ---- the restored world is live: edit it, get a new immutable version, export ----------------------------
        target = before["entity_ids"][0]
        edit = client.post(f"/api/worlds/{wid}/commands", json={
            "command_type": "move", "entity_id": target, "params": {"position": [2.5, 0.0, -1.2]}, "description": "after restore"})
        assert edit.status_code == 201, edit.text
        edited = _snapshot(client, wid)
        old, new = set(before["versions"]), set(edited["versions"])      # the API lists newest first: compare as sets
        assert old < new and len(new - old) == 1, "history must be append-only: old versions untouched, exactly one new"
        parents = {v[2] for v in old}
        head = next(v for v in old if v[0] not in parents)               # the restored HEAD: nobody's parent
        assert next(iter(new - old))[2] == head[0], "the new version's parent is the restored HEAD"
        assert edited["worldir"] != before["worldir"] and edited["entity_ids"] == before["entity_ids"]

        export_after = client.post(f"/api/worlds/{wid}/export", json={"format": "glb"})
        assert export_after.status_code == 200, export_after.text

    # ---- the new state is itself backed up and restorable ---------------------------------------------------------
    second = tmp_path / "backup" / "b.tar.gz"
    bk.create_backup(second, db_path=b / "app.db", worldstore_root=b / "worldstore", storage_root=b / "artifacts")
    again = bk.restore_backup(second, db_path=tmp_path / "c" / "app.db", worldstore_root=tmp_path / "c" / "worldstore",
                              storage_root=tmp_path / "c" / "artifacts")
    assert again["consistency"]["problems"] == []
    assert again["consistency"]["checked"]["world_versions"] == len(before["versions"]) + 1


def test_restore_refuses_to_start_over_the_original_deployment(tmp_path, monkeypatch):
    a = tmp_path / "a"
    a.mkdir()
    with _application(a, monkeypatch) as client:
        assert client.post("/api/worlds", json={"name": "x"}).status_code == 201
    archive = tmp_path / "a.tar.gz"
    bk.create_backup(archive, db_path=a / "app.db", worldstore_root=a / "worldstore", storage_root=a / "artifacts")
    with pytest.raises(bk.BackupError, match="refusing to restore over"):
        bk.restore_backup(archive, db_path=a / "app.db", worldstore_root=a / "worldstore", storage_root=a / "artifacts")
