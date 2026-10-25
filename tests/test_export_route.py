"""POST /api/worlds/{id}/export for the six core formats, through the real route: export -> download -> the downloaded
bytes are the exported content (hash matches), per-version selection works, and unavailable/unsupported formats
fail with a typed status instead of a silent empty file.

Drives apps.api.routes_export / sdk.reality.export on a version persisted through worldstore.WorldStore.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
from fastapi.testclient import TestClient

from tests.test_export_pipeline_e2e import _promoted_world

CORE = ("gltf", "usda", "blender", "cityjson", "citygml", "ifc", "geojson")


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


def _world_with_geometry(client, tmp_path) -> tuple[str, str]:
    from worldstore.store import WorldStore

    wid = client.post("/api/worlds", json={"name": "exportable"}).json()["id"]
    wir = _promoted_world()
    wir.id = wid
    stored = WorldStore(tmp_path / "ws").save_version(wir, parent=None)

    async def _seed():
        import apps.api.db as db_mod
        from apps.api.models import World, WorldVersion

        async with db_mod.get_sessionmaker()() as db:
            db.add(WorldVersion(id=stored.version_id, world_id=wid, parent_version_id=None,
                                artifact_uri=stored.artifact_uri, artifact_hash=stored.artifact_hash,
                                source_session_ids=[], changed_entity_ids=[]))
            (await db.get(World, wid)).current_version_id = stored.version_id
            await db.commit()

    asyncio.run(_seed())
    return wid, stored.version_id


@pytest.mark.parametrize("fmt", CORE)
def test_core_format_exports_downloads_and_hashes(client, tmp_path, fmt):
    if fmt == "ifc":
        pytest.importorskip("ifcopenshell")
    wid, vid = _world_with_geometry(client, tmp_path)
    r = client.post(f"/api/worlds/{wid}/export", json={"format": fmt})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["version_id"] == vid
    assert len(body["entities_exported"]) == 4 and body["entities_skipped"] == []
    dl = client.get(body["download_url"])
    assert dl.status_code == 200 and dl.content
    stored_digest = body["artifact_uri"].removeprefix("sha256://")
    assert hashlib.sha256(dl.content).hexdigest() == stored_digest
    again = client.post(f"/api/worlds/{wid}/export", json={"format": fmt}).json()
    assert again["artifact_uri"] == body["artifact_uri"], "same version + format must export identical bytes"


def test_ifc_download_is_a_real_ifc_file(client, tmp_path):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    wid, _ = _world_with_geometry(client, tmp_path)
    body = client.post(f"/api/worlds/{wid}/export", json={"format": "ifc"}).json()
    data = client.get(body["download_url"]).content
    path = tmp_path / "out.ifc"
    path.write_bytes(data)
    assert len(ifcopenshell.open(str(path)).by_type("IfcWall")) >= 1


def test_unknown_format_is_422_not_a_silent_file(client, tmp_path):
    wid, _ = _world_with_geometry(client, tmp_path)
    assert client.post(f"/api/worlds/{wid}/export", json={"format": "fbx"}).status_code == 422


def test_ifc_without_ifcopenshell_is_501_not_an_empty_file(client, tmp_path, monkeypatch):
    wid, _ = _world_with_geometry(client, tmp_path)
    import builtins

    real_import = builtins.__import__

    def _no_ifc(name, *a, **k):
        if name.split(".")[0] == "ifcopenshell":
            raise ImportError("No module named 'ifcopenshell'")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_ifc)
    r = client.post(f"/api/worlds/{wid}/export", json={"format": "ifc"})
    assert r.status_code == 501 and "ifcopenshell" in r.text.lower()
