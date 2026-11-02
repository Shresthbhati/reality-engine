"""GET /api/worlds/{id}/artifacts/{sha256}: world-scoped, content-addressed texture reads."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")
pytest.importorskip("greenlet")
from fastapi.testclient import TestClient

from world_ir.artifact_store import FileArtifactStore

PNG = b"\x89PNG\r\n\x1a\n" + b"fake-png-body"


@pytest.fixture()
def env(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    store = FileArtifactStore(tmp_path / "ws" / "pipeline-artifacts")
    worlds: dict[str, SimpleNamespace] = {}
    import apps.api.routes_artifacts as ra
    monkeypatch.setattr(ra, "_load_world_or_409", lambda row: worlds[row.world_id])
    with TestClient(main_mod.app) as c:
        yield SimpleNamespace(client=c, db=db_mod, store=store, worlds=worlds)


def _world(env, blob: bytes | None, referenced: bool = True) -> tuple[str, str]:
    """Create a world with a head version; its single entity references `blob` as a texture."""
    from apps.api.models import World, WorldVersion, new_id

    wid = env.client.post("/api/worlds", json={"name": "w"}).json()["id"]
    uri, digest = env.store.put(blob) if blob is not None else ("artifact://" + "0" * 64, "0" * 64)
    tex = {"uri": uri, "hash": digest} if referenced else None
    env.worlds[wid] = SimpleNamespace(
        entities={"e1": SimpleNamespace(custom_properties={"appearance": {"texture": tex}})})

    async def go():
        async with env.db.get_sessionmaker()() as s:
            w = await s.get(World, wid)
            v = WorldVersion(id=new_id("ver"), world_id=wid)
            s.add(v)
            await s.flush()
            w.current_version_id = v.id
            await s.commit()
    asyncio.run(go())
    return wid, digest


def test_serves_referenced_png_immutable(env):
    wid, digest = _world(env, PNG)
    r = env.client.get(f"/api/worlds/{wid}/artifacts/{digest}")
    assert r.status_code == 200 and r.content == PNG
    assert r.headers["content-type"] == "image/png"
    assert "immutable" in r.headers["cache-control"]


def test_non_png_is_octet_stream(env):
    wid, digest = _world(env, b"not a png")
    r = env.client.get(f"/api/worlds/{wid}/artifacts/{digest}")
    assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"


@pytest.mark.parametrize("bad", ["abc", "G" * 64, "A" * 64, "0" * 63, "0" * 65, "..%2f" + "0" * 59])
def test_bad_hash_rejected(env, bad):
    wid, _ = _world(env, PNG)
    assert env.client.get(f"/api/worlds/{wid}/artifacts/{bad}").status_code in (404, 422)
    if "%" not in bad:
        assert env.client.get(f"/api/worlds/{wid}/artifacts/{bad}").status_code == 422


def test_unknown_world_404(env):
    assert env.client.get(f"/api/worlds/wld_nope/artifacts/{'0' * 64}").status_code == 404


def test_other_worlds_artifact_404(env):
    _, other_digest = _world(env, PNG)
    wid2, _ = _world(env, b"\x89PNG\r\n\x1a\nsecond")
    assert env.client.get(f"/api/worlds/{wid2}/artifacts/{other_digest}").status_code == 404


def test_referenced_but_missing_blob_404(env):
    wid, digest = _world(env, None)
    assert env.client.get(f"/api/worlds/{wid}/artifacts/{digest}").status_code == 404
