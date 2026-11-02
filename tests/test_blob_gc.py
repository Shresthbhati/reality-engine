"""Task 1.3: permanent delete removes rows + WorldStore versions + ONLY unreferenced blobs.

Two worlds share blobs two ways: an uploaded photo (same sha256 in both worlds' evidence) and a duplicated
world (same points/cameras sha256 blobs and the same artifact:// geometry blob in the WorldStore). Purging one must
leave the shared blobs readable by the other; purging the last holder finally removes them.
"""

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
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from apps.api import storage
from apps.api.models import Evidence, EvidenceDecision, Session, Upload, WorldVersion, new_id
from world_ir.artifact_store import FileArtifactStore
from world_ir.world_v1 import WorldIR


@pytest.fixture()
def env(tmp_path: Path, monkeypatch):
    url = f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)
    import apps.api.jobs as jobs_mod

    async def _no_work(db):
        return None
    monkeypatch.setattr(jobs_mod, "process_next_job", _no_work)

    def run(fn):
        async def go():
            eng = create_async_engine(url)
            try:
                async with async_sessionmaker(eng, expire_on_commit=False)() as s:
                    return await fn(s)
            finally:
                await eng.dispose()
        return asyncio.run(go())

    with TestClient(main_mod.app) as c:
        yield SimpleNamespace(client=c, run=run, tmp=tmp_path, ws=FileArtifactStore(tmp_path / "ws" / "artifacts"))


def blob_exists(digest: str) -> bool:
    return storage.resolve_artifact(f"sha256://{digest}") is not None


def attach_photo(env, wid: str, data: bytes, name: str) -> str:
    """A session on the world with one uploaded+evidence photo whose blob is `data` (content-addressed)."""
    digest, _ = storage.store_bytes(data)
    sid, uid, eid = new_id("ses"), new_id("upl"), new_id("evd")

    async def go(s):
        s.add(Session(id=sid, name="cap", world_id=wid))
        await s.flush()
        s.add(Upload(id=uid, session_id=sid, filename=name, artifact_uri=f"sha256://{digest}"))
        s.add(Evidence(id=eid, name=name, type="photo", session_id=sid, upload_id=uid,
                       artifact_uri=f"sha256://{digest}"))
        await s.flush()
        s.add(EvidenceDecision(evidence_id=eid, decision="accepted"))
        await s.commit()
    env.run(go)
    return digest


def make_world(env, name, tex=None) -> str:
    from apps.api import worldstore_service as wss
    wid = env.client.post("/api/worlds", json={"name": name}).json()["id"]
    meta = {"tex": env.ws.put(tex)[0]} if tex else {}

    async def go(s):
        await wss.commit_version(s, world_id=wid, world=WorldIR(name=name, metadata=meta), parent=None,
                                 points=b"points-" + name.encode(), cameras=b"cameras-" + name.encode())
    env.run(go)
    return wid


def version_blobs(env, wid) -> tuple[str, str]:
    async def go(s):
        v = (await s.execute(select(WorldVersion).where(WorldVersion.world_id == wid))).scalars().one()
        return storage.blob_digest(v.points_artifact_uri), storage.blob_digest(v.cameras_artifact_uri)
    return env.run(go)


def count(env, model, **where) -> int:
    async def go(s):
        q = select(func.count()).select_from(model)
        for k, v in where.items():
            q = q.where(getattr(model, k) == v)
        return await s.scalar(q)
    return env.run(go)


def store_blob_files(env) -> set[str]:
    return {p.stem for p in (env.tmp / "ws" / "artifacts").glob("??/*.bin")}


def purge(env, wid) -> dict:
    r = env.client.delete(f"/api/worlds/{wid}", params={"permanent": "true"})
    assert r.status_code == 200, r.text
    return r.json()


def test_purge_removes_only_unreferenced_upload_blobs(env):
    a = make_world(env, "A")
    b = make_world(env, "B")
    shared = attach_photo(env, a, b"the same photo bytes", "a.jpg")
    assert attach_photo(env, b, b"the same photo bytes", "b.jpg") == shared      # identical sha256 in both worlds
    only_a = attach_photo(env, a, b"photo only world A has", "a2.jpg")
    export_blob, _ = storage.store_bytes(b"an export nobody row-references")          # not a candidate: never touched

    report = purge(env, a)
    assert report["sessions_removed"] == 2 and report["evidence_removed"] == 2
    assert only_a in report["blobs_removed"] and shared in report["blobs_kept_shared"]
    assert not blob_exists(only_a)
    assert blob_exists(shared), "blob still referenced by world B's evidence must survive"
    assert blob_exists(export_blob), "an unreferenced blob that was not part of the purge is never deleted"
    # world A's rows are gone, world B's are intact
    assert count(env, Evidence) == 1 and count(env, Upload) == 1 and count(env, Session) == 1
    assert count(env, EvidenceDecision) == 1
    assert env.client.get(f"/api/worlds/{b}").status_code == 200
    # the last holder finally lets the blob go
    report_b = purge(env, b)
    assert shared in report_b["blobs_removed"] and not blob_exists(shared)
    assert blob_exists(export_blob)


def test_purge_keeps_blobs_shared_with_a_duplicate_until_the_last_world_goes(env):
    src = make_world(env, "Src", tex=b"shared geometry texture")
    dup = env.client.post(f"/api/worlds/{src}/duplicate").json()["id"]
    pts, cams = version_blobs(env, src)
    assert version_blobs(env, dup) == (pts, cams)                                  # same sha256 blobs, no copies
    before = store_blob_files(env)

    r = purge(env, src)
    assert len(r["versions_removed"]) == 1 and {pts, cams} <= set(r["blobs_kept_shared"])
    assert blob_exists(pts) and blob_exists(cams), "the duplicate still references these"
    after = store_blob_files(env)
    # only the source's own world JSON blob (different bytes: different world id) was removed; the texture survives
    assert len(before - after) == 1 and len(after) == len(before) - 1
    assert env.ws.put(b"shared geometry texture")[1] in after
    # the duplicate still loads from the WorldStore and through the API
    assert env.client.get(f"/api/worlds/{dup}/versions").json()["items"][0]["is_current"] is True
    from apps.api import worldstore_service as wss
    vid = env.client.get(f"/api/worlds/{dup}").json()["current_version_id"]
    assert wss.get_store().verify_version(vid) == []

    r2 = purge(env, dup)
    assert pts in r2["blobs_removed"] and cams in r2["blobs_removed"]
    assert not blob_exists(pts) and not blob_exists(cams)
    assert store_blob_files(env) == set()                                           # texture + JSON gone with the last holder
    assert wss.get_store().list_versions() == []


def test_failed_purge_deletes_no_blobs(env):
    """A refused purge (job still stopping) must leave rows AND blobs exactly as they were."""
    from apps.api.models import Job, utcnow
    a = make_world(env, "A")
    digest = attach_photo(env, a, b"precious", "p.jpg")
    pts, _ = version_blobs(env, a)

    async def plant(s):
        s.add(Job(id=new_id("job"), type="reconstruct_session", entity_type="world", entity_id=a,
                  status="running", heartbeat_at=utcnow()))
        await s.commit()
    env.run(plant)
    assert env.client.delete(f"/api/worlds/{a}", params={"permanent": "true"}).status_code == 409
    assert blob_exists(digest) and blob_exists(pts) and count(env, Evidence) == 1
    assert len(store_blob_files(env)) == 1


def test_gc_blobs_unit_only_candidates_and_respects_references(env):
    d_ref, _ = storage.store_bytes(b"referenced")
    d_free, _ = storage.store_bytes(b"free candidate")
    d_other, _ = storage.store_bytes(b"free but not asked about")
    a = make_world(env, "A")
    attach_photo(env, a, b"referenced", "r.jpg")

    async def go(s):
        return await storage.gc_blobs(s, {d_ref, d_free})
    out = env.run(go)
    assert out == {"removed": [d_free], "kept_shared": [d_ref]}
    assert blob_exists(d_ref) and blob_exists(d_other) and not blob_exists(d_free)
    assert storage.blob_digest("sha256://../../etc") is None and storage.blob_digest("artifact://" + "a" * 64) is None
