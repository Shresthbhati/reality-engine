"""Task 1.2: world lifecycle API (rename, duplicate, archive, restore, soft delete, purge, status filter).

Every operation writes exactly one activity event; deleting cancels the world's jobs first; a purge refuses while a
job is still stopping and never deletes anything it refused.
"""

from __future__ import annotations

import asyncio
import importlib
from datetime import timedelta
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

from apps.api.models import ActivityEvent, Job, World, WorldVersion, new_id, utcnow
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

    async def _no_work(db):          # the in-process worker must not race the rows these tests plant
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


def make_world(env, name="House", *, with_version=True, tex: bytes | None = b"texture-bytes") -> str:
    """Create a world through the API; optionally commit a real WorldStore version (+ points/cameras blobs)."""
    wid = env.client.post("/api/worlds", json={"name": name, "description": "d"}).json()["id"]
    if with_version:
        from apps.api import worldstore_service as wss
        meta = {}
        if tex is not None:
            meta["tex"] = env.ws.put(tex)[0]

        async def go(s):
            await wss.commit_version(s, world_id=wid, world=WorldIR(name=name, metadata=meta), parent=None,
                                     points=b"points-" + name.encode(), cameras=b"cameras-" + name.encode(),
                                     report={"registered": 3})
        env.run(go)
    return wid


def events(env, wid) -> list[str]:
    async def go(s):
        rows = (await s.execute(select(ActivityEvent).where(ActivityEvent.entity_id == wid)
                                .order_by(ActivityEvent.created_at))).scalars().all()
        return [r.type for r in rows]
    return env.run(go)


def ids(env, status=None) -> list[str]:
    r = env.client.get("/api/worlds", params={"status": status} if status else None)
    assert r.status_code == 200
    return [w["id"] for w in r.json()["items"]]


def add_job(env, wid, status, **kw) -> str:
    jid = new_id("job")

    async def go(s):
        s.add(Job(id=jid, type="reconstruct_session", entity_type="world", entity_id=wid, status=status,
                  heartbeat_at=utcnow(), **kw))
        await s.commit()
    env.run(go)
    return jid


def job(env, jid) -> dict:
    return env.client.get(f"/api/jobs/{jid}").json()


def test_patch_renames_and_keeps_identity(env):
    wid = make_world(env)
    r = env.client.patch(f"/api/worlds/{wid}", json={"name": "Renamed", "description": "new"})
    assert r.status_code == 200 and r.json()["name"] == "Renamed" and r.json()["id"] == wid
    got = env.client.get(f"/api/worlds/{wid}").json()
    assert got["description"] == "new" and got["current_version_id"]
    assert events(env, wid)[-1] == "world.updated"
    assert env.client.patch(f"/api/worlds/{wid}", json={}).status_code == 422          # nothing to change
    assert env.client.patch(f"/api/worlds/{wid}", json={"name": "   "}).status_code in (400, 422)
    assert env.client.patch("/api/worlds/wld_nope", json={"name": "x"}).status_code == 404
    assert events(env, wid).count("world.updated") == 1                                # failures wrote no events


def test_archive_restore_and_status_filter(env):
    a, b = make_world(env, "A"), make_world(env, "B")
    assert set(ids(env)) == {a, b}
    r = env.client.post(f"/api/worlds/{a}/archive")
    assert r.status_code == 200 and r.json()["status"] == "archived" and r.json()["archived_at"]
    assert ids(env) == [b] and ids(env, "active") == [b] and ids(env, "archived") == [a] and ids(env, "trashed") == []
    assert env.client.get("/api/worlds", params={"status": "bogus"}).status_code == 422
    assert env.client.post(f"/api/worlds/{a}/archive").json()["status"] == "archived"   # idempotent
    assert events(env, a).count("world.archived") == 1
    r = env.client.post(f"/api/worlds/{a}/restore")
    assert r.json()["status"] == "active" and r.json()["archived_at"] is None
    assert set(ids(env)) == {a, b}
    assert env.client.post(f"/api/worlds/{a}/restore").status_code == 409               # already active
    assert [e for e in events(env, a) if e != "world.created"] == ["world.archived", "world.restored"]


def test_soft_delete_cancels_jobs_first_then_trashes_and_restores(env):
    wid = make_world(env)
    other = make_world(env, "Other")
    queued = add_job(env, wid, "queued")
    running = add_job(env, wid, "running")
    done = add_job(env, wid, "succeeded")
    other_job = add_job(env, other, "queued")
    r = env.client.delete(f"/api/worlds/{wid}")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "trashed" and body["deleted_at"]
    assert body["jobs_cancelled"] == [queued] and body["jobs_cancel_requested"] == [running]
    assert job(env, queued)["status"] == "cancelled"
    assert job(env, running)["cancel_requested"] is True            # worker stops at its next stage boundary
    assert job(env, done)["status"] == "succeeded"                  # finished work is never rewritten
    assert job(env, other_job)["status"] == "queued"                # another world's jobs are untouched
    assert ids(env) == [other] and ids(env, "trashed") == [wid]
    assert env.client.get(f"/api/worlds/{wid}").json()["status"] == "trashed"      # still readable (recoverable)
    assert env.client.patch(f"/api/worlds/{wid}", json={"name": "x"}).status_code == 409
    assert env.client.post(f"/api/worlds/{wid}/archive").status_code == 409
    assert env.client.post(f"/api/worlds/{wid}/duplicate").status_code == 409
    assert env.client.delete(f"/api/worlds/{wid}").status_code == 409                # already trashed
    assert events(env, wid).count("world.trashed") == 1
    r = env.client.post(f"/api/worlds/{wid}/restore")
    assert r.json()["status"] == "active" and r.json()["deleted_at"] is None
    assert events(env, wid)[-1] == "world.restored" and ids(env, "trashed") == []


def test_duplicate_is_copy_on_write_and_independent(env):
    src = make_world(env, "Src")
    src_info = env.client.get(f"/api/worlds/{src}").json()
    n_blobs = len(list((env.tmp / "ws" / "artifacts").glob("??/*.bin")))
    store_files = len(list((env.tmp / "artifacts").rglob("*"))) if (env.tmp / "artifacts").exists() else 0
    r = env.client.post(f"/api/worlds/{src}/duplicate")
    assert r.status_code == 201
    dup = r.json()
    assert dup["id"] != src and dup["name"] == "Src (copy)" and dup["status"] == "active"
    assert dup["duplicated_from"] == src and dup["current_version_id"]
    assert dup["current_version_id"] != src_info["current_version_id"]
    # copy-on-write: the only new store blob is the re-saved WorldIR JSON (new world id); geometry/points are shared
    assert len(list((env.tmp / "ws" / "artifacts").glob("??/*.bin"))) == n_blobs + 1
    assert (len(list((env.tmp / "artifacts").rglob("*"))) if (env.tmp / "artifacts").exists() else 0) == store_files

    async def rows(s):
        v = {w: (await s.execute(select(WorldVersion).where(WorldVersion.world_id == w))).scalars().one()
             for w in (src, dup["id"])}
        return {w: (r.points_artifact_uri, r.cameras_artifact_uri, r.report, r.label) for w, r in v.items()}
    got = env.run(rows)
    assert got[src][:3] == got[dup["id"]][:3] and got[src][0].startswith("sha256://")
    assert got[src][3] is None and "Duplicate of Src" in got[dup["id"]][3]
    # the duplicate's version reads back through the normal routes
    assert env.client.get(f"/api/worlds/{dup['id']}/versions").json()["items"][0]["is_current"] is True
    assert set(ids(env)) == {src, dup["id"]}
    # independence: renaming the copy leaves the source alone
    env.client.patch(f"/api/worlds/{dup['id']}", json={"name": "Other name"})
    assert env.client.get(f"/api/worlds/{src}").json()["name"] == "Src"
    assert events(env, dup["id"]) == ["world.duplicated", "world.updated"]
    assert events(env, src).count("world.duplicated") == 0                 # the event belongs to the new world
    assert env.client.post("/api/worlds/wld_nope/duplicate").status_code == 404


def test_duplicate_world_without_a_version(env):
    src = make_world(env, "Empty", with_version=False)
    r = env.client.post(f"/api/worlds/{src}/duplicate")
    assert r.status_code == 201 and r.json()["current_version_id"] is None


def test_permanent_delete_removes_world_and_keeps_audit_event(env):
    wid = make_world(env)
    keep = make_world(env, "Keep")
    add_job(env, wid, "queued")
    r = env.client.delete(f"/api/worlds/{wid}", params={"permanent": "true"})
    assert r.status_code == 200 and r.json()["permanently_deleted"] is True
    assert len(r.json()["versions_removed"]) == 1
    assert env.client.get(f"/api/worlds/{wid}").status_code == 404
    assert ids(env, "trashed") == [] and ids(env) == [keep]
    assert env.client.get(f"/api/worlds/{keep}/versions").json()["items"]        # neighbours untouched
    assert events(env, wid)[-1] == "world.purged"
    left = env.run(lambda s: s.scalar(select(func.count()).select_from(Job).where(Job.entity_id == wid)))
    assert left == 0
    assert env.client.delete(f"/api/worlds/{wid}", params={"permanent": "true"}).status_code == 404


def test_permanent_delete_refuses_while_a_job_is_still_stopping(env):
    wid = make_world(env)
    running = add_job(env, wid, "running")
    r = env.client.delete(f"/api/worlds/{wid}", params={"permanent": "true"})
    assert r.status_code == 409 and running in r.json()["detail"]
    assert env.client.get(f"/api/worlds/{wid}").status_code == 200               # nothing was deleted
    assert env.client.get(f"/api/worlds/{wid}/versions").json()["items"]
    assert job(env, running)["cancel_requested"] is True                          # but the stop was requested
    # once the worker has finished stopping (status cancelled), the purge goes through
    async def stop(s):
        j = await s.get(Job, running)
        j.status, j.completed_at = "cancelled", utcnow()
        await s.commit()
    env.run(stop)
    assert env.client.delete(f"/api/worlds/{wid}", params={"permanent": "true"}).status_code == 200


def test_trash_sweep_purges_only_worlds_trashed_over_30_days(env):
    from apps.api.world_lifecycle import purge_expired_trash
    old, recent, active = make_world(env, "Old"), make_world(env, "Recent"), make_world(env, "Active")
    env.client.delete(f"/api/worlds/{old}")
    env.client.delete(f"/api/worlds/{recent}")

    async def age(s):
        (await s.get(World, old)).deleted_at = utcnow() - timedelta(days=31)
        (await s.get(World, recent)).deleted_at = utcnow() - timedelta(days=29)
        await s.commit()
    env.run(age)

    async def sweep(s):
        return [p["id"] for p in await purge_expired_trash(s)]
    assert env.run(sweep) == [old]
    assert env.client.get(f"/api/worlds/{old}").status_code == 404
    assert env.client.get(f"/api/worlds/{recent}").json()["status"] == "trashed"
    assert env.client.get(f"/api/worlds/{active}").json()["status"] == "active"
    assert env.run(sweep) == []                                                    # idempotent


def test_sweep_is_wired_into_the_worker_loop():
    import inspect
    import apps.api.jobs as jobs_mod
    src = inspect.getsource(jobs_mod.worker_loop)
    assert "sweep_trash" in src and callable(jobs_mod.sweep_trash)
