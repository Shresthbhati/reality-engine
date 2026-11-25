"""Workstream 4: deterministic failure injection (fakes only) and the never-reports-success invariant.

Worker death during artifact persistence, API shutdown during a run, cache/checkpoint corruption, a missing or corrupted model
artifact, a fusion failure, an export failing midway -- and, for every way a job can end badly (failed, timed out, cancelled, lost),
that neither the job status nor any summary field the API serves ever says it succeeded.
"""
from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import select, update

import apps.api.jobs as jobs
from apps.api.models import Job, JobLog, utcnow

FAKE = "FAKE_JOB"
pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture()
async def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'jobs.db').as_posix()}")
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("JOB_HEARTBEAT_INTERVAL_SECONDS", "0.2")
    monkeypatch.setenv("JOB_STALE_AFTER_SECONDS", "0.5")
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "30")
    import apps.api.db as db_mod
    import apps.api.worldstore_service as ws_svc

    await db_mod.dispose_db()
    await db_mod.init_db()
    ws_svc._stores.clear()
    ws_svc._store = None
    yield db_mod.get_sessionmaker()
    await db_mod.dispose_db()


async def _add_job(maker, job_id="job_t1", job_type=FAKE, *, max_attempts=3):
    async with maker() as db:
        db.add(Job(id=job_id, type=job_type, entity_type="session", entity_id="ses_x", team_id="default", max_attempts=max_attempts))
        await db.commit()
    return job_id


async def _row(maker, job_id="job_t1") -> Job:
    async with maker() as db:
        return (await db.execute(select(Job).where(Job.id == job_id).execution_options(populate_existing=True))).scalar_one()


async def _age(maker, job_id="job_t1"):
    async with maker() as db:
        await db.execute(update(Job).where(Job.id == job_id).values(heartbeat_at=utcnow() - timedelta(seconds=60)))
        await db.commit()


async def _run(maker, worker="w1"):
    async with maker() as db:
        return await jobs.process_next_job(db, worker_id=worker)


def _never_success(job: Job, api: dict | None = None) -> None:
    """The invariant: every field that could be read as 'it worked' says otherwise."""
    assert job.status not in ("succeeded", "partial")
    assert job.stage not in ("done", "done_with_warnings") and (job.progress or 0) < 100.0
    p = job.payload or {}
    assert p.get("outcome") not in ("succeeded", "partial"), p
    assert not (p.get("validated") is True and job.status != "cancelled"), p
    if api is not None:
        assert api["status"] == job.status and api["photos_placed"] is None
        assert (api["payload"] or {}).get("outcome") not in ("succeeded", "partial")


# ---- worker death during artifact persistence ---------------------------------------------------------------------------

async def test_worker_death_after_artifact_write_requeues_then_fails_but_never_succeeds(env):
    from apps.api.storage import store_bytes

    await _add_job(env, max_attempts=2)
    for attempt in (1, 2):
        async with env() as db:
            job = await jobs.claim_next_job(db, worker_id=f"dead{attempt}")
            assert job is not None
            store_bytes(b"points written, version never committed %d" % attempt)       # the artifact is on disk...
            job.payload = {"outcome": "succeeded", "validated": True, "adopted": True}    # ...the handler had claimed success...
            await db.commit()                                                              # ...and then the process died: no grading
        await _age(env)
        async with env() as db:
            assert await jobs.reap_stale_jobs(db) == 1
        row = await _row(env)
        _never_success(row)
        assert row.payload["outcome_claimed_by_handler"] == "succeeded"
        if attempt == 1:
            assert row.status == "queued" and row.worker_id is None and row.payload["outcome"] == "retrying"
    assert row.status == "failed" and "after 2 attempt" in row.error and row.payload["outcome"] == "failed"


async def test_a_requeued_job_after_worker_death_completes_exactly_once(env, monkeypatch):
    await _add_job(env)
    async with env() as db:
        await jobs.claim_next_job(db, worker_id="dead")
    await _age(env)
    async with env() as db:
        await jobs.reap_stale_jobs(db)

    async def ok(db, job):
        return "succeeded"

    monkeypatch.setitem(jobs._HANDLERS, FAKE, ok)
    job = await _run(env, "w2")
    assert job.status == "succeeded" and job.attempts == 2 and job.worker_id == "w2"
    assert await _run(env, "w3") is None


# ---- API shutdown during reconstruction -----------------------------------------------------------------------------------

async def test_shutdown_mid_run_hands_the_job_back_and_stops_the_pulse(env, monkeypatch):
    started, finished = asyncio.Event(), []

    async def slow(db, job):
        started.set()
        try:
            await asyncio.sleep(60)
        finally:
            finished.append("handler-cancelled")      # the handler is stopped, not orphaned
        return "succeeded"

    monkeypatch.setitem(jobs._HANDLERS, FAKE, slow)
    await _add_job(env, max_attempts=3)
    task = asyncio.create_task(_run(env))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()                                      # the API / worker is stopped
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.3)
    row = await _row(env)
    assert row.status == "queued" and row.worker_id is None and row.attempts == 1, "recoverable, not stuck 'running'"
    _never_success(row)
    assert finished == ["handler-cancelled"]
    assert not [t for t in asyncio.all_tasks() if "_run_job_heartbeat" in repr(t.get_coro()) and not t.done()], "no zombie heartbeat"

    async def ok(db, job):
        return "succeeded"

    monkeypatch.setitem(jobs._HANDLERS, FAKE, ok)
    assert (await _run(env, "w2")).status == "succeeded"                                  # resumed after the restart


async def test_shutdown_on_the_last_attempt_marks_failed_never_succeeded(env, monkeypatch):
    started = asyncio.Event()

    async def slow(db, job):
        job.payload = {"outcome": "succeeded", "validated": True}
        started.set()
        await asyncio.sleep(60)

    monkeypatch.setitem(jobs._HANDLERS, FAKE, slow)
    await _add_job(env, max_attempts=1)
    task = asyncio.create_task(_run(env))
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    row = await _row(env)
    assert row.status == "failed" and "shut down" in row.error
    _never_success(row)


# ---- the invariant: every bad ending, in the DB row and in what the API serves ---------------------------------------------

async def _api_view(maker, job_id):
    from apps.api import routes_jobs

    async with maker() as db:
        return await routes_jobs.get_job(job_id, db)


@pytest.mark.parametrize("ending", ["raises", "timeout", "cancelled", "bad_outcome", "claimed_success_then_raises"])
async def test_failed_timed_out_or_cancelled_jobs_never_report_success(env, monkeypatch, ending):
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "1")

    async def handler(db, job):
        if ending in ("claimed_success_then_raises", "timeout", "cancelled"):
            job.payload = {"outcome": "succeeded", "validated": True, "adopted": True, "cameras_registered": 7}
        if ending == "raises" or ending == "claimed_success_then_raises":
            raise RuntimeError("fusion exploded")
        if ending == "timeout":
            await asyncio.sleep(30)
        if ending == "cancelled":
            raise jobs._JobCancelled("operator cancelled")
        return "ok-ish"                                   # bad_outcome: not a defined outcome

    monkeypatch.setitem(jobs._HANDLERS, FAKE, handler)
    await _add_job(env, max_attempts=1)
    job = await asyncio.wait_for(_run(env), 20)
    assert job.status == ("cancelled" if ending == "cancelled" else "failed")
    api = await _api_view(env, "job_t1")
    _never_success(job, api)
    assert api["terminal"] is True and (api["failure_reason"] if job.status == "failed" else True)


async def test_cancel_requested_during_a_failing_run_grades_cancelled_not_success(env, monkeypatch):
    async def handler(db, job):
        job.payload = {"outcome": "succeeded", "validated": True}
        await db.execute(update(Job).where(Job.id == job.id).values(cancel_requested=True))
        await db.commit()
        raise RuntimeError("late failure")

    monkeypatch.setitem(jobs._HANDLERS, FAKE, handler)
    await _add_job(env, max_attempts=1)
    job = await _run(env)
    assert job.status == "cancelled"
    _never_success(job, await _api_view(env, "job_t1"))


# ---- cache / checkpoint corruption ----------------------------------------------------------------------------------------

def _stage_result(outputs):
    from engine.pipeline.stages.base import StageResult

    t = datetime.now(timezone.utc)
    return StageResult("depth", "succeeded", "h" * 64, outputs, {}, [], None, t, t, None)


async def test_a_corrupted_stage_checkpoint_is_a_miss_so_the_stage_reruns(env):
    from apps.api.stage_checkpoints import DbCheckpointStore
    from apps.api.storage import store_bytes

    await _add_job(env)
    digest, path = store_bytes(b"depth map bytes")
    store = DbCheckpointStore(env, "default", loop=asyncio.get_running_loop())
    await store._record("job_t1", _stage_result({"depth": digest}))
    assert await store._lookup("job_t1", "depth", "h" * 64) is not None
    path.write_bytes(b"DEPTH MAP BYTES")                 # bit rot: same size, wrong content
    assert await store._lookup("job_t1", "depth", "h" * 64) is None
    path.write_bytes(b"short")                           # truncation
    assert await store._lookup("job_t1", "depth", "h" * 64) is None
    path.unlink()                                        # deletion
    assert await store._lookup("job_t1", "depth", "h" * 64) is None


async def test_store_bytes_heals_a_damaged_blob_and_a_crash_mid_write_leaves_none(tmp_path, monkeypatch):
    from apps.api import storage

    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "a"))
    data = b"x" * 1000
    digest, path = storage.store_bytes(data)
    path.write_bytes(b"x" * 10)                          # truncated by a crash
    storage.store_bytes(data)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest and storage.verify_artifact(f"sha256://{digest}")

    import pathlib
    real_replace = pathlib.Path.replace

    def crash(self, target):
        raise OSError("disk pulled")

    monkeypatch.setattr(pathlib.Path, "replace", crash)
    other = b"y" * 500
    with pytest.raises(OSError):
        storage.store_bytes(other)
    monkeypatch.setattr(pathlib.Path, "replace", real_replace)
    d2 = hashlib.sha256(other).hexdigest()
    assert storage.resolve_artifact(f"sha256://{d2}") is None, "a partial write never appears as a complete artifact"
    assert not list((tmp_path / "a").rglob("*.tmp")), "no temp debris"


def test_depth_cache_treats_damaged_entries_and_index_as_misses(tmp_path):
    import numpy as np

    from engine.multiengine.depth_cache import DepthCache

    cache = DepthCache(tmp_path / "dc")
    key = DepthCache.key("eng", "a" * 64, "b" * 64)
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    cache.put(key, arr)
    assert np.array_equal(cache.get(key), arr)
    blob = next((tmp_path / "dc" / "artifacts").rglob("*"))
    files = [p for p in (tmp_path / "dc" / "artifacts").rglob("*") if p.is_file()]
    files[0].write_bytes(b"garbage")
    assert cache.get(key) is None
    (tmp_path / "dc" / "index.json").write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None and blob is not None


# ---- model artifact missing / corrupted; fusion failure -------------------------------------------------------------------

def test_a_corrupted_model_checkpoint_is_refused_and_a_missing_one_is_reported(tmp_path, monkeypatch):
    from perception import model_registry as mr

    # Isolate the process-wide registry so the probe entry cannot leak into other tests
    # (test_model_licence_gate walks every registry entry and requires a licence).
    monkeypatch.setattr(mr, "_REGISTRY", dict(mr._REGISTRY))

    ckpt = tmp_path / "w.pt"
    ckpt.write_bytes(b"real weights")
    good = hashlib.sha256(b"real weights").hexdigest()
    mr.record_pretrained("inj_model", "depth", "manual", ckpt, sha256=good)
    mr.require_verified(ckpt)                                   # intact: fine
    ckpt.write_bytes(b"real weightZ")                           # corrupted on disk
    with pytest.raises(mr.CheckpointMismatch):
        mr.require_verified(ckpt)
    assert mr.checkpoint_for("no_such_model_xyz", cache_dir=tmp_path) is None
    spec = mr.get("inj_model").to_dict()
    assert spec["present"] is True and spec["sha256_verified"] is False


def test_a_failing_model_load_leaves_the_pool_clean_and_usable():
    from engine.hardware.model_pool import ModelPool

    pool = ModelPool(budget_mb=3000)

    def missing(name):
        raise FileNotFoundError(f"{name} weights missing")

    with pytest.raises(FileNotFoundError):
        with pool.acquire("dpt_hybrid", missing):
            pass
    assert pool.loaded() == [] and pool.loads == {} and pool.used_mb == 0
    with pool.acquire("dpt_hybrid", lambda n: object()) as m:
        assert m is not None
    assert pool.loaded() == ["dpt_hybrid"]


def test_a_fusion_failure_or_a_stage_timeout_can_never_grade_succeeded(tmp_path):
    import time

    from engine.pipeline.stages.base import StageContext, StageResult, StageSpec
    from engine.pipeline.stages.runner import run_stages

    class S:
        def __init__(self, name, required, fn):
            self.spec, self._fn = StageSpec(name=name, version="1", resource_class="cpu", required=required), fn

        def run(self, ctx):
            self._fn()
            t = datetime.now(timezone.utc)
            return StageResult(self.spec.name, "succeeded", "", {}, {}, [], None, t, t, None)

    def boom():
        raise ValueError("fusion: singular covariance")

    for required, want in ((True, "failed"), (False, "partial")):
        stages = [S("pose", True, lambda: None), S("fusion", required, boom)]
        ctx = StageContext("j", tmp_path, {}, {}, {})
        m = run_stages(stages, ctx)
        assert m.stages[1].status == "failed" and "singular covariance" in m.stages[1].error
        assert m.grade([s.spec for s in stages]) == want

    stages = [S("fusion", True, lambda: time.sleep(1.0))]
    m = run_stages(stages, StageContext("j", tmp_path, {}, {}, {}), stage_timeouts={"fusion": 0.05})
    assert m.stages[0].status == "failed" and m.stages[0].metrics["timed_out"] == 1.0
    assert m.grade([stages[0].spec]) == "failed"


# ---- export failing midway ------------------------------------------------------------------------------------------------

def test_an_export_that_fails_midway_presents_no_file_and_records_nothing(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("aiosqlite")
    import importlib

    from fastapi.testclient import TestClient

    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("ENABLE_WORKER", "false")
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    from provenance import Provenance
    from sdk import reality
    from world_ir import Entity, EntityType
    from world_ir.world_v1 import WorldIR
    from worldstore.store import WorldStore

    with TestClient(main_mod.app, raise_server_exceptions=False) as client:
        wid = client.post("/api/worlds", json={"name": "Export world"}).json()["id"]
        wir = WorldIR(id=wid)
        wir.entities["e1"] = Entity(id="e1", type=EntityType.STRUCTURE, name="Shed", provenance=Provenance.RECONSTRUCTED, confidence=0.8)
        stored = WorldStore(tmp_path / "ws").save_version(wir, parent=None)

        async def seed():
            from apps.api.models import World, WorldVersion

            async with db_mod.get_sessionmaker()() as db:
                db.add(WorldVersion(id=stored.version_id, world_id=wid, parent_version_id=None, artifact_uri=stored.artifact_uri,
                                    artifact_hash=stored.artifact_hash, source_session_ids=[], changed_entity_ids=[]))
                (await db.get(World, wid)).current_version_id = stored.version_id
                await db.commit()

        asyncio.run(seed())

        def exporter_dies_midway(world, fmt, artifact_store=None):
            raise RuntimeError("exporter died after writing half the file")

        monkeypatch.setattr(reality, "export", exporter_dies_midway)
        r = client.post(f"/api/worlds/{wid}/export", json={"format": "gltf"})
        assert r.status_code >= 500
        assert "download_url" not in r.text and "artifact_uri" not in r.text

        async def exports():
            from apps.api.models import ExportRecord

            async with db_mod.get_sessionmaker()() as db:
                return (await db.execute(select(ExportRecord))).scalars().all()

        assert asyncio.run(exports()) == [], "no ExportRecord for a failed export"
        assert not [p for p in (tmp_path / "artifacts").rglob("*") if p.is_file()], "no blob left that could be served as the export"
