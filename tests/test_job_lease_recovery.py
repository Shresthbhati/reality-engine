"""Phase 3/5: job leases, fencing, the reaper's compare-and-swap, outcome grading, retry limits, timeouts, cancellation.

Synthetic verification: a temp SQLite database, synthetic ``Job`` rows and fake handlers registered into ``jobs._HANDLERS``.
No reconstruction runs; what is checked is the job lifecycle contract itself.
"""
from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
import pytest_asyncio
from sqlalchemy import func, select, update

import apps.api.jobs as jobs
from apps.api.models import Job, JobLog, utcnow

pytestmark = pytest.mark.asyncio
FAKE = "FAKE_JOB"


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


async def _add_job(maker, job_id="job_t1", job_type=FAKE, *, max_attempts=3, status="queued", **kw) -> str:
    async with maker() as db:
        db.add(Job(id=job_id, type=job_type, entity_type="session", entity_id="ses_x", team_id="default",
                   max_attempts=max_attempts, status=status, **kw))
        await db.commit()
    return job_id


async def _row(maker, job_id) -> Job:
    async with maker() as db:
        return (await db.execute(select(Job).where(Job.id == job_id).execution_options(populate_existing=True))).scalar_one()


async def _log_count(maker, job_id, stage) -> int:
    async with maker() as db:
        return (await db.execute(select(func.count()).select_from(JobLog).where(JobLog.job_id == job_id, JobLog.stage == stage))).scalar_one()


async def _age_heartbeat(maker, job_id, seconds=60) -> None:
    async with maker() as db:
        await db.execute(update(Job).where(Job.id == job_id).values(heartbeat_at=utcnow() - timedelta(seconds=seconds)))
        await db.commit()


async def _run(maker, worker_id="w1"):
    async with maker() as db:
        return await jobs.process_next_job(db, worker_id=worker_id)


# ---- claim ----------------------------------------------------------------------------------------------------------

async def test_only_one_of_many_concurrent_workers_claims_a_job(env):
    await _add_job(env)

    async def attempt(i):
        async with env() as db:
            return await jobs.claim_next_job(db, worker_id=f"w{i}")

    results = await asyncio.gather(*(attempt(i) for i in range(5)))
    assert sum(r is not None for r in results) == 1
    row = await _row(env, "job_t1")
    assert row.status == "running" and row.attempts == 1 and row.worker_id in {f"w{i}" for i in range(5)}


# ---- fencing: a worker that lost its lease cannot write ------------------------------------------------------------

@pytest.mark.parametrize("heartbeat", ["0.2", "60"], ids=["lease-loss-detected-by-heartbeat", "lease-loss-caught-by-final-write"])
async def test_a_zombie_worker_cannot_overwrite_the_new_owners_state(env, monkeypatch, heartbeat):
    monkeypatch.setenv("JOB_HEARTBEAT_INTERVAL_SECONDS", heartbeat)
    started, release = asyncio.Event(), asyncio.Event()

    async def handler(db, job):
        if job.attempts == 1:                      # worker A: stalls while it is reaped and replaced
            started.set()
            await release.wait()
            return "succeeded"
        return "succeeded"                         # worker B: finishes at once

    monkeypatch.setitem(jobs._HANDLERS, FAKE, handler)
    await _add_job(env)

    a = asyncio.create_task(_run(env, "worker-A"))
    await asyncio.wait_for(started.wait(), 5)
    await _age_heartbeat(env, "job_t1")            # A looks dead ...
    async with env() as db:
        assert await jobs.reap_stale_jobs(db) == 1  # ... so it is reaped and requeued
    assert (await _row(env, "job_t1")).status == "queued"

    b = await _run(env, "worker-B")                # B claims attempt 2 and completes it
    assert b.status == "succeeded" and b.worker_id == "worker-B" and b.attempts == 2

    release.set()                                  # A wakes up (or is told it lost the lease) and tries to finish
    a_result = await asyncio.wait_for(a, 10)

    row = await _row(env, "job_t1")
    assert (row.status, row.worker_id, row.attempts) == ("succeeded", "worker-B", 2), "the zombie overwrote the new owner"
    # The zombie returns the row as it really stands when it gives up (requeued, or B's result): never its own stale copy.
    assert a_result.worker_id in (None, "worker-B") and a_result.status in ("queued", "succeeded")
    assert await _log_count(env, "job_t1", "failed") == 0


async def test_a_zombie_heartbeat_does_not_keep_another_workers_job_alive(env):
    """After the job is re-claimed by B, A's heartbeat pulses must not refresh the row."""
    await _add_job(env, status="running", worker_id="worker-B", attempts=2, heartbeat_at=utcnow() - timedelta(seconds=60))
    lost = asyncio.Event()
    beat = asyncio.create_task(jobs._run_job_heartbeat("job_t1", 0.2, lease=("worker-A", 1), lost=lost))
    await asyncio.wait_for(lost.wait(), 5)
    await asyncio.wait_for(beat, 5)
    row = await _row(env, "job_t1")
    beat_at = row.heartbeat_at if row.heartbeat_at.tzinfo else row.heartbeat_at.replace(tzinfo=utcnow().tzinfo)
    assert utcnow() - beat_at > timedelta(seconds=30), "the zombie's pulse refreshed another worker's job"


# ---- the reaper is compare-and-swap ---------------------------------------------------------------------------------

async def test_the_reaper_never_takes_a_job_that_heartbeated_after_it_was_read(env):
    await _add_job(env, status="running", worker_id="worker-A", attempts=1, heartbeat_at=utcnow() - timedelta(seconds=60))
    async with env() as reaper:
        real_execute, fired = reaper.execute, {"done": False}

        async def execute(stmt, *a, **k):
            res = await real_execute(stmt, *a, **k)
            if not fired["done"] and str(stmt).lstrip().upper().startswith("SELECT"):
                fired["done"] = True               # the live worker heartbeats AFTER the reaper read its row
                async with env() as other:
                    await other.execute(update(Job).where(Job.id == "job_t1").values(heartbeat_at=utcnow()))
                    await other.commit()
            return res

        reaper.execute = execute
        assert await jobs.reap_stale_jobs(reaper) == 0
    row = await _row(env, "job_t1")
    assert (row.status, row.worker_id, row.attempts) == ("running", "worker-A", 1)
    assert await _log_count(env, "job_t1", "recovered") == 0


async def test_crash_recovery_requeues_until_the_attempt_limit_then_fails(env, monkeypatch):
    async def ok(db, job):
        return "succeeded"

    monkeypatch.setitem(jobs._HANDLERS, FAKE, ok)
    await _add_job(env, status="running", worker_id="dead", attempts=1, max_attempts=2, heartbeat_at=utcnow() - timedelta(seconds=60))
    async with env() as db:
        assert await jobs.reap_stale_jobs(db) == 1
    assert (await _row(env, "job_t1")).status == "queued"

    claimed = await _run(env, "w2")                # the recovered job runs to success on its second attempt
    assert claimed.status == "succeeded" and claimed.attempts == 2

    await _add_job(env, "job_t2", status="running", worker_id="dead", attempts=2, max_attempts=2,
                   heartbeat_at=utcnow() - timedelta(seconds=60))
    async with env() as db:
        assert await jobs.reap_stale_jobs(db) == 1
    last = await _row(env, "job_t2")
    assert last.status == "failed" and "after 2 attempt" in last.error


# ---- success is graded, never assumed ------------------------------------------------------------------------------

@pytest.mark.parametrize("outcome", [None, "ok", "SUCCEEDED", 1])
async def test_an_unrecognised_handler_outcome_is_never_success(env, monkeypatch, outcome):
    async def handler(db, job):
        return outcome

    monkeypatch.setitem(jobs._HANDLERS, FAKE, handler)
    await _add_job(env)
    job = await _run(env)
    assert job.status == "failed" and "unrecognised outcome" in job.error
    assert (job.payload or {}).get("failure_kind") == "invalid_outcome"
    assert job.attempts == 1, "a deterministic failure must not be retried"


async def test_a_reconstruction_cannot_succeed_without_a_verifiable_world_version(env, monkeypatch):
    from apps.api import worldstore_service as ws_svc
    from apps.api.models import World
    from world_ir.world_v1 import WorldIR

    state: dict = {}

    async def handler(db, job):
        job.payload = {**(job.payload or {}), **state}
        return "succeeded"

    monkeypatch.setitem(jobs._HANDLERS, jobs.RECONSTRUCT_SESSION, handler)

    await _add_job(env, "job_none", jobs.RECONSTRUCT_SESSION)                       # claims success, recorded no version
    r1 = await _run(env)
    assert r1.status == "failed" and "no world version" in r1.error

    state["version_id"] = "v-ghost"                                                   # points at a version that does not exist
    await _add_job(env, "job_ghost", jobs.RECONSTRUCT_SESSION)
    r2 = await _run(env)
    assert r2.status == "failed" and "no such version row" in r2.error

    async with env() as db:                                                           # a real, verifiable version
        db.add(World(id="wld_real", name="real", team_id="default"))
        await db.commit()
        version = await ws_svc.commit_version(db, world_id="wld_real", world=WorldIR(id="wld_real"), parent=None)
        await db.commit()
    state["version_id"] = version.id
    await _add_job(env, "job_good", jobs.RECONSTRUCT_SESSION)
    r3 = await _run(env)
    assert r3.status == "succeeded", r3.error

    for artifact in (ws_svc.worldstore_root() / "artifacts").rglob("*"):            # the stored bytes rot
        if artifact.is_file():
            artifact.write_bytes(b"corrupt")
    await _add_job(env, "job_rotten", jobs.RECONSTRUCT_SESSION)
    r4 = await _run(env)
    assert r4.status == "failed" and "verification" in r4.error


async def test_a_query_between_grading_and_the_fenced_write_does_not_defeat_it(env, monkeypatch):
    """Regression: the completion step queries the database after the status is set. If that autoflushed the dirty job
    row, the status would be written unconditionally and the compare-and-swap (WHERE status = 'running') would then find
    nothing to swap, reporting a normal completion as a lost lease (found by CI on the durable-queue test)."""
    async def ok(db, job):
        return "succeeded"

    async def completion_that_queries(db, job):
        await db.execute(select(Job.id).where(Job.id == job.id))   # any query autoflushes pending changes

    monkeypatch.setitem(jobs._HANDLERS, FAKE, ok)
    monkeypatch.setattr(jobs, "_emit_completion", completion_that_queries)
    await _add_job(env)
    job = await _run(env)
    assert job.status == "succeeded" and job.worker_id == "w1" and job.attempts == 1


# ---- retries, timeout, cancellation -------------------------------------------------------------------------------

async def test_retryable_failures_stop_at_the_attempt_limit(env, monkeypatch):
    async def boom(db, job):
        raise RuntimeError("transient")

    monkeypatch.setitem(jobs._HANDLERS, FAKE, boom)
    await _add_job(env, max_attempts=2)
    first = await _run(env)
    assert first.status == "queued" and first.attempts == 1 and first.worker_id is None, "a retry releases the lease"
    second = await _run(env)
    assert second.status == "failed" and second.attempts == 2 and "transient" in second.error
    assert await _run(env) is None, "a failed job is never picked up again"


async def test_a_timeout_is_a_failure_and_never_a_success(env, monkeypatch):
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "1")

    async def wedged(db, job):
        await asyncio.sleep(30)
        return "succeeded"

    monkeypatch.setitem(jobs._HANDLERS, FAKE, wedged)
    await _add_job(env, max_attempts=1)
    job = await asyncio.wait_for(_run(env), 15)
    assert job.status == "failed" and "timed out" in job.error


async def test_a_cancel_request_stops_a_running_job_and_it_grades_cancelled(env, monkeypatch):
    started = asyncio.Event()

    async def cooperative(db, job):
        started.set()
        for _ in range(200):
            await asyncio.sleep(0.05)
            await jobs._throw_if_cancelled(db, job)
        return "succeeded"

    monkeypatch.setitem(jobs._HANDLERS, FAKE, cooperative)
    await _add_job(env)
    task = asyncio.create_task(_run(env))
    await asyncio.wait_for(started.wait(), 5)
    async with env() as db:
        await db.execute(update(Job).where(Job.id == "job_t1").values(cancel_requested=True))
        await db.commit()
    job = await asyncio.wait_for(task, 10)
    assert job.status == "cancelled" and "cancelled by operator" in job.error
