"""Dead-worker recovery (``reap_stale_jobs`` -> ``get_dead_worker_jobs`` branch) is compare-and-swap, not read-modify-write.

Each test interleaves a rival write between the reaper's SELECT and its UPDATE (the window the old branch lost races in),
or runs two reapers at once, and asserts the rival's state stands and no log row is written for a swap that did not happen.

Synthetic verification: temp SQLite database, synthetic ``Job`` rows, a real exited subprocess PID as the "dead worker".
SQLite serialises writers, so this proves the CAS predicate; it is NOT a PostgreSQL concurrency validation.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import timedelta

import pytest
import pytest_asyncio
from sqlalchemy import func, select, update

import apps.api.jobs as jobs
from apps.api.models import Job, JobLog, utcnow
from reconstruction import proc

pytestmark = pytest.mark.asyncio
FAKE = "FAKE_JOB"


@pytest_asyncio.fixture()
async def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'jobs.db').as_posix()}")
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("JOB_STALE_AFTER_SECONDS", "3600")  # the stale-heartbeat branch must not be what recovers these
    import apps.api.db as db_mod
    import apps.api.worldstore_service as ws_svc

    await db_mod.dispose_db()
    await db_mod.init_db()
    ws_svc._stores.clear()
    ws_svc._store = None
    yield db_mod.get_sessionmaker()
    for job_id in list(proc._worker_pids):
        proc.unregister_worker(job_id)
    await db_mod.dispose_db()


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


async def _add_running(maker, job_id="job_d1", *, worker="worker-A", attempts=1, max_attempts=3, job_type=FAKE) -> str:
    async with maker() as db:
        db.add(Job(id=job_id, type=job_type, entity_type="session", entity_id="ses_x", team_id="default", status="running",
                   worker_id=worker, attempts=attempts, max_attempts=max_attempts, heartbeat_at=utcnow(), started_at=utcnow()))
        await db.commit()
    proc.register_worker(job_id, _dead_pid())
    assert not proc.is_worker_alive(job_id)
    return job_id


async def _row(maker, job_id="job_d1") -> Job:
    async with maker() as db:
        return (await db.execute(select(Job).where(Job.id == job_id).execution_options(populate_existing=True))).scalar_one()


async def _logs(maker, job_id="job_d1") -> int:
    async with maker() as db:
        return (await db.execute(select(func.count()).select_from(JobLog).where(JobLog.job_id == job_id))).scalar_one()


async def _reap(maker) -> int:
    async with maker() as db:
        return await jobs.reap_stale_jobs(db)


async def _reap_with_rival(maker, rival):
    """Run the reaper; ``rival(db)`` fires once, right after the reaper's per-job SELECT and before its UPDATE."""
    async with maker() as reaper:
        real_execute, fired = reaper.execute, {"done": False}

        async def execute(stmt, *a, **k):
            res = await real_execute(stmt, *a, **k)
            if not fired["done"] and "jobs.id =" in str(stmt) and str(stmt).lstrip().upper().startswith("SELECT"):
                fired["done"] = True
                async with maker() as other:
                    await rival(other)
                    await other.commit()
            return res

        reaper.execute = execute
        n = await jobs.reap_stale_jobs(reaper)
    assert fired["done"], "the interleaving never happened; the test proved nothing"
    return n


# ---- the plain path ------------------------------------------------------------------------------------------------

async def test_a_dead_worker_job_is_requeued_with_one_log_and_the_registry_is_cleaned(env):
    await _add_running(env)
    assert await _reap(env) == 1
    row = await _row(env)
    assert (row.status, row.worker_id, row.attempts) == ("queued", None, 1)
    assert await _logs(env) == 1
    assert "job_d1" not in proc._worker_pids                       # orphan cleanup still ran


async def test_a_dead_worker_job_out_of_attempts_fails_and_is_not_requeued(env):
    await _add_running(env, attempts=3, max_attempts=3)
    assert await _reap(env) == 1
    row = await _row(env)
    assert row.status == "failed" and "after 3 attempt" in row.error and row.completed_at is not None
    assert await _reap(env) == 0                                    # terminal: a second reaper pass does nothing
    assert await _logs(env) == 1


# ---- races ---------------------------------------------------------------------------------------------------------

async def test_worker_completes_while_the_reaper_is_executing(env):
    await _add_running(env)

    async def complete(db):
        await db.execute(update(Job).where(Job.id == "job_d1").values(status="succeeded", stage="complete", progress=100.0,
                                                                       completed_at=utcnow()))

    assert await _reap_with_rival(env, complete) == 0
    row = await _row(env)
    assert (row.status, row.worker_id, row.attempts) == ("succeeded", "worker-A", 1)
    assert await _logs(env) == 0                                    # no "recovered" log for a swap that did not happen


async def test_worker_heartbeats_between_select_and_update(env):
    await _add_running(env)

    async def beat(db):
        await db.execute(update(Job).where(Job.id == "job_d1").values(heartbeat_at=utcnow() + timedelta(seconds=1)))

    assert await _reap_with_rival(env, beat) == 0
    row = await _row(env)
    assert (row.status, row.worker_id, row.attempts) == ("running", "worker-A", 1)
    assert await _logs(env) == 0


async def test_job_is_reclaimed_by_another_worker_while_the_original_is_dead_or_slow(env):
    await _add_running(env)

    async def reclaim(db):  # reaped elsewhere, requeued, and claimed again by worker B before this reaper's UPDATE
        await db.execute(update(Job).where(Job.id == "job_d1").values(status="running", worker_id="worker-B", attempts=2,
                                                                       heartbeat_at=utcnow()))

    assert await _reap_with_rival(env, reclaim) == 0
    row = await _row(env)
    assert (row.status, row.worker_id, row.attempts) == ("running", "worker-B", 2), "the reaper clobbered the new attempt"
    assert await _logs(env) == 0


async def test_two_reapers_racing_over_one_dead_job_flip_it_exactly_once(env):
    await _add_running(env)
    results = await asyncio.gather(_reap(env), _reap(env))
    assert sorted(results) == [0, 1], results
    row = await _row(env)
    assert (row.status, row.attempts) == ("queued", 1)             # requeued once; attempts not double-counted
    assert await _logs(env) == 1


async def test_worker_crashes_right_after_claiming_before_the_handler_runs(env, monkeypatch):
    ran = []

    async def handler(db, job):
        ran.append(job.attempts)
        return "succeeded"

    monkeypatch.setitem(jobs._HANDLERS, FAKE, handler)
    await _add_running(env)                                         # claimed (attempt 1), registered, then the process died
    assert await _reap(env) == 1
    async with env() as db:
        done = await jobs.process_next_job(db, worker_id="worker-B")
    assert (done.status, done.attempts, done.worker_id) == ("succeeded", 2, "worker-B") and ran == [2]


async def test_a_crashed_reconstruction_never_reads_as_succeeded_and_stops_at_the_attempt_limit(env):
    await _add_running(env, job_type=jobs.RECONSTRUCT_SESSION, attempts=3, max_attempts=3)
    assert await _reap(env) == 1
    row = await _row(env)
    assert row.status == "failed" and row.status != "succeeded" and row.error
    assert not (row.payload or {}).get("version_id")               # a crash never invents a world version


async def test_timeout_and_cancellation_together_grade_cancelled_not_succeeded(env, monkeypatch):
    monkeypatch.setenv("JOB_TIMEOUT_SECONDS", "0.3")

    async def slow(db, job):
        await asyncio.sleep(5)
        return "succeeded"

    monkeypatch.setitem(jobs._HANDLERS, FAKE, slow)
    async with env() as db:
        db.add(Job(id="job_c1", type=FAKE, entity_type="session", entity_id="ses_x", team_id="default", status="queued",
                   max_attempts=1, cancel_requested=True))
        await db.commit()
    async with env() as db:
        done = await jobs.process_next_job(db, worker_id="worker-A")
    assert done.status in ("cancelled", "failed") and done.status != "succeeded"
    assert done.status == "cancelled", "operator intent (cancel) must win over a simultaneous timeout"
