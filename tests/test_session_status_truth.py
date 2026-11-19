"""A session must not read ``processing`` after its reconstruction job has stopped without a result.

``POST /api/sessions/{id}/reconstruct`` sets the session to ``processing``; only a successful run set it back (to
``complete``). A failed, cancelled or crashed run left it ``processing`` forever, so the UI showed a permanent spinner for
work that was no longer happening. The status is settled in the SAME transaction as the job's final (fenced) write or the
reaper's compare-and-swap, so it can neither be lost nor written by a worker that lost its lease.

Synthetic verification: temp SQLite database, synthetic rows, fake handlers. SQLite only.
"""
from __future__ import annotations

from datetime import timedelta

import pytest
import pytest_asyncio
from sqlalchemy import select

import apps.api.jobs as jobs
from apps.api.models import Job, Session, utcnow

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture()
async def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'jobs.db').as_posix()}")
    monkeypatch.setenv("WORLDSTORE_ROOT", str(tmp_path / "ws"))
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("JOB_STALE_AFTER_SECONDS", "0.5")
    import apps.api.db as db_mod
    import apps.api.worldstore_service as ws_svc

    await db_mod.dispose_db()
    await db_mod.init_db()
    ws_svc._stores.clear()
    ws_svc._store = None
    yield db_mod.get_sessionmaker()
    await db_mod.dispose_db()


async def _seed(maker, *, job_status="queued", attempts=0, max_attempts=1, **job_kw):
    async with maker() as db:
        db.add(Session(id="ses_1", name="S", status="processing", team_id="default"))
        db.add(Job(id="job_1", type=jobs.RECONSTRUCT_SESSION, entity_type="session", entity_id="ses_1", team_id="default",
                   status=job_status, attempts=attempts, max_attempts=max_attempts, **job_kw))
        await db.commit()


async def _session_status(maker) -> str:
    async with maker() as db:
        return (await db.execute(select(Session.status).where(Session.id == "ses_1").execution_options(populate_existing=True))).scalar_one()


async def _run(maker):
    async with maker() as db:
        return await jobs.process_next_job(db, worker_id="w1")


async def test_a_failed_reconstruction_does_not_leave_the_session_processing(env, monkeypatch):
    async def boom(db, job):
        raise RuntimeError("solver diverged")

    monkeypatch.setitem(jobs._HANDLERS, jobs.RECONSTRUCT_SESSION, boom)
    await _seed(env)
    done = await _run(env)
    assert done.status == "failed"
    assert await _session_status(env) == "failed"


async def test_a_retryable_failure_keeps_the_session_processing_because_the_job_will_run_again(env, monkeypatch):
    async def boom(db, job):
        raise RuntimeError("transient")

    monkeypatch.setitem(jobs._HANDLERS, jobs.RECONSTRUCT_SESSION, boom)
    await _seed(env, max_attempts=3)
    done = await _run(env)
    assert done.status == "queued"
    assert await _session_status(env) == "processing"


async def test_a_cancelled_reconstruction_returns_the_session_to_a_re_runnable_state(env, monkeypatch):
    async def never(db, job):
        raise AssertionError("must not run: cancelled before start")

    monkeypatch.setitem(jobs._HANDLERS, jobs.RECONSTRUCT_SESSION, never)
    await _seed(env, cancel_requested=True)
    done = await _run(env)
    assert done.status == "cancelled"
    assert await _session_status(env) == "created"


async def test_the_reaper_failing_a_crashed_job_at_the_attempt_limit_settles_the_session(env):
    await _seed(env, job_status="running", attempts=1, max_attempts=1, worker_id="dead",
                heartbeat_at=utcnow() - timedelta(seconds=60))
    async with env() as db:
        assert await jobs.reap_stale_jobs(db) == 1
    assert await _session_status(env) == "failed"


async def test_the_reaper_requeueing_a_crashed_job_leaves_the_session_processing(env):
    await _seed(env, job_status="running", attempts=1, max_attempts=3, worker_id="dead",
                heartbeat_at=utcnow() - timedelta(seconds=60))
    async with env() as db:
        assert await jobs.reap_stale_jobs(db) == 1
    assert await _session_status(env) == "processing"


async def test_a_completed_session_is_never_downgraded_by_a_late_failure(env):
    await _seed(env, job_status="running", attempts=1, max_attempts=1, worker_id="dead",
                heartbeat_at=utcnow() - timedelta(seconds=60))
    async with env() as db:
        s = await db.get(Session, "ses_1")
        s.status = "complete"                      # e.g. another run already finished this session
        await db.commit()
    async with env() as db:
        await jobs.reap_stale_jobs(db)
    assert await _session_status(env) == "complete"
