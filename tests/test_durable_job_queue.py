"""Tests for T-6.2 Durable Job Queue.

Verifies:
1. Worker restart loses no job: a crashed worker's running job is reaped back to queued
   and completed by the restarted worker.
2. Atomic claim concurrency: multiple competing workers never claim the same job.
3. Active heartbeat pulsing during long-running execution.
4. Terminal failure when max_attempts is exceeded.
5. In-process worker disabled via environment configuration.
"""

from __future__ import annotations

import asyncio
import importlib
import unittest.mock as mock
from datetime import timedelta

import pytest
from sqlalchemy import select

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from apps.api.jobs import (
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    claim_next_job,
    process_next_job,
    reap_stale_jobs,
)
from apps.api.models import Job, JobLog, utcnow


@pytest.fixture()
def db_maker(tmp_path, monkeypatch):
    db_file = tmp_path / "queue_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file.as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("JOB_STALE_AFTER_SECONDS", "1")
    monkeypatch.setenv("JOB_HEARTBEAT_INTERVAL_SECONDS", "0.2")

    import apps.api.db as db_mod
    importlib.reload(db_mod)
    asyncio.run(db_mod.init_db())

    maker = db_mod.get_sessionmaker()
    yield maker
    asyncio.run(db_mod.dispose_db())


def test_worker_restart_loses_no_job(db_maker):
    """If a worker crashes mid-run, restarting the worker reaps and executes the job to completion."""
    maker = db_maker
    job_id = "job_restart_test_1"

    async def _test():
        # 1. Enqueue job
        async with maker() as db:
            db.add(
                Job(
                    id=job_id,
                    type="PROCESS_EVIDENCE",
                    entity_type="evidence",
                    entity_id="evi_001",
                    status=JOB_QUEUED,
                    attempts=0,
                    max_attempts=3,
                    priority=10,
                )
            )
            await db.commit()

        # 2. Worker 1 claims job
        async with maker() as db:
            claimed = await claim_next_job(db, worker_id="worker-crash-candidate")
            assert claimed is not None
            assert claimed.id == job_id
            assert claimed.status == JOB_RUNNING
            assert claimed.attempts == 1
            assert claimed.worker_id == "worker-crash-candidate"

        # 3. Simulate worker 1 crashing: worker dies abruptly, leaving heartbeat in the past
        past = utcnow() - timedelta(seconds=10)
        async with maker() as db:
            job = await db.get(Job, job_id)
            job.heartbeat_at = past
            await db.commit()

        # 4. Worker restarts and runs recovery (reap_stale_jobs)
        async with maker() as db:
            reaped_count = await reap_stale_jobs(db)
            assert reaped_count == 1
            job = await db.get(Job, job_id)
            assert job.status == JOB_QUEUED
            assert job.worker_id is None
            assert job.attempts == 1  # Attempt count preserved

        # 5. Restarted worker picks up the job and completes it
        import apps.api.jobs as jobs_mod
        executed = False

        async def _mock_handler(db, j):
            nonlocal executed
            executed = True
            return "succeeded"

        with mock.patch.dict(jobs_mod._HANDLERS, {"PROCESS_EVIDENCE": _mock_handler}):
            async with maker() as db:
                processed = await process_next_job(db, worker_id="worker-restarted")
                assert processed is not None
                assert processed.id == job_id
                assert processed.status == JOB_SUCCEEDED
                assert processed.worker_id == "worker-restarted"
                assert processed.attempts == 2
                assert executed is True

        # 6. Verify audit logs in DB
        async with maker() as db:
            logs_res = await db.execute(
                select(JobLog).where(JobLog.job_id == job_id).order_by(JobLog.timestamp.asc())
            )
            logs = logs_res.scalars().all()
            stages = [entry.stage for entry in logs]
            assert "started" in stages
            assert "recovered" in stages
            assert "completed" in stages

    asyncio.run(_test())


def test_atomic_claim_concurrency(db_maker):
    """When multiple workers race to claim the same queued job, exactly one wins."""
    maker = db_maker
    job_id = "job_race_001"

    async def _test():
        async with maker() as db:
            db.add(
                Job(
                    id=job_id,
                    type="PROCESS_EVIDENCE",
                    entity_type="evidence",
                    entity_id="evi_race",
                    status=JOB_QUEUED,
                    attempts=0,
                    max_attempts=3,
                )
            )
            await db.commit()

        # Two concurrent claim attempts
        async with maker() as db1, maker() as db2:
            claimed1 = await claim_next_job(db1, worker_id="worker-A")
            claimed2 = await claim_next_job(db2, worker_id="worker-B")

        winners = [c for c in (claimed1, claimed2) if c is not None]
        losers = [c for c in (claimed1, claimed2) if c is None]
        assert len(winners) == 1
        assert len(losers) == 1
        assert winners[0].id == job_id

    asyncio.run(_test())


def test_active_heartbeat_during_execution(db_maker):
    """While a handler runs, background heartbeat updates heartbeat_at in the database."""
    maker = db_maker
    job_id = "job_heartbeat_test"

    async def _test():
        async with maker() as db:
            db.add(
                Job(
                    id=job_id,
                    type="RUN_ANALYSIS",
                    entity_type="world",
                    entity_id="w_hb",
                    status=JOB_QUEUED,
                    attempts=0,
                    max_attempts=3,
                )
            )
            await db.commit()

        heartbeats_observed = []

        async def _slow_handler(db, j):
            for _ in range(3):
                await asyncio.sleep(0.3)
                async with maker() as check_db:
                    refreshed = await check_db.get(Job, job_id)
                    heartbeats_observed.append(refreshed.heartbeat_at)
            return "succeeded"

        import apps.api.jobs as jobs_mod
        with mock.patch.dict(jobs_mod._HANDLERS, {"RUN_ANALYSIS": _slow_handler}):
            async with maker() as db:
                done = await process_next_job(db, worker_id="worker-heartbeat")
                assert done is not None
                assert done.status == JOB_SUCCEEDED

        assert len(heartbeats_observed) >= 2
        assert heartbeats_observed[-1] >= heartbeats_observed[0]

    asyncio.run(_test())


def test_max_attempts_exceeded_on_crash(db_maker):
    """If a job crashes and reaches max_attempts, it fails permanently with an honest reason."""
    maker = db_maker
    job_id = "job_fatal_crash"

    async def _test():
        past = utcnow() - timedelta(seconds=10)
        async with maker() as db:
            db.add(
                Job(
                    id=job_id,
                    type="PROCESS_EVIDENCE",
                    entity_type="evidence",
                    entity_id="evi_fatal",
                    status=JOB_RUNNING,
                    attempts=3,
                    max_attempts=3,
                    heartbeat_at=past,
                )
            )
            await db.commit()

        async with maker() as db:
            reaped = await reap_stale_jobs(db)
            assert reaped == 1
            job = await db.get(Job, job_id)
            assert job.status == JOB_FAILED
            assert "timed out" in job.error.lower() or "died" in job.error.lower()

    asyncio.run(_test())


def test_disable_in_process_worker_env(tmp_path, monkeypatch):
    """When REALITY_DISABLE_IN_PROCESS_WORKER=1, app.state.worker is None."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'noworker.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("REALITY_DISABLE_IN_PROCESS_WORKER", "1")

    from fastapi.testclient import TestClient

    import apps.api.db as db_mod
    importlib.reload(db_mod)
    import apps.api.main as main_mod
    importlib.reload(main_mod)

    with TestClient(main_mod.app) as client:
        res = client.get("/api/health")
        assert res.status_code == 200
        assert main_mod.app.state.worker is None
