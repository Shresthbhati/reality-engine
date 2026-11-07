"""T-6.2: the job queue survives worker restarts and several workers; a separate GPU worker can consume it.

Facts under test: a claim is atomic (two workers never both run a job), a job stranded by a dead worker is requeued and finished
by the next one, a job that keeps killing its worker is failed instead of looping, and a long-running job keeps heartbeating so
another worker's reaper cannot take it.
"""
from __future__ import annotations

import asyncio
import importlib
from datetime import timedelta

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def queue(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("REALITY_EMBEDDED_WORKER", "0")          # the API only enqueues; this test plays the worker
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    import apps.api.jobs as jobs_mod

    with TestClient(main_mod.app) as c:
        yield c, db_mod, jobs_mod


def _run(db_mod, fn):
    async def go():
        async with db_mod.get_sessionmaker()() as db:
            return await fn(db)

    return asyncio.run(go())


def _job(job_id, **kw):
    from apps.api.models import Job, utcnow

    base = dict(id=job_id, type="QTEST", entity_type="evidence", entity_id="ev_x", status="queued", attempts=0, max_attempts=3)
    base.update(kw)
    if base["status"] == "running":
        base.setdefault("heartbeat_at", utcnow())
    return Job(**base)


def test_the_embedded_worker_can_be_turned_off_for_a_separate_worker_process(queue):
    client, _, _ = queue
    assert getattr(client.app.state, "worker", None) is None


def test_two_workers_never_both_run_the_same_job(queue, monkeypatch):
    _, db_mod, jobs_mod = queue
    calls = []

    async def handler(db, job):
        calls.append(job.id)
        await asyncio.sleep(0.05)
        return "succeeded"

    monkeypatch.setitem(jobs_mod._HANDLERS, "QTEST", handler)

    async def scenario():
        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            db.add(_job("job_once"))
            await db.commit()
        async with maker() as a, maker() as b:
            return await asyncio.gather(jobs_mod.process_next_job(a), jobs_mod.process_next_job(b))

    results = asyncio.run(scenario())
    assert len([r for r in results if r is not None]) == 1, "exactly one worker may claim the job"
    assert calls == ["job_once"]

    async def check(db):
        from apps.api.models import Job

        j = await db.get(Job, "job_once")
        return j.status, j.attempts

    assert _run(db_mod, check) == ("succeeded", 1)


def test_a_job_stranded_by_a_dead_worker_is_finished_by_the_next_one(queue, monkeypatch):
    _, db_mod, jobs_mod = queue
    from apps.api.models import Job, utcnow

    async def handler(db, job):
        return "succeeded"

    monkeypatch.setitem(jobs_mod._HANDLERS, "QTEST", handler)

    async def scenario(db):
        db.add(_job("job_orphan", status="running", worker_id="worker-dead", attempts=1,
                    heartbeat_at=utcnow() - timedelta(hours=1)))
        await db.commit()
        reaped = await jobs_mod.reap_stale_jobs(db)           # what a restarted worker does first
        done = await jobs_mod.process_next_job(db)
        j = await db.get(Job, "job_orphan")
        return reaped, done is not None, j.status, j.attempts, j.worker_id

    assert _run(db_mod, scenario) == (1, True, "succeeded", 2, jobs_mod.WORKER_ID)


def test_a_job_that_keeps_killing_its_worker_is_failed_not_looped(queue):
    _, db_mod, jobs_mod = queue
    from apps.api.models import Job, utcnow

    async def scenario(db):
        db.add(_job("job_poison", status="running", worker_id="worker-dead", attempts=3, max_attempts=3,
                    heartbeat_at=utcnow() - timedelta(hours=1)))
        await db.commit()
        await jobs_mod.reap_stale_jobs(db)
        j = await db.get(Job, "job_poison")
        return j.status, j.error, j.completed_at is not None

    status, error, completed = _run(db_mod, scenario)
    assert status == "failed" and "worker lost" in error and completed


def test_a_running_job_keeps_heartbeating_so_another_reaper_leaves_it_alone(queue, monkeypatch):
    _, db_mod, jobs_mod = queue
    from apps.api.models import Job, utcnow

    monkeypatch.setattr(jobs_mod, "_stale_after_seconds", lambda: 0.06)      # heartbeat every 20 ms
    old = utcnow() - timedelta(hours=1)

    async def scenario():
        maker = db_mod.get_sessionmaker()
        async with maker() as db:
            db.add(_job("job_long", status="running", worker_id=jobs_mod.WORKER_ID, attempts=1, heartbeat_at=old))
            await db.commit()
        task = asyncio.create_task(jobs_mod._keep_heartbeat("job_long"))
        await asyncio.sleep(0.4)
        task.cancel()
        async with maker() as db:
            return (await db.get(Job, "job_long")).heartbeat_at

    beat = asyncio.run(scenario())
    assert beat.replace(tzinfo=None) > old.replace(tzinfo=None) + timedelta(minutes=30)
