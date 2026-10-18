"""GET /api/jobs/{id} reports the recorded timestamps (None when not yet reached)."""

from __future__ import annotations

import asyncio
import importlib
import unittest.mock as mock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'app.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    import apps.api.db as db_mod

    importlib.reload(db_mod)
    import apps.api.main as main_mod

    importlib.reload(main_mod)
    with TestClient(main_mod.app) as c:
        yield c


def test_job_detail_carries_timestamps_and_none_for_unreached(client):
    import apps.api.db as db_mod
    import apps.api.jobs as jobs_mod
    from apps.api.models import Job

    async def _noop(_db):
        return None

    async def _seed():
        async with db_mod.get_sessionmaker()() as db:
            db.add(Job(id="job_ts001", type="RECONSTRUCT_SESSION", entity_type="session",
                       entity_id="ses_x", status="queued"))
            await db.commit()

    asyncio.run(_seed())
    with mock.patch.object(jobs_mod, "process_next_job", _noop):
        body = client.get("/api/jobs/job_ts001").json()
    assert body["created_at"] is not None
    assert body["started_at"] is None and body["completed_at"] is None
