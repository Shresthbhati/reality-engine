"""P103: JobStageRun table round-trips and enforces its uniqueness constraint."""

from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timezone

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from apps.api.models import JobStageRun


@pytest.fixture()
def db_maker(tmp_path, monkeypatch):
    db_file = tmp_path / "stage_runs_test.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file.as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))

    import apps.api.db as db_mod
    importlib.reload(db_mod)
    asyncio.run(db_mod.init_db())
    yield db_mod.get_sessionmaker()
    asyncio.run(db_mod.dispose_db())


def _row(**over) -> JobStageRun:
    kw = dict(
        id="jsr_1", job_id="job_1", team_id="team_1", name="depth", version="1.2.0",
        inputs_hash="a" * 64, status="succeeded", attempt=1,
        outputs={"depth": {"sha": "b" * 64}}, metrics={"rmse": 0.5}, warnings=["w1"],
        error="boom", peak_rss_mb=123.5,
        started_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc),
    )
    kw.update(over)
    return JobStageRun(**kw)


def test_insert_and_read_back_every_field(db_maker):
    async def _t():
        async with db_maker() as s:
            s.add(_row())
            await s.commit()
        async with db_maker() as s:
            r = (await s.execute(select(JobStageRun))).scalar_one()
            assert (r.id, r.job_id, r.team_id, r.name, r.version) == ("jsr_1", "job_1", "team_1", "depth", "1.2.0")
            assert (r.inputs_hash, r.status, r.attempt) == ("a" * 64, "succeeded", 1)
            assert r.outputs == {"depth": {"sha": "b" * 64}}
            assert r.metrics == {"rmse": 0.5}
            assert r.warnings == ["w1"]
            assert r.error == "boom"
            assert r.peak_rss_mb == 123.5
            assert r.started_at.replace(tzinfo=timezone.utc) == datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
            assert r.ended_at.replace(tzinfo=timezone.utc) == datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc)

    asyncio.run(_t())


def test_attempt_defaults_to_one_and_nullables(db_maker):
    async def _t():
        async with db_maker() as s:
            s.add(JobStageRun(id="jsr_2", job_id="job_1", name="x", version="1",
                              inputs_hash="c" * 64, status="running"))
            await s.commit()
            r = (await s.execute(select(JobStageRun).where(JobStageRun.id == "jsr_2"))).scalar_one()
            assert r.attempt == 1
            assert r.team_id is None and r.outputs is None and r.error is None

    asyncio.run(_t())


def test_duplicate_job_name_hash_attempt_raises(db_maker):
    async def _t():
        async with db_maker() as s:
            s.add(_row())
            await s.commit()
        async with db_maker() as s:
            s.add(_row(id="jsr_dup"))
            with pytest.raises(IntegrityError):
                await s.commit()
        async with db_maker() as s:  # a different attempt is allowed
            s.add(_row(id="jsr_ok", attempt=2))
            await s.commit()

    asyncio.run(_t())
