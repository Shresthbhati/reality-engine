"""P104: DbCheckpointStore reuses only succeeded stage rows whose output artifacts still exist."""

from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timezone

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from sqlalchemy import select

from apps.api.models import JobStageRun
from engine.pipeline.stages.base import StageResult

H1, H2 = "a" * 64, "c" * 64


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'ckpt.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    asyncio.run(db_mod.init_db())
    from apps.api.stage_checkpoints import DbCheckpointStore
    yield db_mod.get_sessionmaker(), DbCheckpointStore
    asyncio.run(db_mod.dispose_db())


def _result(inputs_hash: str, outputs: dict[str, str], status: str = "succeeded") -> StageResult:
    return StageResult(
        name="depth", status=status, inputs_hash=inputs_hash, outputs=outputs,
        metrics={"rmse": 0.5}, warnings=["w1"], error=None,
        started_at=datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 1, 0, 0, 5, tzinfo=timezone.utc), peak_rss_mb=12.5,
    )


def _artifact(data: bytes) -> str:
    from apps.api.storage import store_bytes
    return store_bytes(data)[0]


def test_hit_round_trips_every_field(env):
    maker, Store = env
    sha = _artifact(b"depth-map")
    store = Store(maker, "team_1")
    store.record("job_1", _result(H1, {"depth": sha}))
    got = store.lookup("job_1", "depth", H1)
    assert got is not None
    assert (got.name, got.status, got.inputs_hash, got.outputs) == ("depth", "succeeded", H1, {"depth": sha})
    assert got.metrics == {"rmse": 0.5} and got.warnings == ["w1"] and got.error is None and got.peak_rss_mb == 12.5
    assert got.started_at.replace(tzinfo=timezone.utc) == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert (got.ended_at - got.started_at).total_seconds() == 5


def test_miss_on_different_inputs_hash(env):
    maker, Store = env
    store = Store(maker, "team_1")
    store.record("job_1", _result(H1, {"depth": _artifact(b"x")}))
    assert store.lookup("job_1", "depth", H2) is None


def test_miss_when_artifact_file_deleted(env):
    from apps.api.storage import resolve_artifact
    maker, Store = env
    sha = _artifact(b"will-vanish")
    store = Store(maker, "team_1")
    store.record("job_1", _result(H1, {"depth": sha}))
    assert store.lookup("job_1", "depth", H1) is not None
    resolve_artifact(f"sha256://{sha}").unlink()
    assert store.lookup("job_1", "depth", H1) is None


def test_failed_row_is_never_reused(env):
    maker, Store = env
    store = Store(maker, "team_1")
    store.record("job_1", _result(H1, {}, status="failed"))
    assert store.lookup("job_1", "depth", H1) is None


def test_attempt_increments_and_latest_succeeded_wins(env):
    maker, Store = env
    store = Store(maker, "team_1")
    store.record("job_1", _result(H1, {}, status="failed"))
    store.record("job_1", _result(H1, {"depth": _artifact(b"ok")}))
    store.record("job_1", _result(H2, {}, status="failed"))  # other hash: its own attempt counter

    async def _rows():
        async with maker() as s:
            rows = (await s.execute(select(JobStageRun).order_by(JobStageRun.started_at, JobStageRun.id))).scalars().all()
            return sorted((r.inputs_hash, r.attempt, r.status, r.team_id) for r in rows)
    assert asyncio.run(_rows()) == [
        (H1, 1, "failed", "team_1"), (H1, 2, "succeeded", "team_1"), (H2, 1, "failed", "team_1"),
    ]
    assert store.lookup("job_1", "depth", H1).status == "succeeded"


def test_works_from_worker_thread_under_running_loop(env):
    """The real call path: the sync runner executes in asyncio.to_thread while the event loop is alive."""
    maker, Store = env
    sha = _artifact(b"threaded")

    async def _go():
        store = Store(maker, "team_1", loop=asyncio.get_running_loop())
        await asyncio.to_thread(store.record, "job_1", _result(H1, {"depth": sha}))
        return await asyncio.to_thread(store.lookup, "job_1", "depth", H1)
    assert asyncio.run(_go()).outputs == {"depth": sha}
