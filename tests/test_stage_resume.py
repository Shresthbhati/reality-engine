"""P109: a killed worker's rerun reuses finished stages via DbCheckpointStore and only reruns the interrupted one."""

from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timezone

import pytest

pytest.importorskip("sqlalchemy")
pytest.importorskip("aiosqlite")

from sqlalchemy import select

from apps.api.models import JobStageRun
from engine.pipeline.stages.base import StageContext, StageResult, StageSpec
from engine.pipeline.stages.runner import run_stages


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{(tmp_path / 'resume.db').as_posix()}")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "artifacts"))
    import apps.api.db as db_mod
    importlib.reload(db_mod)
    asyncio.run(db_mod.init_db())
    from apps.api.stage_checkpoints import DbCheckpointStore
    yield db_mod.get_sessionmaker(), DbCheckpointStore, tmp_path
    asyncio.run(db_mod.dispose_db())


class _FakeStage:
    def __init__(self, name: str, calls: list[int], crash_first: bool = False) -> None:
        self.spec = StageSpec(name=name, version="1", resource_class="cpu", required=True)
        self._calls, self._crash_first = calls, crash_first

    def run(self, ctx: StageContext) -> StageResult:
        from apps.api.storage import store_bytes
        self._calls[0] += 1
        if self._crash_first and self._calls[0] == 1:
            raise SystemExit("worker killed")  # not an Exception: run_stages must not catch it
        sha = store_bytes(f"{self.spec.name}-output".encode())[0]
        now = datetime.now(timezone.utc)
        return StageResult(
            name=self.spec.name, status="succeeded", inputs_hash="", outputs={f"{self.spec.name}_out": sha},
            metrics={}, warnings=[], error=None, started_at=now, ended_at=now, peak_rss_mb=None,
        )


def test_crashed_run_reuses_finished_stages(env):
    maker, Store, tmp_path = env
    counters = [[0], [0], [0]]
    stages = [_FakeStage("s1", counters[0]), _FakeStage("s2", counters[1]), _FakeStage("s3", counters[2], crash_first=True)]
    store = Store(maker, "team_1")

    def _ctx() -> StageContext:
        return StageContext(job_id="job_1", workdir=tmp_path, artifacts={}, params={})

    with pytest.raises(SystemExit):
        run_stages(stages, _ctx(), checkpoints=store)
    assert [c[0] for c in counters] == [1, 1, 1]

    manifest = run_stages(stages, _ctx(), checkpoints=store)
    assert [c[0] for c in counters] == [1, 1, 2]
    assert manifest.grade([s.spec for s in stages]) == "succeeded"

    async def _rows():
        async with maker() as s:
            rows = (await s.execute(select(JobStageRun).where(JobStageRun.job_id == "job_1"))).scalars().all()
            return sorted((r.name, r.status) for r in rows)
    assert asyncio.run(_rows()) == [("s1", "succeeded"), ("s2", "succeeded"), ("s3", "succeeded")]
