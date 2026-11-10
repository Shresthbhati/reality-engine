"""P104: database-backed ``CheckpointStore`` for the stage runner (``job_stage_runs`` rows, P103).

Sync/async bridge: ``CheckpointStore.lookup/record`` are synchronous (the runner is plain sync code) but the app
session factory is async. The runner executes inside ``asyncio.to_thread`` (the pattern ``apps/api/jobs.py`` uses for
all heavy work), so the store is called from a worker thread while the app's event loop is alive. We therefore run each
store call as a coroutine on that loop via ``asyncio.run_coroutine_threadsafe(...).result()`` when ``loop`` is given,
reusing the one async engine/session factory (works for aiosqlite and asyncpg). With no ``loop`` (scripts, tests, no
running loop in the calling thread) it falls back to ``asyncio.run``. No second sync engine or extra driver is needed.
Do NOT call the store directly from the event-loop thread itself: it would block that loop (and deadlock if ``loop``
is that same loop).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any

from sqlalchemy import func, select

from apps.api.models import JobStageRun, new_id
from apps.api.storage import resolve_artifact
from engine.pipeline.stages.base import StageResult


def _uri(sha: str) -> str:
    return sha if sha.startswith("sha256://") else f"sha256://{sha}"


class DbCheckpointStore:
    def __init__(self, session_factory: Callable[[], Any], team_id: str | None,
                 *, loop: asyncio.AbstractEventLoop | None = None, versions: dict[str, str] | None = None) -> None:
        self._sf = session_factory
        self._team_id = team_id
        self._loop = loop
        self._versions = versions or {}  # stage name -> spec.version; StageResult does not carry the version

    def _run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        if self._loop is not None:
            return asyncio.run_coroutine_threadsafe(coro, self._loop).result()
        return asyncio.run(coro)

    def lookup(self, job_id: str, name: str, inputs_hash: str) -> StageResult | None:
        return self._run(self._lookup(job_id, name, inputs_hash))

    def record(self, job_id: str, result: StageResult) -> None:
        self._run(self._record(job_id, result))

    async def _lookup(self, job_id: str, name: str, inputs_hash: str) -> StageResult | None:
        async with self._sf() as s:
            rows = (await s.execute(
                select(JobStageRun)
                .where(JobStageRun.job_id == job_id, JobStageRun.name == name,
                       JobStageRun.inputs_hash == inputs_hash, JobStageRun.status == "succeeded",
                       JobStageRun.team_id == self._team_id)
                .order_by(JobStageRun.attempt.desc())
            )).scalars().all()
        for r in rows:
            outputs = dict(r.outputs or {})
            if all(resolve_artifact(_uri(sha)) is not None for sha in outputs.values()):
                return StageResult(
                    name=r.name, status=r.status, inputs_hash=r.inputs_hash, outputs=outputs,
                    metrics=dict(r.metrics or {}), warnings=list(r.warnings or []), error=r.error,
                    started_at=r.started_at, ended_at=r.ended_at, peak_rss_mb=r.peak_rss_mb,
                )
        return None

    async def _record(self, job_id: str, result: StageResult) -> None:
        async with self._sf() as s:
            prev = (await s.execute(
                select(func.max(JobStageRun.attempt)).where(
                    JobStageRun.job_id == job_id, JobStageRun.name == result.name,
                    JobStageRun.inputs_hash == result.inputs_hash)
            )).scalar()
            s.add(JobStageRun(
                id=new_id("jsr"), job_id=job_id, team_id=self._team_id, name=result.name,
                version=self._versions.get(result.name, ""), inputs_hash=result.inputs_hash,
                status=result.status, attempt=(prev or 0) + 1, outputs=dict(result.outputs),
                metrics=dict(result.metrics), warnings=list(result.warnings), error=result.error,
                peak_rss_mb=result.peak_rss_mb, started_at=result.started_at, ended_at=result.ended_at,
            ))
            await s.commit()
