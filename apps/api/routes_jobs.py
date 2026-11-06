"""Routes: jobs, notifications, activity."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import get_db, get_sessionmaker
from apps.api.auth import AuthContext, get_auth_context
from apps.api.tenancy import entity_clause, job_scope
from apps.api.models import ActivityEvent, Evidence, Job, Notification
from apps.api.models import utcnow



jobs = APIRouter(prefix="/api/jobs", tags=["jobs"], dependencies=[Depends(job_scope)])

_TERMINAL = ("succeeded", "partial", "failed", "cancelled")


def _sse(event: str, data: dict) -> str:
    return "\n".join((f"event: {event}", f"data: {json.dumps(data)}", "", ""))


async def _photos_total(db: AsyncSession, j: Job) -> int | None:
    """How many photographs the job is working on (session jobs only); None when not applicable."""
    if j.entity_type != "session":
        return None
    n = (await db.execute(
        select(func.count()).select_from(Evidence).where(Evidence.session_id == j.entity_id, Evidence.type == "photo")
    )).scalar_one()
    return int(n)


async def _progress_view(db: AsyncSession, j: Job) -> dict:
    """The fields a client needs to draw progress. `progress` is stage-weighted (coarse, monotonic, 100 only on
    success/partial); `photos_placed` is known only after the run, so it is None while building -- never guessed."""
    placed = (j.payload or {}).get("cameras_registered") if j.status in _TERMINAL else None
    counts = (j.payload or {}).get("counts") or (j.payload or {}).get("compile") or {}
    failure_reason = None
    if j.status == "failed":
        failure_reason = j.error or (j.payload or {}).get("reason") or (j.payload or {}).get("error")
    return {
        "status": j.status,
        "stage": j.stage,
        "stage_counts": counts,
        "failure_reason": failure_reason,
        "progress": j.progress,
        "photos_total": await _photos_total(db, j),
        "photos_placed": placed,
        "terminal": j.status in _TERMINAL,
    }


@jobs.get("")
async def list_jobs(
    status: str | None = None, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    q = select(Job).order_by(Job.created_at.desc()).limit(100)
    if (scope := entity_clause(Job.entity_type, Job.entity_id, auth)) is not None:
        q = q.where(scope)
    if status:
        q = q.where(Job.status == status)
    res = await db.execute(q)
    return {
        "items": [
            {
                "id": j.id,
                "type": j.type,
                "entity_type": j.entity_type,
                "entity_id": j.entity_id,
                "status": j.status,
                "stage": j.stage,
                "stage_counts": (j.payload or {}).get("counts") or (j.payload or {}).get("compile") or {},
                "failure_reason": (j.error or (j.payload or {}).get("reason")) if j.status == "failed" else None,
                "attempts": j.attempts,
                "error": j.error,
                "worker_id": j.worker_id,
                "created_at": j.created_at.isoformat() if j.created_at else None,
                "started_at": j.started_at.isoformat() if j.started_at else None,
                "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            }
            for j in res.scalars().all()
        ]
    }


@jobs.get("/{job_id}")
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    j = await db.get(Job, job_id)
    if j is None:
        raise HTTPException(404, "Job not found")
    return {
        "id": j.id,
        "type": j.type,
        "status": j.status,
        "stage": j.stage,
        "attempts": j.attempts,
        "error": j.error,
        "worker_id": j.worker_id,
        "cancel_requested": j.cancel_requested,
        **{k: v for k, v in (await _progress_view(db, j)).items() if k not in ("status", "stage")},
        "entity_type": j.entity_type,
        "entity_id": j.entity_id,
        # Stage outputs (e.g. a reconstruction's world/version ids) so
        # terminal jobs carry their real result, not just a status.
        "payload": j.payload,
        # recorded timestamps (None until the job reaches that point) -- the pipeline view shows these, never invented durations
        "created_at": j.created_at.isoformat() if j.created_at else None,
        "started_at": j.started_at.isoformat() if j.started_at else None,
        "completed_at": j.completed_at.isoformat() if j.completed_at else None,
    }


@jobs.get("/{job_id}/events")
async def job_events(job_id: str, interval: float = 1.0) -> StreamingResponse:
    """Server-Sent Events: one `progress` event whenever status/stage/progress changes, then the stream closes after
    the terminal event. Clients that cannot use SSE poll GET /api/jobs/{id}, which carries the same fields."""
    interval = min(max(interval, 0.05), 10.0)
    maker = get_sessionmaker()
    async with maker() as db:
        if await db.get(Job, job_id) is None:
            raise HTTPException(404, "Job not found")

    async def stream():
        last = None
        while True:
            async with maker() as db:
                j = await db.get(Job, job_id)
                if j is None:
                    yield _sse("gone", {})
                    return
                view = await _progress_view(db, j)
            if view != last:
                last = view
                yield _sse("progress", view)
            if view["terminal"]:
                return
            await asyncio.sleep(interval)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@jobs.post("/{job_id}/cancel", status_code=200)
async def cancel_job(job_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Cancel a job. Queued jobs stop immediately (cancelled); running
    jobs are flagged, their owned subprocess trees are terminated at
    once, and they stop at the next stage boundary (202 while the worker
    winds down). Repeat cancels of an already-cancelled job return the
    same state (idempotent); other terminal jobs cannot be cancelled
    (409). Never claims success before ownership is established: the
    202 carries the live running state, not a completion."""
    from reconstruction.proc import terminate_job_procs

    j = await db.get(Job, job_id)
    if j is None:
        raise HTTPException(404, "Job not found")
    if j.status == "cancelled":
        return {"id": j.id, "status": j.status, "cancel_requested": True}
    if j.status == "queued":
        j.status = "cancelled"
        j.completed_at = utcnow()
        await db.commit()
        return {"id": j.id, "status": j.status}
    if j.status == "running":
        j.cancel_requested = True
        await db.commit()
        killed = terminate_job_procs(job_id)
        return JSONResponse(
            {
                "id": j.id,
                "status": j.status,
                "cancel_requested": True,
                "processes_terminated": killed,
            },
            status_code=202,
        )
    raise HTTPException(409, f"Job already {j.status}; only queued/running jobs can be cancelled")


notifications = APIRouter(prefix="/api/notifications", tags=["notifications"])


@notifications.get("")
async def list_notifications(
    db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    q = select(Notification).order_by(Notification.created_at.desc()).limit(50)
    if (scope := entity_clause(Notification.entity_type, Notification.entity_id, auth)) is not None:
        q = q.where(scope)
    res = await db.execute(q)
    return {
        "items": [
            {
                "id": n.id,
                "type": n.type,
                "title": n.title,
                "body": n.body,
                "entity_type": n.entity_type,
                "entity_id": n.entity_id,
                "read_at": n.read_at.isoformat() if n.read_at else None,
                "created_at": n.created_at.isoformat() if n.created_at else None,
            }
            for n in res.scalars().all()
        ]
    }


activity = APIRouter(prefix="/api/activity", tags=["activity"])


@activity.get("")
async def list_activity(
    db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    q = select(ActivityEvent).order_by(ActivityEvent.created_at.desc()).limit(100)
    if (scope := entity_clause(ActivityEvent.entity_type, ActivityEvent.entity_id, auth)) is not None:
        q = q.where(scope)
    res = await db.execute(q)
    return {
        "items": [
            {
                "id": a.id,
                "type": a.type,
                "entity_type": a.entity_type,
                "entity_id": a.entity_id,
                "summary": a.summary,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in res.scalars().all()
        ]
    }
