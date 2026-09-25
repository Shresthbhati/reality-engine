"""Routes: jobs, notifications, activity."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import get_db
from apps.api.models import ActivityEvent, Job, Notification
from apps.api.models import utcnow



jobs = APIRouter(prefix="/api/jobs", tags=["jobs"])


@jobs.get("")
async def list_jobs(status: str | None = None, db: AsyncSession = Depends(get_db)) -> dict:
    q = select(Job).order_by(Job.created_at.desc()).limit(100)
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
        "entity_type": j.entity_type,
        "entity_id": j.entity_id,
        # Stage outputs (e.g. a reconstruction's world/version ids) so
        # terminal jobs carry their real result, not just a status.
        "payload": j.payload,
    }


@jobs.post("/{job_id}/cancel", status_code=200)
async def cancel_job(job_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Cancel a job. Queued jobs stop immediately (cancelled); running
    jobs are flagged and stop at their next stage boundary, reported
    with 202 while the worker winds down. Terminal jobs cannot be
    cancelled (409)."""
    j = await db.get(Job, job_id)
    if j is None:
        raise HTTPException(404, "Job not found")
    if j.status == "queued":
        j.status = "cancelled"
        j.completed_at = utcnow()
        await db.commit()
        return {"id": j.id, "status": j.status}
    if j.status == "running":
        j.cancel_requested = True
        await db.commit()
        return JSONResponse(
            {"id": j.id, "status": j.status, "cancel_requested": True},
            status_code=202,
        )
    raise HTTPException(409, f"Job already {j.status}; only queued/running jobs can be cancelled")


notifications = APIRouter(prefix="/api/notifications", tags=["notifications"])


@notifications.get("")
async def list_notifications(db: AsyncSession = Depends(get_db)) -> dict:
    res = await db.execute(
        select(Notification).order_by(Notification.created_at.desc()).limit(50)
    )
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
async def list_activity(db: AsyncSession = Depends(get_db)) -> dict:
    res = await db.execute(
        select(ActivityEvent).order_by(ActivityEvent.created_at.desc()).limit(100)
    )
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
