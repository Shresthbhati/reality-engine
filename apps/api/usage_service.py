"""Usage metering service for multi-tenant Reality Engine deployments (T-6.5).

Computes GPU-minutes, storage bytes, and export metrics per tenant.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import (
    Asset,
    Evidence,
    ExportRecord,
    Job,
    Session,
    Upload,
    World,
    WorldVersion,
    utcnow,
)


def _tenant_filter(col, tenant_id: str):
    """Filter condition for tenant ownership (unowned/None counts as 'default')."""
    if tenant_id == "default":
        return or_(col.is_(None), col == "default")
    return col == tenant_id


async def get_tenant_usage(
    db: AsyncSession,
    tenant_id: str,
    since: datetime | None = None,
    until: datetime | None = None,
) -> dict[str, Any]:
    """Calculate compute (GPU/CPU minutes), storage bytes, and export metrics for a tenant."""
    # 1. Compute & Job Metrics
    session_ids_q = select(Session.id).where(_tenant_filter(Session.team_id, tenant_id))
    world_ids_q = select(World.id).where(_tenant_filter(World.team_id, tenant_id))
    evidence_ids_q = select(Evidence.id).where(_tenant_filter(Evidence.team_id, tenant_id))

    job_cond = or_(
        _tenant_filter(Job.team_id, tenant_id),
        (Job.entity_type == "session") & Job.entity_id.in_(session_ids_q),
        (Job.entity_type == "world") & Job.entity_id.in_(world_ids_q),
        (Job.entity_type == "evidence") & Job.entity_id.in_(evidence_ids_q),
    )
    if since is not None:
        job_cond = job_cond & (Job.created_at >= since)
    if until is not None:
        job_cond = job_cond & (Job.created_at <= until)

    jobs_stmt = select(Job).where(job_cond)
    jobs_res = await db.execute(jobs_stmt)
    jobs = jobs_res.scalars().all()

    total_gpu_seconds = 0.0
    total_cpu_seconds = 0.0
    jobs_by_status: dict[str, int] = {}
    jobs_by_type: dict[str, int] = {}

    for job in jobs:
        jobs_by_status[job.status] = jobs_by_status.get(job.status, 0) + 1
        jobs_by_type[job.type] = jobs_by_type.get(job.type, 0) + 1

        if job.started_at and job.completed_at:
            duration = max(0.0, (job.completed_at - job.started_at).total_seconds())
        elif job.started_at and job.status == "running":
            duration = max(0.0, (utcnow() - job.started_at).total_seconds())
        elif job.completed_at and job.created_at:
            duration = max(0.0, (job.completed_at - job.created_at).total_seconds())
        else:
            duration = 0.0

        payload = job.payload or {}
        if "gpu_seconds" in payload:
            gpu_s = float(payload["gpu_seconds"])
            cpu_s = max(0.0, duration - gpu_s)
        elif payload.get("gpu") is True or payload.get("dense_gpu") is True:
            gpu_s = duration
            cpu_s = 0.0
        elif job.type in ("RECONSTRUCT_SESSION", "COMPILE_WORLD"):
            try:
                import torch
                cuda_avail = torch.cuda.is_available()
            except ImportError:
                cuda_avail = False
            if cuda_avail or os.environ.get("REALITY_FORCE_GPU_METERING") == "1":
                gpu_s = duration
                cpu_s = 0.0
            else:
                gpu_s = 0.0
                cpu_s = duration
        else:
            gpu_s = 0.0
            cpu_s = duration

        total_gpu_seconds += gpu_s
        total_cpu_seconds += cpu_s

    total_compute_seconds = total_gpu_seconds + total_cpu_seconds

    # 2. Storage Metrics
    up_q = select(
        func.coalesce(func.sum(Upload.size), 0),
        func.count(Upload.id),
    ).where(_tenant_filter(Upload.team_id, tenant_id))
    if since:
        up_q = up_q.where(Upload.created_at >= since)
    if until:
        up_q = up_q.where(Upload.created_at <= until)
    uploads_bytes, uploads_count = (await db.execute(up_q)).one()

    ev_q = select(
        func.coalesce(func.sum(Evidence.size), 0),
        func.count(Evidence.id),
    ).where(_tenant_filter(Evidence.team_id, tenant_id))
    if since:
        ev_q = ev_q.where(Evidence.created_at >= since)
    if until:
        ev_q = ev_q.where(Evidence.created_at <= until)
    evidence_bytes, evidence_count = (await db.execute(ev_q)).one()

    as_q = select(func.count(Asset.id)).where(_tenant_filter(Asset.team_id, tenant_id))
    if since:
        as_q = as_q.where(Asset.created_at >= since)
    if until:
        as_q = as_q.where(Asset.created_at <= until)
    assets_count = (await db.execute(as_q)).scalar() or 0

    w_q = select(func.count(World.id)).where(_tenant_filter(World.team_id, tenant_id))
    if since:
        w_q = w_q.where(World.created_at >= since)
    if until:
        w_q = w_q.where(World.created_at <= until)
    worlds_count = (await db.execute(w_q)).scalar() or 0

    wv_q = select(func.count(WorldVersion.id)).join(
        World, WorldVersion.world_id == World.id
    ).where(_tenant_filter(World.team_id, tenant_id))
    if since:
        wv_q = wv_q.where(WorldVersion.created_at >= since)
    if until:
        wv_q = wv_q.where(WorldVersion.created_at <= until)
    versions_count = (await db.execute(wv_q)).scalar() or 0

    # 3. Export Metrics
    exp_cond = _tenant_filter(ExportRecord.tenant_id, tenant_id)
    if since:
        exp_cond = exp_cond & (ExportRecord.created_at >= since)
    if until:
        exp_cond = exp_cond & (ExportRecord.created_at <= until)

    exp_tot_q = select(
        func.coalesce(func.sum(ExportRecord.size_bytes), 0),
        func.count(ExportRecord.id),
        func.max(ExportRecord.created_at),
    ).where(exp_cond)
    exports_bytes, exports_count, last_export_at = (await db.execute(exp_tot_q)).one()

    exp_fmt_q = select(
        ExportRecord.format,
        func.count(ExportRecord.id),
    ).where(exp_cond).group_by(ExportRecord.format)
    exports_by_format = dict((await db.execute(exp_fmt_q)).all())

    total_storage_bytes = int(uploads_bytes) + int(evidence_bytes) + int(exports_bytes)
    total_mb = round(total_storage_bytes / (1024 * 1024), 2)
    total_gb = round(total_storage_bytes / (1024 * 1024 * 1024), 4)

    return {
        "tenant_id": tenant_id,
        "as_of": utcnow().isoformat(),
        "period": {
            "since": since.isoformat() if since else None,
            "until": until.isoformat() if until else None,
        },
        "compute": {
            "gpu_seconds": round(total_gpu_seconds, 2),
            "gpu_minutes": round(total_gpu_seconds / 60.0, 2),
            "cpu_seconds": round(total_cpu_seconds, 2),
            "cpu_minutes": round(total_cpu_seconds / 60.0, 2),
            "total_compute_seconds": round(total_compute_seconds, 2),
            "total_compute_minutes": round(total_compute_seconds / 60.0, 2),
            "jobs_count": len(jobs),
            "jobs_by_status": jobs_by_status,
            "jobs_by_type": jobs_by_type,
        },
        "storage": {
            "total_bytes": total_storage_bytes,
            "total_mb": total_mb,
            "total_gb": total_gb,
            "breakdown_bytes": {
                "uploads": int(uploads_bytes),
                "evidence": int(evidence_bytes),
                "exports": int(exports_bytes),
            },
            "counts": {
                "uploads": uploads_count,
                "evidence": evidence_count,
                "assets": assets_count,
                "worlds": worlds_count,
                "versions": versions_count,
                "exports": exports_count,
            },
        },
        "exports": {
            "total_count": exports_count,
            "total_bytes": int(exports_bytes),
            "by_format": exports_by_format,
            "last_export_at": last_export_at.isoformat() if last_export_at else None,
        },
    }
