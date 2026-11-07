"""Per-tenant usage metering (plan T-6.5): storage held, compute time used, exports made. Billing comes later; these are its inputs.

Everything is derived from rows the app already writes, scoped to the tenant by the same rule the rest of the API uses
(``tenancy.team_stamp``: the default tenant is the unowned rows, any other tenant its own team id).

* storage is a gauge (what the tenant holds now), so ``since`` does not apply to it;
* compute is the wall-clock time of finished jobs (started to completed). The GPU share is not measured separately;
* exports are the successful export calls (``world.exported`` activity events).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import ActivityEvent, Evidence, Job, World
from apps.api.tenancy import ENTITY_MODELS


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _team_clause(model, team: Optional[str]):
    return model.team_id == team if team else model.team_id.is_(None)


async def tenant_usage(db: AsyncSession, tenant_id: str, since: Optional[datetime] = None) -> Dict[str, object]:
    team = None if tenant_id == "default" else tenant_id
    since_utc = _utc(since) if since else None

    bytes_, items = (await db.execute(
        select(func.coalesce(func.sum(Evidence.size), 0), func.count(Evidence.id)).where(_team_clause(Evidence, team))
    )).one()

    seconds, by_status, jobs = 0.0, {}, 0
    for etype, model in ENTITY_MODELS.items():
        if not hasattr(model, "team_id"):
            continue
        rows = (await db.execute(
            select(Job.status, Job.started_at, Job.completed_at).where(
                Job.entity_type == etype, Job.entity_id.in_(select(model.id).where(_team_clause(model, team))),
                Job.started_at.is_not(None), Job.completed_at.is_not(None))
        )).all()
        for status, started, completed in rows:
            if since_utc and _utc(completed) < since_utc:
                continue
            jobs += 1
            by_status[status] = by_status.get(status, 0) + 1
            seconds += max(0.0, (_utc(completed) - _utc(started)).total_seconds())

    q = select(ActivityEvent.metadata_json, ActivityEvent.created_at).where(
        ActivityEvent.type == "world.exported", ActivityEvent.entity_id.in_(select(World.id).where(_team_clause(World, team))))
    by_format: Dict[str, int] = {}
    count = 0
    for meta, created in (await db.execute(q)).all():
        if since_utc and _utc(created) < since_utc:
            continue
        count += 1
        fmt = str((meta or {}).get("format", "unknown"))
        by_format[fmt] = by_format.get(fmt, 0) + 1

    return {
        "tenant": tenant_id,
        "since": since_utc.isoformat() if since_utc else None,
        "storage": {"bytes": int(bytes_), "evidence_items": int(items), "note": "what the tenant holds now; not windowed by `since`"},
        "compute": {"jobs": jobs, "seconds": round(seconds, 3), "minutes": round(seconds / 60.0, 3), "by_status": by_status,
                    "note": "wall-clock time of finished jobs; the GPU share is not measured separately"},
        "exports": {"count": count, "by_format": by_format},
    }
