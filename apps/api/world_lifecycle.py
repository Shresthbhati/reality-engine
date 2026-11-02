"""World lifecycle (doc 13 / P12): archive, restore, soft delete, permanent purge, and the 30-day trash sweep.

Rules kept here, once, for the routes and the periodic sweep:
  * deleting never leaves work running against a world: queued jobs are cancelled outright, running jobs are
    flagged and their subprocesses terminated BEFORE the status changes;
  * a permanent purge refuses while a job is still winding down (it could adopt a version into a deleted world);
  * purge removes rows, WorldStore versions and only the blobs no other row/version references.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import worldstore_service
from apps.api.models import (
    ActivityEvent, CaptureRequest, CompletionProposal, Evidence, EvidenceDecision, Job, Location, Notification,
    Session, SiteComposite, SitePlacement, Upload, World, WorldIntent, WorldMeasurement, WorldVersion, new_id, utcnow,
)
from apps.api.routes_assistant import AssistantPlan
from apps.api.storage import blob_digest, gc_blobs

log = logging.getLogger("reality.api.lifecycle")

TRASH_RETENTION_DAYS = 30


class JobsStillRunning(RuntimeError):
    """A job for this world has been told to stop but has not yet; permanent delete must wait."""


def record_event(db: AsyncSession, kind: str, w: World, summary: str, meta: dict | None = None) -> None:
    """The one audit-trail row every lifecycle operation writes (added to the caller's transaction)."""
    db.add(ActivityEvent(id=new_id("act"), type=f"world.{kind}", entity_type="world", entity_id=w.id,
                         actor_id=w.owner_id, summary=summary[:500], metadata_json=meta))


def _world_job_filter(world_id: str):
    sess_ids = select(Session.id).where(Session.world_id == world_id)
    ev_ids = select(Evidence.id).where(Evidence.session_id.in_(sess_ids))
    return or_(
        (Job.entity_type == "world") & (Job.entity_id == world_id),
        (Job.entity_type == "session") & Job.entity_id.in_(sess_ids),
        (Job.entity_type == "evidence") & Job.entity_id.in_(ev_ids),
    )


async def cancel_world_jobs(db: AsyncSession, world_id: str) -> dict:
    """Cancel every queued/running job tied to the world (directly, via its sessions, or via their evidence).
    Same semantics as POST /api/jobs/{id}/cancel: queued -> cancelled now; running -> flagged + processes killed,
    final status written by the worker at its next stage boundary. Caller commits."""
    from .jobs import terminate_job_procs  # via jobs: apps.api must not import reconstruction directly

    rows = (await db.execute(select(Job).where(_world_job_filter(world_id),
                                               Job.status.in_(("queued", "running"))))).scalars().all()
    cancelled, running = [], []
    for j in rows:
        if j.status == "queued":
            j.status, j.completed_at = "cancelled", utcnow()
            j.error = "cancelled: world deleted"
            cancelled.append(j.id)
        else:
            j.cancel_requested = True
            terminate_job_procs(j.id)
            running.append(j.id)
    return {"cancelled": cancelled, "cancel_requested": running}


async def purge_world(db: AsyncSession, world: World) -> dict:
    """Permanently delete the world and everything only it owns. Commits. Raises JobsStillRunning (nothing deleted)."""
    wid = world.id
    jobs = await cancel_world_jobs(db, wid)
    await db.commit()                          # keep cancellations (and any cancel request) even if we refuse below
    if jobs["cancel_requested"]:
        raise JobsStillRunning(f"job(s) {jobs['cancel_requested']} are still stopping; retry shortly")
    name, owner = world.name, world.owner_id
    sess_ids = list((await db.execute(select(Session.id).where(Session.world_id == wid))).scalars())
    ev_rows = (await db.execute(select(Evidence).where(Evidence.session_id.in_(sess_ids)))).scalars().all() \
        if sess_ids else []
    up_rows = (await db.execute(select(Upload).where(Upload.session_id.in_(sess_ids)))).scalars().all() \
        if sess_ids else []
    ver_rows = (await db.execute(select(WorldVersion).where(WorldVersion.world_id == wid))).scalars().all()
    candidates = {d for d in (
        [blob_digest(r.artifact_uri) for r in (*ev_rows, *up_rows)]
        + [blob_digest(u) for v in ver_rows for u in (v.points_artifact_uri, v.cameras_artifact_uri)]
        + [blob_digest(world.thumbnail_uri)]) if d}
    ev_ids = [e.id for e in ev_rows]
    s_ids, e_ids = sess_ids or [""], ev_ids or [""]

    await db.execute(delete(Job).where(_world_job_filter(wid)))
    await db.execute(delete(EvidenceDecision).where(EvidenceDecision.evidence_id.in_(e_ids)))
    await db.execute(delete(Location).where(or_(
        (Location.entity_type == "world") & (Location.entity_id == wid),
        (Location.entity_type == "session") & Location.entity_id.in_(s_ids),
        (Location.entity_type == "evidence") & Location.entity_id.in_(e_ids))))
    await db.execute(delete(Notification).where(or_(
        (Notification.entity_type == "world") & (Notification.entity_id == wid),
        (Notification.entity_type == "session") & Notification.entity_id.in_(s_ids),
        (Notification.entity_type == "evidence") & Notification.entity_id.in_(e_ids))))
    await db.execute(delete(Evidence).where(Evidence.session_id.in_(s_ids)))
    await db.execute(delete(Upload).where(Upload.session_id.in_(s_ids)))
    await db.execute(delete(Session).where(Session.world_id == wid))
    for model in (WorldMeasurement, CompletionProposal, CaptureRequest, AssistantPlan, SitePlacement, SiteComposite):
        await db.execute(delete(model).where(model.world_id == wid))
    await db.execute(delete(WorldIntent).where(WorldIntent.world_id == wid))
    await db.execute(delete(WorldVersion).where(WorldVersion.world_id == wid))
    await db.delete(world)
    db.add(ActivityEvent(id=new_id("act"), type="world.purged", entity_type="world", entity_id=wid,
                         actor_id=owner, summary=f"World '{name}' permanently deleted"[:500],
                         metadata_json={"sessions": len(sess_ids), "evidence": len(ev_ids)}))
    await db.commit()
    # Rows are gone and committed; only now touch files, so a failed DB step never costs a blob.
    store = worldstore_service.purge_world_store(wid)
    blobs = await gc_blobs(db, candidates)
    return {"id": wid, "sessions_removed": len(sess_ids), "evidence_removed": len(ev_ids),
            "versions_removed": store["versions_removed"], "store_blobs_removed": store["blobs_removed"],
            "store_blobs_kept_shared": store["blobs_kept_shared"],
            "blobs_removed": blobs["removed"], "blobs_kept_shared": blobs["kept_shared"]}


async def purge_expired_trash(db: AsyncSession, *, older_than_days: int = TRASH_RETENTION_DAYS, now=None) -> list[dict]:
    """The daily sweep: permanently delete worlds trashed more than ``older_than_days`` ago. A world whose jobs are
    still stopping is skipped (retried next sweep); one failing world never blocks the others."""
    cutoff = (now or utcnow()) - timedelta(days=older_than_days)
    due = (await db.execute(select(World).where(World.status == "trashed", World.deleted_at.is_not(None),
                                                World.deleted_at < cutoff))).scalars().all()
    done = []
    for w in due:
        try:
            done.append(await purge_world(db, w))
        except JobsStillRunning:
            log.info("trash purge of %s deferred: jobs still stopping", w.id)
        except Exception:   # noqa: BLE001
            log.exception("trash purge of %s failed", w.id)
            await db.rollback()
    return done
