"""Routes: health/diagnostics and Sessions (incl. location, trajectory, reconstruct)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import jobs as jobrunner
from apps.api.db import get_db
from apps.api.auth import AuthContext, get_auth_context
from apps.api.tenancy import session_scope, team_stamp, visible
from apps.api.models import (
    ActivityEvent,
    Evidence,
    Job,
    Location,
    Session,
    World,
    new_id,
    utcnow,
)
from apps.api.storage import measure_storage_usage

# --------------------------------------------------------------------------
# Schemas (API DTOs — distinct from domain types and UI state)
# --------------------------------------------------------------------------


class LocationIn(BaseModel):
    latitude: float
    longitude: float
    altitude: float | None = None
    accuracy: float | None = None
    heading: float | None = None
    speed: float | None = None
    source: str = "gnss"
    coordinate_reference_system: str = "WGS84"
    captured_at: datetime | None = None


class SessionIn(BaseModel):
    name: str
    device_metadata: dict | None = None
    coordinate_reference_system: str = "WGS84"
    captured_at: datetime | None = None
    location: LocationIn | None = None


class EvidenceItemIn(BaseModel):
    id: str
    captured_at: datetime | float | None = None
    latitude: float | None = None
    longitude: float | None = None
    metadata: dict | None = None


class GroupSessionsIn(BaseModel):
    items: list[EvidenceItemIn] | None = None
    evidence_ids: list[str] | None = None
    max_time_gap_s: float = 1800.0
    max_distance_m: float = 500.0
    group_prefix: str = "session"


class DeduplicateSessionIn(BaseModel):
    hash_type: str = "phash"
    threshold: int | None = None
    dry_run: bool = True


class SessionOut(BaseModel):
    id: str
    name: str
    status: str
    world_id: str | None
    coordinate_reference_system: str
    created_at: datetime | None = None
    captured_at: datetime | None = None
    uploaded_at: datetime | None = None
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None
    location: LocationIn | None = None
    evidence_count: int = 0


def _location_out(loc: Location | None) -> LocationIn | None:
    if loc is None:
        return None
    return LocationIn(
        latitude=loc.latitude,
        longitude=loc.longitude,
        altitude=loc.altitude,
        accuracy=loc.accuracy,
        heading=loc.heading,
        speed=loc.speed,
        source=loc.source,
        captured_at=loc.captured_at,
    )


def _session_out(s: Session, loc: Location | None = None) -> SessionOut:
    return SessionOut(
        id=s.id,
        name=s.name,
        status=s.status,
        world_id=s.world_id,
        coordinate_reference_system=s.coordinate_reference_system,
        created_at=s.created_at,
        captured_at=s.captured_at,
        uploaded_at=s.uploaded_at,
        processing_started_at=s.processing_started_at,
        processing_completed_at=s.processing_completed_at,
        location=_location_out(loc),
    )


# --------------------------------------------------------------------------
# Health / diagnostics — actual service status only
# --------------------------------------------------------------------------

health = APIRouter(prefix="/api", tags=["health"])


@health.get("/health")
async def health_check() -> dict:
    return {"status": "ok", "time": utcnow().isoformat()}


@health.get("/status")
async def status(db: AsyncSession = Depends(get_db)) -> dict:
    counts = {}
    for name, model in (("sessions", Session), ("evidence", Evidence), ("jobs", Job), ("worlds", World)):
        res = await db.execute(select(func.count()).select_from(model))
        counts[name] = res.scalar_one()

    running_res = await db.execute(select(func.count()).select_from(Job).where(Job.status == "running"))
    counts["jobs_running"] = running_res.scalar_one()

    failed_res = await db.execute(select(func.count()).select_from(Job).where(Job.status == "failed"))
    counts["jobs_failed"] = failed_res.scalar_one()

    storage_info = measure_storage_usage()

    reconstruction = False
    try:
        import reconstruction.orchestrator  # noqa: F401
        reconstruction = True
    except Exception:
        reconstruction = False
    return {
        "database": "ok",
        "worker": "running",
        "counts": counts,
        "storage": storage_info,
        "backends": {"reconstruction": reconstruction},
    }


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

sessions = APIRouter(prefix="/api/sessions", tags=["sessions"], dependencies=[Depends(session_scope)])


async def _latest_location(db: AsyncSession, entity_type: str, entity_id: str) -> Location | None:
    res = await db.execute(
        select(Location)
        .where(Location.entity_type == entity_type, Location.entity_id == entity_id)
        .order_by(Location.captured_at.desc())
        .limit(1)
    )
    return res.scalar_one_or_none()


@sessions.post("", response_model=SessionOut, status_code=201)
async def create_session(
    body: SessionIn, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> SessionOut:
    from apps.api.validation import (
        require_finite,
        require_json_size,
        require_latitude,
        require_longitude,
        require_name,
    )

    name = require_name(body.name, "session name")
    require_json_size(body.device_metadata, "device_metadata")
    if len(body.coordinate_reference_system) > 64:
        raise HTTPException(422, "coordinate_reference_system must be at most 64 chars")
    location = None
    if body.location is not None:
        loc = body.location
        location = {
            "latitude": require_latitude(loc.latitude),
            "longitude": require_longitude(loc.longitude),
            "altitude": require_finite(loc.altitude, "altitude") if loc.altitude is not None else None,
            "accuracy": require_finite(loc.accuracy, "accuracy") if loc.accuracy is not None else None,
            "heading": require_finite(loc.heading, "heading") if loc.heading is not None else None,
            "speed": require_finite(loc.speed, "speed") if loc.speed is not None else None,
            "source": loc.source,
        }
    s = Session(
        id=new_id("ses"),
        name=name,
        status="created",
        owner_id=auth.user_id,
        team_id=team_stamp(auth),
        device_metadata=body.device_metadata,
        coordinate_reference_system=body.coordinate_reference_system,
        captured_at=body.captured_at,
    )
    db.add(s)
    await db.flush()
    if location is not None:
        db.add(
            Location(
                id=new_id("loc"),
                entity_type="session",
                entity_id=s.id,
                **location,
                captured_at=body.location.captured_at or utcnow(),
            )
        )
    db.add(
        ActivityEvent(
            id=new_id("act"),
            type="session.created",
            entity_type="session",
            entity_id=s.id,
            summary=f"Session '{s.name}' created",
        )
    )
    await db.commit()
    await db.refresh(s)
    loc = await _latest_location(db, "session", s.id)
    return _session_out(s, loc)


@sessions.get("")
async def list_sessions(
    world_id: str | None = None, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    q = select(Session).order_by(Session.created_at.desc()).limit(200)
    if (scope := visible(Session, auth)) is not None:
        q = q.where(scope)
    if world_id:
        q = q.where(Session.world_id == world_id)
    res = await db.execute(q)
    items = res.scalars().all()
    ev_counts = dict(
        (await db.execute(
            select(Evidence.session_id, func.count()).group_by(Evidence.session_id)
        )).all()
    )
    out = []
    for s in items:
        loc = await _latest_location(db, "session", s.id)
        dto = _session_out(s, loc)
        dto.evidence_count = ev_counts.get(s.id, 0)
        out.append(dto.model_dump(mode="json"))
    return {"items": out}


@sessions.get("/{session_id}", response_model=SessionOut)
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)) -> SessionOut:
    s = await db.get(Session, session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    loc = await _latest_location(db, "session", session_id)
    return _session_out(s, loc)


@sessions.get("/{session_id}/location")
async def session_location(session_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    loc = await _latest_location(db, "session", session_id)
    if loc is None:
        raise HTTPException(404, "Location unavailable — no GPS data was captured")
    return _location_out(loc).model_dump(mode="json")


class LocationBatchIn(BaseModel):
    points: list[LocationIn]


@sessions.post("/{session_id}/locations")
async def add_session_locations(
    session_id: str,
    body: LocationBatchIn,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    from apps.api.validation import require_finite, require_latitude, require_longitude

    s = await db.get(Session, session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    created = []
    for pt in body.points:
        loc = Location(
            id=new_id("loc"),
            entity_type="session",
            entity_id=session_id,
            latitude=require_latitude(pt.latitude),
            longitude=require_longitude(pt.longitude),
            altitude=require_finite(pt.altitude, "altitude") if pt.altitude is not None else None,
            accuracy=require_finite(pt.accuracy, "accuracy") if pt.accuracy is not None else None,
            heading=require_finite(pt.heading, "heading") if pt.heading is not None else None,
            speed=require_finite(pt.speed, "speed") if pt.speed is not None else None,
            source=pt.source,
            coordinate_reference_system=pt.coordinate_reference_system,
            captured_at=pt.captured_at or utcnow(),
        )
        db.add(loc)
        created.append(loc.id)
    await db.commit()
    return {"added": len(created), "location_ids": created}


@sessions.get("/{session_id}/trajectory")
async def session_trajectory(session_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    s = await db.get(Session, session_id)
    if s is None:
        raise HTTPException(404, "Session not found")

    res = await db.execute(
        select(Location)
        .where(Location.entity_type == "session", Location.entity_id == session_id)
        .order_by(Location.captured_at.asc())
    )
    locs = res.scalars().all()

    if not locs:
        # Check evidence attached to this session with locations
        ev_res = await db.execute(
            select(Location)
            .join(Evidence, Evidence.id == Location.entity_id)
            .where(Location.entity_type == "evidence", Evidence.session_id == session_id)
            .order_by(Location.captured_at.asc())
        )
        locs = ev_res.scalars().all()

    if not locs:
        return {
            "available": False,
            "session_id": session_id,
            "reason": "No trajectory points captured for this session",
            "points": [],
        }

    points = [
        {
            "latitude": loc.latitude,
            "longitude": loc.longitude,
            "altitude": loc.altitude,
            "accuracy": loc.accuracy,
            "heading": loc.heading,
            "speed": loc.speed,
            "source": loc.source,
            "coordinate_reference_system": loc.coordinate_reference_system,
            "captured_at": loc.captured_at.isoformat() if loc.captured_at else None,
        }
        for loc in locs
    ]
    return {
        "available": True,
        "session_id": session_id,
        "count": len(points),
        "points": points,
    }


@sessions.post("/{session_id}/reconstruct")
async def reconstruct_session(session_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    from sqlalchemy import select as _select

    from apps.api.models import Job as _Job

    s = await db.get(Session, session_id)
    if s is None:
        raise HTTPException(404, "Session not found")

    # A reconstruction already queued/running for this session is a
    # duplicate request, not a new job: 409 instead of a second version
    # pipeline. A terminal (succeeded/partial/failed/cancelled) job does
    # not block a retry.
    existing = await db.execute(
        _select(_Job).where(
            _Job.entity_type == "session",
            _Job.entity_id == session_id,
            _Job.type == jobrunner.RECONSTRUCT_SESSION,
            _Job.status.in_(["queued", "running"]),
        )
    )
    if existing.scalars().first():
        raise HTTPException(409, "Reconstruction already in progress for this session")

    job = jobrunner.enqueue_job(db, jobrunner.RECONSTRUCT_SESSION, "session", s.id, team_id=s.team_id)
    s.status = "processing"
    s.processing_started_at = utcnow()
    await db.commit()
    return {"job_id": job.id, "type": job.type, "status": job.status}


@sessions.post("/group")
async def group_sessions_endpoint(
    body: GroupSessionsIn,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Group evidence items into capture sessions based on temporal and GPS spatial continuity.

    Splits capture batches when the time gap exceeds max_time_gap_s (default 30 min)
    or GPS distance exceeds max_distance_m (default 500m)."""
    from evidence.session_grouping import group_evidence_by_session

    items_to_group = []

    if body.evidence_ids:
        res = await db.execute(
            select(Evidence).where(Evidence.id.in_(body.evidence_ids))
        )
        db_evs = res.scalars().all()
        loc_res = await db.execute(
            select(Location).where(
                Location.entity_type == "evidence",
                Location.entity_id.in_(body.evidence_ids),
            )
        )
        locs = {loc.entity_id: loc for loc in loc_res.scalars().all()}
        for ev in db_evs:
            loc = locs.get(ev.id)
            gps_tuple = (loc.latitude, loc.longitude) if loc else None
            meta = dict(ev.metadata_json or {})
            if gps_tuple and "gps" not in meta:
                meta["gps"] = gps_tuple
            items_to_group.append({
                "id": ev.id,
                "captured_at": ev.processed_at.timestamp() if ev.processed_at else None,
                "metadata": meta,
            })

    if body.items:
        for it in body.items:
            t = it.captured_at.timestamp() if isinstance(it.captured_at, datetime) else it.captured_at
            gps = (it.latitude, it.longitude) if it.latitude is not None and it.longitude is not None else None
            meta = dict(it.metadata or {})
            if gps and "gps" not in meta:
                meta["gps"] = gps
            items_to_group.append({
                "id": it.id,
                "captured_at": t,
                "metadata": meta,
            })

    groups = group_evidence_by_session(
        items_to_group,
        max_time_gap_s=body.max_time_gap_s,
        max_distance_m=body.max_distance_m,
        group_prefix=body.group_prefix,
    )

    return {
        "total_items": len(items_to_group),
        "group_count": len(groups),
        "groups": [g.to_dict() for g in groups],
    }


@sessions.post("/{session_id}/deduplicate")
async def deduplicate_session(
    session_id: str,
    body: DeduplicateSessionIn | None = None,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Analyze and deduplicate photos in a session using perceptual hashing (pHash/dHash).

    Returns retained photos, duplicate pairs with distance and reason, and quality scores.
    If dry_run is False, marks near_duplicate metadata on duplicate Evidence records."""
    from apps.api.models import Evidence
    from apps.api.storage import resolve_artifact
    from evidence.dedup import deduplicate_images, DEFAULT_PHASH_THRESHOLD, DEFAULT_DHASH_THRESHOLD

    cfg = body or DeduplicateSessionIn()
    s = await db.get(Session, session_id)
    if s is None:
        raise HTTPException(404, "Session not found")

    res = await db.execute(
        select(Evidence).where(Evidence.session_id == session_id, Evidence.type == "photo")
    )
    ev_list = res.scalars().all()
    threshold = cfg.threshold or (DEFAULT_PHASH_THRESHOLD if cfg.hash_type == "phash" else DEFAULT_DHASH_THRESHOLD)
    if not ev_list:
        return {
            "session_id": session_id,
            "total_input": 0,
            "retained_count": 0,
            "duplicate_count": 0,
            "retained_ids": [],
            "duplicate_ids": [],
            "duplicates": [],
            "hash_type": cfg.hash_type,
            "threshold": threshold,
        }

    items = []
    qualities = {}
    ev_by_id = {}
    for ev in ev_list:
        ev_by_id[ev.id] = ev
        p = resolve_artifact(ev.artifact_uri)
        if p and p.exists():
            items.append((ev.id, p))
            sharpness = (ev.metadata_json or {}).get("image_facts", {}).get("sharpness", 0.0)
            qualities[ev.id] = float(sharpness or 0.0)

    dedup = deduplicate_images(items, max_distance=cfg.threshold, hash_type=cfg.hash_type, quality_scores=qualities)

    if not cfg.dry_run and dedup.duplicates:
        dup_map = {d.duplicate_id: d for d in dedup.duplicates}
        for ev_id, d in dup_map.items():
            ev = ev_by_id.get(ev_id)
            if ev:
                meta = dict(ev.metadata_json or {})
                meta["near_duplicate_of"] = d.canonical_id
                meta["phash_distance"] = d.distance
                meta["dedup_reason"] = d.reason
                ev.metadata_json = meta
        await db.commit()

    return {
        "session_id": session_id,
        "total_input": dedup.total_input,
        "retained_count": len(dedup.retained_ids),
        "duplicate_count": dedup.duplicate_count,
        "retained_ids": dedup.retained_ids,
        "duplicate_ids": dedup.duplicate_ids,
        "duplicates": [d.to_dict() for d in dedup.duplicates],
        "hash_type": dedup.hash_type,
        "threshold": dedup.threshold,
    }
