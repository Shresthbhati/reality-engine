"""Routes: health/diagnostics and Sessions (incl. location, trajectory, reconstruct)."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import jobs as jobrunner
from apps.api.db import get_db
from apps.api.models import (
    ActivityEvent,
    Evidence,
    Job,
    Location,
    Session,
    new_id,
    utcnow,
)

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
    captured_at: datetime | None = None


class SessionIn(BaseModel):
    name: str
    device_metadata: dict | None = None
    coordinate_reference_system: str = "WGS84"
    captured_at: datetime | None = None
    location: LocationIn | None = None


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
    for name, model in (("sessions", Session), ("evidence", Evidence), ("jobs", Job)):
        res = await db.execute(select(func.count()).select_from(model))
        counts[name] = res.scalar_one()
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
        "backends": {"reconstruction": reconstruction},
    }


# --------------------------------------------------------------------------
# Sessions
# --------------------------------------------------------------------------

sessions = APIRouter(prefix="/api/sessions", tags=["sessions"])


async def _latest_location(db: AsyncSession, entity_type: str, entity_id: str) -> Location | None:
    res = await db.execute(
        select(Location)
        .where(Location.entity_type == entity_type, Location.entity_id == entity_id)
        .order_by(Location.captured_at.desc())
        .limit(1)
    )
    return res.scalar_one_or_none()


@sessions.post("", response_model=SessionOut, status_code=201)
async def create_session(body: SessionIn, db: AsyncSession = Depends(get_db)) -> SessionOut:
    from apps.api.validation import (
        require_finite,
        require_json_size,
        require_latitude,
        require_longitude,
        require_name,
    )

    name = require_name(body.name, "session name")
    require_json_size(body.device_metadata, "device_metadata")
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
    world_id: str | None = None, db: AsyncSession = Depends(get_db)
) -> dict:
    q = select(Session).order_by(Session.created_at.desc()).limit(200)
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


@sessions.get("/{session_id}/trajectory")
async def session_trajectory(session_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    s = await db.get(Session, session_id)
    if s is None:
        raise HTTPException(404, "Session not found")
    # Real trajectory artifacts only — honest empty state until they exist.
    return {"available": False, "reason": "No trajectory artifacts processed for this session", "points": []}


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

    job = jobrunner.enqueue_job(db, jobrunner.RECONSTRUCT_SESSION, "session", s.id)
    s.status = "processing"
    s.processing_started_at = utcnow()
    await db.commit()
    return {"job_id": job.id, "type": job.type, "status": job.status}