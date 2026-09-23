"""Routes: worlds (WorldStore remains authoritative for computational data)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import get_db
from apps.api.models import (
    ActivityEvent,
    Location,
    Session,
    World,
    WorldVersion,
    new_id,
)



worlds = APIRouter(prefix="/api/worlds", tags=["worlds"])


class WorldIn(BaseModel):
    name: str
    description: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    coverage: dict | None = None


@worlds.post("", status_code=201)
async def create_world(body: WorldIn, db: AsyncSession = Depends(get_db)) -> dict:
    w = World(
        id=new_id("wld"),
        name=body.name,
        description=body.description,
        latitude=body.latitude,
        longitude=body.longitude,
        coverage=body.coverage,
    )
    db.add(w)
    db.add(
        ActivityEvent(
            id=new_id("act"),
            type="world.created",
            entity_type="world",
            entity_id=w.id,
            summary=f"World '{w.name}' created",
        )
    )
    await db.commit()
    await db.refresh(w)
    return {"id": w.id, "name": w.name}


@worlds.get("")
async def list_worlds(db: AsyncSession = Depends(get_db)) -> dict:
    res = await db.execute(select(World).order_by(World.updated_at.desc()).limit(200))
    ses_counts = dict(
        (await db.execute(
            select(Session.world_id, func.count()).group_by(Session.world_id)
        )).all()
    )
    return {
        "items": [
            {
                "id": w.id,
                "name": w.name,
                "description": w.description,
                "status": w.status,
                "latitude": w.latitude,
                "longitude": w.longitude,
                "current_version_id": w.current_version_id,
                "session_count": ses_counts.get(w.id, 0),
                "created_at": w.created_at.isoformat() if w.created_at else None,
            }
            for w in res.scalars().all()
        ]
    }


@worlds.get("/{world_id}")
async def get_world(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    session_count = await db.execute(
        select(func.count()).select_from(Session).where(Session.world_id == world_id)
    )
    return {
        "id": w.id,
        "name": w.name,
        "description": w.description,
        "status": w.status,
        "latitude": w.latitude,
        "longitude": w.longitude,
        "current_version_id": w.current_version_id,
        "session_count": session_count.scalar_one(),
        "created_at": w.created_at.isoformat() if w.created_at else None,
    }


@worlds.get("/{world_id}/versions")
async def list_world_versions(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Application mirror of WorldStore lineage. Empty until computation
    actually creates versions — never synthesized here."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    res = await db.execute(
        select(WorldVersion).where(WorldVersion.world_id == world_id).order_by(WorldVersion.created_at.desc())
    )
    return {
        "items": [
            {
                "id": v.id,
                "world_id": v.world_id,
                "parent_version_id": v.parent_version_id,
                "artifact_uri": v.artifact_uri,
                "artifact_hash": v.artifact_hash,
                "source_session_ids": v.source_session_ids or [],
                "changed_entity_ids": v.changed_entity_ids or [],
                "created_at": v.created_at.isoformat() if v.created_at else None,
                "is_current": v.id == w.current_version_id,
            }
            for v in res.scalars().all()
        ]
    }


@worlds.post("/{world_id}/attach/{session_id}")
async def attach_session(
    world_id: str, session_id: str, db: AsyncSession = Depends(get_db)
) -> dict:
    w = await db.get(World, world_id)
    s = await db.get(Session, session_id)
    if w is None or s is None:
        raise HTTPException(404, "World or Session not found")
    s.world_id = w.id
    await db.commit()
    return {"session_id": s.id, "world_id": w.id}


@worlds.get("/{world_id}/coverage")
async def world_coverage(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    res = await db.execute(
        select(Location).where(
            Location.entity_type == "session",
            Location.entity_id.in_(
                select(Session.id).where(Session.world_id == world_id)
            ),
        )
    )
    points = [
        {"lat": l.latitude, "lon": l.longitude, "accuracy": l.accuracy}
        for l in res.scalars().all()
    ]
    if w.coverage:
        return {"available": True, "coverage": w.coverage, "session_points": points}
    if points:
        return {"available": True, "coverage": None, "session_points": points}
    return {"available": False, "reason": "No spatial data captured for this world"}
