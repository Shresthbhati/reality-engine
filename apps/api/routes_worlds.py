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
