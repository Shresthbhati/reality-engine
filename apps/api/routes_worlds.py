"""Routes: worlds (WorldStore remains authoritative for computational data)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import worldstore_service
from apps.api.db import get_db
from apps.api.models import (
    ActivityEvent,
    Location,
    Session,
    World,
    WorldVersion,
    new_id,
)
from apps.api.storage import resolve_artifact
from world_ir.diff import diff_worlds
from worldstore.store import WorldStoreError



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


def _version_dict(v: WorldVersion, current_version_id: str | None) -> dict:
    return {
        "id": v.id,
        "world_id": v.world_id,
        "parent_version_id": v.parent_version_id,
        "artifact_uri": v.artifact_uri,
        "artifact_hash": v.artifact_hash,
        "source_session_ids": v.source_session_ids or [],
        "changed_entity_ids": v.changed_entity_ids or [],
        "changed_geometry_ids": v.changed_geometry_ids or [],
        "created_at": v.created_at.isoformat() if v.created_at else None,
        "is_current": v.id == current_version_id,
    }


@worlds.get("/{world_id}/versions")
async def list_world_versions(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Application mirror of WorldStore lineage. Reconciled against the
    real WorldStore on every read (the CLI can write versions directly
    into the same store root, outside this process) -- never synthesized
    here."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    await worldstore_service.resync_versions(db, world_id)
    res = await db.execute(
        select(WorldVersion).where(WorldVersion.world_id == world_id).order_by(WorldVersion.created_at.desc())
    )
    return {"items": [_version_dict(v, w.current_version_id) for v in res.scalars().all()]}


@worlds.get("/{world_id}/versions/{version_id}")
async def get_world_version(
    world_id: str, version_id: str, db: AsyncSession = Depends(get_db)
) -> dict:
    """The real, versioned WorldIR for a specific version -- not the
    application World row."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    try:
        world = worldstore_service.get_store().load_version(version_id)
    except WorldStoreError as exc:
        raise HTTPException(404, str(exc)) from exc
    return world.to_dict()


@worlds.get("/{world_id}/worldir")
async def world_worldir(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """The real, current WorldIR for this world (from WorldStore) -- an
    honest empty state until a version has actually been committed, never
    the application World row reshaped to look like one."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    if not w.current_version_id:
        raise HTTPException(404, "No version compiled yet for this world")
    world = worldstore_service.get_store().load_version(w.current_version_id)
    return world.to_dict()


@worlds.get("/{world_id}/points")
async def world_points(world_id: str, db: AsyncSession = Depends(get_db)):
    version = await worldstore_service.get_current_version_row(db, world_id)
    if version is None or not version.points_artifact_uri:
        raise HTTPException(404, "No points artifact for this world's current version")
    path = resolve_artifact(version.points_artifact_uri)
    if path is None:
        raise HTTPException(410, "Points artifact missing from store")
    return FileResponse(path, media_type="application/octet-stream", filename="points.ply")


@worlds.get("/{world_id}/cameras")
async def world_cameras(world_id: str, db: AsyncSession = Depends(get_db)):
    version = await worldstore_service.get_current_version_row(db, world_id)
    if version is None or not version.cameras_artifact_uri:
        raise HTTPException(404, "No cameras artifact for this world's current version")
    path = resolve_artifact(version.cameras_artifact_uri)
    if path is None:
        raise HTTPException(410, "Cameras artifact missing from store")
    return FileResponse(path, media_type="application/json", filename="cameras.json")


@worlds.get("/{world_id}/report")
async def world_report(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    version = await worldstore_service.get_current_version_row(db, world_id)
    if version is None or version.report is None:
        raise HTTPException(404, "No pipeline report for this world's current version")
    return version.report


@worlds.get("/{world_id}/versions/{version_id}/diff")
async def world_version_diff(
    world_id: str, version_id: str, against: str, db: AsyncSession = Depends(get_db)
) -> dict:
    """Real structural diff between two committed versions (world_ir.diff),
    not a UI-computed approximation."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    store = worldstore_service.get_store()
    try:
        a = store.load_version(version_id)
        b = store.load_version(against)
    except WorldStoreError as exc:
        raise HTTPException(404, str(exc)) from exc
    return diff_worlds(a, b).to_dict()


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
