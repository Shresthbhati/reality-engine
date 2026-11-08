"""Routes for Multi-User Real-Time Collaborative Telepresence & Live Spatial Markup.

Endpoints:
- GET    /api/worlds/{world_id}/collab/presence: List currently active peers in the 3D model.
- POST   /api/worlds/{world_id}/collab/heartbeat: Keep alive presence and update 3D camera pose.
- GET    /api/worlds/{world_id}/collab/events: SSE real-time presence and markup update stream.
- GET    /api/worlds/{world_id}/collab/markups: List persistent 3D spatial markups and pins.
- POST   /api/worlds/{world_id}/collab/markups: Add or update 3D spatial markup.
- DELETE /api/worlds/{world_id}/collab/markups/{markup_id}: Delete a 3D spatial markup.
- GET    /api/worlds/{world_id}/collab/export/bcf: Export collaborative markups as buildingSMART BCF 2.1 zip.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncGenerator, List, Optional, Tuple
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import World, WorldVersion
from apps.api.routes_worlds import _load_world_or_409, _version_row
from apps.api.worldstore_service import get_store
from engine.pipeline.collab_presence import (
    MarkupStatus,
    MarkupTrade,
    MarkupType,
    PeerPresence,
    SpatialMarkup,
    delete_world_markup,
    export_markups_to_bcf,
    get_presence_registry,
    load_world_markups,
    save_world_markup,
)

log = logging.getLogger("reality.api.collab")

collab_router = APIRouter(prefix="/api/worlds", tags=["collab"])


class HeartbeatRequest(BaseModel):
    client_id: str = Field(..., description="Unique client session ID")
    user_name: str = Field(..., description="Display name of collaborator")
    role: str = Field("architect", description="Trade or discipline: architect, structural_engineer, mep_engineer, etc.")
    color: str = Field("#3b82f6", description="Avatar / cursor color hex token")
    camera_position: List[float] = Field(..., min_length=3, max_length=3, description="[x, y, z] camera coordinate")
    look_at: List[float] = Field(..., min_length=3, max_length=3, description="[x, y, z] look-at target")
    field_of_view: float = Field(60.0, ge=20.0, le=120.0, description="Camera FOV in degrees")


class CreateMarkupRequest(BaseModel):
    version_id: Optional[str] = Field(None, description="World version ID")
    markup_type: str = Field("pin", description="pin, stroke, measurement")
    trade: str = Field("general", description="architecture, structure, mep, safety, general")
    title: str = Field(..., min_length=1, max_length=120, description="Title of spatial note or issue")
    description: str = Field("", max_length=1000, description="Detailed description or notes")
    author_name: str = Field("Collaborator", description="Author display name")
    author_id: str = Field("user-1", description="Author user ID")
    position: List[float] = Field(..., min_length=3, max_length=3, description="[x, y, z] 3D anchor coordinate")
    normal: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0], min_length=3, max_length=3)
    stroke_points: List[List[float]] = Field(default_factory=list, description="Optional 3D stroke path coordinates")
    camera_viewpoint: dict = Field(default_factory=dict, description="Camera position, direction, fov")
    color: str = Field("#ef4444", description="Pin or stroke color")
    status: str = Field("open", description="open, in_review, resolved")


@collab_router.get("/{world_id}/collab/presence")
async def get_presence(
    world_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Retrieve list of currently active collaborators in this 3D model."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    registry = get_presence_registry()
    peers = registry.get_active_peers(world_id)
    return {
        "world_id": world_id,
        "peer_count": len(peers),
        "peers": [p.to_dict() for p in peers],
    }


@collab_router.post("/{world_id}/collab/heartbeat")
async def post_heartbeat(
    world_id: str,
    req: HeartbeatRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Register or refresh client 3D camera pose and presence keep-alive."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    registry = get_presence_registry()
    peer = registry.update_peer(
        world_id=world_id,
        client_id=req.client_id,
        user_name=req.user_name,
        role=req.role,
        color=req.color,
        camera_position=(req.camera_position[0], req.camera_position[1], req.camera_position[2]),
        look_at=(req.look_at[0], req.look_at[1], req.look_at[2]),
        field_of_view=req.field_of_view,
    )
    all_peers = registry.get_active_peers(world_id)
    return {
        "status": "ok",
        "peer": peer.to_dict(),
        "active_peers": [p.to_dict() for p in all_peers],
    }


@collab_router.get("/{world_id}/collab/events")
async def collab_events_stream(
    world_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
):
    """Server-Sent Events (SSE) stream for real-time presence and spatial markup synchronization."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    async def event_generator() -> AsyncGenerator[str, None]:
        registry = get_presence_registry()
        # Stream initial presence snapshot
        initial_peers = registry.get_active_peers(world_id)
        yield f"event: presence\ndata: {json.dumps({'peers': [p.to_dict() for p in initial_peers]})}\n\n"

        iteration = 0
        while True:
            if await request.is_disconnected():
                break

            await asyncio.sleep(1.0)
            iteration += 1

            # Broadcast current active peers
            active = registry.get_active_peers(world_id)
            yield f"event: presence\ndata: {json.dumps({'peers': [p.to_dict() for p in active]})}\n\n"

            # Ping keep-alive
            if iteration % 15 == 0:
                yield f"event: ping\ndata: {json.dumps({'timestamp': time.time()})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@collab_router.get("/{world_id}/collab/markups")
async def list_markups(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Retrieve all persistent 3D spatial markups and annotations."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    markups = load_world_markups(world_ir)
    return {
        "world_id": world_id,
        "markup_count": len(markups),
        "markups": [m.to_dict() for m in markups],
    }


@collab_router.post("/{world_id}/collab/markups")
async def create_markup(
    world_id: str,
    req: CreateMarkupRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Create or update a 3D spatial markup and save into the WorldStore."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, req.version_id)
    world_ir = _load_world_or_409(v_row)

    markup = SpatialMarkup(
        markup_id=f"markup-{uuid.uuid4().hex[:8]}",
        world_id=world_id,
        markup_type=MarkupType(req.markup_type),
        trade=MarkupTrade(req.trade),
        title=req.title,
        description=req.description,
        author_name=req.author_name,
        author_id=req.author_id,
        position=(req.position[0], req.position[1], req.position[2]),
        normal=(req.normal[0], req.normal[1], req.normal[2]),
        stroke_points=[(pt[0], pt[1], pt[2]) for pt in req.stroke_points if len(pt) >= 3],
        camera_viewpoint=req.camera_viewpoint,
        color=req.color,
        status=MarkupStatus(req.status),
    )

    all_markups = save_world_markup(world_ir, markup)

    # Persist updated WorldIR into storage
    store = get_store()
    new_version = store.save_version(world_ir, parent=v_row.id if v_row else None)

    # Insert DB mirror row so HEAD pointer and DB stay consistent
    db.add(
        WorldVersion(
            id=new_version.version_id,
            world_id=w.id,
            parent_version_id=new_version.parent,
            label="Collaborative markup added",
            artifact_uri=new_version.artifact_uri,
            artifact_hash=new_version.artifact_hash,
            report=dict(v_row.report) if v_row is not None and v_row.report else None,
            points_artifact_uri=v_row.points_artifact_uri if v_row is not None else None,
            cameras_artifact_uri=v_row.cameras_artifact_uri if v_row is not None else None,
        )
    )

    # Update world row pointer
    w.current_version_id = new_version.version_id
    await db.commit()

    return {
        "status": "created",
        "markup": markup.to_dict(),
        "total_markups": len(all_markups),
        "version_id": new_version.version_id,
    }


@collab_router.delete("/{world_id}/collab/markups/{markup_id}")
async def delete_markup(
    world_id: str,
    markup_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Delete a spatial markup from the world model."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    removed = delete_world_markup(world_ir, markup_id)
    if not removed:
        raise HTTPException(404, f"Markup '{markup_id}' not found in world '{world_id}'")

    # Persist updated WorldIR
    store = get_store()
    new_version = store.save_version(world_ir, parent=v_row.id if v_row else None)

    # Insert DB mirror row so HEAD pointer and DB stay consistent
    db.add(
        WorldVersion(
            id=new_version.version_id,
            world_id=w.id,
            parent_version_id=new_version.parent,
            label="Collaborative markup deleted",
            artifact_uri=new_version.artifact_uri,
            artifact_hash=new_version.artifact_hash,
            report=dict(v_row.report) if v_row is not None and v_row.report else None,
            points_artifact_uri=v_row.points_artifact_uri if v_row is not None else None,
            cameras_artifact_uri=v_row.cameras_artifact_uri if v_row is not None else None,
        )
    )

    w.current_version_id = new_version.version_id
    await db.commit()

    return {
        "status": "deleted",
        "markup_id": markup_id,
        "version_id": new_version.version_id,
    }


@collab_router.get("/{world_id}/collab/export/bcf")
async def export_collab_bcf(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Export all 3D spatial annotations as standard buildingSMART BCF 2.1 zip package."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    markups = load_world_markups(world_ir)
    bcf_zip_bytes = export_markups_to_bcf(markups, world_id=world_id)

    filename = f"spatial_markups_{world_id}.bcfzip"
    return Response(
        content=bcf_zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
