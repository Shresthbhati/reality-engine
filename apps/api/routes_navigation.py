"""Routes for Semantic Multi-Room Navigation & Walkthrough Tours (Track B)."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import World
from apps.api.worldstore_service import get_current_worldir, get_store
from engine.pipeline.walkthrough_tour import (
    ArchitecturalTour,
    NavigationData,
    RoomWaypoint,
    SpatialBookmark,
    build_world_navigation_data,
    detect_room_at_position,
    generate_architectural_tour,
    interpolate_tour_pose,
)

log = logging.getLogger("reality.api.navigation")

navigation = APIRouter(prefix="/api/worlds", tags=["navigation"])


class BookmarkCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Display name for bookmark")
    room_id: Optional[str] = Field(None, description="Associated room ID if known")
    room_name: Optional[str] = Field(None, description="Associated room display name")
    position: List[float] = Field(..., min_length=3, max_length=3, description="Camera 3D position [x, y, z]")
    yaw_rad: float = Field(0.0, description="Azimuth rotation in radians")
    pitch_rad: float = Field(0.0, description="Elevation angle in radians")
    fov_deg: float = Field(60.0, ge=20.0, le=120.0, description="Field of view in degrees")
    tags: List[str] = Field(default_factory=list, description="Optional search/category tags")


class PathRequest(BaseModel):
    start_pos: List[float] = Field(..., min_length=3, max_length=3)
    target_pos: Optional[List[float]] = Field(None, min_length=3, max_length=3)
    target_room_id: Optional[str] = Field(None)


class TourSampleRequest(BaseModel):
    elapsed_seconds: float = Field(..., ge=0.0)


async def _get_world_and_check(world_id: str, db: AsyncSession, auth: AuthContext) -> tuple[World, Any]:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(status_code=404, detail=f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    wir = await get_current_worldir(db, world_id)
    if wir is None:
        raise HTTPException(status_code=404, detail=f"WorldIR for world '{world_id}' not available")
    return w, wir


@navigation.get("/{world_id}/navigation")
async def get_world_navigation(
    world_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Dict[str, Any]:
    """Retrieve multi-room semantic navigation manifest, rooms, doors, and automated tour."""
    w, wir = await _get_world_and_check(world_id, db, auth)
    nav_data = build_world_navigation_data(wir, project_name=w.name or "Architectural Space")
    return nav_data.to_dict()


@navigation.get("/{world_id}/bookmarks")
async def list_world_bookmarks(
    world_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> List[Dict[str, Any]]:
    """List saved spatial bookmarks for a world."""
    _, wir = await _get_world_and_check(world_id, db, auth)
    bookmarks = []
    if "spatial_bookmarks" in wir.metadata:
        for b_dict in wir.metadata["spatial_bookmarks"]:
            try:
                bmk = SpatialBookmark.from_dict(b_dict)
                bookmarks.append(bmk.to_dict())
            except Exception:
                continue
    if not bookmarks:
        # Fallback to auto-generated room bookmarks
        _, generated = generate_architectural_tour(wir)
        bookmarks = [b.to_dict() for b in generated]
    return bookmarks


@navigation.post("/{world_id}/bookmarks")
async def create_world_bookmark(
    world_id: str,
    req: BookmarkCreateRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Dict[str, Any]:
    """Create and persist a new spatial viewpoint bookmark."""
    w, wir = await _get_world_and_check(world_id, db, auth)

    bmk = SpatialBookmark(
        bookmark_id=f"bmk_{req.name.lower().replace(' ', '_')}_{len(wir.metadata.get('spatial_bookmarks', [])) + 1}",
        name=req.name,
        room_id=req.room_id,
        room_name=req.room_name,
        position=req.position,
        yaw_rad=req.yaw_rad,
        pitch_rad=req.pitch_rad,
        fov_deg=req.fov_deg,
        tags=req.tags,
    )

    current_bmks = wir.metadata.get("spatial_bookmarks", [])
    current_bmks.append(bmk.to_dict())
    wir.metadata["spatial_bookmarks"] = current_bmks

    # Save version to WorldStore and update World row
    store = get_store()
    sv = store.save_version(wir, parent=w.current_version_id)
    w.current_version_id = sv.version_id
    await db.commit()

    return bmk.to_dict()


@navigation.post("/{world_id}/navigation/path")
async def calculate_navigation_path(
    world_id: str,
    req: PathRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Dict[str, Any]:
    """Calculate a collision-aware waypoint path from start position to target room/point."""
    _, wir = await _get_world_and_check(world_id, db, auth)
    nav_data = build_world_navigation_data(wir)

    start_pos = req.start_pos
    start_room = detect_room_at_position(start_pos, nav_data.rooms)

    # Determine target coordinates
    target_pos = req.target_pos
    target_room = None
    if req.target_room_id:
        target_room = next((r for r in nav_data.rooms if r.room_id == req.target_room_id), None)
        if target_room:
            target_pos = [target_room.center[0], target_room.floor_elevation_m + 1.65, target_room.center[2]]

    if not target_pos:
        raise HTTPException(status_code=400, detail="Must supply target_pos or a valid target_room_id")

    # Generate waypoints: start -> doorway(s) -> target
    waypoints: list[list[float]] = [start_pos]

    # If crossing between different rooms, route through connecting door
    if start_room and target_room and start_room.room_id != target_room.room_id:
        # Find opening associated with either room
        common_door = None
        for d in nav_data.doors:
            d_pos = d["position"]
            if start_room.contains_point(d_pos, margin=0.6) or target_room.contains_point(d_pos, margin=0.6):
                common_door = [d_pos[0], start_pos[1], d_pos[2]]
                break
        if common_door:
            waypoints.append(common_door)

    waypoints.append(target_pos)

    # Calculate total path distance
    total_dist_m = sum(
        math.hypot(waypoints[i + 1][0] - waypoints[i][0], waypoints[i + 1][2] - waypoints[i][2])
        for i in range(len(waypoints) - 1)
    )

    return {
        "world_id": world_id,
        "start_room": start_room.name if start_room else "Space",
        "target_room": target_room.name if target_room else "Space",
        "distance_m": round(total_dist_m, 2),
        "waypoints": waypoints,
    }
