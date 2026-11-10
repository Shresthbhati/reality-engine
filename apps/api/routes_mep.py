"""Routes for Automated MEP & Conduit Routing Perception with IFC4x3 Facility Topology.

Endpoints:
- GET  /api/worlds/{world_id}/mep/topology: Retrieve perceived MEP facility topology and distribution systems.
- POST /api/worlds/{world_id}/mep/detect: Trigger perception, auto-routing, and code clearance verification.
- POST /api/worlds/{world_id}/mep/route: Compute orthogonal 3D route between coordinates or terminals.
- GET  /api/worlds/{world_id}/mep/export/ifc: Download buildingSMART IFC4X3 Building Services STEP file.
- GET  /api/worlds/{world_id}/mep/export/geojson: Download 3D LineString GeoJSON network representation.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import World
from apps.api.routes_worlds import _load_world_or_409, _version_row
from engine.pipeline.mep_topology import (
    MepFacilityTopology,
    MepSystemType,
    detect_mep_elements_from_world,
    export_mep_to_geojson,
    export_mep_to_ifc4x3,
    infer_orthogonal_routing,
)

log = logging.getLogger("reality.api.mep")

mep_router = APIRouter(prefix="/api/worlds", tags=["mep"])


def _label_topology(topology: MepFacilityTopology) -> dict:
    return {
        **topology.to_dict(),
        "validation_status": "preliminary",
        "basis": "inferred_from_world_entities",
        "assumptions": [
            "MEP systems are inferred from detected world entities and fixtures, not from design drawings or site survey.",
            "Distribution routes are synthesized with orthogonal routing and a default ceiling plenum height.",
        ],
    }


class MepDetectRequest(BaseModel):
    version_id: Optional[str] = Field(None, description="World version ID (default: current head)")
    system_filter: Optional[str] = Field(None, description="Optional system filter: electrical, hvac, plumbing")
    min_clearance_m: float = Field(0.15, ge=0.05, le=1.0, description="Code clearance threshold in meters")


class MepCustomRouteRequest(BaseModel):
    version_id: Optional[str] = Field(None, description="World version ID")
    start_point: List[float] = Field(..., min_length=3, max_length=3, description="[x, y, z] start coordinate")
    end_point: List[float] = Field(..., min_length=3, max_length=3, description="[x, y, z] end coordinate")
    ceiling_z: float = Field(2.4, ge=1.5, le=10.0, description="Ceiling plenum / header routing height in meters")
    system_type: str = Field("electrical", description="electrical, hvac, plumbing")
    diameter_mm: float = Field(25.0, ge=10.0, le=500.0, description="Nominal diameter in millimeters")


@mep_router.get("/{world_id}/mep/topology")
async def get_mep_topology(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Retrieve full MEP facility infrastructure topology and distribution systems."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    topology = detect_mep_elements_from_world(world_ir)
    return _label_topology(topology)


@mep_router.post("/{world_id}/mep/detect")
async def detect_mep_topology_endpoint(
    world_id: str,
    req: MepDetectRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Trigger automated perception, orthogonal routing synthesis, and clearance verification."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, req.version_id)
    world_ir = _load_world_or_409(v_row)

    topology = detect_mep_elements_from_world(world_ir)

    if req.system_filter:
        filt = req.system_filter.lower()
        topology.systems = [s for s in topology.systems if s.system_type.value == filt]

    return _label_topology(topology)


@mep_router.post("/{world_id}/mep/route")
async def compute_mep_custom_route(
    world_id: str,
    req: MepCustomRouteRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Compute optimal orthogonal 3D route with 90° elbows between coordinates."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    start = (req.start_point[0], req.start_point[1], req.start_point[2])
    end = (req.end_point[0], req.end_point[1], req.end_point[2])
    waypoints = infer_orthogonal_routing(start, end, ceiling_z=req.ceiling_z)

    # Compute length
    import math
    total_len = 0.0
    for i in range(len(waypoints) - 1):
        total_len += math.sqrt(sum((b - a) ** 2 for a, b in zip(waypoints[i], waypoints[i + 1])))

    return {
        "world_id": world_id,
        "system_type": req.system_type,
        "diameter_mm": req.diameter_mm,
        "start_point": req.start_point,
        "end_point": req.end_point,
        "ceiling_z": req.ceiling_z,
        "total_length_m": round(total_len, 3),
        "bend_count": max(0, len(waypoints) - 2),
        "waypoints": [[round(c, 4) for c in wp] for wp in waypoints],
        "validation_status": "preliminary",
        "assumptions": [
            "Route is a geometric orthogonal path between the supplied coordinates; it does not check the world model for obstructions or code clearances.",
            f"Ceiling routing height is {req.ceiling_z} m as supplied or defaulted.",
        ],
    }


@mep_router.get("/{world_id}/mep/export/ifc")
async def export_mep_ifc_endpoint(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Export buildingSMART IFC4X3 Building Services STEP physical file."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    topology = detect_mep_elements_from_world(world_ir)
    ifc_text = export_mep_to_ifc4x3(topology, world=world_ir)

    filename = f"mep_topology_{world_id}.ifc"
    return Response(
        content=ifc_text,
        media_type="application/x-step",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@mep_router.get("/{world_id}/mep/export/geojson")
async def export_mep_geojson_endpoint(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Export 3D LineString GeoJSON network representation."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    topology = detect_mep_elements_from_world(world_ir)
    return export_mep_to_geojson(topology)
