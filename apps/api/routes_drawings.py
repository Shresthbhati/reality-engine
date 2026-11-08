"""Routes for Architectural Section & Elevation Drawings (DXF & Vector SVG).

Endpoints:
- GET  /api/worlds/{world_id}/drawings/presets: List preset architectural drawing cut planes.
- POST /api/worlds/{world_id}/drawings/generate: Generate 2D section/elevation drawing data.
- GET  /api/worlds/{world_id}/drawings/render/svg: Render high-DPI vector SVG drawing sheet.
- GET  /api/worlds/{world_id}/drawings/export/dxf: Export AutoCAD-compatible DXF file.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import World
from apps.api.routes_worlds import _load_world_or_409, _version_row
from engine.pipeline.section_elevation_drawings import (
    ArchitecturalDrawing,
    DrawingType,
    SheetSize,
    export_drawing_to_dxf,
    export_drawing_to_svg,
    generate_architectural_drawing,
)

log = logging.getLogger("reality.api.drawings")

drawings_router = APIRouter(prefix="/api/worlds", tags=["drawings"])


class GenerateDrawingRequest(BaseModel):
    version_id: Optional[str] = Field(None, description="World version ID (default: current head)")
    drawing_type: str = Field("section_longitudinal", description="section_longitudinal, section_transverse, elevation_north, elevation_south, etc.")
    cut_offset_m: Optional[float] = Field(None, description="Optional custom cutting plane offset in meters")
    projection_depth_m: float = Field(12.0, ge=1.0, le=100.0, description="Depth behind section plane for projection lines")
    scale_label: str = Field("1:100", description="Architectural scale: 1:20, 1:50, 1:100, 1:200")
    sheet_size: str = Field("A2", description="Drawing sheet size: A1, A2, A3, ARCH_D")
    drawing_title: Optional[str] = Field(None, description="Custom title override for title block")


@drawings_router.get("/{world_id}/drawings/presets")
async def list_drawing_presets(
    world_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """List standard architectural section and elevation drawing presets."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    presets = [
        {
            "id": "section_longitudinal",
            "title": "Building Section A-A (Longitudinal)",
            "type": "section_longitudinal",
            "axis": "X (looking North/South along Z)",
            "description": "Crosses main architectural spine cutting east-west across slabs and structural partitions.",
        },
        {
            "id": "section_transverse",
            "title": "Building Section B-B (Transverse)",
            "type": "section_transverse",
            "axis": "Z (looking East/West along X)",
            "description": "Perpendicular cross-section cutting north-south through rooms and service vertical cores.",
        },
        {
            "id": "elevation_north",
            "title": "North Exterior Elevation",
            "type": "elevation_north",
            "axis": "Exterior frontage (looking South)",
            "description": "Full architectural facade elevation showing fenestration, openings, and parapet line.",
        },
        {
            "id": "elevation_south",
            "title": "South Exterior Elevation",
            "type": "elevation_south",
            "axis": "Exterior frontage (looking North)",
            "description": "Rear/courtyard architectural elevation with storey datum markers and fenestration.",
        },
        {
            "id": "elevation_east",
            "title": "East Exterior Elevation",
            "type": "elevation_east",
            "axis": "Side elevation (looking West)",
            "description": "Side gable/elevation displaying structural height and roof profile.",
        },
        {
            "id": "elevation_west",
            "title": "West Exterior Elevation",
            "type": "elevation_west",
            "axis": "Side elevation (looking East)",
            "description": "Side elevation displaying building profile, foundation, and datum levels.",
        },
    ]

    return {
        "world_id": world_id,
        "available_scales": ["1:20", "1:50", "1:100", "1:200"],
        "available_sheets": ["A1", "A2", "A3", "ARCH_D"],
        "presets": presets,
    }


@drawings_router.post("/{world_id}/drawings/generate")
async def generate_drawing_endpoint(
    world_id: str,
    req: GenerateDrawingRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Generate 2D architectural building section or elevation drawing."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, req.version_id)
    if v_row is None:
        raise HTTPException(404, f"World '{world_id}' has no committed versions")

    world_ir = _load_world_or_409(v_row)

    try:
        dtype = DrawingType(req.drawing_type)
    except ValueError:
        dtype = DrawingType.SECTION_LONGITUDINAL

    try:
        ssize = SheetSize(req.sheet_size)
    except ValueError:
        ssize = SheetSize.A2

    drawing = generate_architectural_drawing(
        world=world_ir,
        drawing_type=dtype,
        cut_offset_m=req.cut_offset_m,
        projection_depth_m=req.projection_depth_m,
        scale_label=req.scale_label,
        sheet_size=ssize,
        drawing_title=req.drawing_title,
    )

    return drawing.to_dict()


@drawings_router.get("/{world_id}/drawings/render/svg")
async def render_drawing_svg(
    world_id: str,
    version_id: Optional[str] = Query(None),
    drawing_type: str = Query("section_longitudinal"),
    cut_offset_m: Optional[float] = Query(None),
    projection_depth_m: float = Query(12.0),
    scale_label: str = Query("1:100"),
    sheet_size: str = Query("A2"),
    theme: str = Query("dark", description="dark or light"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Render 2D architectural section or elevation drawing to SVG."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    if v_row is None:
        raise HTTPException(404, f"World '{world_id}' has no committed versions")

    world_ir = _load_world_or_409(v_row)

    try:
        dtype = DrawingType(drawing_type)
    except ValueError:
        dtype = DrawingType.SECTION_LONGITUDINAL

    try:
        ssize = SheetSize(sheet_size)
    except ValueError:
        ssize = SheetSize.A2

    drawing = generate_architectural_drawing(
        world=world_ir,
        drawing_type=dtype,
        cut_offset_m=cut_offset_m,
        projection_depth_m=projection_depth_m,
        scale_label=scale_label,
        sheet_size=ssize,
    )

    svg_str = export_drawing_to_svg(drawing, project_name=w.name, theme=theme)
    return Response(content=svg_str, media_type="image/svg+xml")


@drawings_router.get("/{world_id}/drawings/export/dxf")
async def export_drawing_dxf(
    world_id: str,
    version_id: Optional[str] = Query(None),
    drawing_type: str = Query("section_longitudinal"),
    cut_offset_m: Optional[float] = Query(None),
    projection_depth_m: float = Query(12.0),
    scale_label: str = Query("1:100"),
    sheet_size: str = Query("A2"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Export 2D architectural drawing to AutoCAD-compatible DXF (R12/R2000)."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    if v_row is None:
        raise HTTPException(404, f"World '{world_id}' has no committed versions")

    world_ir = _load_world_or_409(v_row)

    try:
        dtype = DrawingType(drawing_type)
    except ValueError:
        dtype = DrawingType.SECTION_LONGITUDINAL

    try:
        ssize = SheetSize(sheet_size)
    except ValueError:
        ssize = SheetSize.A2

    drawing = generate_architectural_drawing(
        world=world_ir,
        drawing_type=dtype,
        cut_offset_m=cut_offset_m,
        projection_depth_m=projection_depth_m,
        scale_label=scale_label,
        sheet_size=ssize,
    )

    dxf_str = export_drawing_to_dxf(drawing, project_name=w.name)
    fname = f"{drawing.drawing_id}_{drawing_type}.dxf"

    return Response(
        content=dxf_str,
        media_type="application/dxf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
