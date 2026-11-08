"""Routes for Automated Issue & Clash Detection and buildingSMART BCF Export/Import.

Endpoints:
- GET /api/worlds/{world_id}/clashes: Return detected clashes and coordination issues.
- POST /api/worlds/{world_id}/clashes/detect: Run clash detection with custom tolerances.
- GET /api/worlds/{world_id}/bcf/export: Download standard BCF 2.1 archive (.bcfzip).
- POST /api/worlds/{world_id}/bcf/import: Ingest external BCF package from Navisworks/Solibri.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import World
from apps.api.routes_worlds import _load_world_or_409, _version_row
from engine.pipeline.clash_detection import (
    ClashDetectionOptions,
    detect_world_clashes,
    generate_bcf_zip,
    parse_bcf_zip,
)

log = logging.getLogger("reality.api.bcf")

bcf_router = APIRouter(prefix="/api/worlds", tags=["clash_bcf"])


class ClashDetectRequest(BaseModel):
    version_id: Optional[str] = Field(None, description="Target world version (default: head)")
    hard_tolerance_m: float = Field(0.01, ge=0.001, le=0.5, description="Minimum overlap for hard clash")
    clearance_buffer_m: float = Field(0.05, ge=0.01, le=1.0, description="Minimum required clearance")
    check_duplicates: bool = Field(True, description="Check for duplicate entity volumes")
    check_clearance: bool = Field(True, description="Check for clearance buffer violations")


@bcf_router.get("/{world_id}/clashes")
async def get_world_clashes(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID (default: current head)"),
    hard_tolerance_m: float = Query(0.01, ge=0.001, le=0.5),
    clearance_buffer_m: float = Query(0.05, ge=0.01, le=1.0),
    check_duplicates: bool = Query(True),
    check_clearance: bool = Query(True),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """Retrieve automated clash detection and trade coordination report for a world."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    row = await _version_row(db, world_id, version_id, auth=auth)
    world_ir = _load_world_or_409(row)

    opts = ClashDetectionOptions(
        hard_clash_tolerance_m=hard_tolerance_m,
        clearance_buffer_m=clearance_buffer_m,
        check_duplicates=check_duplicates,
        check_clearance=check_clearance,
    )

    report = detect_world_clashes(world_ir, world_id=world_id, options=opts)
    return report.to_dict()


@bcf_router.post("/{world_id}/clashes/detect")
async def run_clash_detection(
    world_id: str,
    req: ClashDetectRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """Run parameterized multi-trade clash detection on a world."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    row = await _version_row(db, world_id, req.version_id, auth=auth)
    world_ir = _load_world_or_409(row)

    opts = ClashDetectionOptions(
        hard_clash_tolerance_m=req.hard_tolerance_m,
        clearance_buffer_m=req.clearance_buffer_m,
        check_duplicates=req.check_duplicates,
        check_clearance=req.check_clearance,
    )

    report = detect_world_clashes(world_ir, world_id=world_id, options=opts)
    return report.to_dict()


@bcf_router.get("/{world_id}/bcf/export")
async def export_bcf_package(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    hard_tolerance_m: float = Query(0.01, ge=0.001, le=0.5),
    clearance_buffer_m: float = Query(0.05, ge=0.01, le=1.0),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Generate and download an open buildingSMART BCF 2.1 zip package (.bcfzip)."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    row = await _version_row(db, world_id, version_id, auth=auth)
    world_ir = _load_world_or_409(row)

    opts = ClashDetectionOptions(
        hard_clash_tolerance_m=hard_tolerance_m,
        clearance_buffer_m=clearance_buffer_m,
    )
    report = detect_world_clashes(world_ir, world_id=world_id, options=opts)
    bcf_bytes = generate_bcf_zip(report, author="Reality Engine Clash Perception")

    filename = f"world_{world_id[:8]}_clashes.bcfzip"
    return Response(
        content=bcf_bytes,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Type": "application/zip",
        },
    )


@bcf_router.post("/{world_id}/bcf/import")
async def import_bcf_package(
    world_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict[str, Any]:
    """Import an external BCF 2.1/3.0 zip package and parse coordination topics."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    content = await file.read()
    if not content:
        raise HTTPException(400, "Empty BCF file uploaded")

    topics = parse_bcf_zip(content)
    return {
        "world_id": world_id,
        "filename": file.filename,
        "imported_topic_count": len(topics),
        "topics": topics,
    }
