"""Routes for Construction Site Progress & As-Built vs. Plan Verification."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import World
from apps.api.worldstore_service import get_current_worldir
from engine.pipeline.construction_progress import (
    ComplianceStatus,
    ConstructionTimelineReport,
    SiteComplianceReport,
    TimelineMilestone,
    Trade,
    compute_construction_timeline,
    export_construction_progress_pdf,
    verify_as_built_against_plan,
)
from world_ir.temporal import TemporalEpoch

log = logging.getLogger("reality.api.construction")

construction = APIRouter(prefix="/api/construction", tags=["construction"])


class VerifyPlanRequest(BaseModel):
    design_world_id: str = Field(..., description="ID of the planned design world (BIM / architectural baseline)")
    site_world_id: str = Field(..., description="ID of the as-built site world reconstructed from capture")
    tolerance_m: float = Field(0.05, ge=0.001, le=1.0, description="Positional tolerance in meters (default 5 cm)")
    auto_align: bool = Field(True, description="Whether to compute rigid Procrustes registration between design and site")


class TimelineRequest(BaseModel):
    design_world_id: str = Field(..., description="ID of the design world baseline")
    epoch_world_ids: List[str] = Field(..., min_length=1, description="Chronological sequence of site capture world IDs")
    tolerance_m: float = Field(0.05, ge=0.001, le=1.0, description="Positional tolerance in meters")


class ReportPdfRequest(BaseModel):
    design_world_id: str = Field(...)
    site_world_id: str = Field(...)
    project_name: str = Field("Construction Site As-Built Verification")
    tolerance_m: float = Field(0.05, ge=0.001, le=1.0)


async def _get_world_and_check(world_id: str, db: AsyncSession, auth: AuthContext) -> tuple[World, Any]:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(status_code=404, detail=f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    wir = await get_current_worldir(db, world_id)
    if wir is None:
        raise HTTPException(status_code=404, detail=f"WorldIR for world '{world_id}' not available")
    return w, wir


@construction.post("/verify")
async def verify_plan(
    req: VerifyPlanRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Dict[str, Any]:
    """Verify an as-built site capture against a planned design model.

    Evaluates element compliance, detects misalignments, missing elements, and unplanned site objects.
    """
    _, design_wir = await _get_world_and_check(req.design_world_id, db, auth)
    _, site_wir = await _get_world_and_check(req.site_world_id, db, auth)

    report = verify_as_built_against_plan(
        design_world=design_wir,
        site_world=site_wir,
        default_tolerance_m=req.tolerance_m,
        auto_align=req.auto_align,
    )
    return report.to_dict()


@construction.post("/timeline")
async def timeline_progression(
    req: TimelineRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Dict[str, Any]:
    """Track multi-epoch site construction progression over time."""
    _, design_wir = await _get_world_and_check(req.design_world_id, db, auth)

    epochs: List[TemporalEpoch] = []
    for wid in req.epoch_world_ids:
        w, wir = await _get_world_and_check(wid, db, auth)
        timestamp = w.created_at.isoformat() if w.created_at else "2026-01-01T00:00:00Z"
        epochs.append(TemporalEpoch(epoch_id=wid, timestamp_iso=timestamp, world=wir, description=w.name or wid))

    timeline_report = compute_construction_timeline(
        epochs=epochs,
        design_world=design_wir,
        tolerance_m=req.tolerance_m,
    )
    return timeline_report.to_dict()


@construction.post("/report/pdf")
async def generate_pdf_report(
    req: ReportPdfRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Generate and download a formal Construction Progress Verification PDF report."""
    _, design_wir = await _get_world_and_check(req.design_world_id, db, auth)
    _, site_wir = await _get_world_and_check(req.site_world_id, db, auth)

    report = verify_as_built_against_plan(
        design_world=design_wir,
        site_world=site_wir,
        default_tolerance_m=req.tolerance_m,
        auto_align=True,
    )

    pdf_bytes = export_construction_progress_pdf(report, project_name=req.project_name)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'attachment; filename="construction_progress_report.pdf"',
            "Content-Type": "application/pdf",
        },
    )
