"""API routes for Automated Architectural Bill of Quantities (BoQ) & Material Takeoff.

Endpoints:
- GET  /api/worlds/{world_id}/takeoff: Retrieve standard CSI MasterFormat 2020 BoQ & MTO.
- POST /api/worlds/{world_id}/takeoff/estimate: Recalculate BoQ with custom rates & waste factors.
- GET  /api/worlds/{world_id}/takeoff/export/csv: Download tabular CSV spreadsheet.
- GET  /api/worlds/{world_id}/takeoff/export/pdf: Download formatted PDF-1.4 BoQ report.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import World
from apps.api.routes_worlds import _load_world_or_409, _version_row
from engine.pipeline.material_takeoff import (
    DEFAULT_UNIT_RATES,
    TakeoffReport,
    compute_world_material_takeoff,
    export_takeoff_to_csv,
    export_takeoff_to_pdf,
)

log = logging.getLogger("reality.api.takeoff")


def _takeoff_assumptions(rates_overridden: bool, waste_overridden: bool) -> list[str]:
    return [
        "Quantities are derived from the reconstructed world model, not from surveyed measurements.",
        "Unit rates are user-supplied overrides where given, otherwise generic DEFAULT_UNIT_RATES (not a regional price quote)."
        if rates_overridden
        else "Unit rates are the generic DEFAULT_UNIT_RATES (not a regional price quote).",
        "Waste allowances are user-supplied overrides where given, otherwise the per-item defaults."
        if waste_overridden
        else "Waste allowances are the per-item defaults.",
    ]


def _label_takeoff(report: TakeoffReport, rates_overridden: bool = False, waste_overridden: bool = False) -> dict:
    return {
        **report.to_dict(),
        "validation_status": "preliminary",
        "assumptions": _takeoff_assumptions(rates_overridden, waste_overridden),
    }

takeoff_router = APIRouter(prefix="/api/worlds", tags=["takeoff"])


class CustomEstimateRequest(BaseModel):
    version_id: Optional[str] = Field(None, description="World version ID")
    rate_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Dictionary mapping CSI code to custom rate details or total rate",
    )
    waste_overrides: Optional[Dict[str, float]] = Field(
        default=None,
        description="Dictionary mapping CSI code to custom waste percentage allowance",
    )
    project_name: Optional[str] = Field(None, description="Custom project title for the report")


@takeoff_router.get("/{world_id}/takeoff")
async def get_material_takeoff(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Compute and retrieve CSI MasterFormat Bill of Quantities & Material Takeoff."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    report = compute_world_material_takeoff(
        world=world_ir,
        project_name=w.name or "Architectural Reality Model",
    )

    return _label_takeoff(report)


@takeoff_router.post("/{world_id}/takeoff/estimate")
async def calculate_custom_estimate(
    world_id: str,
    req: CustomEstimateRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Calculate customized Material Takeoff & Cost Estimate applying user-defined rates."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, req.version_id)
    world_ir = _load_world_or_409(v_row)

    report = compute_world_material_takeoff(
        world=world_ir,
        rate_overrides=req.rate_overrides,
        waste_overrides=req.waste_overrides,
        project_name=req.project_name or w.name or "Architectural Reality Model",
    )

    return _label_takeoff(report, bool(req.rate_overrides), bool(req.waste_overrides))


@takeoff_router.get("/{world_id}/takeoff/export/csv")
async def export_takeoff_csv(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Export standard tabular Bill of Quantities CSV spreadsheet."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    report = compute_world_material_takeoff(
        world=world_ir,
        project_name=w.name or "Architectural Reality Model",
    )

    csv_text = export_takeoff_to_csv(report)
    filename = f"takeoff_boq_{world_id}.csv"

    return Response(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@takeoff_router.get("/{world_id}/takeoff/export/pdf")
async def export_takeoff_pdf(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Export formatted PDF-1.4 Bill of Quantities and Material Takeoff report."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    report = compute_world_material_takeoff(
        world=world_ir,
        project_name=w.name or "Architectural Reality Model",
    )

    pdf_bytes = export_takeoff_to_pdf(
        report=report,
        project_name=w.name or "Architectural Reality Model",
    )
    filename = f"takeoff_boq_{world_id}.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
