"""FastAPI routes for Building Envelope Thermal Performance & ASHRAE 90.1 Compliance."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import World
from apps.api.routes_worlds import _load_world_or_409, _version_row
from engine.pipeline.thermal_envelope import (
    ClimateZone,
    ThermalEnvelopeReport,
    evaluate_thermal_envelope,
    export_thermal_envelope_certificate_pdf,
)

log = logging.getLogger("reality.api.thermal_envelope")

thermal_envelope_router = APIRouter(prefix="/api/worlds", tags=["thermal-envelope"])


class ThermalEvaluationRequest(BaseModel):
    version_id: Optional[str] = Field(None, description="World version ID")
    climate_zone: str = Field(default="5A_cold", description="ASHRAE Climate Zone")
    design_delta_t_k: float = Field(default=25.0, description="Heating design temperature delta in Kelvin")
    custom_u_overrides: Optional[dict[str, float]] = Field(default=None, description="Optional custom U-factors")
    project_name: Optional[str] = Field(default=None, description="Optional project title")


def _parse_climate_zone(zone_str: str) -> ClimateZone:
    for cz in ClimateZone:
        if cz.value == zone_str or cz.name.lower() == zone_str.lower():
            return cz
    return ClimateZone.ZONE_5


@thermal_envelope_router.get("/{world_id}/thermal-envelope")
async def get_thermal_envelope_report(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    climate_zone: str = Query(default="5A_cold", description="ASHRAE Climate Zone"),
    design_delta_t_k: float = Query(default=25.0, description="Design temperature difference (K)"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Retrieve Building Envelope Thermal Performance & ASHRAE 90.1 compliance audit."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    cz = _parse_climate_zone(climate_zone)
    report = evaluate_thermal_envelope(
        world=world_ir,
        climate_zone=cz,
        design_delta_t_k=design_delta_t_k,
        project_name=w.name or "Architectural Reality Model",
    )
    return report.to_dict()


@thermal_envelope_router.post("/{world_id}/thermal-envelope/evaluate")
async def evaluate_custom_thermal_scenario(
    world_id: str,
    payload: ThermalEvaluationRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Evaluate envelope with custom U-factors, climate zone, and design temperature difference."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, payload.version_id)
    world_ir = _load_world_or_409(v_row)

    cz = _parse_climate_zone(payload.climate_zone)
    project_title = payload.project_name or w.name or "Architectural Reality Model"
    report = evaluate_thermal_envelope(
        world=world_ir,
        climate_zone=cz,
        design_delta_t_k=payload.design_delta_t_k,
        custom_u_overrides=payload.custom_u_overrides,
        project_name=project_title,
    )
    return report.to_dict()


@thermal_envelope_router.get("/{world_id}/thermal-envelope/certificate")
async def download_thermal_certificate(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    climate_zone: str = Query(default="5A_cold"),
    auditor_name: str = Query(default="Reality Engine Certified Energy Auditor"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Download formal ASHRAE 90.1 / IECC Thermal Envelope Certificate PDF."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    cz = _parse_climate_zone(climate_zone)
    report = evaluate_thermal_envelope(
        world=world_ir,
        climate_zone=cz,
        project_name=w.name or "Architectural Reality Model",
    )
    pdf_bytes = export_thermal_envelope_certificate_pdf(report, auditor_name=auditor_name)
    clean_world_id = world_id.replace(" ", "_")[:12]
    filename = f"thermal_envelope_certificate_{clean_world_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
