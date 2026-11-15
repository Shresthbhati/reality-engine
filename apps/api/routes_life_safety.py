"""FastAPI routes for Life Safety & ADA Accessibility Compliance."""

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
from engine.pipeline.life_safety_compliance import (
    LifeSafetyReport,
    audit_life_safety_compliance,
    export_life_safety_certificate_pdf,
)

log = logging.getLogger("reality.api.life_safety")

life_safety_router = APIRouter(prefix="/api/worlds", tags=["life-safety"])


def _label_life_safety(report: LifeSafetyReport, occupancy_group: str, is_sprinklered: bool) -> dict:
    return {
        **report.to_dict(),
        "validation_status": "preliminary",
        "assumptions": [
            f"Occupancy group is {occupancy_group}.",
            "Building is sprinklered." if is_sprinklered else "Building is not sprinklered.",
            "Egress travel distances are computed from the reconstructed world model, not a surveyed floor plan.",
        ],
    }


class LifeSafetyAuditRequest(BaseModel):
    version_id: Optional[str] = Field(None, description="World version ID")
    occupancy_group: str = Field(default="Business (Group B)", description="IBC Occupancy classification group")
    is_sprinklered: bool = Field(default=True, description="Whether the building has automatic sprinkler protection")
    project_name: Optional[str] = Field(default=None, description="Optional custom project title")


@life_safety_router.get("/{world_id}/life-safety")
async def get_life_safety_report(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    occupancy_group: str = Query(default="Business (Group B)", description="IBC Occupancy Group"),
    is_sprinklered: bool = Query(default=True, description="Automatic fire suppression"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Retrieve Life Safety & ADA accessibility compliance audit for the world."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    report = audit_life_safety_compliance(
        world=world_ir,
        occupancy_group=occupancy_group,
        is_sprinklered=is_sprinklered,
        project_name=w.name or "Architectural Reality Model",
    )
    return _label_life_safety(report, occupancy_group, is_sprinklered)


@life_safety_router.post("/{world_id}/life-safety/audit")
async def run_custom_life_safety_audit(
    world_id: str,
    payload: LifeSafetyAuditRequest,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Run audit with customized occupancy group and fire suppression parameters."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, payload.version_id)
    world_ir = _load_world_or_409(v_row)

    project_title = payload.project_name or w.name or "Architectural Reality Model"
    report = audit_life_safety_compliance(
        world=world_ir,
        occupancy_group=payload.occupancy_group,
        is_sprinklered=payload.is_sprinklered,
        project_name=project_title,
    )
    return _label_life_safety(report, payload.occupancy_group, payload.is_sprinklered)


@life_safety_router.get("/{world_id}/life-safety/egress-path")
async def get_egress_routes(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    is_sprinklered: bool = Query(default=True),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> list[dict[str, Any]]:
    """Retrieve egress routes and waypoints from remote room points to exit doors."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    report = audit_life_safety_compliance(
        world=world_ir,
        is_sprinklered=is_sprinklered,
        project_name=w.name or "Architectural Reality Model",
    )
    return [r.to_dict() for r in report.egress_routes]


@life_safety_router.get("/{world_id}/life-safety/certificate")
async def download_life_safety_certificate(
    world_id: str,
    version_id: Optional[str] = Query(None, description="World version ID"),
    occupancy_group: str = Query(default="Business (Group B)"),
    is_sprinklered: bool = Query(default=True),
    architect_name: str = Query(default="Automated preliminary screening (not professionally reviewed)"),
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Download formal Life Safety & ADA Accessibility Compliance Certificate PDF."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, f"World '{world_id}' not found")
    enforce_project_isolation(w, auth)

    v_row = await _version_row(db, world_id, version_id)
    world_ir = _load_world_or_409(v_row)

    report = audit_life_safety_compliance(
        world=world_ir,
        occupancy_group=occupancy_group,
        is_sprinklered=is_sprinklered,
        project_name=w.name or "Architectural Reality Model",
    )
    pdf_bytes = export_life_safety_certificate_pdf(report, architect_name=architect_name)
    clean_world_id = world_id.replace(" ", "_")[:12]
    filename = f"life_safety_certificate_{clean_world_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
