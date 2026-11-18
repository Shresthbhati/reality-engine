"""World design options, version labels, camera views, and still renders (T-4b.2, T-4b.3, T-4b.6 / RE-ARCH-000009).

Extracted from routes_worlds.py into a modular router providing:
  - GET /{world_id}/options: list design options and survey versions
  - POST /{world_id}/options: create a named design option
  - PATCH /{world_id}/versions/{version_id}/label: label or rename a version
  - GET /{world_id}/options/compare: compare design options / as-built versions
  - POST /{world_id}/views: save a camera view pose
  - GET /{world_id}/views: list saved camera views
  - DELETE /{world_id}/views/{view_id}: delete a saved camera view
  - GET /{world_id}/render: render a still PNG from a saved camera view
"""

from __future__ import annotations

import math
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, Role, get_auth_context
from apps.api.db import get_db
from apps.api.models import World, WorldCameraView, WorldVersion, new_id
from apps.api.tenancy import enforce_project_isolation

options_router = APIRouter()


class DesignOptionIn(BaseModel):
    label: str
    base_version: Optional[str] = None
    description: Optional[str] = None
    surface_finishes: Optional[list[dict]] = None


class VersionLabelIn(BaseModel):
    label: str


class CameraViewIn(BaseModel):
    name: str
    position: dict
    target: dict
    fov_deg: float = 60.0


async def _resolve_version_row(
    db: AsyncSession, world_id: str, version_id: Optional[str], auth: Optional[AuthContext] = None
) -> WorldVersion:
    """Internal helper to resolve a world version mirror row."""
    from apps.api.routes_worlds import _version_row
    return await _version_row(db, world_id, version_id, auth=auth)


def _load_world_ir(row: WorldVersion):
    """Internal helper to load WorldIR from a version row."""
    from apps.api.routes_worlds import _load_world_or_409
    return _load_world_or_409(row)


@options_router.get("/{world_id}/options")
async def list_design_options(
    world_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """List all saved design options and as-built survey versions for a world (T-4b.3)."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)

    res = await db.execute(
        select(WorldVersion).where(WorldVersion.world_id == world_id).order_by(WorldVersion.created_at.asc())
    )
    versions = res.scalars().all()
    options = []
    for i, v in enumerate(versions):
        is_as_built = (v.label == "as-built") or (v.parent_version_id is None and (v.label is None or v.label == "as-built"))
        lbl = v.label or ("as-built" if is_as_built else f"Option {chr(65 + max(0, i - 1))}")
        options.append({
            "version_id": v.id,
            "label": lbl,
            "parent_version_id": v.parent_version_id,
            "is_as_built": is_as_built,
            "is_current": v.id == w.current_version_id,
            "created_at": v.created_at.isoformat() if v.created_at else None,
            "changed_entities_count": len(v.changed_entity_ids or []),
        })
    return {"world_id": world_id, "options": options}


@options_router.post("/{world_id}/options")
async def create_design_option(
    world_id: str,
    body: DesignOptionIn,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Create a new named design option with a plain-language label derived from a base version (T-4b.3)."""
    if auth.role == Role.VIEWER:
        raise HTTPException(403, "Read-only access cannot create design options")
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)

    base_row = await _resolve_version_row(db, world_id, body.base_version, auth=auth)
    worldir = _load_world_ir(base_row)

    from world_ir.schema_v1 import Material, Provenance, SurfaceFinish
    changed_entity_ids = []
    if body.surface_finishes:
        for sf in body.surface_finishes:
            ent_id = sf.get("entity_id")
            if not ent_id or ent_id not in worldir.entities:
                raise HTTPException(422, f"entity '{ent_id}' not found in world")
            ent = worldir.entities[ent_id]
            changed_entity_ids.append(ent_id)

            finish_str = sf.get("surface_finish", "smooth")
            try:
                finish_val = SurfaceFinish(finish_str)
            except ValueError:
                finish_val = SurfaceFinish.SMOOTH

            color_rgb = tuple(float(c) for c in sf.get("color_rgb", [0.8, 0.8, 0.8]))
            mat_name = str(sf.get("name") or sf.get("material_name") or "Option Finish")
            mat_class = str(sf.get("class_name") or sf.get("material") or "finish")

            mat_id = f"mat-{ent.id}-{uuid4().hex[:6]}"
            new_mat = Material(
                id=mat_id,
                name=mat_name,
                class_name=mat_class,
                surface_finish=finish_val,
                color_rgb=color_rgb,
                provenance=Provenance.GENERATED,
                confidence=1.0,
            )
            worldir.materials[mat_id] = new_mat
            ent.material_ids = [mat_id]
            ent.custom_properties = dict(ent.custom_properties or {})
            ent.custom_properties["surface_finish"] = finish_val.value
            ent.custom_properties["material"] = mat_class
            ent.custom_properties["material_label"] = mat_name
            ent.custom_properties["design_layer"] = "finish"

    from world_ir.world_v1 import Scenario
    scn_id = f"scn-{uuid4().hex[:8]}"
    worldir.scenarios[scn_id] = Scenario(
        id=scn_id,
        name=body.label.strip(),
        description=body.description or "",
    )

    from apps.api import worldstore_service
    version_row = await worldstore_service.commit_version(
        db,
        world_id=w.id,
        world=worldir,
        parent=base_row.id,
        label=body.label.strip(),
        allow_branch=True,
    )

    return {
        "world_id": w.id,
        "version_id": version_row.id,
        "label": version_row.label,
        "parent_version_id": version_row.parent_version_id,
        "created_at": version_row.created_at.isoformat() if version_row.created_at else None,
        "changed_entity_ids": changed_entity_ids,
    }


@options_router.patch("/{world_id}/versions/{version_id}/label")
async def set_version_label(
    world_id: str,
    version_id: str,
    body: VersionLabelIn,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Assign or rename a plain-language label on a world version (T-4b.3)."""
    if auth.role == Role.VIEWER:
        raise HTTPException(403, "Read-only access cannot rename versions")
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)

    row = await _resolve_version_row(db, world_id, version_id, auth=auth)
    row.label = body.label.strip()
    await db.commit()
    await db.refresh(row)
    return {"version_id": row.id, "label": row.label}


@options_router.get("/{world_id}/options/compare")
async def compare_design_options(
    world_id: str,
    base: str = "as-built",
    target: str = "as-built",
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Compare as-built survey vs named design option, or two options (T-4b.3)."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)

    row_base = await _resolve_version_row(db, world_id, base, auth=auth)
    row_target = await _resolve_version_row(db, world_id, target, auth=auth)

    world_base = _load_world_ir(row_base)
    world_target = _load_world_ir(row_target)

    finish_diffs = []
    for eid, ent_target in world_target.entities.items():
        ent_base = world_base.entities.get(eid)
        base_props = ent_base.custom_properties if ent_base else {}
        target_props = ent_target.custom_properties or {}

        base_finish = base_props.get("surface_finish")
        target_finish = target_props.get("surface_finish")
        base_mat = base_props.get("material") or base_props.get("material_label")
        target_mat = target_props.get("material") or target_props.get("material_label")

        if base_finish != target_finish or base_mat != target_mat:
            finish_diffs.append({
                "entity_id": eid,
                "name": ent_target.name,
                "type": ent_target.type.value if hasattr(ent_target.type, "value") else str(ent_target.type),
                "base": {"surface_finish": base_finish, "material": base_mat},
                "target": {"surface_finish": target_finish, "material": target_mat},
            })

    added_ids = [eid for eid in world_target.entities if eid not in world_base.entities]
    removed_ids = [eid for eid in world_base.entities if eid not in world_target.entities]

    return {
        "world_id": world_id,
        "base": {
            "version_id": row_base.id,
            "label": row_base.label or "as-built",
            "created_at": row_base.created_at.isoformat() if row_base.created_at else None,
            "entity_count": len(world_base.entities),
        },
        "target": {
            "version_id": row_target.id,
            "label": row_target.label or target,
            "created_at": row_target.created_at.isoformat() if row_target.created_at else None,
            "entity_count": len(world_target.entities),
        },
        "summary": {
            "finishes_changed_count": len(finish_diffs),
            "added_entities_count": len(added_ids),
            "removed_entities_count": len(removed_ids),
        },
        "finish_differences": finish_diffs,
        "added_entity_ids": added_ids,
        "removed_entity_ids": removed_ids,
    }


@options_router.post("/{world_id}/views")
async def create_camera_view(
    world_id: str,
    body: CameraViewIn,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Save a camera view pose for client deck still renders and presentations (T-4b.6)."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)
    if auth.role == Role.VIEWER:
        raise HTTPException(403, "Read-only access cannot save camera views")

    if not body.name or len(body.name) > 120:
        raise HTTPException(422, "name must be a non-empty string <= 120 chars")

    for k, val in (("position", body.position), ("target", body.target)):
        if not isinstance(val, dict) or not all(axis in val for axis in ("x", "y", "z")):
            raise HTTPException(422, f"{k} must be an object with numeric x, y, z")
        for axis in ("x", "y", "z"):
            v = val[axis]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise HTTPException(422, f"{k}.{axis} must be a finite number")

    if not (10.0 <= body.fov_deg <= 150.0):
        raise HTTPException(422, "fov_deg must be within [10, 150]")

    view = WorldCameraView(
        id=new_id("view"),
        world_id=w.id,
        name=body.name.strip(),
        position={axis: float(body.position[axis]) for axis in ("x", "y", "z")},
        target={axis: float(body.target[axis]) for axis in ("x", "y", "z")},
        fov_deg=float(body.fov_deg),
    )
    db.add(view)
    await db.commit()
    await db.refresh(view)
    return {
        "id": view.id,
        "world_id": view.world_id,
        "name": view.name,
        "position": view.position,
        "target": view.target,
        "fov_deg": view.fov_deg,
        "created_at": view.created_at.isoformat() if view.created_at else None,
    }


@options_router.get("/{world_id}/views")
async def list_camera_views(
    world_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """List all saved camera views for a world (T-4b.6)."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)

    res = await db.execute(
        select(WorldCameraView).where(WorldCameraView.world_id == world_id).order_by(WorldCameraView.created_at.asc())
    )
    views = res.scalars().all()
    return {
        "world_id": world_id,
        "items": [
            {
                "id": v.id,
                "name": v.name,
                "position": v.position,
                "target": v.target,
                "fov_deg": v.fov_deg,
                "created_at": v.created_at.isoformat() if v.created_at else None,
            }
            for v in views
        ],
    }


@options_router.delete("/{world_id}/views/{view_id}")
async def delete_camera_view(
    world_id: str,
    view_id: str,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Delete a saved camera view (T-4b.6)."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)
    if auth.role == Role.VIEWER:
        raise HTTPException(403, "Read-only access cannot delete camera views")

    view = await db.get(WorldCameraView, view_id)
    if view is None or view.world_id != world_id:
        raise HTTPException(404, "Camera view not found")

    await db.delete(view)
    await db.commit()
    return {"status": "deleted", "id": view_id}


@options_router.get("/{world_id}/render")
async def render_world_view(
    world_id: str,
    view_id: Optional[str] = None,
    version: Optional[str] = None,
    width: int = 1280,
    height: int = 720,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Render still PNG from saved camera view for client decks (T-4b.6)."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)

    row = await _resolve_version_row(db, world_id, version, auth=auth)
    worldir = _load_world_ir(row)

    view_name = "Default Perspective"
    pos = {"x": 0.0, "y": 1.6, "z": -4.0}
    tgt = {"x": 0.0, "y": 1.2, "z": 0.0}
    fov = 60.0

    if view_id is not None:
        res = await db.execute(
            select(WorldCameraView).where(WorldCameraView.world_id == world_id, WorldCameraView.id == view_id)
        )
        view = res.scalars().first()
        if view is None:
            res = await db.execute(
                select(WorldCameraView).where(WorldCameraView.world_id == world_id, WorldCameraView.name == view_id)
            )
            view = res.scalars().first()
        if view is None:
            raise HTTPException(404, f"Saved camera view '{view_id}' not found")
        view_name = view.name
        pos = view.position
        tgt = view.target
        fov = view.fov_deg

    from apps.api.render_service import render_still_png
    png_bytes = render_still_png(
        world=worldir,
        view_name=view_name,
        position=pos,
        target=tgt,
        fov_deg=fov,
        width=max(320, min(3840, width)),
        height=max(240, min(2160, height)),
        world_name=w.name,
        version_label=row.label or "as-built",
    )

    safe_view = "".join(c for c in view_name if c.isalnum() or c in ("-", "_")).strip() or "view"
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "Content-Disposition": f'inline; filename="{w.id}_{safe_view}.png"',
            "Cache-Control": "private, max-age=3600",
        },
    )
