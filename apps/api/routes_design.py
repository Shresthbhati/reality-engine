"""Furniture design edits (plan T-4b.1): move, rotate, delete, replace from a built-in catalog.

Each route is one committed edit (a new immutable world version, so Undo/Redo from T-6.7 cover it) made through the same
``commit_world_correction`` path as every other correction: validated, tenant-checked, stale-parent-guarded, and recorded for rule 18.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, get_auth_context
from apps.api.db import get_db
from apps.api.routes_worlds import CommitRequest, commit_world_correction
from apps.api.tenancy import world_scope
from world_ir.furniture_catalog import catalog_items

catalog = APIRouter(prefix="/api/furniture-catalog", tags=["design"])
design = APIRouter(prefix="/api/worlds/{world_id}/furniture/{entity_id}", tags=["design"], dependencies=[Depends(world_scope)])


class MoveIn(BaseModel):
    x: float
    z: float
    y: float | None = None
    parent_version_id: str | None = None


class RotateIn(BaseModel):
    yaw_rad: float
    parent_version_id: str | None = None


class ReplaceIn(BaseModel):
    catalog_id: str
    parent_version_id: str | None = None


class DeleteIn(BaseModel):
    parent_version_id: str | None = None


@catalog.get("")
async def get_catalog() -> dict:
    return {"items": catalog_items(), "note": "Nominal sizes for design, not measurements. Items are drawn from boxes; no 3D models yet."}


async def _commit(world_id: str, entity_id: str, op: dict, parent: str | None, message: str, db: AsyncSession, auth: AuthContext) -> dict:
    res = await commit_world_correction(
        world_id, CommitRequest(entity_id=entity_id, changes={"design_edit": op}, parent_version_id=parent, commit_message=message), db, auth)
    res["operation"] = f"design_{op['op']}"
    return res


@design.post("/move")
async def move_furniture(world_id: str, entity_id: str, body: MoveIn, db: AsyncSession = Depends(get_db),
                         auth: AuthContext = Depends(get_auth_context)) -> dict:
    return await _commit(world_id, entity_id, {"op": "move", "x": body.x, "z": body.z, "y": body.y}, body.parent_version_id,
                         f"Move {entity_id}", db, auth)


@design.post("/rotate")
async def rotate_furniture(world_id: str, entity_id: str, body: RotateIn, db: AsyncSession = Depends(get_db),
                           auth: AuthContext = Depends(get_auth_context)) -> dict:
    return await _commit(world_id, entity_id, {"op": "rotate", "yaw_rad": body.yaw_rad}, body.parent_version_id,
                         f"Rotate {entity_id}", db, auth)


@design.post("/replace")
async def replace_furniture(world_id: str, entity_id: str, body: ReplaceIn, db: AsyncSession = Depends(get_db),
                            auth: AuthContext = Depends(get_auth_context)) -> dict:
    return await _commit(world_id, entity_id, {"op": "replace", "catalog_id": body.catalog_id}, body.parent_version_id,
                         f"Replace {entity_id} with {body.catalog_id}", db, auth)


@design.post("/delete")
async def delete_furniture(world_id: str, entity_id: str, body: DeleteIn | None = None, db: AsyncSession = Depends(get_db),
                           auth: AuthContext = Depends(get_auth_context)) -> dict:
    return await _commit(world_id, entity_id, {"op": "delete"}, body.parent_version_id if body else None,
                         f"Delete {entity_id}", db, auth)
