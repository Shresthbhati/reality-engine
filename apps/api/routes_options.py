"""Named design options (plan T-4b.3): save, list, open, rename, delete, compare with the as-built survey, export side by side.

Logic and rules live in ``world_options``; this module is the HTTP surface. Exports reuse the world exporter per version, so an
option's files are exactly what exporting that version would give.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import world_options as wo
from apps.api import worldstore_service
from apps.api.auth import AuthContext, get_auth_context
from apps.api.db import get_db
from apps.api.models import World
from apps.api.tenancy import world_scope
from world_ir.diff import diff_worlds

options = APIRouter(prefix="/api/worlds/{world_id}/options", tags=["options"], dependencies=[Depends(world_scope)])

MAX_EXPORT = 6


class NameIn(BaseModel):
    name: str


class ExportIn(BaseModel):
    refs: list[str]                 # "as-built", "current" or option ids
    format: str = "glb"


async def _world(db: AsyncSession, world_id: str) -> World:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    return w


def _guard(exc: wo.OptionError):
    return HTTPException(exc.status, str(exc))


@options.get("")
async def list_options(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    return await wo.list_options(db, await _world(db, world_id))


@options.post("", status_code=201)
async def save_option(world_id: str, body: NameIn, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> dict:
    try:
        return await wo.save_option(db, await _world(db, world_id), body.name, auth.user_id)
    except wo.OptionError as exc:
        raise _guard(exc) from None


@options.get("/compare")
async def compare(world_id: str, a: str = wo.AS_BUILT, b: str = "current", db: AsyncSession = Depends(get_db)) -> dict:
    """What differs between two versions, in design words, plus the structural diff. ``a`` and ``b`` are ``as-built``, ``current`` or option ids."""
    w = await _world(db, world_id)
    try:
        (name_a, va), (name_b, vb) = await wo.resolve(db, w, a), await wo.resolve(db, w, b)
    except wo.OptionError as exc:
        raise _guard(exc) from None
    store = worldstore_service.get_store()
    try:
        wa, wb = store.load_version(va), store.load_version(vb)
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(409, f"A version could not be read: {exc}") from None
    return {"a": {"ref": a, "name": name_a, "version_id": va}, "b": {"ref": b, "name": name_b, "version_id": vb},
            "summary": wo.summarise(wa, wb), "diff": diff_worlds(wa, wb).to_dict()}


@options.post("/export")
async def export_options(world_id: str, body: ExportIn, db: AsyncSession = Depends(get_db)) -> dict:
    """Export several options (and/or the as-built survey) in one call, so they can be handed over side by side."""
    from apps.api.routes_export import ExportRequest, export_world

    if not body.refs or len(body.refs) > MAX_EXPORT or len(set(body.refs)) != len(body.refs):
        raise HTTPException(422, f"Choose 1 to {MAX_EXPORT} different options to export")
    w = await _world(db, world_id)
    items = []
    for ref in body.refs:
        try:
            name, version = await wo.resolve(db, w, ref)
        except wo.OptionError as exc:
            raise _guard(exc) from None
        res = await export_world(world_id, ExportRequest(format=body.format, version=version), db)
        items.append({"ref": ref, "name": name, "version_id": version, "format": body.format, "artifact_uri": res["artifact_uri"],
                      "download_url": res["download_url"], "content_hash": res["content_hash"]})
    return {"format": body.format, "items": items}


@options.patch("/{option_id}")
async def rename_option(world_id: str, option_id: str, body: NameIn, db: AsyncSession = Depends(get_db)) -> dict:
    try:
        return await wo.rename_option(db, await _world(db, world_id), option_id, body.name)
    except wo.OptionError as exc:
        raise _guard(exc) from None


@options.delete("/{option_id}")
async def delete_option(world_id: str, option_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    try:
        return await wo.delete_option(db, await _world(db, world_id), option_id)
    except wo.OptionError as exc:
        raise _guard(exc) from None


@options.post("/{option_id}/open")
async def open_option(world_id: str, option_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    try:
        return await wo.open_option(db, await _world(db, world_id), option_id)
    except wo.OptionError as exc:
        raise _guard(exc) from None
