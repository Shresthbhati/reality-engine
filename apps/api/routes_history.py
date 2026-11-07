"""GET /api/worlds/{id}/history, POST /undo, POST /redo (plan T-6.7). Logic and rules live in ``world_history``."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import world_history
from apps.api.db import get_db
from apps.api.models import World
from apps.api.tenancy import world_scope

history = APIRouter(prefix="/api/worlds/{world_id}", tags=["history"], dependencies=[Depends(world_scope)])


async def _world(db: AsyncSession, world_id: str) -> World:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    return w


@history.get("/history")
async def get_history(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    return await world_history.state(db, await _world(db, world_id))


@history.post("/undo")
async def post_undo(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    try:
        return await world_history.undo(db, await _world(db, world_id))
    except world_history.HistoryError as exc:
        raise HTTPException(exc.status, str(exc)) from None


@history.post("/redo")
async def post_redo(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    try:
        return await world_history.redo(db, await _world(db, world_id))
    except world_history.HistoryError as exc:
        raise HTTPException(exc.status, str(exc)) from None
