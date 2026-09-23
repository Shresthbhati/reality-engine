"""Routes: procedural room construction (Phase 4 of the golden loop).

"Create Room" is real end to end: a validated RoomGrammarSpec is run
through the same procedural/room_grammar.py the reality CLI uses,
merged into the world's current WorldIR (or a fresh one), and
committed as a new WorldStore version through worldstore_service --
the same single write path reconstruction uses. No frontend-only
state anywhere in this chain.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import worldstore_service
from apps.api.db import get_db
from apps.api.models import ActivityEvent, World, new_id
from procedural.room_grammar import RoomGrammarSpec, WallOpening, generate_room
from world_ir.world_v1 import WorldIR

procedural = APIRouter(prefix="/api/worlds/{world_id}", tags=["procedural"])


class OpeningIn(BaseModel):
    wall: str
    kind: str
    lateral_offset: float
    width: float
    bottom: float
    top: float


class RoomIn(BaseModel):
    name: str
    width: float
    depth: float
    height: float
    openings: list[OpeningIn] = []
    seed: int = 0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _spec_from_body(body: RoomIn) -> RoomGrammarSpec:
    try:
        return RoomGrammarSpec(
            name=body.name,
            width=body.width,
            depth=body.depth,
            height=body.height,
            openings=[WallOpening(**o.model_dump()) for o in body.openings],
            seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@procedural.post("/rooms", status_code=201)
async def create_room(
    world_id: str, body: RoomIn, db: AsyncSession = Depends(get_db)
) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")

    spec = _spec_from_body(body)
    current = await worldstore_service.get_current_worldir(db, world_id)
    before_ids = set(current.entities) if current is not None else set()
    try:
        world: WorldIR = generate_room(spec, world=current, origin=body.origin)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    added_ids = [eid for eid in world.entities if eid not in before_ids]

    version = await worldstore_service.commit_version(
        db,
        world_id=world_id,
        world=world,
        parent=w.current_version_id,
        source_session_ids=[],
    )
    db.add(
        ActivityEvent(
            id=new_id("act"),
            type="room.created",
            entity_type="world",
            entity_id=world_id,
            summary=f"Room '{spec.name}' created",
        )
    )
    await db.commit()

    return {
        "version_id": version.id,
        "room_entity_id": f"{spec.name}-room",
        "entity_ids_added": added_ids,
    }
