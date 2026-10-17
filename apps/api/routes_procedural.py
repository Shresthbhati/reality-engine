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
from procedural.urban_grammar import (
    CityGrammarSpec,
    CorridorGrammarSpec,
    generate_city,
    generate_corridor,
)
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
    import math

    if not body.name or not body.name.strip():
        raise HTTPException(422, "room name must be a non-empty string")
    if len(body.name) > 200:
        raise HTTPException(422, "room name must be at most 200 chars")
    if len(body.openings) > 200:
        raise HTTPException(413, "too many wall openings (at most 200)")
    numerics = {
        "width": body.width,
        "depth": body.depth,
        "height": body.height,
        **{f"origin[{i}]": v for i, v in enumerate(body.origin)},
    }
    for opening in body.openings:
        numerics.update({
            f"opening.lateral_offset": opening.lateral_offset,
            f"opening.width": opening.width,
            f"opening.bottom": opening.bottom,
            f"opening.top": opening.top,
        })
    for label, value in numerics.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise HTTPException(422, f"room {label} must be a number")
        if not math.isfinite(value):
            raise HTTPException(422, f"room {label} must be finite (no NaN/infinity)")
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
    result = await _generate_and_commit(
        db, w, world_id, "room", spec.name,
        lambda current: generate_room(spec, world=current, origin=body.origin),
    )
    return {**result, "room_entity_id": f"{spec.name}-room"}


class CorridorIn(BaseModel):
    name: str
    length: float
    width: float = 2.0
    height: float = 2.6
    axis: str = "x"
    connects: list[str] = []
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)


class CityIn(BaseModel):
    name: str
    blocks_x: int = 3
    blocks_y: int = 3
    lot_size: float = 20.0
    street_width: float = 8.0
    min_height: float = 6.0
    max_height: float = 30.0
    seed: int = 0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)


def _check_name_and_numbers(body: BaseModel) -> None:
    import math

    if not body.name or not body.name.strip() or len(body.name) > 200:
        raise HTTPException(422, "name must be 1-200 characters")
    for key, value in body.model_dump().items():
        for v in value if isinstance(value, tuple) else [value]:
            if isinstance(v, float) and not math.isfinite(v):
                raise HTTPException(422, f"{key} must be finite (no NaN/infinity)")


@procedural.post("/corridors", status_code=201)
async def create_corridor(
    world_id: str, body: CorridorIn, db: AsyncSession = Depends(get_db)
) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    _check_name_and_numbers(body)
    if len(body.connects) > 50:
        raise HTTPException(413, "too many connected rooms (at most 50)")
    try:
        spec = CorridorGrammarSpec(
            name=body.name, length=body.length, width=body.width,
            height=body.height, axis=body.axis, connects=body.connects,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    result = await _generate_and_commit(
        db, w, world_id, "corridor", spec.name,
        lambda current: generate_corridor(spec, world=current, origin=body.origin),
    )
    return {**result, "corridor_entity_id": f"{spec.name}-corridor"}


@procedural.post("/cities", status_code=201)
async def create_city(
    world_id: str, body: CityIn, db: AsyncSession = Depends(get_db)
) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    _check_name_and_numbers(body)
    try:
        spec = CityGrammarSpec(
            name=body.name, blocks_x=body.blocks_x, blocks_y=body.blocks_y,
            lot_size=body.lot_size, street_width=body.street_width,
            min_height=body.min_height, max_height=body.max_height, seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    result = await _generate_and_commit(
        db, w, world_id, "city", spec.name,
        lambda current: generate_city(spec, world=current, origin=body.origin),
    )
    return {**result, "city_entity_id": f"{spec.name}-city"}


async def _generate_and_commit(db, w, world_id, kind, name, generate) -> dict:
    """Run a grammar against the world's current WorldIR and commit the
    result as a new WorldStore version (the one write path)."""
    current = await worldstore_service.get_current_worldir(db, world_id)
    before_ids = set(current.entities) if current is not None else set()
    try:
        world: WorldIR = generate(current)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    added_ids = [eid for eid in world.entities if eid not in before_ids]

    base_head = w.current_version_id
    try:
        version = await worldstore_service.commit_version(
            db,
            world_id=world_id,
            world=world,
            parent=base_head,
            source_session_ids=[],
            expect_parent=base_head,
        )
    except worldstore_service.ConcurrentModificationError as exc:
        raise HTTPException(409, str(exc)) from exc
    db.add(
        ActivityEvent(
            id=new_id("act"),
            type=f"{kind}.created",
            entity_type="world",
            entity_id=world_id,
            summary=f"{kind.capitalize()} '{name}' created",
        )
    )
    await db.commit()
    return {"version_id": version.id, "entity_ids_added": added_ids}
