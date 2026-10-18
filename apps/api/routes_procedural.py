"""Routes: procedural room construction (Phase 4 of the golden loop).

"Create Room" is real end to end: a validated RoomGrammarSpec is run
through the same procedural/room_grammar.py the reality CLI uses,
merged into the world's current WorldIR (or a fresh one), and
committed as a new WorldStore version through worldstore_service --
the same single write path reconstruction uses. No frontend-only
state anywhere in this chain.
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import worldstore_service
from apps.api.db import get_db
from apps.api.models import ActivityEvent, World, new_id
from procedural.room_grammar import RoomGrammarSpec, WallOpening
from procedural import spec_store
from procedural.urban_grammar import CityGrammarSpec, CorridorGrammarSpec
from world_ir.world_v1 import WorldIR

procedural = APIRouter(prefix="/api/worlds/{world_id}", tags=["procedural"])
templates_router = APIRouter(prefix="/api/procedural", tags=["procedural"])


@templates_router.get("/templates")
async def list_room_templates() -> dict:
    """Default room dimensions (data only; the UI just fills form fields)."""
    from procedural.templates import ROOM_TEMPLATES

    return {
        "label": "Default dimensions only; edit before building.",
        "templates": [{"name": n, **dims} for n, dims in ROOM_TEMPLATES.items()],
    }


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

    spec = _room_spec(body)
    result = await _create(db, w, world_id, spec)
    return {**result, "room_entity_id": spec.entity_id}


class AttachIn(BaseModel):
    to: str
    side: str


class CorridorIn(BaseModel):
    name: str
    length: float
    width: float = 2.0
    height: float = 2.6
    axis: str = "x"
    connects: list[str] = []
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    attach: Optional[AttachIn] = None


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
    spec = _corridor_spec(body)
    result = await _create(db, w, world_id, spec)
    return {**result, "corridor_entity_id": spec.entity_id}


@procedural.post("/cities", status_code=201)
async def create_city(
    world_id: str, body: CityIn, db: AsyncSession = Depends(get_db)
) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    spec = _city_spec(body)
    result = await _create(db, w, world_id, spec)
    return {**result, "city_entity_id": spec.entity_id}


async def _generate_and_commit(db, w, world_id, kind, name, generate, verb="created") -> dict:
    """Run a grammar against the world's current WorldIR and commit the
    result as a new WorldStore version (the one write path)."""
    current = await worldstore_service.get_current_worldir(db, world_id)
    before_ids = set(current.entities) if current is not None else set()
    try:
        world: WorldIR = generate(current)
    except spec_store.UnknownSpec as exc:
        raise HTTPException(404, f"no generated object {exc}") from exc
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
            type=f"{kind}.{verb}",
            entity_type="world",
            entity_id=world_id,
            summary=f"{kind.capitalize()} '{name}' {verb}",
        )
    )
    await db.commit()
    return {"version_id": version.id, "entity_ids_added": added_ids}


# ---------------------------------------------------------------------------
# Stored specs (C1): spec builders, create/edit/delete/list
# ---------------------------------------------------------------------------


def _room_spec(body: RoomIn) -> spec_store.GeneratedSpec:
    _spec_from_body(body)  # full validation (HTTP 422/413)
    return spec_store.GeneratedSpec(
        "room", body.name,
        params={"width": body.width, "depth": body.depth, "height": body.height,
                "openings": [o.model_dump() for o in body.openings], "seed": body.seed},
        seed=body.seed, origin=list(body.origin))


def _corridor_spec(body: CorridorIn) -> spec_store.GeneratedSpec:
    _check_name_and_numbers(body)
    if len(body.connects) > 50:
        raise HTTPException(413, "too many connected rooms (at most 50)")
    try:
        CorridorGrammarSpec(
            name=body.name, length=body.length, width=body.width,
            height=body.height, axis=body.axis, connects=body.connects)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    connects = list(body.connects)
    attach = body.attach.model_dump() if body.attach else None
    if attach and attach["to"] not in connects:
        connects.append(attach["to"])
    try:
        return spec_store.GeneratedSpec(
            "corridor", body.name,
            params={"length": body.length, "width": body.width, "height": body.height,
                    "axis": body.axis, "connects": connects},
            origin=list(body.origin), attach=attach)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _city_spec(body: CityIn) -> spec_store.GeneratedSpec:
    _check_name_and_numbers(body)
    try:
        CityGrammarSpec(
            name=body.name, blocks_x=body.blocks_x, blocks_y=body.blocks_y,
            lot_size=body.lot_size, street_width=body.street_width,
            min_height=body.min_height, max_height=body.max_height, seed=body.seed)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return spec_store.GeneratedSpec(
        "city", body.name,
        params={"blocks_x": body.blocks_x, "blocks_y": body.blocks_y,
                "lot_size": body.lot_size, "street_width": body.street_width,
                "min_height": body.min_height, "max_height": body.max_height,
                "seed": body.seed},
        seed=body.seed, origin=list(body.origin))


_BUILDERS = {
    "room": (RoomIn, _room_spec),
    "corridor": (CorridorIn, _corridor_spec),
    "city": (CityIn, _city_spec),
}


def _name_in_use(world: Optional[WorldIR], name: str) -> bool:
    if world is None:
        return False
    taken = {s.name for s in spec_store.load_specs(world).values()}
    legacy = {f"{name}-room", f"{name}-corridor", f"{name}-city", f"{name}-floor"}
    return name in taken or bool(legacy & set(world.entities))


async def _create(db, w, world_id, spec: spec_store.GeneratedSpec) -> dict:
    def gen(current):
        old = (spec_store.load_specs(current) if current is not None else {}).get(spec.entity_id)
        if (old is not None and old.kind == spec.kind and old.params == spec.params
                and old.attach == spec.attach and (spec.attach or old.origin == spec.origin)):
            return current  # byte-identical resubmit: idempotent, commit dedups
        if _name_in_use(current, spec.name):
            raise HTTPException(
                409, f"a generated object named {spec.name!r} already exists; "
                     "PUT to edit it or DELETE it first")
        return spec_store.generate(current, spec)

    return await _generate_and_commit(db, w, world_id, spec.kind, spec.name, gen)


@procedural.get("/generated")
async def list_generated(world_id: str, db: AsyncSession = Depends(get_db)) -> list:
    if await db.get(World, world_id) is None:
        raise HTTPException(404, "World not found")
    current = await worldstore_service.get_current_worldir(db, world_id)
    if current is None:
        return []
    return [
        {"entity_id": sid, "kind": s.kind, "name": s.name, "params": s.params,
         "origin": s.origin, "attach": s.attach, "depends_on": s.depends_on}
        for sid, s in spec_store.load_specs(current).items()
    ]


class _Patch(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoomPatch(_Patch):
    width: Optional[float] = None
    depth: Optional[float] = None
    height: Optional[float] = None
    openings: Optional[list[OpeningIn]] = None
    seed: Optional[int] = None
    origin: Optional[tuple[float, float, float]] = None


class CorridorPatch(_Patch):
    length: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    axis: Optional[str] = None
    connects: Optional[list[str]] = None
    origin: Optional[tuple[float, float, float]] = None
    attach: Optional[AttachIn] = None


class CityPatch(_Patch):
    blocks_x: Optional[int] = None
    blocks_y: Optional[int] = None
    lot_size: Optional[float] = None
    street_width: Optional[float] = None
    min_height: Optional[float] = None
    max_height: Optional[float] = None
    seed: Optional[int] = None
    origin: Optional[tuple[float, float, float]] = None


def _find_spec(current: Optional[WorldIR], kind: str, name: str):
    sid = f"{name}-{kind}"
    specs = spec_store.load_specs(current) if current is not None else {}
    if sid in specs and specs[sid].kind == kind:
        return sid, specs
    if current is not None and sid in current.entities:
        raise HTTPException(
            409, f"{kind} {name!r} was created before specs were stored; it has "
                 "no stored parameters to regenerate from (recreate it to enable edits)")
    raise HTTPException(404, f"{kind} {name!r} not found")


def _edit_route(path: str, kind: str, patch_model):
    async def edit(world_id: str, name: str, patch,
                   db: AsyncSession = Depends(get_db)) -> dict:
        w = await db.get(World, world_id)
        if w is None:
            raise HTTPException(404, "World not found")
        stats: dict = {}

        def gen(current):
            sid, specs = _find_spec(current, kind, name)
            old = specs[sid]
            body_cls, build = _BUILDERS[kind]
            fields = {**old.params, "origin": tuple(old.origin), "attach": old.attach}
            if kind == "corridor" and old.attach:
                fields["connects"] = [c for c in old.params.get("connects", [])
                                      if c != old.attach["to"]]
            edits = patch.model_dump(exclude_unset=True)
            try:
                new = build(body_cls(name=name, **{**fields, **edits}))
            except ValidationError as exc:
                raise HTTPException(422, exc.errors(include_url=False,
                                                    include_context=False,
                                                    include_input=False)) from exc
            new_params = {**new.params, "origin": new.origin, "attach": new.attach}
            world, changed, removed, unchanged = spec_store.regenerate(
                current, sid, new_params)
            stats.update(changed=changed, removed=removed, unchanged=unchanged)
            return world

        result = await _generate_and_commit(db, w, world_id, kind, name, gen, verb="updated")
        return {
            "version_id": result["version_id"],
            "changed": len(stats["changed"]),
            "removed": len(stats["removed"]),
            "unchanged": len(stats["unchanged"]),
            "entity_ids_changed": stats["changed"],
        }

    edit.__name__ = f"edit_{kind}"
    # `from __future__ import annotations` makes annotations strings that
    # can't see this local; assign the real body model so FastAPI treats it
    # as the request body.
    edit.__annotations__["patch"] = patch_model
    procedural.put(f"/{path}/{{name}}")(edit)


def _delete_route(path: str, kind: str):
    async def delete(world_id: str, name: str, cascade: bool = False,
                     db: AsyncSession = Depends(get_db)) -> dict:
        w = await db.get(World, world_id)
        if w is None:
            raise HTTPException(404, "World not found")
        stats: dict = {}

        def gen(current):
            sid, specs = _find_spec(current, kind, name)
            deps = spec_store.dependents(specs, sid, include_connects=True)
            if deps and not cascade:
                names = [specs[d].name for d in deps]
                raise HTTPException(
                    409, f"{kind} {name!r} has dependents: {', '.join(names)}; "
                         "delete them first or pass ?cascade=true")
            world, removed = spec_store.delete_specs(current, [sid, *deps])
            stats.update(removed=removed, cascaded=[specs[d].name for d in deps])
            return world

        result = await _generate_and_commit(db, w, world_id, kind, name, gen, verb="deleted")
        return {"version_id": result["version_id"], "removed": len(stats["removed"]),
                "entity_ids_removed": stats["removed"], "cascaded": stats["cascaded"]}

    delete.__name__ = f"delete_{kind}"
    procedural.delete(f"/{path}/{{name}}")(delete)


for _path, _kind, _patch in (("rooms", "room", RoomPatch),
                             ("corridors", "corridor", CorridorPatch),
                             ("cities", "city", CityPatch)):
    _edit_route(_path, _kind, _patch)
    _delete_route(_path, _kind)
