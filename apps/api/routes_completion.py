"""Routes: completion mode (Phase D) -- GENERATED interior proposals inside a reconstructed shell.

    POST /api/worlds/{id}/completion/proposals            create 1-3 alternatives (stored, nothing committed)
    GET  /api/worlds/{id}/completion/proposals            list
    GET  /api/worlds/{id}/completion/proposals/{pid}/diff existing-vs-proposed (read-only)
    POST /api/worlds/{id}/completion/proposals/{pid}/accept   commit via the procedural write path
    POST /api/worlds/{id}/completion/proposals/{pid}/reject

Honesty: proposed geometry is always provenance GENERATED (never OBSERVED/RECONSTRUCTED). The footprint is the
axis-aligned bounding rectangle of the non-generated RECONSTRUCTED/OBSERVED shell, reported as such. Units are
`model_units` unless the world has a metric/authored measurement.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from apps.api import worldstore_service
from apps.api.db import get_db
from apps.api.models import CompletionProposal, World, WorldMeasurement, new_id
from apps.api.routes_procedural import ApartmentRoomIn, _generate_and_commit, _name_in_use
from procedural import layout_solver, spec_store
from world_ir.schema_v1 import EntityType, Provenance
from world_ir.world_v1 import WorldIR

completion = APIRouter(prefix="/api/worlds/{world_id}/completion", tags=["completion"])

_SHELL_TYPES = (EntityType.FLOOR, EntityType.WALL, EntityType.ROOM)
_SHELL_PROVENANCE = (Provenance.RECONSTRUCTED, Provenance.OBSERVED)
MIN_SIDE = 2.0

TEMPLATES = {
    "studio": {
        "rooms": [
            {"id": "living", "kind": "living", "min_area": 16, "max_area": 35},
            {"id": "kitchen", "kind": "kitchen", "min_area": 5, "max_area": 12},
            {"id": "bath", "kind": "bathroom", "min_area": 3.5, "max_area": 8},
        ],
        "entrance": "living",
        "adjacency": [["kitchen", "living", "preferred"]],
    },
    "one_bed": {
        "rooms": [
            {"id": "living", "kind": "living", "min_area": 16, "max_area": 35},
            {"id": "kitchen", "kind": "kitchen", "min_area": 5, "max_area": 12},
            {"id": "bath", "kind": "bathroom", "min_area": 3.5, "max_area": 8},
            {"id": "bed", "kind": "bedroom", "min_area": 9, "max_area": 20},
        ],
        "entrance": "living",
        "adjacency": [["kitchen", "living", "preferred"]],
    },
}


class ProposalIn(BaseModel):
    name: str = "interior"
    template: Optional[str] = None
    rooms: Optional[list[ApartmentRoomIn]] = None
    entrance: Optional[str] = None
    adjacency: list[tuple[str, str, str]] = []
    seed: int = 0
    alternatives: int = Field(default=2, ge=1, le=3)


# ---- shell ---------------------------------------------------------------------------------------------------------


def shell_footprint(world: Optional[WorldIR]) -> dict:
    """Measured footprint of the reconstructed shell, or HTTP 422 saying exactly why there is none."""
    if world is None:
        raise HTTPException(422, "world has no committed version, so there is no reconstructed shell to complete")
    xs, zs, floor_tops, floor_ys, ids = [], [], [], [], []
    for eid, e in world.entities.items():
        if e.type not in _SHELL_TYPES or e.provenance not in _SHELL_PROVENANCE:
            continue
        if e.custom_properties.get("generated_by"):
            continue
        for gid in e.geometry_ids:
            g = world.geometries.get(gid)
            if g is None or g.bounds_min is None or g.bounds_max is None:
                continue
            vals = (g.bounds_min.x, g.bounds_max.x, g.bounds_min.z, g.bounds_max.z, g.bounds_min.y, g.bounds_max.y)
            if not all(math.isfinite(v) for v in vals):
                continue
            ids.append(eid)
            xs += [g.bounds_min.x, g.bounds_max.x]
            zs += [g.bounds_min.z, g.bounds_max.z]
            (floor_tops if e.type is EntityType.FLOOR else floor_ys).append(
                g.bounds_max.y if e.type is EntityType.FLOOR else g.bounds_min.y)
    if not ids:
        raise HTTPException(
            422, "no usable shell: completion needs RECONSTRUCTED or OBSERVED floor/wall/room entities with "
                 "measured geometry bounds (generated entities and entities without bounds do not count)")
    w, d = max(xs) - min(xs), max(zs) - min(zs)
    if w < MIN_SIDE or d < MIN_SIDE:
        raise HTTPException(422, f"shell footprint {w:.2f} x {d:.2f} is smaller than {MIN_SIDE} x {MIN_SIDE}; "
                                 "too small to hold a room program")
    return {"x_min": min(xs), "z_min": min(zs), "z_max": max(zs), "width": w, "depth": d,
            "floor_y": min(floor_tops) if floor_tops else min(floor_ys),
            "method": "axis_aligned_bbox", "shell_entities": sorted(set(ids))}


async def _unit(db: AsyncSession, world_id: str) -> str:
    rows = (await db.execute(select(WorldMeasurement.scale_state).where(
        WorldMeasurement.world_id == world_id))).scalars().all()
    return "m" if any(s in ("metric", "authored") for s in rows) else "model_units"


def _same_footprint(a: dict, b: dict) -> bool:
    return all(math.isclose(a[k], b[k], abs_tol=1e-6) for k in ("x_min", "z_min", "z_max", "width", "depth", "floor_y"))


# ---- helpers -------------------------------------------------------------------------------------------------------


def _program(body: ProposalIn) -> dict:
    if body.rooms:
        if not body.entrance:
            raise HTTPException(422, "entrance is required when rooms are supplied")
        return {"rooms": [r.model_dump() for r in body.rooms], "entrance": body.entrance,
                "adjacency": [list(a) for a in body.adjacency]}
    t = body.template or "studio"
    if t not in TEMPLATES:
        raise HTTPException(422, f"unknown template {t!r}; one of {sorted(TEMPLATES)} or supply rooms")
    return TEMPLATES[t]


def _solve(body: ProposalIn, prog: dict, fp: dict) -> list[dict]:
    """CPU-bound: returns one stored-spec dict per distinct plan."""
    w, d = math.floor(fp["width"] * 10) / 10, math.floor(fp["depth"] * 10) / 10
    try:
        spec = layout_solver.ApartmentSpec(
            name=body.name, rooms=prog["rooms"], entrance=prog["entrance"], footprint=(w, d),
            adjacency=[tuple(a) for a in prog["adjacency"]], seed=body.seed)
        result = layout_solver.solve_layout(spec, body.alternatives)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not result.ok:
        raise HTTPException(422, "no room program fits the measured footprint: " + "; ".join(result.violations))
    origin = [fp["x_min"], -fp["z_max"], fp["floor_y"]]    # grammar-local y = -world z
    out = []
    for i, plan in enumerate(result.plans):
        params = spec.to_dict()
        params.pop("name")
        params["rects"] = {k: [float(v) for v in r] for k, r in plan.rects.items()}
        params["plan_index"] = i
        out.append(asdict(spec_store.GeneratedSpec("apartment", body.name, params=params, seed=body.seed,
                                                   origin=origin)))
    return out


def _plan_out(r: CompletionProposal) -> Optional[dict]:
    """Proposed room rectangles [x, y, w, d] (grammar-local, origin = shell corner) from the stored spec.
    Doors are not stored by the solver, so none are returned."""
    p = (r.spec or {}).get("params") or {}
    if not p.get("rects"):
        return None
    return {"rects": p["rects"], "kinds": {x["id"]: x["kind"] for x in p.get("rooms", [])},
            "footprint": list(p["footprint"]) if p.get("footprint") else None, "provenance": "GENERATED"}


def _row_out(r: CompletionProposal) -> dict:
    return {"id": r.id, "plan": _plan_out(r), "world_id": r.world_id, "status": r.status, "alt_index": r.alt_index, "seed": r.seed,
            "base_version_id": r.base_version_id, "accepted_version_id": r.accepted_version_id,
            "summary": r.summary, "label": "GENERATED proposal -- not observed, not reconstructed",
            "created_at": r.created_at.isoformat() if r.created_at else None}


async def _get(db, world_id, pid) -> CompletionProposal:
    row = await db.get(CompletionProposal, pid)
    if row is None or row.world_id != world_id:
        raise HTTPException(404, "Proposal not found")
    return row


def _stamped_generate(current: Optional[WorldIR], gspec: spec_store.GeneratedSpec, pid: str) -> WorldIR:
    if current is None:
        raise HTTPException(422, "world has no committed version")
    if _name_in_use(current, gspec.name, gspec.kind):
        raise HTTPException(409, f"a generated object named {gspec.name!r} already exists in this world")
    world = spec_store.generate(current, gspec)
    for eid in spec_store.owned_ids(world, gspec.entity_id):
        world.entities[eid].custom_properties["completion_proposal"] = pid
    return world


# ---- routes --------------------------------------------------------------------------------------------------------


@completion.post("/proposals", status_code=201)
async def create_proposals(world_id: str, body: ProposalIn, db: AsyncSession = Depends(get_db)) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    if not body.name.strip() or len(body.name) > 200:
        raise HTTPException(422, "name must be 1-200 characters")
    current = await worldstore_service.get_current_worldir(db, world_id)
    fp = shell_footprint(current)
    unit = await _unit(db, world_id)
    prog = _program(body)
    specs = await run_in_threadpool(_solve, body, prog, fp)
    summary = {"footprint": {k: v for k, v in fp.items() if k != "shell_entities"}, "unit": unit,
               "shell_entity_ids": fp["shell_entities"], "program": prog, "name": body.name}
    rows = []
    for i, s in enumerate(specs):
        row = CompletionProposal(id=new_id("cmp"), world_id=world_id, base_version_id=w.current_version_id,
                                 alt_index=i, seed=body.seed, spec=s, summary=summary)
        db.add(row)
        rows.append(row)
    await db.commit()
    return {"proposals": [_row_out(r) for r in rows], "found": len(rows), "requested": body.alternatives,
            "note": "Proposals are stored only; the committed world is unchanged until one is accepted."}


@completion.get("/proposals")
async def list_proposals(world_id: str, db: AsyncSession = Depends(get_db)) -> list:
    if await db.get(World, world_id) is None:
        raise HTTPException(404, "World not found")
    rows = (await db.execute(select(CompletionProposal).where(CompletionProposal.world_id == world_id)
                             .order_by(CompletionProposal.created_at, CompletionProposal.alt_index))).scalars().all()
    return [_row_out(r) for r in rows]


@completion.get("/proposals/{pid}/diff")
async def proposal_diff(world_id: str, pid: str, db: AsyncSession = Depends(get_db)) -> dict:
    if await db.get(World, world_id) is None:
        raise HTTPException(404, "World not found")
    row = await _get(db, world_id, pid)
    current = await worldstore_service.get_current_worldir(db, world_id)
    if current is None:
        raise HTTPException(422, "world has no committed version")
    gspec = spec_store.GeneratedSpec(**row.spec)

    def compute() -> dict:
        base = WorldIR.from_dict(current.to_dict())
        new = _stamped_generate(WorldIR.from_dict(current.to_dict()), gspec, pid)
        added = sorted(e for e in new.entities if e not in base.entities)
        removed = sorted(e for e in base.entities if e not in new.entities)
        changed = sorted(e for e in base.entities if e in new.entities
                         and base.entities[e].to_dict() != new.entities[e].to_dict())
        count = lambda ids, w_: dict(Counter(w_.entities[i].provenance.value for i in ids))  # noqa: E731
        return {"added": added, "changed": changed, "removed": removed,
                "counts": {"added": len(added), "changed": len(changed), "removed": len(removed)},
                "provenance": {"existing": count(base.entities, base), "added": count(added, new)}}

    out = await run_in_threadpool(compute)
    return {"proposal_id": pid, "status": row.status, "unit": row.summary.get("unit"),
            "footprint": row.summary["footprint"],
            "generated_label": "All added entities are GENERATED proposals; none is observed or reconstructed.",
            **out}


@completion.post("/proposals/{pid}/accept")
async def accept_proposal(world_id: str, pid: str, db: AsyncSession = Depends(get_db)) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    row = await _get(db, world_id, pid)
    if row.status != "proposed":
        raise HTTPException(409, f"proposal is already {row.status}")
    current = await worldstore_service.get_current_worldir(db, world_id)
    if not _same_footprint(shell_footprint(current), {**row.summary["footprint"]}):
        raise HTTPException(409, "the reconstructed shell changed since this proposal was made; create a new proposal")
    gspec = spec_store.GeneratedSpec(**row.spec)
    result = await _generate_and_commit(
        db, w, world_id, "completion", gspec.name, lambda cur: _stamped_generate(cur, gspec, pid), verb="accepted")
    row = await _get(db, world_id, pid)
    row.status, row.accepted_version_id = "accepted", result["version_id"]
    await db.commit()
    return {**result, "proposal": _row_out(row), "provenance": "GENERATED"}


@completion.post("/proposals/{pid}/reject")
async def reject_proposal(world_id: str, pid: str, db: AsyncSession = Depends(get_db)) -> dict:
    if await db.get(World, world_id) is None:
        raise HTTPException(404, "World not found")
    row = await _get(db, world_id, pid)
    if row.status != "proposed":
        raise HTTPException(409, f"proposal is already {row.status}")
    row.status = "rejected"
    await db.commit()
    return _row_out(row)
