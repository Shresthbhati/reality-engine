"""Routes: saved viewport measurements (Phase B5, measurement mode).

The value of a measurement is never trusted from the client: it is recomputed
from the points by world_ir.measure (single source of truth) and a client value
that disagrees is rejected. The unit follows the world's scale state, so an
unscaled reconstruction cannot be stored as metres, and an uncertainty exists
only when the caller was given one (authored geometry is exact, i.e. 0).
"""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import get_db
from apps.api.models import World, WorldMeasurement, new_id
from world_ir import measure

measurements = APIRouter(prefix="/api/worlds/{world_id}/measurements", tags=["measurements"])

MAX_PER_WORLD = 500


class MeasurementIn(BaseModel):
    kind: str
    points: list
    scale_state: str
    method: str
    #: optional; if given it must equal the server-computed value.
    value: float | None = None
    #: only when the data supplied one; ignored for authored geometry (exactly 0).
    uncertainty: float | None = None


def _row(m: WorldMeasurement) -> dict:
    return {
        "id": m.id,
        "world_id": m.world_id,
        "kind": m.kind,
        "points": m.points,
        "value": m.value,
        "unit": m.unit,
        "uncertainty": m.uncertainty,
        "method": m.method,
        "scale_state": m.scale_state,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


async def _require_world(db: AsyncSession, world_id: str) -> World:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    return w


@measurements.get("")
async def list_measurements(world_id: str, db: AsyncSession = Depends(get_db)) -> list[dict]:
    await _require_world(db, world_id)
    res = await db.execute(
        select(WorldMeasurement)
        .where(WorldMeasurement.world_id == world_id)
        .order_by(WorldMeasurement.created_at, WorldMeasurement.id)
        .limit(MAX_PER_WORLD)
    )
    return [_row(m) for m in res.scalars()]


@measurements.post("", status_code=201)
async def create_measurement(
    world_id: str, body: MeasurementIn, db: AsyncSession = Depends(get_db)
) -> dict:
    await _require_world(db, world_id)

    if isinstance(body.points, list) and len(body.points) > measure.POINTS_MAX.get(body.kind, 0) > 0:
        raise HTTPException(413, f"too many points (at most {measure.POINTS_MAX[body.kind]} for {body.kind})")
    try:
        value = measure.compute(body.kind, body.points)
        unit = measure.unit_for(body.kind, body.scale_state)
        points = [list(p) for p in measure.validate_points(body.kind, body.points)]
    except measure.MeasureError as exc:
        raise HTTPException(422, str(exc)) from exc

    if body.value is not None and not measure.values_agree(body.value, value):
        raise HTTPException(
            422, f"value {body.value!r} disagrees with the value computed from the points ({value!r})"
        )

    method = (body.method or "").strip()
    if not method or len(method) > 200:
        raise HTTPException(422, "method must be a non-empty string of at most 200 chars")

    if body.scale_state == "authored":
        uncertainty: float | None = 0.0  # authored geometry is exact by construction
    elif body.uncertainty is None:
        uncertainty = None
    elif not math.isfinite(body.uncertainty) or body.uncertainty < 0:
        raise HTTPException(422, "uncertainty must be a finite number >= 0 (or omitted)")
    else:
        uncertainty = float(body.uncertainty)

    count = (
        await db.execute(
            select(func.count()).select_from(WorldMeasurement).where(WorldMeasurement.world_id == world_id)
        )
    ).scalar_one()
    if count >= MAX_PER_WORLD:
        raise HTTPException(413, f"too many measurements for this world (at most {MAX_PER_WORLD})")

    m = WorldMeasurement(
        id=new_id("msr"),
        world_id=world_id,
        kind=body.kind,
        points=points,
        value=value,
        unit=unit,
        uncertainty=uncertainty,
        method=method,
        scale_state=body.scale_state,
    )
    db.add(m)
    await db.commit()
    await db.refresh(m)
    return _row(m)


@measurements.delete("/{measurement_id}")
async def delete_measurement(
    world_id: str, measurement_id: str, db: AsyncSession = Depends(get_db)
) -> dict:
    await _require_world(db, world_id)
    m = await db.get(WorldMeasurement, measurement_id)
    if m is None or m.world_id != world_id:
        raise HTTPException(404, "Measurement not found")
    await db.delete(m)
    await db.commit()
    return {"id": measurement_id, "deleted": True}
