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
    #: a tape measurement of the same quantity (length kinds only); marks the dimension VERIFIED.
    verified_value: float | None = None


def _row(m: WorldMeasurement, scale_dimension_state: str | None = None) -> dict:
    scale_state = m.scale_state
    verified = m.verified_value is not None
    requested = m.state or ("VERIFIED" if verified else "RECONSTRUCTED" if scale_state == "unknown" else "MEASURED")
    dim = measure.make_dimension(
        m.verified_value if verified else m.value,
        state=requested, basis=("tape measurement confirmed by user" if verified else m.method),
        scale_state=scale_state, scale_dimension_state=None if verified else scale_dimension_state,
        uncertainty_m=None if verified else m.uncertainty, kind=m.kind)
    return {
        # D1: the shared dimension shape; the flat keys below stay for existing consumers.
        "dimension": dim,
        "state": dim["state"],
        "basis": dim["basis"],
        "uncertainty_m": dim["uncertainty_m"],
        "verified_value": m.verified_value,
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


async def _scale_dimension_state(db: AsyncSession, world_id: str) -> str | None:
    """metadata.scale.dimension_state of the world's current version (ESTIMATED after an accepted prior); None if
    the world has no readable version -- then there is no cap to apply."""
    from apps.api.routes_worlds import _load_world_or_409, _version_row

    try:
        scale = ((_load_world_or_409(await _version_row(db, world_id, None)).metadata or {}).get("scale") or {})
    except HTTPException:
        return None
    return scale.get("dimension_state")


async def _require_world(db: AsyncSession, world_id: str) -> World:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    return w


@measurements.get("")
async def list_measurements(world_id: str, db: AsyncSession = Depends(get_db)) -> list[dict]:
    await _require_world(db, world_id)
    cap = await _scale_dimension_state(db, world_id)
    res = await db.execute(
        select(WorldMeasurement)
        .where(WorldMeasurement.world_id == world_id)
        .order_by(WorldMeasurement.created_at, WorldMeasurement.id)
        .limit(MAX_PER_WORLD)
    )
    return [_row(m, cap) for m in res.scalars()]


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

    verified = body.verified_value
    if verified is not None:
        if body.kind in measure.AREA_KINDS or body.kind == "angle" or body.scale_state == "unknown":
            raise HTTPException(422, "a tape-verified value needs a length kind in a metric or authored world")
        if not math.isfinite(verified) or (verified <= 0 and body.kind != "elevation_difference"):
            raise HTTPException(422, "verified_value must be a finite positive number")
    cap = await _scale_dimension_state(db, world_id)
    state = "VERIFIED" if verified is not None else measure.effective_state(
        "RECONSTRUCTED" if body.scale_state == "unknown" else "MEASURED", body.scale_state, cap)

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
        state=state,
        verified_value=verified,
    )
    db.add(m)
    await db.commit()
    await db.refresh(m)
    return _row(m, cap)


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
