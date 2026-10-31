"""Sites: user-directed composition of reconstructed worlds on a metric grid.

Placement is chosen by the user. This is NOT automatic geo-registration: nothing here infers where a world is on
Earth or how two worlds relate; the grid is just a layout surface.
"""

from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import get_db
from apps.api.models import Site, SitePlacement, World, new_id
from apps.api.routes_worlds import _load_world_or_409, _version_row
from world_ir.schema_v1 import EntityType

sites = APIRouter(prefix="/api/sites", tags=["sites"])

_F = {"allow_inf_nan": False}


class SiteIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    grid_cell_m: float = Field(default=50.0, ge=1, le=5000, **_F)


class PlacementIn(BaseModel):
    world_id: str
    label: str | None = Field(default=None, max_length=120)
    cell_x: int = Field(ge=-500, le=500)
    cell_z: int = Field(ge=-500, le=500)
    offset_x_m: float = Field(default=0.0, **_F)
    offset_z_m: float = Field(default=0.0, **_F)
    rotation_deg: float = Field(default=0.0, **_F)
    scale: float = Field(default=1.0, gt=0, le=100, **_F)


class PlacementPatch(BaseModel):
    label: str | None = Field(default=None, max_length=120)
    cell_x: int | None = Field(default=None, ge=-500, le=500)
    cell_z: int | None = Field(default=None, ge=-500, le=500)
    offset_x_m: float | None = Field(default=None, **_F)
    offset_z_m: float | None = Field(default=None, **_F)
    rotation_deg: float | None = Field(default=None, **_F)
    scale: float | None = Field(default=None, gt=0, le=100, **_F)


async def _world_footprint(db: AsyncSession, world_id: str) -> tuple[float, float] | None:
    """(width_m, depth_m) of the world's HEAD WorldIR via the existing measured-shell footprint, else None.

    Never invented: no head version / unreadable artifact / no measured shell -> None (source "unknown").
    """
    try:
        world = _load_world_or_409(await _version_row(db, world_id, None))
    except HTTPException:
        return None
    return _extent(world)


def _extent(world) -> tuple[float, float] | None:
    """Plan extent of the room footprint polygons (the same measured outline the room summary reports).

    The compiler's interior frame is Z-up while the canonical frame is Y-up, and a floor polygon is flat along its
    vertical axis, so the two axes with the largest spread are the horizontal ones in either convention.
    ``shell_footprint`` (completion) assumes Y-up and drops inferred walls, so it is not used here."""
    pts = []
    for e in world.entities.values():
        if e.type is not EntityType.ROOM:
            continue
        for p in e.custom_properties.get("footprint_polygon_xyz_m") or []:
            if len(p) == 3 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in p):
                pts.append(p)
    if len(pts) < 3:
        return None
    spans = sorted((max(p[i] for p in pts) - min(p[i] for p in pts) for i in range(3)), reverse=True)
    return (spans[0], spans[1]) if spans[1] > 0 else None


def _placement_dict(p: SitePlacement, cell_m: float, world_name: str | None, fp: tuple[float, float] | None) -> dict:
    w, d = (fp[0] * p.scale, fp[1] * p.scale) if fp else (None, None)
    return {
        "id": p.id, "site_id": p.site_id, "world_id": p.world_id, "world_name": world_name, "label": p.label,
        "cell_x": p.cell_x, "cell_z": p.cell_z, "offset_x_m": p.offset_x_m, "offset_z_m": p.offset_z_m,
        "rotation_deg": p.rotation_deg, "scale": p.scale,
        "center_m": {"x": (p.cell_x + 0.5) * cell_m + p.offset_x_m, "z": (p.cell_z + 0.5) * cell_m + p.offset_z_m},
        "footprint": {"width_m": w, "depth_m": d, "source": "worldir" if fp else "unknown"},
    }


def _overlaps(placements: list[dict]) -> list[list[str]]:
    # ponytail: axis-aligned rectangles, rotation_deg ignored (rotated footprints can over/under-report);
    # upgrade to separating-axis tests on rotated rects if tight layouts matter.
    out = []
    for i, a in enumerate(placements):
        for b in placements[i + 1:]:
            fa, fb = a["footprint"], b["footprint"]
            if fa["width_m"] is None or fb["width_m"] is None:
                continue
            if (abs(a["center_m"]["x"] - b["center_m"]["x"]) * 2 < fa["width_m"] + fb["width_m"]
                    and abs(a["center_m"]["z"] - b["center_m"]["z"]) * 2 < fa["depth_m"] + fb["depth_m"]):
                out.append([a["id"], b["id"]])
    return out


async def _site_or_404(db: AsyncSession, site_id: str) -> Site:
    s = await db.get(Site, site_id)
    if s is None:
        raise HTTPException(404, "Site not found")
    return s


async def _render(db: AsyncSession, site: Site, p: SitePlacement) -> dict:
    w = await db.get(World, p.world_id)
    return _placement_dict(p, site.grid_cell_m, w.name if w else None, await _world_footprint(db, p.world_id))


def _site_dict(s: Site) -> dict:
    return {"id": s.id, "name": s.name, "grid_cell_m": s.grid_cell_m, "created_at": s.created_at.isoformat()}


@sites.get("")
async def list_sites(db: AsyncSession = Depends(get_db)) -> list[dict]:
    counts = dict((await db.execute(select(SitePlacement.site_id, func.count()).group_by(SitePlacement.site_id))).all())
    rows = (await db.execute(select(Site).order_by(Site.created_at.desc()))).scalars().all()
    return [{"id": s.id, "name": s.name, "grid_cell_m": s.grid_cell_m, "placement_count": counts.get(s.id, 0)}
            for s in rows]


@sites.post("", status_code=201)
async def create_site(body: SiteIn, db: AsyncSession = Depends(get_db)) -> dict:
    s = Site(id=new_id("site"), name=body.name, grid_cell_m=body.grid_cell_m)
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return _site_dict(s)


@sites.get("/{site_id}")
async def get_site(site_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    s = await _site_or_404(db, site_id)
    rows = (await db.execute(select(SitePlacement).where(SitePlacement.site_id == site_id)
                             .order_by(SitePlacement.created_at, SitePlacement.id))).scalars().all()
    placements = [await _render(db, s, p) for p in rows]
    return {**_site_dict(s), "placements": placements, "overlaps": _overlaps(placements)}


@sites.delete("/{site_id}", status_code=204)
async def delete_site(site_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    s = await _site_or_404(db, site_id)
    # explicit: SQLite does not enforce ON DELETE CASCADE unless foreign_keys is on
    await db.execute(delete(SitePlacement).where(SitePlacement.site_id == site_id))
    await db.delete(s)
    await db.commit()
    return Response(status_code=204)


@sites.post("/{site_id}/placements", status_code=201)
async def add_placement(site_id: str, body: PlacementIn, db: AsyncSession = Depends(get_db)) -> dict:
    s = await _site_or_404(db, site_id)
    if await db.get(World, body.world_id) is None:
        raise HTTPException(404, "World not found")
    p = SitePlacement(id=new_id("plc"), site_id=site_id,
                      **{**body.model_dump(), "rotation_deg": body.rotation_deg % 360.0})
    db.add(p)
    await db.commit()
    await db.refresh(p)
    return await _render(db, s, p)


@sites.patch("/{site_id}/placements/{placement_id}")
async def patch_placement(site_id: str, placement_id: str, body: PlacementPatch,
                          db: AsyncSession = Depends(get_db)) -> dict:
    s = await _site_or_404(db, site_id)
    p = await db.get(SitePlacement, placement_id)
    if p is None or p.site_id != site_id:
        raise HTTPException(404, "Placement not found")
    for k, v in body.model_dump(exclude_unset=True).items():
        if v is None and k != "label":
            raise HTTPException(422, f"{k} cannot be null")
        setattr(p, k, v % 360.0 if k == "rotation_deg" else v)
    await db.commit()
    await db.refresh(p)
    return await _render(db, s, p)


@sites.delete("/{site_id}/placements/{placement_id}", status_code=204)
async def delete_placement(site_id: str, placement_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    await _site_or_404(db, site_id)
    p = await db.get(SitePlacement, placement_id)
    if p is None or p.site_id != site_id:
        raise HTTPException(404, "Placement not found")
    await db.delete(p)
    await db.commit()
    return Response(status_code=204)
