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
from apps.api.auth import AuthContext, get_auth_context
from apps.api.tenancy import assert_visible, site_scope, team_stamp, visible
from typing import Literal

from apps.api import worldstore_service
from apps.api.models import Site, SiteAnchor, SiteComposite, SitePlacement, World, new_id, utcnow
from apps.api.routes_worlds import _load_world_or_409, _version_row
from world_ir.compose import ComposeError, _up_axis, compose_worlds
from world_ir.connectors import is_generated, make_connector
from world_ir.schema_v1 import EntityType

sites = APIRouter(prefix="/api/sites", tags=["sites"], dependencies=[Depends(site_scope)])

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


async def _world_visible(db: AsyncSession, world_id: str, auth: AuthContext) -> bool:
    """False only for a placed world that exists and belongs to another tenant (its name/footprint must not leak)."""
    try:
        await assert_visible(db, World, world_id, auth)
    except HTTPException:
        return False
    return True


def _site_dict(s: Site) -> dict:
    return {"id": s.id, "name": s.name, "grid_cell_m": s.grid_cell_m, "created_at": s.created_at.isoformat()}


@sites.get("")
async def list_sites(db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> list[dict]:
    counts = dict((await db.execute(select(SitePlacement.site_id, func.count()).group_by(SitePlacement.site_id))).all())
    q = select(Site).order_by(Site.created_at.desc())
    if (scope := visible(Site, auth)) is not None:
        q = q.where(scope)
    rows = (await db.execute(q)).scalars().all()
    return [{"id": s.id, "name": s.name, "grid_cell_m": s.grid_cell_m, "placement_count": counts.get(s.id, 0)}
            for s in rows]


@sites.post("", status_code=201)
async def create_site(
    body: SiteIn, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    s = Site(id=new_id("site"), team_id=team_stamp(auth), name=body.name, grid_cell_m=body.grid_cell_m)
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return _site_dict(s)


@sites.get("/{site_id}")
async def get_site(site_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> dict:
    s = await _site_or_404(db, site_id)
    rows = (await db.execute(select(SitePlacement).where(SitePlacement.site_id == site_id)
                             .order_by(SitePlacement.created_at, SitePlacement.id))).scalars().all()
    placements = [await _render(db, s, p) for p in rows if await _world_visible(db, p.world_id, auth)]
    return {**_site_dict(s), "placements": placements, "overlaps": _overlaps(placements),
            "anchor": _anchor_dict(await db.get(SiteAnchor, site_id))}


@sites.delete("/{site_id}", status_code=204)
async def delete_site(site_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    s = await _site_or_404(db, site_id)
    # explicit: SQLite does not enforce ON DELETE CASCADE unless foreign_keys is on
    await db.execute(delete(SitePlacement).where(SitePlacement.site_id == site_id))
    await db.execute(delete(SiteAnchor).where(SiteAnchor.site_id == site_id))
    await db.execute(delete(SiteComposite).where(SiteComposite.site_id == site_id))
    await db.delete(s)
    await db.commit()
    return Response(status_code=204)


@sites.post("/{site_id}/placements", status_code=201)
async def add_placement(
    site_id: str, body: PlacementIn, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    s = await _site_or_404(db, site_id)
    await assert_visible(db, World, body.world_id, auth)
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
                          db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> dict:
    s = await _site_or_404(db, site_id)
    p = await db.get(SitePlacement, placement_id)
    if p is None or p.site_id != site_id or not await _world_visible(db, p.world_id, auth):
        raise HTTPException(404, "Placement not found")
    for k, v in body.model_dump(exclude_unset=True).items():
        if v is None and k != "label":
            raise HTTPException(422, f"{k} cannot be null")
        setattr(p, k, v % 360.0 if k == "rotation_deg" else v)
    await db.commit()
    await db.refresh(p)
    return await _render(db, s, p)


@sites.delete("/{site_id}/placements/{placement_id}", status_code=204)
async def delete_placement(
    site_id: str, placement_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> Response:
    await _site_or_404(db, site_id)
    p = await db.get(SitePlacement, placement_id)
    if p is None or p.site_id != site_id or not await _world_visible(db, p.world_id, auth):
        raise HTTPException(404, "Placement not found")
    await db.delete(p)
    await db.commit()
    return Response(status_code=204)


# ---- anchor -------------------------------------------------------------------------------------------------------

class AnchorIn(BaseModel):
    lat: float = Field(ge=-90, le=90, **_F)
    lon: float = Field(ge=-180, le=180, **_F)
    rotation_deg: float = Field(default=0.0, **_F)


def _anchor_dict(a: SiteAnchor | None) -> dict | None:
    return None if a is None else {"lat": a.lat, "lon": a.lon, "rotation_deg": a.rotation_deg}


@sites.get("/{site_id}/anchor")
async def get_anchor(site_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    await _site_or_404(db, site_id)
    return _anchor_dict(await db.get(SiteAnchor, site_id)) or {"lat": None, "lon": None, "rotation_deg": None}


@sites.put("/{site_id}/anchor")
async def put_anchor(site_id: str, body: AnchorIn, db: AsyncSession = Depends(get_db)) -> dict:
    await _site_or_404(db, site_id)
    rot = body.rotation_deg % 360.0
    a = await db.get(SiteAnchor, site_id)
    if a is None:
        a = SiteAnchor(site_id=site_id, lat=body.lat, lon=body.lon, rotation_deg=rot)
        db.add(a)
    else:
        a.lat, a.lon, a.rotation_deg, a.updated_at = body.lat, body.lon, rot, utcnow()
    await db.commit()
    return _anchor_dict(a)


# ---- compose ------------------------------------------------------------------------------------------------------

class ConnectorIn(BaseModel):
    kind: Literal["driveway", "street", "corridor"]
    from_placement_id: str
    to_placement_id: str
    width_m: float | None = Field(default=None, ge=0.5, le=60, **_F)


class ComposeIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    kind: Literal["house", "neighbourhood"]
    connectors: list[ConnectorIn] = Field(default_factory=list, max_length=50)


_EDGE_MARGIN_M = 0.5


def _edge_point(centre: tuple[float, float], toward: tuple[float, float],
                half: tuple[float, float] | None) -> tuple[float, float]:
    """Where the line from ``centre`` toward ``toward`` leaves the footprint rectangle (half-width, half-depth), plus a
    small margin. Unknown footprint -> the centre itself. ponytail: axis-aligned, ignores the placement's rotation."""
    if half is None:
        return centre
    dx, dz = toward[0] - centre[0], toward[1] - centre[1]
    if dx == 0 and dz == 0:
        return centre
    t = min((half[0] / abs(dx)) if dx else float("inf"), (half[1] / abs(dz)) if dz else float("inf"))
    n = (dx * dx + dz * dz) ** 0.5
    t = min(1.0, t + _EDGE_MARGIN_M / n)
    return centre[0] + dx * t, centre[1] + dz * t


@sites.post("/{site_id}/compose")
async def compose_site(
    site_id: str, body: ComposeIn, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    site = await _site_or_404(db, site_id)
    rows = (await db.execute(select(SitePlacement).where(SitePlacement.site_id == site_id)
                             .order_by(SitePlacement.created_at, SitePlacement.id))).scalars().all()
    if not rows:
        raise HTTPException(422, "Site has no placements to compose")
    placed: dict[str, str] = {}
    sources, centres, halves = [], {}, {}
    for p in rows:
        if p.world_id in placed:
            raise HTTPException(422, f"World '{p.world_id}' is placed twice (placements {placed[p.world_id]} and "
                                     f"{p.id}); compose supports one placement per world")
        placed[p.world_id] = p.id
        w = await db.get(World, p.world_id)
        await assert_visible(db, World, p.world_id, auth)  # a placement may reference a world this caller cannot see
        if w is None or not w.current_version_id:
            raise HTTPException(422, f"World '{p.world_id}' (placement {p.id}) has no head version to compose")
        row = await _version_row(db, p.world_id, None, auth)
        c = _placement_dict(p, site.grid_cell_m, None, None)["center_m"]
        centres[p.id] = (c["x"], c["z"])
        fp = await _world_footprint(db, p.world_id)
        halves[p.id] = (fp[0] * p.scale / 2, fp[1] * p.scale / 2) if fp else None
        sources.append((p.world_id, p.label or w.name, _load_world_or_409(row), c["x"], c["z"], p.rotation_deg,
                        p.scale, row.id))
    try:
        vertical = _up_axis(sources[0][2], f"world '{sources[0][0]}'")
        connectors = []
        for i, c in enumerate(body.connectors):
            if c.from_placement_id not in centres or c.to_placement_id not in centres:
                raise HTTPException(422, f"connector {i}: placement is not on this site")
            a, b = centres[c.from_placement_id], centres[c.to_placement_id]
            # run edge to edge: a centre-to-centre strip would cut through both buildings
            start = _edge_point(a, b, halves[c.from_placement_id])
            end = _edge_point(b, a, halves[c.to_placement_id])
            connectors.append(make_connector(c.kind, start, end, c.width_m,
                                             connector_id=f"gen/{i}-{c.kind}", vertical=vertical))
        world = compose_worlds(sources, name=body.name, kind=body.kind, connectors=connectors)
    except ComposeError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:  # e.g. coincident connector endpoints
        raise HTTPException(422, str(exc))

    comp = await db.get(SiteComposite, (site_id, body.kind, body.name))
    if comp is not None and await db.get(World, comp.world_id) is not None:
        world_id = comp.world_id
    else:
        w = World(id=new_id("wld"), owner_id=auth.user_id, team_id=site.team_id or team_stamp(auth), name=body.name,
                  description=f"Composed {body.kind} from site '{site.name}' ({len(sources)} worlds). Not a capture.")
        db.add(w)
        if comp is None:
            db.add(SiteComposite(site_id=site_id, kind=body.kind, name=body.name, world_id=w.id))
        else:
            comp.world_id = w.id
        await db.commit()
        world_id = w.id
    ver = await worldstore_service.commit_version(db, world_id=world_id, world=world, parent=None)
    return {"world_id": world_id, "version_id": ver.id, "kind": body.kind, "source_count": len(sources),
            "entity_count": len(world.entities),
            "generated_count": sum(1 for e in world.entities.values() if is_generated(e)),
            "warnings": world.metadata.get("compose_warnings", [])}
