"""/api/geo: import real map context (OpenStreetMap via Overpass) for a place as its own world."""

from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import worldstore_service
from apps.api.db import get_db
from apps.api.models import World, new_id
from apps.api.routes_worlds import _worldstore_root
from geo import osm_context as osm

geo = APIRouter(prefix="/api/geo", tags=["geo"])


class GeoContextIn(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    radius_m: float = Field(default=300, ge=50, le=1000)
    name: str | None = Field(default=None, max_length=200)


def _make_client() -> httpx.Client | None:
    """Overridable seam (tests inject a MockTransport client); None = a real client."""
    return None


@geo.post("/context")
async def import_context(body: GeoContextIn, db: AsyncSession = Depends(get_db)) -> dict:
    cache = _worldstore_root() / "osm-cache"
    try:
        data, source = await asyncio.to_thread(
            osm.fetch_overpass, body.lat, body.lon, body.radius_m, client=_make_client(), cache_dir=cache)
    except osm.OverpassRateLimited as exc:
        raise HTTPException(429, str(exc)) from exc
    except osm.OverpassError as exc:
        raise HTTPException(502, str(exc)) from exc
    name = (body.name or "").strip() or f"OSM context {body.lat:.4f}, {body.lon:.4f}"
    elements = data["elements"]
    world = osm.osm_to_world(elements, body.lat, body.lon, name, radius_m=body.radius_m)

    w = (await db.execute(select(World).where(World.name == name))).scalars().first()
    if w is None:
        w = World(id=new_id("wld"), name=name, description=f"{osm.ATTRIBUTION} (ODbL). Imported map context, not a capture.",
                  latitude=body.lat, longitude=body.lon)
        db.add(w)
        await db.commit()
    row = await worldstore_service.commit_version(db, world_id=w.id, world=world, parent=None)
    return {
        "world_id": w.id, "version_id": row.id, "counts_by_kind": osm.counts_by_kind(world),
        "estimated_count": osm.estimated_count(world), "element_count": len(elements),
        "attribution": osm.ATTRIBUTION, "source": source,
    }
