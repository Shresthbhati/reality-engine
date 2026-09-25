"""Routes: spatial/scene-graph queries (Phase 7 of the golden loop).

Thin HTTP wrappers around the real, already-tested query engines --
sdk.reality.spatial_index() (grid-based nearest/within_radius/within_region,
engine/scene_graph/spatial_index.py) and sdk.reality.scene_graph()
(containment/relationship queries, engine/scene_graph/graph.py). No query
logic lives here; this module only parses HTTP params, builds the index
from the world's current WorldIR, and serializes real results.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import get_db
from apps.api.routes_worlds import _load_world_or_409, _version_row
from sdk import reality
from world_ir.schema_v1 import EntityType

query = APIRouter(prefix="/api/worlds/{world_id}/query", tags=["query"])


def _type_predicate(type_filter: str | None):
    if not type_filter:
        return None
    try:
        wanted = EntityType(type_filter)
    except ValueError as exc:
        raise HTTPException(422, f"Unknown entity type '{type_filter}'") from exc
    return lambda entity: entity.type == wanted


@query.get("/nearest")
async def query_nearest(
    world_id: str,
    x: float, y: float, z: float,
    k: int = Query(5, ge=1, le=100),
    type: str | None = None,
    version: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """The k nearest indexed entities to (x, y, z), real Euclidean distance."""
    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)
    index = reality.spatial_index(world)
    predicate = _type_predicate(type)
    results = index.nearest((x, y, z), k=k, predicate=predicate)
    return {
        "version_id": row.id,
        "results": [
            {**entity.to_dict(), "distance_m": distance}
            for entity, distance in results
        ],
    }


@query.get("/within_radius")
async def query_within_radius(
    world_id: str,
    x: float, y: float, z: float,
    radius: float = Query(..., ge=0),
    type: str | None = None,
    version: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Every indexed entity within `radius` meters of (x, y, z)."""
    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)
    index = reality.spatial_index(world)
    predicate = _type_predicate(type)
    results = index.within_radius((x, y, z), radius=radius, predicate=predicate)
    return {
        "version_id": row.id,
        "results": [
            {**entity.to_dict(), "distance_m": distance}
            for entity, distance in results
        ],
    }


@query.get("/contents/{entity_id}")
async def query_contents(
    world_id: str, entity_id: str, version: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Entities the given entity CONTAINS (e.g. a room's furniture)."""
    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)
    if entity_id not in world.entities:
        raise HTTPException(404, f"Entity '{entity_id}' not found in version '{row.id}'")
    graph = reality.scene_graph(world)
    contents = graph.contents_of(entity_id)
    return {
        "version_id": row.id,
        "entity_id": entity_id,
        "contents": [e.to_dict() for e in contents],
    }


@query.get("/container/{entity_id}")
async def query_container(
    world_id: str, entity_id: str, version: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """The entity that CONTAINS the given entity, if any."""
    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)
    if entity_id not in world.entities:
        raise HTTPException(404, f"Entity '{entity_id}' not found in version '{row.id}'")
    graph = reality.scene_graph(world)
    container = graph.container_of(entity_id)
    return {
        "version_id": row.id,
        "entity_id": entity_id,
        "container": container.to_dict() if container is not None else None,
    }
