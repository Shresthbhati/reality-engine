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
from apps.api.tenancy import world_scope
from apps.api.routes_worlds import _load_world_or_409, _version_row
from sdk import reality
from world_ir.schema_v1 import EntityType

query = APIRouter(prefix="/api/worlds/{world_id}/query", tags=["query"], dependencies=[Depends(world_scope)])


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


_CHAIN_KINDS = ("part_of", "inside", "on", "attached_to")


def _norm(s: str) -> str:
    return " ".join(str(s).lower().replace("_", " ").replace("/", " ").split())


def find_entities(world, q: str) -> list[dict]:
    """Entities whose type, label, ontology alias or attribute value matches ``q``, each with its containment chain
    (via part_of / inside / on / attached_to), 3D position and basis. Unknown attributes (value None) never match."""
    from engine.pipeline.facade import ontology_path_for

    needle = _norm(q)
    if not needle:
        return []
    path = ontology_path_for(q)
    wanted_path = None if path.startswith("fallback/") else path
    out = []
    for e in world.entities.values():
        attrs = (e.extra_fields or {}).get("attributes") or {}
        hit_attr = next((n for n, a in attrs.items() if a.get("value") and needle in _norm(a["value"])), None)
        user = next((a for a in attrs.values() if a.get("basis") == "user_stated" and a.get("value") and needle in _norm(a["value"])), None)
        hit = (wanted_path is not None and e.type == wanted_path) or needle in _norm(e.type) \
            or any(needle in _norm(lbl) for lbl in e.semantic_labels) or hit_attr is not None
        if not hit:
            continue
        chain, cur, seen = [], e, {e.id}
        while True:
            nxt = next((r for r in cur.relationships if r.kind in _CHAIN_KINDS and r.target_id in world.entities
                        and r.target_id not in seen), None)
            if nxt is None:
                break
            cur = world.entities[nxt.target_id]; seen.add(cur.id)
            chain.append({"id": cur.id, "type": cur.type, "via": nxt.kind})
        pos = None
        if e.transform is not None:
            m = e.transform.matrix
            pos = [m[0][3], m[1][3], m[2][3]]
        out.append({"id": e.id, "type": e.type, "chain": chain, "position": pos,
                    "basis": "user_stated" if user else "visual",
                    "interior": (e.extra_fields or {}).get("interior")})
    return out


@query.get("/find")
async def query_find(
    world_id: str, q: str = Query(..., min_length=1), version: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """\"Where is X\": entities matching ``q`` with their room -> furniture -> part chain and position."""
    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)
    return {"version_id": row.id, "q": q, "results": find_entities(world, q)}


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
