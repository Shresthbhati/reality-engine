"""Base types and utilities for spatial indexing (shared between flat and accelerated implementations)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR

__all__ = [
    "Point",
    "Bounds",
    "IndexedEntity",
    "UnlocalizedEntity",
    "entity_position",
    "_entity_bounds",
    "_distance_sq",
    "_aabb_overlaps",
    "_point_in_aabb",
]

Point = Tuple[float, float, float]
Bounds = Tuple[Point, Point]


def _distance_sq(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _aabb_overlaps(a: Bounds, b: Bounds) -> bool:
    (a_min, a_max), (b_min, b_max) = a, b
    return all(a_min[i] <= b_max[i] and b_min[i] <= a_max[i] for i in range(3))


def _point_in_aabb(point: Point, bounds: Bounds) -> bool:
    lo, hi = bounds
    return all(lo[i] <= point[i] <= hi[i] for i in range(3))


@dataclass(frozen=True)
class IndexedEntity:
    entity_id: str
    position: Point
    #: None when the entity has a position but no real geometry bounds
    #: (e.g. transform-only placement with no AABB) -- region/overlap
    #: queries treat such an entity as a zero-volume point.
    bounds: Optional[Bounds]


@dataclass(frozen=True)
class UnlocalizedEntity:
    entity_id: str
    reason: str


def entity_position(world: WorldIR, entity: Entity) -> Optional[Point]:
    if entity.transform and "position" in entity.transform:
        p = entity.transform["position"]
        return (float(p["x"]), float(p["y"]), float(p["z"]))
    for gid in entity.geometry_ids:
        geom = world.geometries.get(gid)
        if geom is not None and geom.bounds_min is not None and geom.bounds_max is not None:
            return (
                (geom.bounds_min.x + geom.bounds_max.x) / 2.0,
                (geom.bounds_min.y + geom.bounds_max.y) / 2.0,
                (geom.bounds_min.z + geom.bounds_max.z) / 2.0,
            )
    return None


def _entity_bounds(world: WorldIR, entity: Entity) -> Optional[Bounds]:
    for gid in entity.geometry_ids:
        geom = world.geometries.get(gid)
        if geom is not None and geom.bounds_min is not None and geom.bounds_max is not None:
            return (
                (geom.bounds_min.x, geom.bounds_min.y, geom.bounds_min.z),
                (geom.bounds_max.x, geom.bounds_max.y, geom.bounds_max.z),
            )
    return None