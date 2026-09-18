"""Shared geometric/grid primitives for the spatial-query layer.

Split out of `spatial_index.py` when `SpatialIndex` was upgraded to use
the same chunk-grid bucketing `spatial_tiling.py` already implemented
(city-scale campaign, spatial index architecture item): `spatial_index.py`
now builds an internal `SpatialTiling` for its `within_region`, so the two
modules can no longer import from each other without a cycle. Everything
either one needs is here instead.
"""

from __future__ import annotations

from typing import Optional, Tuple

from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR

Point = Tuple[float, float, float]
Bounds = Tuple[Point, Point]
ChunkKey = Tuple[int, int, int]


def _distance_sq(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


def _aabb_overlaps(a: Bounds, b: Bounds) -> bool:
    (a_min, a_max), (b_min, b_max) = a, b
    return all(a_min[i] <= b_max[i] and b_min[i] <= a_max[i] for i in range(3))


def _point_in_aabb(point: Point, bounds: Bounds) -> bool:
    lo, hi = bounds
    return all(lo[i] <= point[i] <= hi[i] for i in range(3))


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


def _chunk_key(point: Point, chunk_size: float) -> ChunkKey:
    return (
        int(point[0] // chunk_size),
        int(point[1] // chunk_size),
        int(point[2] // chunk_size),
    )


def _chunk_range(bounds_min: Point, bounds_max: Point, chunk_size: float):
    """Every chunk key whose cell overlaps [bounds_min, bounds_max].

    Only safe to call when the caller controls both the box AND the
    chunk_size together (e.g. indexing one entity's own AABB into the
    chunks it touches, where the box is that entity's own small extent).
    NEVER call this with a query-time box whose size is independent of
    chunk_size -- a fine grid (small chunk_size) queried with a large
    box enumerates every cell in the box combinatorially, most of them
    empty, which is an O((box_size/chunk_size)^3) blowup regardless of
    how few entities are actually indexed. `_chunks_in_box` below is the
    query-safe alternative: bounded by the number of *populated* chunks,
    never by the query box's volume.
    """
    lo = _chunk_key(bounds_min, chunk_size)
    hi = _chunk_key(bounds_max, chunk_size)
    keys = []
    for ix in range(lo[0], hi[0] + 1):
        for iy in range(lo[1], hi[1] + 1):
            for iz in range(lo[2], hi[2] + 1):
                keys.append((ix, iy, iz))
    return keys


def _chunks_in_box(populated_keys, bounds_min: Point, bounds_max: Point, chunk_size: float):
    """Every key in `populated_keys` (an iterable of ChunkKey, typically
    a chunk-map's `.keys()`) whose cell overlaps [bounds_min, bounds_max].

    Bounded by len(populated_keys), not by the box's volume -- the
    query-time counterpart to `_chunk_range` (see its docstring for why
    that one is unsafe here): a query box can be arbitrarily large
    relative to chunk_size without blowing up, since only chunks that
    actually contain entities are ever considered.
    """
    lo = _chunk_key(bounds_min, chunk_size)
    hi = _chunk_key(bounds_max, chunk_size)
    for key in populated_keys:
        if lo[0] <= key[0] <= hi[0] and lo[1] <= key[1] <= hi[1] and lo[2] <= key[2] <= hi[2]:
            yield key
