"""Lazy spatial query over a WorldVersionHandle (WORLDOS real-data
city-scale checkpoint, Task 2): wires the lazy tile layer into the SAME
production query engine (`engine.scene_graph.spatial_index.SpatialIndex`)
instead of leaving `WorldVersionHandle.query_region()` as a parallel,
weaker API.

Each function loads only the tile artifacts whose manifest AABB can
possibly contain a matching entity, builds a real (but partial) WorldIR
from just that data, and hands it to a real `SpatialIndex` -- so
`nearest()`/`within_radius()`/`within_region()` keep their exact
semantics (deterministic tie-breaking, real Euclidean distance,
geometry-AABB overlap, coordinate-frame handling) instead of a
re-derived or approximate query path.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from engine.scene_graph.spatial_index import SpatialIndex
from world_ir.coordinates import Frame
from world_ir.schema_v1 import Entity, Geometry
from world_ir.world_v1 import WorldIR

from .tiles import WorldVersionHandle

Point = Tuple[float, float, float]


def _partial_world(handle: WorldVersionHandle, entities: Dict[str, Entity], geometries: Dict[str, Geometry]) -> WorldIR:
    """A real WorldIR containing only the loaded subset -- SpatialIndex
    only ever reads `.entities`/`.geometries`/`.coordinate_frame`, so a
    partial world is a legitimate input, not a stub."""
    return WorldIR(
        id=handle.manifest.world_id,
        entities=entities,
        geometries=geometries,
        coordinate_frame=Frame(handle.manifest.coordinate_frame),
    )


def _index_for_region(handle: WorldVersionHandle, bounds_min: Point, bounds_max: Point) -> SpatialIndex:
    entities, geometries = handle.query_region_with_geometries(bounds_min, bounds_max)
    world = _partial_world(handle, entities, geometries)
    return SpatialIndex(world, chunk_size=handle.manifest.tile_size)


def lazy_within_region(handle: WorldVersionHandle, bounds_min: Point, bounds_max: Point) -> List[Entity]:
    """`SpatialIndex.within_region()`, loading only tiles overlapping
    the queried box."""
    return _index_for_region(handle, bounds_min, bounds_max).within_region(bounds_min, bounds_max)


def lazy_within_radius(
    handle: WorldVersionHandle, point: Point, radius: float, predicate: Optional[Callable[[Entity], bool]] = None,
) -> List[Tuple[Entity, float]]:
    """`SpatialIndex.within_radius()`, loading only tiles that can
    possibly contain a point within `radius` of `point`."""
    box_min = tuple(point[i] - radius for i in range(3))
    box_max = tuple(point[i] + radius for i in range(3))
    return _index_for_region(handle, box_min, box_max).within_radius(point, radius, predicate)


def lazy_nearest(
    handle: WorldVersionHandle, point: Point, k: int = 1, predicate: Optional[Callable[[Entity], bool]] = None,
) -> List[Tuple[Entity, float]]:
    """`SpatialIndex.nearest()`, ring-expanding the loaded region (in
    tile_size steps) until `k` results are found AND provably correct --
    same termination proof `SpatialIndex.nearest()` itself uses: the
    k-th candidate's distance is within the searched radius, or the
    search box already covers every tile in the manifest."""
    if k <= 0 or not handle.manifest.tiles:
        return []
    tile_size = handle.manifest.tile_size
    world_min, world_max = handle._overall_bounds()
    radius = tile_size
    while True:
        box_min = tuple(point[i] - radius for i in range(3))
        box_max = tuple(point[i] + radius for i in range(3))
        covers_world = all(box_min[i] <= world_min[i] and box_max[i] >= world_max[i] for i in range(3))
        index = _index_for_region(handle, box_min, box_max)
        results = index.nearest(point, k=k, predicate=predicate)
        if covers_world:
            return results
        if len(results) >= k and results[k - 1][1] <= radius:
            return results
        radius *= 2.0
