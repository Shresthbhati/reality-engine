"""Geometric spatial index/query engine over WorldIR (scale campaign,
Phases 31/32/33: Query Engine, Spatial Indexing, World Search).

`engine/scene_graph/graph.py` (`SceneGraph`) already answers relationship
questions ("what's inside Room 4?") by walking `Relationship` edges.
Nothing in the repo before this module could answer a *geometric*
question -- "what's near this point", "what's inside this region",
"what's the closest entity to X" -- because nothing indexed entity
positions at all. This module is that missing half.

Position resolution per entity (in priority order, first available wins):
  1. `entity.transform["position"]` -- an explicit placement (what every
     exporter already reads).
  2. The centroid of the first geometry with real `bounds_min`/
     `bounds_max` -- the same "real, non-fabricated data" the physics
     compiler and Blender exporter already use.
An entity with neither is UNLOCALIZED and is excluded from the index
with an explicit reason recorded, never silently dropped or given a
fabricated (0, 0, 0) position.

Index structure (city-scale campaign, "select based on actual workload
and benchmark the choice"): this used to be a flat list scanned in full
on every query -- a real, correct, but O(n) implementation, deliberately
left that way until a real workload existed to benchmark against. That
workload now exists (`benchmarks/spatial_index_bench.py` has the
measured numbers), so `SpatialIndex` now buckets entities into the same
uniform chunk grid `SpatialTiling` (spatial_tiling.py) already
implements for region queries, and reuses that module directly for
`within_region` rather than re-deriving the chunking logic. `nearest`
and `within_radius` use their own position-only grid (see
`_position_chunks`) since -- like the old flat-scan versions -- they
only ever cared about entity position, not geometry bounds.
`chunk_size` is auto-picked from the indexed positions' spread (see
`_auto_chunk_size`) unless the caller has a better estimate for their
world's scale.

All position/vector arithmetic here is plain Python floats -- no numpy,
consistent with world_ir/coordinates.py's stated reasoning that a
foundation-layer module this small isn't worth a heavy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR

from .spatial_common import (
    Bounds, ChunkKey, Point, _aabb_overlaps, _chunk_key, _chunk_range,
    _chunks_in_box, _distance_sq, _entity_bounds, _point_in_aabb, entity_position,
)
from .spatial_tiling import SpatialTiling

#: Fallback grid resolution when there's nothing to derive one from (an
#: empty world, or a single entity) -- arbitrary but harmless, since an
#: empty/singleton grid degenerates to visiting one chunk regardless.
_DEFAULT_CHUNK_SIZE = 10.0


def _auto_chunk_size(positions: List[Point]) -> float:
    """Aim for roughly one entity per grid cell: chunk_size = the
    bounding box's largest axis span divided by cbrt(entity count). A
    world with zero spread (one entity, or many at the same point) falls
    back to `_DEFAULT_CHUNK_SIZE` rather than dividing by zero."""
    if not positions:
        return _DEFAULT_CHUNK_SIZE
    mins = [min(p[axis] for p in positions) for axis in range(3)]
    maxs = [max(p[axis] for p in positions) for axis in range(3)]
    span = max(maxs[axis] - mins[axis] for axis in range(3))
    if span <= 0.0:
        return _DEFAULT_CHUNK_SIZE
    cube_root_n = len(positions) ** (1.0 / 3.0)
    return max(span / cube_root_n, 1e-6)


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


class SpatialIndex:
    """Built once from a WorldIR snapshot. Does not observe later
    mutations to `world` -- rebuild after edits, same contract as
    `SceneGraph` (also a plain snapshot wrapper).

    `chunk_size` overrides the auto-picked grid resolution (see
    `_auto_chunk_size`) -- pass one when the caller already knows the
    world's rough scale (e.g. "this is a city, use ~50m cells")."""

    def __init__(self, world: WorldIR, chunk_size: Optional[float] = None):
        self.world = world
        indexed: List[IndexedEntity] = []
        unlocalized: List[UnlocalizedEntity] = []
        for entity_id in sorted(world.entities):
            entity = world.entities[entity_id]
            position = entity_position(world, entity)
            if position is None:
                unlocalized.append(UnlocalizedEntity(
                    entity_id=entity_id,
                    reason="no transform.position and no geometry with real bounds_min/bounds_max",
                ))
                continue
            indexed.append(IndexedEntity(
                entity_id=entity_id, position=position, bounds=_entity_bounds(world, entity),
            ))
        # Deterministic order: sort by (x, y, z, entity_id) so query
        # results with tied distances break ties the same way every run.
        indexed.sort(key=lambda e: (e.position[0], e.position[1], e.position[2], e.entity_id))
        self._entities: Tuple[IndexedEntity, ...] = tuple(indexed)
        self._unlocalized: Tuple[UnlocalizedEntity, ...] = tuple(unlocalized)

        self.chunk_size = chunk_size if chunk_size is not None else _auto_chunk_size(
            [e.position for e in indexed]
        )
        # Position-only grid for nearest()/within_radius() -- one home
        # chunk per entity, same as the old flat scan's disregard for
        # geometry bounds on those two queries.
        self._position_chunks: Dict[ChunkKey, List[IndexedEntity]] = {}
        for e in indexed:
            self._position_chunks.setdefault(_chunk_key(e.position, self.chunk_size), []).append(e)
        # Bounds-aware grid (entities indexed under every chunk their
        # AABB touches) for within_region() -- reuse SpatialTiling rather
        # than re-deriving that logic here.
        self._tiling = SpatialTiling(world, self.chunk_size)

    @property
    def unlocalized(self) -> Tuple[UnlocalizedEntity, ...]:
        """Entities that could not be placed in the index -- surfaced
        explicitly rather than silently omitted (Phase 27 "world
        completeness" vocabulary: an unlocalized entity is a completeness
        gap, not a non-existent one)."""
        return self._unlocalized

    def __len__(self) -> int:
        return len(self._entities)

    def nearest(
        self, point: Point, k: int = 1, predicate: Optional[Callable[[Entity], bool]] = None,
    ) -> List[Tuple[Entity, float]]:
        """The `k` closest indexed entities to `point`, nearest first, as
        (Entity, distance) pairs. `distance` is real Euclidean distance
        (not squared) so callers get a directly usable unit value.
        `predicate` filters by the resolved Entity (semantic class,
        confidence, provenance, etc.) before ranking.

        Implementation: expands a search box outward in grid-chunk rings
        until it has `k` candidates AND the k-th candidate's distance is
        provably within the searched box (or every indexed chunk has been
        visited, whichever comes first) -- standard grid-based kNN, exact
        (not approximate)."""
        if k <= 0 or not self._entities:
            return []
        candidates: Dict[str, Tuple[Entity, float]] = {}
        visited: Set[ChunkKey] = set()
        radius = self.chunk_size
        # The chunk range walked each iteration includes empty cells (no
        # entity happens to live there), so "visited >= chunk count" is
        # not a valid stopping signal -- only "every chunk that actually
        # HAS entities has been visited" proves nothing closer remains.
        all_occupied_chunks = self._position_chunks.keys()
        while True:
            box_min = (point[0] - radius, point[1] - radius, point[2] - radius)
            box_max = (point[0] + radius, point[1] + radius, point[2] + radius)
            for chunk in _chunks_in_box(all_occupied_chunks, box_min, box_max, self.chunk_size):
                if chunk in visited:
                    continue
                visited.add(chunk)
                for indexed in self._position_chunks.get(chunk, ()):
                    entity = self.world.entities[indexed.entity_id]
                    if predicate is not None and not predicate(entity):
                        continue
                    candidates[indexed.entity_id] = (entity, _distance_sq(point, indexed.position))
            covers_world = all(chunk in visited for chunk in all_occupied_chunks)
            if len(candidates) >= k or covers_world:
                ranked = sorted(candidates.values(), key=lambda pair: (pair[1], pair[0].id))
                if covers_world:
                    # Every entity that could ever match has been seen --
                    # `ranked` (whether shorter than, equal to, or longer
                    # than k) is final, no radius check needed.
                    return [(e, d ** 0.5) for e, d in ranked[:k]]
                # len(ranked) >= k here (that's how we got past the `if`
                # above without covers_world), so ranked[k - 1] exists --
                # but until covers_world, having exactly/more than k
                # candidates is NOT proof they're the k *nearest*: an
                # unvisited chunk closer than the k-th candidate found so
                # far could still exist, hence the radius check below.
                kth_dist = ranked[k - 1][1] ** 0.5
                if kth_dist <= radius:
                    return [(e, d ** 0.5) for e, d in ranked[:k]]
            radius *= 2.0

    def within_radius(
        self, point: Point, radius: float, predicate: Optional[Callable[[Entity], bool]] = None,
    ) -> List[Tuple[Entity, float]]:
        """Every indexed entity within `radius` of `point` (inclusive),
        sorted nearest first. Same distance semantics as `nearest()`.
        Only visits grid chunks that can contain a point within `radius`
        of `point`, instead of every indexed entity."""
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        radius_sq = radius * radius
        box_min = (point[0] - radius, point[1] - radius, point[2] - radius)
        box_max = (point[0] + radius, point[1] + radius, point[2] + radius)
        results = []
        for chunk in _chunks_in_box(self._position_chunks.keys(), box_min, box_max, self.chunk_size):
            for indexed in self._position_chunks.get(chunk, ()):
                dist_sq = _distance_sq(point, indexed.position)
                if dist_sq > radius_sq:
                    continue
                entity = self.world.entities[indexed.entity_id]
                if predicate is not None and not predicate(entity):
                    continue
                results.append((entity, dist_sq ** 0.5))
        results.sort(key=lambda pair: (pair[1], pair[0].id))
        return results

    def within_region(self, bounds_min: Point, bounds_max: Point) -> List[Entity]:
        """Entities whose position falls inside the AABB [bounds_min,
        bounds_max], sorted by entity id. An entity with real geometry
        bounds also counts if its AABB *overlaps* the region even when
        its centroid falls outside -- a large wall spanning a room
        boundary should still show up in a region query for either side.
        Delegates to `SpatialTiling.query_region`, which indexes each
        entity under every grid chunk its AABB touches."""
        return self._tiling.query_region(bounds_min, bounds_max)
