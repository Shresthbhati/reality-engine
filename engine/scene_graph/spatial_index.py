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

Index structure: a flat list, sorted once at build time by (x, entity_id)
for deterministic iteration and tie-breaking. This is a real, correct,
O(n) implementation -- not an octree/R-tree/BVH. Phase 32 of the scale
campaign explicitly says to "select based on actual workload" and
"benchmark the choice"; with no real measured workload yet (no caller in
this repo queries thousands of entities today), building a tree
structure now would be exactly the premature, unbenchmarked optimization
the project conventions warn against. The complexity ceiling is
documented here and in `.agent/CURRENT_STATE.md` so it is a deliberate,
named tradeoff, not an oversight: replace the flat scan with an octree/
grid-hash once a real caller needs sub-linear query time at real scale.

All position/vector arithmetic here is plain Python floats -- no numpy,
consistent with world_ir/coordinates.py's stated reasoning that a
foundation-layer module this small isn't worth a heavy dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR

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


class SpatialIndex:
    """Built once from a WorldIR snapshot. Does not observe later
    mutations to `world` -- rebuild after edits, same contract as
    `SceneGraph` (also a plain snapshot wrapper)."""

    def __init__(self, world: WorldIR):
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
        confidence, provenance, etc.) before ranking."""
        if k <= 0:
            return []
        candidates = []
        for indexed in self._entities:
            entity = self.world.entities[indexed.entity_id]
            if predicate is not None and not predicate(entity):
                continue
            candidates.append((entity, _distance_sq(point, indexed.position)))
        candidates.sort(key=lambda pair: (pair[1], pair[0].id))
        return [(entity, dist_sq ** 0.5) for entity, dist_sq in candidates[:k]]

    def within_radius(
        self, point: Point, radius: float, predicate: Optional[Callable[[Entity], bool]] = None,
    ) -> List[Tuple[Entity, float]]:
        """Every indexed entity within `radius` of `point` (inclusive),
        sorted nearest first. Same distance semantics as `nearest()`."""
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        radius_sq = radius * radius
        results = []
        for indexed in self._entities:
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
        boundary should still show up in a region query for either side."""
        region: Bounds = (bounds_min, bounds_max)
        results = []
        for indexed in self._entities:
            hit = _point_in_aabb(indexed.position, region)
            if not hit and indexed.bounds is not None:
                hit = _aabb_overlaps(indexed.bounds, region)
            if hit:
                results.append(self.world.entities[indexed.entity_id])
        results.sort(key=lambda e: e.id)
        return results
