"""Spatial tiling / chunk index (P13-01: "spatial cells" half of Large
World support -- docs/future/large-world/LARGE_WORLD.md item 1,
"Spatial partitioning").

Scope, deliberately narrow: this module buckets WorldIR entities into a
uniform grid of chunks in world-frame coordinates and answers region
queries by only scanning the chunks a region actually overlaps, instead of
every entity in the world (`SpatialIndex.within_region`'s documented O(n)
flat scan -- see spatial_index.py's module docstring, which names exactly
this "grid-hash" as the upgrade path once a real caller needs sub-linear
query time).

Explicitly OUT of scope here (per LARGE_WORLD.md, correctly still MISSING):
chunk-based mesh compilation/stitching, LOD chains, and streaming -- those
require real mesh-boundary welding logic this module has no way to verify
correctly without a meshing pipeline to test against. Chunking WHAT gets
indexed (entities) is safe and additive; chunking HOW geometry compiles is
not something to improvise.

Not persisted in WorldIR: like `SpatialIndex`, this is a derived index
built fresh from a snapshot, not a schema field. LARGE_WORLD.md's
"WorldIR gains a partitioning metadata section" is a later step once a
persisted chunk assignment is actually needed by a consumer (e.g. a
streaming viewer) -- premature to add an unused schema field now.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR

from .spatial_index import Bounds, Point, _aabb_overlaps, _entity_bounds, _point_in_aabb, entity_position

ChunkKey = Tuple[int, int, int]


def _chunk_key(point: Point, chunk_size: float) -> ChunkKey:
    return (
        int(point[0] // chunk_size),
        int(point[1] // chunk_size),
        int(point[2] // chunk_size),
    )


def _chunk_range(bounds_min: Point, bounds_max: Point, chunk_size: float) -> List[ChunkKey]:
    """Every chunk key whose cell overlaps [bounds_min, bounds_max]."""
    lo = _chunk_key(bounds_min, chunk_size)
    hi = _chunk_key(bounds_max, chunk_size)
    keys = []
    for ix in range(lo[0], hi[0] + 1):
        for iy in range(lo[1], hi[1] + 1):
            for iz in range(lo[2], hi[2] + 1):
                keys.append((ix, iy, iz))
    return keys


@dataclass(frozen=True)
class TiledEntity:
    entity_id: str
    chunk: ChunkKey
    position: Point


class SpatialTiling:
    """Built once from a WorldIR snapshot, like SpatialIndex/SceneGraph.
    Does not observe later mutations -- rebuild after edits.

    `chunk_size` is in world-frame units (the same units as
    Entity.transform positions / Geometry bounds -- see
    world_ir/coordinates.py for the scale/frame contract). Choose it to
    roughly match one "room/building" scale for city-sized worlds; too
    small over-fragments, too large degenerates back to a flat scan.
    """

    def __init__(self, world: WorldIR, chunk_size: float):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        self.world = world
        self.chunk_size = chunk_size
        self._chunks: Dict[ChunkKey, List[TiledEntity]] = {}
        #: entity_id -> its PRIMARY chunk (the centroid's chunk) -- used by
        #: chunk_of()/entities_in_chunk(); an entity whose geometry spans
        #: multiple chunks is still membership-indexed into all of them
        #: below so query_region() finds it, but has exactly one "home".
        self._entity_chunk: Dict[str, ChunkKey] = {}
        for entity_id in sorted(world.entities):
            entity = world.entities[entity_id]
            position = entity_position(world, entity)
            if position is None:
                continue  # unlocalized -- same exclusion rule as SpatialIndex
            home_key = _chunk_key(position, chunk_size)
            self._entity_chunk[entity_id] = home_key

            tiled = TiledEntity(entity_id=entity_id, chunk=home_key, position=position)
            bounds = _entity_bounds(world, entity)
            if bounds is None:
                self._chunks.setdefault(home_key, []).append(tiled)
            else:
                # Index under EVERY chunk the entity's real geometry AABB
                # touches, not just its centroid's -- a wall spanning a
                # chunk boundary (LARGE_WORLD.md's named failure mode)
                # must be found by a region query that overlaps its
                # geometry even when the region misses its centroid.
                for key in _chunk_range(bounds[0], bounds[1], chunk_size):
                    self._chunks.setdefault(key, []).append(tiled)

    def __len__(self) -> int:
        return len(self._entity_chunk)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def chunk_of(self, entity_id: str) -> Optional[ChunkKey]:
        return self._entity_chunk.get(entity_id)

    def entities_in_chunk(self, chunk: ChunkKey) -> List[Entity]:
        return [self.world.entities[te.entity_id] for te in self._chunks.get(chunk, ())]

    def query_region(self, bounds_min: Point, bounds_max: Point) -> List[Entity]:
        """Same semantics as SpatialIndex.within_region (position-in-AABB,
        or geometry-AABB-overlap for entities with real bounds), but only
        scans entities in chunks overlapping the region instead of every
        indexed entity -- the sub-linear behavior SpatialIndex's flat scan
        deliberately defers."""
        region: Bounds = (bounds_min, bounds_max)
        results = []
        seen = set()
        for chunk in _chunk_range(bounds_min, bounds_max, self.chunk_size):
            for tiled in self._chunks.get(chunk, ()):
                if tiled.entity_id in seen:
                    continue
                entity = self.world.entities[tiled.entity_id]
                hit = _point_in_aabb(tiled.position, region)
                if not hit:
                    bounds = _entity_bounds(self.world, entity)
                    if bounds is not None:
                        hit = _aabb_overlaps(bounds, region)
                if hit:
                    seen.add(tiled.entity_id)
                    results.append(entity)
        results.sort(key=lambda e: e.id)
        return results
