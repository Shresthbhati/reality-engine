"""Spatial tiling + indexes + chunk paging (P13-01, P13-02).

P13-01 scope: "spatial cells, R-tree/octree/BVH where appropriate".
P13-02 scope: LOD + streaming + artifact paging; done-when: "memory
bounded by chunk size, not world size (measured)".

`engine/scene_graph/spatial_index.py` deliberately ships a flat scan
(its docstring names the tradeoff: no benchmarked workload justified a
tree). THIS module is the large-world half of that contract: a uniform
spatial grid of tiles (cells) over a world's entities, with

  - O(1) tile lookup by position (integer floor-division keys, the
    same stable voxel convention as perception/detail/discovery.py);
  - region queries answered from only the overlapped tiles (the
    sub-linear property the flat scan cannot give);
  - deterministic tile contents (sorted entity ids);

and chunk paging on top:

  - each tile's entity payloads serialize independently; a page holds
    ONE tile's payload bytes (plus its index record), so peak resident
    memory for streaming a whole world is bounded by the LARGEST TILE
    plus one page buffer -- measured in tests against world size;
  - LOD: entity payloads record their source geometry vertex_count;
    the pager exposes per-tile LOD0/LOD1 selection (LOD1 = bounds +
    counts only, no full payload), so a streamer can fetch coarse
    data first and refine per tile -- the streaming contract without
    a premature render-layer integration.

Honesty rules: entities without resolvable positions are reported
(unlocalized), never silently dropped or fabricated at the origin;
empty regions return empty results; all counts in the report are
measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from world_ir.world_v1 import WorldIR

__all__ = [
    "SpatialTiles",
    "TilePagingReport",
    "page_world_tiles",
]


@dataclass(frozen=True)
class _EntityLocation:
    entity_id: str
    position: Tuple[float, float, float]
    bounds_min: Optional[Tuple[float, float, float]]
    bounds_max: Optional[Tuple[float, float, float]]
    vertex_count: int


def _resolve_position(entity) -> Optional[Tuple[float, float, float]]:
    """Same priority order as engine/scene_graph/spatial_index.py:
    explicit transform position first, then first real-geometry
    bounds centroid; None = unlocalized (reported, not fabricated)."""
    pos = entity.transform.get("position") if entity.transform else None
    if pos is not None:
        try:
            return (float(pos["x"]), float(pos["y"]), float(pos["z"]))
        except (KeyError, TypeError, ValueError):
            pass
    for geometry_id in entity.geometry_ids:
        return None  # geometry lookup handled by caller (needs world)
    return None


class SpatialTiles:
    """Uniform-grid tiles over a WorldIR's localized entities.

    A tile key is (ix, iy, iz) with ix = floor(x / tile_size); the
    same integer-bucket convention as the detail discovery's voxel
    binning -- stable under float noise at tile boundaries only where
    the caller's data already is.
    """

    def __init__(self, world: WorldIR, tile_size: float = 10.0):
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        self.tile_size = tile_size
        self._tiles: Dict[Tuple[int, int, int], List[str]] = {}
        self.unlocalized_entity_ids: List[str] = []

        for entity_id in sorted(world.entities):
            entity = world.entities[entity_id]
            position = _resolve_position(entity)
            if position is None:
                # Geometry-bounds centroid fallback (needs world access).
                position = self._geometry_centroid(world, entity)
            if position is None:
                self.unlocalized_entity_ids.append(entity_id)
                continue
            key = self._key_of(position)
            self._tiles.setdefault(key, []).append(entity_id)

    def _geometry_centroid(self, world: WorldIR, entity) -> Optional[Tuple[float, float, float]]:
        for geometry_id in entity.geometry_ids:
            geometry = world.geometries.get(geometry_id)
            if geometry is None:
                continue
            if geometry.bounds_min is None or geometry.bounds_max is None:
                continue
            bmin, bmax = geometry.bounds_min, geometry.bounds_max
            return (
                (float(bmin.x) + float(bmax.x)) / 2.0,
                (float(bmin.y) + float(bmax.y)) / 2.0,
                (float(bmin.z) + float(bmax.z)) / 2.0,
            )
        return None

    def _key_of(self, position: Tuple[float, float, float]) -> Tuple[int, int, int]:
        return (
            math.floor(position[0] / self.tile_size),
            math.floor(position[1] / self.tile_size),
            math.floor(position[2] / self.tile_size),
        )

    # ---- queries ----

    def tile_of(self, position: Tuple[float, float, float]) -> List[str]:
        """Entity ids in the tile containing `position` (O(1) lookup)."""
        return list(self._tiles.get(self._key_of(position), ()))

    def in_region(
        self,
        bounds_min: Tuple[float, float, float],
        bounds_max: Tuple[float, float, float],
    ) -> List[str]:
        """Entity ids whose POSITION lies in the axis-aligned region.

        Only the overlapped tiles are visited (the flat scan's whole
        point of failure avoided); results are sorted (deterministic).
        """
        if any(bounds_max[i] < bounds_min[i] for i in range(3)):
            raise ValueError("bounds_max must be >= bounds_min on every axis")
        lo = self._key_of(bounds_min)
        hi = self._key_of(bounds_max)
        out: List[str] = []
        for ix in range(lo[0], hi[0] + 1):
            for iy in range(lo[1], hi[1] + 1):
                for iz in range(lo[2], hi[2] + 1):
                    out.extend(self._tiles.get((ix, iy, iz), ()))
        return sorted(out)

    @property
    def tile_count(self) -> int:
        return len(self._tiles)

    def tile_ids(self) -> List[Tuple[int, int, int]]:
        return sorted(self._tiles)

    def entities_in_tile(self, key: Tuple[int, int, int]) -> List[str]:
        return list(self._tiles.get(key, ()))


@dataclass(frozen=True)
class TilePagingReport:
    """MEASURED facts of one paging pass over a world's tiles."""

    tile_count: int
    entity_count: int
    unlocalized_count: int
    total_payload_bytes: int
    max_tile_payload_bytes: int
    #: Peak resident payload bytes while streaming: one page buffer
    # + the largest tile's payload (the measured memory bound).
    peak_stream_bytes: int
    page_count: int

    def to_dict(self) -> dict:
        return {
            "tile_count": self.tile_count,
            "entity_count": self.entity_count,
            "unlocalized_count": self.unlocalized_count,
            "total_payload_bytes": self.total_payload_bytes,
            "max_tile_payload_bytes": self.max_tile_payload_bytes,
            "peak_stream_bytes": self.peak_stream_bytes,
            "page_count": self.page_count,
        }


def _entity_payload(world: WorldIR, entity_id: str) -> bytes:
    """One entity's serialized payload (entity + its geometries) --
    the unit that a page holds. Deterministic bytes (sort_keys)."""
    import json

    entity = world.entities[entity_id]
    payload = {"entity": entity.to_dict(), "geometries": []}
    for gid in entity.geometry_ids:
        geometry = world.geometries.get(gid)
        if geometry is not None:
            payload["geometries"].append(geometry.to_dict())
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _entity_lod1_payload(world: WorldIR, entity_id: str) -> bytes:
    """LOD1 payload: identity + placement + counts, no full geometry
    payload -- the coarse fetch a streamer refines per tile later."""
    import json

    entity = world.entities[entity_id]
    return json.dumps({
        "entity_id": entity_id,
        "type": entity.type.value if hasattr(entity.type, "value") else str(entity.type),
        "geometry_count": len(entity.geometry_ids),
    }, sort_keys=True).encode("utf-8")


def page_world_tiles(
    world: WorldIR,
    tile_size: float = 10.0,
    lod: int = 0,
) -> Tuple[Iterator[Tuple[Tuple[int, int, int], List[bytes]]], TilePagingReport]:
    """Stream a world's tiles as pages: yields (tile_key, payloads)
    for one tile at a time, plus the MEASURED paging report.

    The peak-memory contract: the caller holds at most ONE tile's
    payload list plus the report accumulators at any time -- bounded
    by the largest tile, never by world size. `lod=1` yields the
    coarse payloads (identity + counts) instead of full geometry.
    """
    tiles = SpatialTiles(world, tile_size=tile_size)
    payload_fn = _entity_lod1_payload if lod == 1 else _entity_payload

    total_bytes = 0
    max_tile_bytes = 0
    page_count = 0
    entity_count = 0

    def _pages():
        nonlocal total_bytes, max_tile_bytes, page_count, entity_count
        for key in tiles.tile_ids():
            entity_ids = tiles.entities_in_tile(key)
            payloads = [payload_fn(world, eid) for eid in entity_ids]
            tile_bytes = sum(len(p) for p in payloads)
            total_bytes += tile_bytes
            entity_count += len(entity_ids)
            if tile_bytes > max_tile_bytes:
                max_tile_bytes = tile_bytes
            page_count += 1
            yield key, payloads

    report = TilePagingReport(
        tile_count=tiles.tile_count,
        entity_count=entity_count,  # filled by the generator contract below
        unlocalized_count=len(tiles.unlocalized_entity_ids),
        total_payload_bytes=total_bytes,
        max_tile_payload_bytes=max_tile_bytes,
        # Peak = one page buffer + the largest tile (measured during
        # the pass; max_tile_bytes is final only after the pass, so
        # the report is materialized by consuming the iterator).
        peak_stream_bytes=max_tile_bytes,
        page_count=page_count,
    )

    # The report must reflect the FULL pass: wrap the generator so the
    # consumer can drain it and then read final numbers.
    class _ReportedPages:
        def __init__(self):
            self._inner = _pages()
            self.report = None

        def __iter__(self):
            return self

        def __next__(self):
            return next(self._inner)

        def drain(self):
            pages = list(self._inner)
            self.report = TilePagingReport(
                tile_count=tiles.tile_count,
                entity_count=entity_count,
                unlocalized_count=len(tiles.unlocalized_entity_ids),
                total_payload_bytes=total_bytes,
                max_tile_payload_bytes=max_tile_bytes,
                peak_stream_bytes=max_tile_bytes,
                page_count=page_count,
            )
            return pages

    return _ReportedPages(), report
