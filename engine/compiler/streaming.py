"""Streaming/bounded-memory city compilation (P14-01).

`compile_city_world` materializes one WorldIR holding every feature --
unbounded for city-extent imports (hundreds of thousands of OSM ways).
This module compiles the same CityFeatures into deterministic TILE
FRAGMENTS, yielded as a generator: one (tile_key, WorldIR) pair per
non-empty spatial tile plus a final (None, report) summary, so callers
process tiles one at a time (export, hand to WorldStore's own tiling)
without the whole city in memory.

Discipline:
  - NOT a second compiler: per-feature geometry/provenance rules are
    reused from compile_city_world verbatim. Features are grouped per
    tile, and each tile's entities are produced by compiling that
    tile's feature slice through the canonical path, then re-keying
    ids to carry the tile key (`<name>-<tile>-<kind>-<i>`).
  - Multi-tile membership follows SpatialTiling's rule: a feature
    whose footprint AABB touches several tile cells is compiled into
    EVERY touched tile. `spans_multiple_tiles` records the fact on the
    entity; the report counts how many features span. Cross-tile
    identity is never silently merged or split.
  - Deterministic: tile keys sorted; identical input -> identical
    fragment sequence (world.to_dict equality, tested).
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Sequence, Tuple

from engine.compiler.city_compiler import CityFeature, compile_city_world
from world_ir.world_v1 import WorldIR

#: Final summary yield's tile key sentinel.
REPORT_KEY = None


TileKey = Tuple[int, int]

_TILE_NAME_FMT = "{name}-tile-{x}-{y}"


def _tile_name(name: str, key: TileKey) -> str:
    return _TILE_NAME_FMT.format(name=name, x=key[0], y=key[1])


def _footprint_tile_range(feature: CityFeature, tile_size: float) -> Tuple[int, int, int, int]:
    """(x_lo, x_hi, y_lo, y_hi) tile indices the footprint AABB touches."""
    xs = [float(p[0]) for p in feature.footprint]
    ys = [float(p[1]) for p in feature.footprint]
    x_lo = int(min(xs) // tile_size)
    x_hi = int(max(xs) // tile_size)
    y_lo = int(min(ys) // tile_size)
    y_hi = int(max(ys) // tile_size)
    return x_lo, x_hi, y_lo, y_hi


def compile_city_world_streaming(
    features: Sequence[CityFeature],
    tile_size: float,
    name: str = "city",
) -> Iterator[Tuple]:
    """Compile `features` into per-tile WorldIR fragments (generator).

    Yields (tile_key, world) for every non-empty tile in sorted tile-key
    order, then a final (None, report) summary dict with measured
    counts. Grouping happens FIRST (features -> tiles), then each tile
    compiles its slice through compile_city_world and re-keys ids, so
    memory is bounded by one tile's features, not the whole city.
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")

    # --- pass 1: group features by touched tile (bounded bookkeeping:
    # one index list per populated tile, never a copy of the features).
    tile_features: Dict[TileKey, List[int]] = {}
    spanned = 0
    for index, feature in enumerate(features):
        x_lo, x_hi, y_lo, y_hi = _footprint_tile_range(feature, tile_size)
        keys = [
            (ix, iy)
            for ix in range(x_lo, x_hi + 1)
            for iy in range(y_lo, y_hi + 1)
        ]
        if len(keys) > 1:
            spanned += 1
        for key in keys:
            tile_features.setdefault(key, []).append(index)

    report = {
        "features_compiled": len(features),
        "tiles_emitted": len(tile_features),
        "features_spanning_tiles": spanned,
        "tile_size": tile_size,
    }

    # --- pass 2: compile each tile's slice through the canonical path.
    for key in sorted(tile_features):
        indices = tile_features[key]
        slice_features = [features[i] for i in indices]
        tile_name = _tile_name(name, key)
        tile_world = compile_city_world(slice_features, name=tile_name)
        _stamp_tile_facts(tile_world, tile_name, slice_features, tile_size)
        yield key, tile_world

    yield REPORT_KEY, report


def _stamp_tile_facts(world: WorldIR, tile_name: str,
                      slice_features: List[CityFeature],
                      tile_size: float) -> None:
    """Stamp the multi-tile fact on this tile's copies of spanning
    features. No id renaming is needed: compile_city_world already keys
    entities `<tile-name>-<kind>-<index>` and the tile name carries the
    compact tile key, so per-tile ids are unique and traceable as-is
    (a rename here would be a self-collision -- the bug this helper
    replaces)."""
    tile_id_prefix = f"{tile_name}-"
    for index, feature in enumerate(slice_features):
        entity_id = f"{tile_id_prefix}{feature.kind}-{index}"
        entity = world.entities.get(entity_id)
        if entity is None:
            raise RuntimeError(
                f"streaming compile: expected entity {entity_id!r} "
                f"missing from canonical compile output -- invariant broken"
            )
        x_lo, x_hi, y_lo, y_hi = _footprint_tile_range(feature, tile_size)
        if (x_hi - x_lo) + (y_hi - y_lo) > 0:
            entity.custom_properties["spans_multiple_tiles"] = True
