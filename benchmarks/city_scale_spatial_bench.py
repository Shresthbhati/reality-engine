"""SYNTHETIC BENCHMARK -- city-scale WorldOS spatial layer.

This is SYNTHETIC data: a deterministic grid of generated entities, NOT
real city capture. These numbers characterize the existing code's
scaling behavior on generated fixtures; they are NOT a claim about
real-world reconstruction performance, which depends on real evidence
density, geometry complexity, and hardware this benchmark does not
model. Do not cite these numbers as real-city performance.

Progressive scale ladder (entity counts are illustrative proxies for
the named scale, not derived from any real survey):
  ROOM               ~20 entities
  BUILDING           ~200 entities
  STREET             ~2,000 entities
  BLOCK              ~5,000 entities
  MULTI_BLOCK        ~20,000 entities
  CITY_REPRESENTATIVE ~100,000 entities

Measures per the mandate: entity count, tile count, WorldStore
on-disk size, save time, load time, query time, incremental update
time, tiles invalidated, tiles reused, and peak memory delta
(tracemalloc -- stdlib only, no profiling dependency).

Run: `python -m benchmarks.city_scale_spatial_bench`
"""

from __future__ import annotations

import json
import random
import tempfile
import time
import tracemalloc
from pathlib import Path

from engine.scene_graph.spatial_index import SpatialIndex
from provenance import Provenance
from world_ir import apply_incremental_update
from world_ir.schema_v1 import Entity, EntityType
from world_ir.spatial_tiles import SpatialTiles
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore

SCALE_LADDER = [
    ("ROOM", 20),
    ("BUILDING", 200),
    ("STREET", 2_000),
    ("BLOCK", 5_000),
    ("MULTI_BLOCK", 20_000),
    ("CITY_REPRESENTATIVE", 100_000),
]

#: A generated city block is ~80m wide; entities spread over a grid of
#: blocks so density stays roughly constant as n grows, matching how a
#: real city adds area, not just density, at larger scales.
BLOCK_SIDE_M = 80.0
ENTITIES_PER_BLOCK_ROW = 40


def _synthetic_world(n: int, seed: int = 42) -> WorldIR:
    rng = random.Random(seed)
    world = WorldIR(id=f"w-synth-{n}")
    for i in range(n):
        block_index = i // (ENTITIES_PER_BLOCK_ROW * ENTITIES_PER_BLOCK_ROW)
        local = i % (ENTITIES_PER_BLOCK_ROW * ENTITIES_PER_BLOCK_ROW)
        row, col = divmod(local, ENTITIES_PER_BLOCK_ROW)
        blocks_per_row = 40
        block_row, block_col = divmod(block_index, blocks_per_row)
        x = block_col * BLOCK_SIDE_M + (col / ENTITIES_PER_BLOCK_ROW) * BLOCK_SIDE_M
        y = block_row * BLOCK_SIDE_M + (row / ENTITIES_PER_BLOCK_ROW) * BLOCK_SIDE_M
        z = rng.uniform(0.0, 3.0)
        world.entities[f"e{i}"] = Entity(
            id=f"e{i}", type=EntityType.STRUCTURE,
            transform={"position": {"x": x, "y": y, "z": z}},
            provenance=Provenance.RECONSTRUCTED,
        )
    return world


def _measure(label: str, fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, dt, peak


def run_one_scale(scale_name: str, n: int) -> dict:
    world, build_time, build_peak = _measure("build", lambda: _synthetic_world(n))

    tiles, tile_time, _ = _measure("tiles", lambda: SpatialTiles(world, tile_size=10.0))
    tile_count = tiles.tile_count

    with tempfile.TemporaryDirectory() as tmp:
        store = WorldStore(Path(tmp))
        (v1, save_time, save_peak) = _measure(
            "save", lambda: store.save_version(world, parent=None, version_id="v-1")
        )
        store_size_bytes = sum(f.stat().st_size for f in Path(tmp).rglob("*") if f.is_file())

        (_loaded, load_time, load_peak) = _measure("load", lambda: store.load_version("v-1"))

        index, index_build_time, index_peak = _measure("index_build", lambda: SpatialIndex(world))
        mid_x = (ENTITIES_PER_BLOCK_ROW / 2) * BLOCK_SIDE_M / ENTITIES_PER_BLOCK_ROW
        (_nearest, query_time, _) = _measure(
            "query", lambda: index.nearest((mid_x, mid_x, 0.0), k=10)
        )

        # A single localized change: move one entity within its own tile.
        changed = Entity(
            id="e0", type=EntityType.STRUCTURE,
            transform={"position": {"x": 1.0, "y": 1.0, "z": 0.0}},
            provenance=Provenance.RECONSTRUCTED,
        )
        (incr_result, incr_time, incr_peak) = _measure(
            "incremental", lambda: apply_incremental_update(world, [changed], tile_size=10.0)
        )
        (_v2, save2_time, _) = _measure(
            "save_v2", lambda: store.save_version(incr_result.new_world, parent="v-1", version_id="v-2")
        )

    return {
        "scale": scale_name,
        "entity_count": n,
        "tile_count": tile_count,
        "worldstore_bytes": store_size_bytes,
        "save_time_s": round(save_time, 4),
        "load_time_s": round(load_time, 4),
        "index_build_time_s": round(index_build_time, 4),
        "query_time_s": round(query_time, 6),
        "incremental_update_time_s": round(incr_time, 4),
        "incremental_save_time_s": round(save2_time, 4),
        "tiles_invalidated": len(incr_result.invalidated_tile_ids),
        "tiles_rebuilt": len(incr_result.rebuilt_tile_ids),
        "tiles_reused": tile_count - len(incr_result.invalidated_tile_ids),
        "reused_entity_count": len(incr_result.reused_entity_ids),
        "build_peak_bytes": build_peak,
        "save_peak_bytes": save_peak,
        "load_peak_bytes": load_peak,
        "index_peak_bytes": index_peak,
        "incremental_peak_bytes": incr_peak,
    }


def main() -> None:
    print("=" * 70)
    print("SYNTHETIC BENCHMARK -- generated fixture data, NOT real city capture")
    print("=" * 70)
    results = []
    for scale_name, n in SCALE_LADDER:
        row = run_one_scale(scale_name, n)
        results.append(row)
        print(json.dumps(row, indent=2))
    return results


if __name__ == "__main__":
    main()
