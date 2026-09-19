"""SYNTHETIC BENCHMARK -- current full-world load vs. new lazy tile
load (worldstore/tiles.py).

SYNTHETIC generated fixture data, NOT real city capture. Do not cite
these numbers as real-world performance -- they characterize the
existing code's scaling behavior on generated grids only.

Compares, at each scale:
  CURRENT FULL LOAD:  WorldStore.load_version() -- deserializes the
                       entire world every time, regardless of query size.
  LAZY TILE LOAD:      worldstore.tiles.open_version() (manifest only)
                       + query_region() over a small area (a handful
                       of tiles) -- proves opening a large world and
                       running a small spatial query does NOT require
                       loading every entity.

Run: `python -m benchmarks.lazy_tile_load_bench`
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from tempfile import TemporaryDirectory

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore
from worldstore.tiles import open_version, save_version_tiled

SCALE_LADDER = [
    ("ROOM", 20),
    ("BUILDING", 200),
    ("STREET", 2_000),
    ("BLOCK", 5_000),
    ("MULTI_BLOCK", 20_000),
    ("CITY_REPRESENTATIVE", 100_000),
]

BLOCK_SIDE_M = 80.0
ENTITIES_PER_BLOCK_ROW = 40


def _synthetic_world(n: int) -> WorldIR:
    world = WorldIR(id=f"w-lazy-{n}")
    for i in range(n):
        block_index = i // (ENTITIES_PER_BLOCK_ROW * ENTITIES_PER_BLOCK_ROW)
        local = i % (ENTITIES_PER_BLOCK_ROW * ENTITIES_PER_BLOCK_ROW)
        row, col = divmod(local, ENTITIES_PER_BLOCK_ROW)
        blocks_per_row = 40
        block_row, block_col = divmod(block_index, blocks_per_row)
        x = block_col * BLOCK_SIDE_M + (col / ENTITIES_PER_BLOCK_ROW) * BLOCK_SIDE_M
        y = block_row * BLOCK_SIDE_M + (row / ENTITIES_PER_BLOCK_ROW) * BLOCK_SIDE_M
        world.entities[f"e{i}"] = Entity(
            id=f"e{i}", type=EntityType.STRUCTURE,
            transform={"position": {"x": x, "y": y, "z": 0.0}},
            provenance=Provenance.RECONSTRUCTED,
        )
    return world


def _measure(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, dt, peak


def run_one_scale(scale_name: str, n: int) -> dict:
    world = _synthetic_world(n)
    with TemporaryDirectory() as tmp:
        store = WorldStore(Path(tmp))
        save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=10.0)

        # CURRENT FULL LOAD: the existing whole-world path.
        (_full, full_load_time, full_load_peak) = _measure(lambda: store.load_version("v-1"))

        # LAZY TILE LOAD: open (manifest only) + a small-region query.
        (handle, open_time, open_peak) = _measure(lambda: open_version(store, "v-1"))
        (small_query_result, small_query_time, small_query_peak) = _measure(
            lambda: handle.query_region((0.0, 0.0, 0.0), (15.0, 15.0, 1.0))
        )

    return {
        "scale": scale_name,
        "entity_count": n,
        "tile_count": len(handle.list_tiles()),
        "full_load_time_s": round(full_load_time, 4),
        "full_load_peak_bytes": full_load_peak,
        "lazy_open_time_s": round(open_time, 6),
        "lazy_open_peak_bytes": open_peak,
        "lazy_small_query_time_s": round(small_query_time, 6),
        "lazy_small_query_peak_bytes": small_query_peak,
        "lazy_small_query_entities_returned": len(small_query_result),
        "speedup_open_vs_full_load": round(full_load_time / max(open_time, 1e-9), 1),
        "speedup_small_query_vs_full_load": round(full_load_time / max(small_query_time, 1e-9), 1),
    }


def main():
    print("=" * 70)
    print("SYNTHETIC BENCHMARK -- generated fixture data, NOT real city capture")
    print("Comparing CURRENT FULL LOAD vs LAZY TILE LOAD")
    print("=" * 70)
    results = []
    for scale_name, n in SCALE_LADDER:
        row = run_one_scale(scale_name, n)
        results.append(row)
        print(json.dumps(row, indent=2))
    return results


if __name__ == "__main__":
    main()
