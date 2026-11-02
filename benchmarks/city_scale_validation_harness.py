"""SYNTHETIC city-scale validation harness (WORLDOS_LIFECYCLE_HEALTH
mission, Mission 3). CLEARLY LABELED SYNTHETIC -- deterministic,
seed-free generated grids, NOT real city capture and NOT "real-city
validation." Do not cite these numbers as real-world performance.

Deterministic world generation at 10^2, 10^3, 10^4, 10^5 entities/tiles.
10^6 is deliberately NOT run in this harness this session: at the
measured per-entity cost of the O(n) whole-world save/load path (see
benchmarks/city_scale_spatial_bench.py's 100K-entity numbers: 45s/53s),
a 10^6-entity run of the FULL comparison (whole-world save+load included)
would take on the order of 10x that -- impractical for a single
interactive session. What DOES scale to 10^6 without that cost is
exactly the O(affected) path this mission is about (lazy open, delta
write, compaction) -- `run_large_scale_delta_only` exercises that path
at 10^5 without paying the whole-world-save cost, and the mechanism is
architecturally identical at 10^6; it was not separately run here.

Measures, at each scale: initial persistence (tiled + partitioned),
lazy open, local (region) query, global (radius covering the whole
world) query, local incremental update, delta write, compaction,
memory (tracemalloc peak), tile-artifact reuse, manifest size, and
chain depth.

Run: `python -m benchmarks.city_scale_validation_harness`
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from tempfile import TemporaryDirectory

from world_ir.incremental import apply_incremental_update
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR
from worldstore.lifecycle import CompactionPolicy, maybe_compact
from worldstore.lazy_query import lazy_within_radius, lazy_within_region
from worldstore.store import WorldStore
from worldstore.tiles import (
    delta_chain_depth,
    open_version,
    save_version_partitioned,
    save_version_partitioned_delta,
    save_version_tiled,
)

SCALES = [
    ("10^2", 100),
    ("10^3", 1_000),
    ("10^4", 10_000),
]
SPACING = 20.0


def _deterministic_world(n: int) -> WorldIR:
    """DETERMINISTIC: same `n` always produces the exact same world
    (no randomness, no wall-clock seed) -- a required property for a
    reproducible validation harness."""
    side = int(n ** 0.5) + 1
    world = WorldIR(id=f"w-validate-{n}")
    count = 0
    for i in range(side):
        for j in range(side):
            if count >= n:
                break
            eid = f"e-{i}-{j}"
            world.entities[eid] = Entity(
                id=eid, transform={"position": {"x": i * SPACING, "y": 0.0, "z": j * SPACING}},
            )
            count += 1
        if count >= n:
            break
    return world


def _measure(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn()
    dt = time.perf_counter() - t0
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, dt, peak


def run_one_scale(label: str, n: int) -> dict:
    world = _deterministic_world(n)
    with TemporaryDirectory() as tmp_tiled_dir, TemporaryDirectory() as tmp_part_dir:
        store_tiled = WorldStore(Path(tmp_tiled_dir))
        store_part = WorldStore(Path(tmp_part_dir))

        (_v1, tiled_save_time, tiled_save_peak) = _measure(
            lambda: save_version_tiled(store_tiled, world, parent=None, version_id="v-1", tile_size=10.0)
        )
        (_v1p, part_save_time, part_save_peak) = _measure(
            lambda: save_version_partitioned(store_part, world, parent=None, version_id="v-1", tile_size=10.0)
        )

        (handle, open_time, open_peak) = _measure(lambda: open_version(store_tiled, "v-1"))
        (local_result, local_time, local_peak) = _measure(
            lambda: lazy_within_region(handle, (0.0, -1.0, 0.0), (15.0, 1.0, 15.0))
        )
        world_span = (int(n ** 0.5) + 1) * SPACING
        (global_result, global_time, global_peak) = _measure(
            lambda: lazy_within_radius(handle, (0.0, 0.0, 0.0), world_span * 1.5)
        )

        moved = Entity(id="e-0-0", transform={"position": {"x": 0.5, "y": 0.0, "z": 0.0}})
        (incr_result, incr_time, incr_peak) = _measure(
            lambda: apply_incremental_update(world, [moved], tile_size=10.0)
        )
        (v2p, delta_time, delta_peak) = _measure(
            lambda: save_version_partitioned_delta(
                store_part, incr_result.new_world, parent="v-1", version_id="v-2",
                tile_size=10.0, incremental_result=incr_result,
            )
        )

        depth_before = delta_chain_depth(store_part, "v-2")
        (compact_outcome, compact_time, compact_peak) = _measure(
            lambda: maybe_compact(store_part, "v-2", CompactionPolicy(max_chain_depth=0))
        )

        v1_handle = open_version(store_tiled, "v-1")
        # Tile reuse proof: rebuild v-2 via save_version_tiled and diff artifact_uris.
        save_version_tiled(
            store_tiled, incr_result.new_world, parent="v-1", version_id="v-2-tiled",
            tile_size=10.0, incremental_result=incr_result,
        )
        v2_handle = open_version(store_tiled, "v-2-tiled")
        reused = sum(
            1 for a, b in zip(
                sorted(v1_handle.manifest.tiles, key=lambda t: t.tile_key),
                sorted(v2_handle.manifest.tiles, key=lambda t: t.tile_key),
            )
            if a.artifact_uri == b.artifact_uri
        )

        manifest_bytes = (Path(tmp_tiled_dir) / "tile_manifests" / "v-1.json").stat().st_size

    return {
        "scale": label,
        "entity_count": n,
        "tile_count": len(handle.list_tiles()),
        "tiled_save_time_s": round(tiled_save_time, 4),
        "tiled_save_peak_bytes": tiled_save_peak,
        "partitioned_save_time_s": round(part_save_time, 4),
        "partitioned_save_peak_bytes": part_save_peak,
        "lazy_open_time_s": round(open_time, 6),
        "local_region_query_time_s": round(local_time, 6),
        "local_region_query_entities": len(local_result),
        "global_radius_query_time_s": round(global_time, 6),
        "global_radius_query_entities": len(global_result),
        "local_incremental_update_time_s": round(incr_time, 6),
        "delta_write_time_s": round(delta_time, 6),
        "delta_chain_depth_before_compaction": depth_before,
        "compaction_time_s": round(compact_time, 6),
        "compaction_succeeded": compact_outcome.compacted,
        "tiles_reused_out_of_total": f"{reused}/{len(v1_handle.manifest.tiles)}",
        "full_manifest_size_bytes": manifest_bytes,
    }


def run_large_scale_delta_only(n: int = 100_000) -> dict:
    """The O(affected) path ONLY, at 10^5 -- skips the O(n) whole-world
    blob entirely (save_version_partitioned, not save_version_tiled),
    so this scale is actually practical to run. Proves the mechanism
    this mission is about (lazy open, delta write, compaction) holds
    at 10^5 without paying the whole-world-save cost."""
    world = _deterministic_world(n)
    with TemporaryDirectory() as tmp:
        store = WorldStore(Path(tmp))
        (_v1, save_time, save_peak) = _measure(
            lambda: save_version_partitioned(store, world, parent=None, version_id="v-1", tile_size=10.0)
        )
        (handle, open_time, open_peak) = _measure(lambda: open_version(store, "v-1"))
        (local_result, local_time, local_peak) = _measure(
            lambda: lazy_within_region(handle, (0.0, -1.0, 0.0), (15.0, 1.0, 15.0))
        )
        moved = Entity(id="e-0-0", transform={"position": {"x": 0.5, "y": 0.0, "z": 0.0}})
        result = apply_incremental_update(world, [moved], tile_size=10.0)
        (_v2, delta_time, delta_peak) = _measure(
            lambda: save_version_partitioned_delta(
                store, result.new_world, parent="v-1", version_id="v-2", tile_size=10.0, incremental_result=result,
            )
        )
        depth = delta_chain_depth(store, "v-2")
        (outcome, compact_time, compact_peak) = _measure(
            lambda: maybe_compact(store, "v-2", CompactionPolicy(max_chain_depth=0))
        )
    return {
        "scale": "10^5 (delta-path only, no whole-world blob)",
        "entity_count": n,
        "tile_count": len(handle.list_tiles()),
        "partitioned_save_time_s": round(save_time, 4),
        "partitioned_save_peak_bytes": save_peak,
        "lazy_open_time_s": round(open_time, 6),
        "local_region_query_time_s": round(local_time, 6),
        "local_region_query_entities": len(local_result),
        "delta_write_time_s": round(delta_time, 6),
        "delta_chain_depth_before_compaction": depth,
        "compaction_time_s": round(compact_time, 6),
        "compaction_succeeded": outcome.compacted,
    }


def main():
    print("=" * 70)
    print("SYNTHETIC city-scale validation harness -- deterministic")
    print("generated grids, NOT real-city validation.")
    print("=" * 70)
    results = []
    for label, n in SCALES:
        row = run_one_scale(label, n)
        results.append(row)
        print(json.dumps(row, indent=2))
    large = run_large_scale_delta_only(100_000)
    results.append(large)
    print(json.dumps(large, indent=2))
    print("10^6 NOT run this session -- see module docstring for why.")
    return results


if __name__ == "__main__":
    main()
