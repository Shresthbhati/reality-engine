"""SYNTHETIC BENCHMARK -- Task 6 of the WORLDOS_REAL_DATA_CITY_SCALE
mission. Honest disclosure up front: there is NO real (sensor-captured)
reconstruction dataset anywhere in this repository -- no .ply/.pcd/.las
files, no camera capture logs. This repo's own quality-state machinery
(world_ir/quality_state.py) exists specifically because everything here
is SYNTHETIC or a TEST DOUBLE, never REAL. "Strongest real reconstruction
dataset available" as asked for by the mission does not exist to run.

What this benchmark actually does instead, honestly:
  1. A synthetic scaling ladder (same convention as
     benchmarks/lazy_tile_load_bench.py) across the full requested
     metric set: full world load, lazy open, local (region) query,
     nearest, radius query, local incremental update, full save, tiled
     save, partitioned save.
  2. The most STRUCTURED synthetic fixture in the repo (the two-room
     scene from tests/test_room_inference.py, run through the REAL
     `sdk.reality.compile_world_from_reconstruction` compile pipeline,
     not a hand-built WorldIR) as a stand-in for "richest available
     scene" -- still SYNTHETIC, clearly labeled as such in every result
     row.

Run: `python -m benchmarks.real_data_city_scale_bench`
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path
from tempfile import TemporaryDirectory

from provenance import Provenance
from world_ir.incremental import apply_incremental_update
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore
from worldstore.tiles import open_version, save_version_partitioned, save_version_tiled

SCALE_LADDER = [
    ("ROOM", 20),
    ("BUILDING", 200),
    ("STREET", 2_000),
    ("BLOCK", 5_000),
    ("MULTI_BLOCK", 20_000),
]

BLOCK_SIDE_M = 80.0
ENTITIES_PER_BLOCK_ROW = 40


def _synthetic_world(n: int) -> WorldIR:
    world = WorldIR(id=f"w-real-city-bench-{n}")
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


def _bytes_under(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def run_one_scale(scale_name: str, n: int) -> dict:
    world = _synthetic_world(n)
    with TemporaryDirectory() as tmp_whole_dir, TemporaryDirectory() as tmp_part_dir:
        store = WorldStore(Path(tmp_whole_dir))
        part_store = WorldStore(Path(tmp_part_dir))

        (_v1, full_save_time, full_save_peak) = _measure(
            lambda: store.save_version(world, parent=None, version_id="v-1")
        )
        (_v1_tiled, tiled_save_time, tiled_save_peak) = _measure(
            lambda: save_version_tiled(store, world, parent="v-1", version_id="v-1-tiled", tile_size=10.0)
        )
        (_v1_part, part_save_time, part_save_peak) = _measure(
            lambda: save_version_partitioned(part_store, world, parent=None, version_id="v-1", tile_size=10.0)
        )

        (_full, full_load_time, full_load_peak) = _measure(lambda: store.load_version("v-1"))
        (handle, open_time, open_peak) = _measure(lambda: open_version(store, "v-1-tiled"))
        (region_result, region_time, region_peak) = _measure(
            lambda: handle.query_region((0.0, 0.0, 0.0), (15.0, 15.0, 1.0))
        )

        from worldstore.lazy_query import lazy_nearest, lazy_within_radius
        (nearest_result, nearest_time, nearest_peak) = _measure(
            lambda: lazy_nearest(handle, (0.0, 0.0, 0.0), k=5)
        )
        (radius_result, radius_time, radius_peak) = _measure(
            lambda: lazy_within_radius(handle, (0.0, 0.0, 0.0), 15.0)
        )

        moved = Entity(
            id="e0", type=EntityType.STRUCTURE,
            transform={"position": {"x": 0.5, "y": 0.5, "z": 0.0}},
            provenance=Provenance.RECONSTRUCTED,
        )
        (incr_result, incr_time, incr_peak) = _measure(
            lambda: apply_incremental_update(world, [moved], tile_size=10.0)
        )

        whole_bytes = _bytes_under(Path(tmp_whole_dir))
        part_bytes = _bytes_under(Path(tmp_part_dir))

    return {
        "scale": scale_name,
        "entity_count": n,
        "tile_count": len(handle.list_tiles()),
        "full_save_time_s": round(full_save_time, 4),
        "tiled_save_time_s": round(tiled_save_time, 4),
        "partitioned_save_time_s": round(part_save_time, 4),
        "full_load_time_s": round(full_load_time, 4),
        "lazy_open_time_s": round(open_time, 6),
        "local_region_query_time_s": round(region_time, 6),
        "local_region_query_entities": len(region_result),
        "nearest_k5_time_s": round(nearest_time, 6),
        "radius_query_time_s": round(radius_time, 6),
        "radius_query_entities": len(radius_result),
        "local_incremental_update_time_s": round(incr_time, 6),
        "incremental_rebuilt_tiles": len(incr_result.rebuilt_tile_ids),
        "whole_world_store_total_bytes": whole_bytes,
        "partitioned_store_total_bytes": part_bytes,
        "speedup_lazy_open_vs_full_load": round(full_load_time / max(open_time, 1e-9), 1),
        "speedup_local_query_vs_full_load": round(full_load_time / max(region_time, 1e-9), 1),
    }


def run_structured_scene_bench() -> dict:
    """The richest STRUCTURED fixture available in the repo, run
    through the real compile pipeline -- still SYNTHETIC (two rooms,
    programmatically generated points), not a real capture."""
    from sdk import reality
    from tests.test_room_inference import _CAMS, _two_room_scene
    from reconstruction.backend.interface import ReconstructedCameraPose

    recon = _two_room_scene()
    recon.camera_poses.extend(
        ReconstructedCameraPose(evidence_id=f"ev-{i}", position=p, rotation=(1.0, 0.0, 0.0, 0.0))
        for i, p in enumerate(_CAMS)
    )
    (compiled, compile_time, compile_peak) = _measure(
        lambda: reality.compile_world_from_reconstruction(recon)
    )
    world, diagnostics = compiled

    with TemporaryDirectory() as tmp:
        store = WorldStore(Path(tmp))
        (_v1_tiled, tiled_save_time, _peak) = _measure(
            lambda: save_version_tiled(store, world, parent=None, version_id="v-1", tile_size=1.0)
        )
        (handle, open_time, _peak) = _measure(lambda: open_version(store, "v-1"))
        (index, index_time, _peak) = _measure(lambda: reality.spatial_index(world))
        (nearest_result, nearest_time, _peak) = _measure(
            lambda: index.nearest((2.0, 1.25, 1.5), k=3)
        )

    return {
        "fixture": "two_room_scene (SYNTHETIC, structured, NOT a real capture)",
        "entity_count": len(world.entities),
        "rooms_detected": diagnostics.rooms_detected,
        "compile_time_s": round(compile_time, 4),
        "tiled_save_time_s": round(tiled_save_time, 6),
        "lazy_open_time_s": round(open_time, 6),
        "full_spatial_index_build_time_s": round(index_time, 6),
        "nearest_k3_time_s": round(nearest_time, 6),
    }


def main():
    print("=" * 70)
    print("SYNTHETIC BENCHMARK -- no real (sensor-captured) dataset exists")
    print("in this repository. Every number below is generated fixture")
    print("data. Do NOT cite these as real-world city performance.")
    print("=" * 70)
    results = {"scaling_ladder": [], "structured_scene": None}
    for scale_name, n in SCALE_LADDER:
        row = run_one_scale(scale_name, n)
        results["scaling_ladder"].append(row)
        print(json.dumps(row, indent=2))
    scene_row = run_structured_scene_bench()
    results["structured_scene"] = scene_row
    print(json.dumps(scene_row, indent=2))
    return results


if __name__ == "__main__":
    main()
