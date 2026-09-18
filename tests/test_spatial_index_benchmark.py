"""Benchmark for spatial index at various entity counts.

This benchmark measures:
- Build time
- Query time (nearest, within_radius, within_region)
- Memory usage (where practical)

Runs at: 100, 1,000, 10,000, 100,000 entities
"""

from __future__ import annotations

import gc
import math
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR

from engine.scene_graph.spatial_index import SpatialIndex, AcceleratedSpatialIndex


Point = Tuple[float, float, float]


def _make_world_with_entities(
    n_entities: int,
    bounds: Tuple[Point, Point] = ((-100, -100, -10), (100, 100, 10)),
    use_geometry: bool = True,
) -> WorldIR:
    """Create a WorldIR with randomly distributed entities."""
    world = WorldIR()
    min_b, max_b = bounds
    
    for i in range(n_entities):
        # Random position within bounds
        x = random.uniform(min_b[0], max_b[0])
        y = random.uniform(min_b[1], max_b[1])
        z = random.uniform(min_b[2], max_b[2])
        
        entity = Entity(
            id=f"ent-{i:06d}",
            type=EntityType.DEBRIS,
            transform={"position": {"x": x, "y": y, "z": z}},
        )
        
        if use_geometry:
            # Add a small geometry box
            geom = Geometry(
                id=f"geom-{i:06d}",
                type=GeometryType.BOX,
                bounds_min=Vector3(x - 0.5, y - 0.5, z - 0.5),
                bounds_max=Vector3(x + 0.5, y + 0.5, z + 0.5),
            )
            entity.geometry_ids = [geom.id]
            world.geometries[geom.id] = geom
        
        world.entities[entity.id] = entity
    
    return world


@dataclass
class BenchmarkResult:
    entity_count: int
    build_time_s: float
    nearest_time_s: float
    within_radius_time_s: float
    within_region_time_s: float
    query_point: Point
    use_accel: bool


def run_benchmark(
    entity_count: int,
    use_accel: bool = True,
    query_point: Point = (0.0, 0.0, 0.0),
) -> BenchmarkResult:
    """Run a single benchmark iteration."""
    
    # Create world
    world = _make_world_with_entities(entity_count)
    
    # Build index (timed)
    gc.collect()
    start = time.perf_counter()
    if use_accel:
        index = AcceleratedSpatialIndex(world, force_accel=True)
    else:
        index = SpatialIndex(world)
    build_time = time.perf_counter() - start
    
    # Verify index has expected number of entities
    assert len(index) == entity_count, f"Index size mismatch: {len(index)} vs {entity_count}"
    
    # Query: nearest (timed)
    gc.collect()
    start = time.perf_counter()
    for _ in range(100):
        index.nearest(query_point, k=5)
    nearest_time = (time.perf_counter() - start) / 100
    
    # Query: within_radius (timed)
    gc.collect()
    start = time.perf_counter()
    for _ in range(100):
        index.within_radius(query_point, radius=20.0)
    within_radius_time = (time.perf_counter() - start) / 100
    
    # Query: within_region (timed)
    gc.collect()
    start = time.perf_counter()
    for _ in range(100):
        index.within_region((-50, -50, -5), (50, 50, 5))
    within_region_time = (time.perf_counter() - start) / 100
    
    return BenchmarkResult(
        entity_count=entity_count,
        build_time_s=build_time,
        nearest_time_s=nearest_time,
        within_radius_time_s=within_radius_time,
        within_region_time_s=within_region_time,
        query_point=query_point,
        use_accel=use_accel,
    )


def print_result(result: BenchmarkResult) -> None:
    """Print a formatted benchmark result."""
    accel_str = "Accelerated" if result.use_accel else "Flat"
    print(f"\n{'='*60}")
    print(f"  {accel_str} Spatial Index Benchmark")
    print(f"  Entities: {result.entity_count:,}")
    print(f"  {'-'*60}")
    print(f"  Build time:        {result.build_time_s*1000:>8.2f} ms")
    print(f"  Nearest (k=5):     {result.nearest_time_s*1000:>8.2f} ms")
    print(f"  Within radius:     {result.within_radius_time_s*1000:>8.2f} ms")
    print(f"  Within region:     {result.within_region_time_s*1000:>8.2f} ms")
    print(f"  {'-'*60}")
    print(f"  Total query time:  {(result.nearest_time_s + result.within_radius_time_s + result.within_region_time_s)*1000:>8.2f} ms")


def main():
    """Run benchmarks at multiple entity counts."""
    random.seed(42)  # Deterministic results
    
    print("Spatial Index Benchmark Suite")
    print("=" * 60)
    print("Environment: Python", sys.version.split()[0])
    print("World bounds: 200x200x20 meters")
    print("Query point: (0, 0, 0)")
    
    entity_counts = [100, 1_000, 10_000, 100_000]
    
    results: List[BenchmarkResult] = []
    
    for count in entity_counts:
        print(f"\n>>> Benchmarking {count:,} entities...")
        
        # Run accelerated version
        result_accel = run_benchmark(count, use_accel=True)
        results.append(result_accel)
        print_result(result_accel)
        
        # Run flat version for comparison (skip 100k as it's too slow)
        if count <= 10_000:
            result_flat = run_benchmark(count, use_accel=False)
            results.append(result_flat)
            print_result(result_flat)
            
            # Speedup
            speedup = result_flat.nearest_time_s / result_accel.nearest_time_s if result_accel.nearest_time_s > 0 else float('inf')
            print(f"  Speedup (nearest): {speedup:.1f}x")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Entities':>10} | {'Mode':>10} | {'Build (ms)':>10} | {'Nearest (ms)':>12} | {'Radius (ms)':>11} | {'Region (ms)':>11}")
    print("-" * 80)
    
    for r in results:
        mode = "Accelerated" if r.use_accel else "Flat"
        print(f"{r.entity_count:>10,} | {mode:>10} | {r.build_time_s*1000:>10.2f} | {r.nearest_time_s*1000:>12.2f} | {r.within_radius_time_s*1000:>11.2f} | {r.within_region_time_s*1000:>11.2f}")
    
    print("=" * 80)
    
    # Verify correctness: both modes should return same results
    print("\nCorrectness check (1000 entities):")
    world = _make_world_with_entities(1000)
    idx_accel = AcceleratedSpatialIndex(world, force_accel=True)
    idx_flat = SpatialIndex(world)
    
    q = (10.0, 5.0, 0.0)
    nearest_accel = idx_accel.nearest(q, k=5)
    nearest_flat = idx_flat.nearest(q, k=5)
    
    accel_ids = [e.id for e, _ in nearest_accel]
    flat_ids = [e.id for e, _ in nearest_flat]
    
    if accel_ids == flat_ids:
        print("  PASS: Accelerated and Flat return identical results")
    else:
        print("  FAIL: Results differ!")
        print(f"  Accelerated: {accel_ids}")
        print(f"  Flat:        {flat_ids}")
        sys.exit(1)
    
    print("\nAll benchmarks completed successfully!")


if __name__ == "__main__":
    main()