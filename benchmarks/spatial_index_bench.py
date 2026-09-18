"""Real benchmarks for SpatialIndex (city-scale campaign, spatial index
architecture item) -- grid-accelerated nearest()/within_radius() against
a brute-force O(n) reference scan, at several world sizes, so the "select
based on actual workload and benchmark the choice" call is backed by
measured numbers instead of an assumption.

Run: `python -m benchmarks.spatial_index_bench`
"""

from __future__ import annotations

import random
from typing import List

from benchmarks.harness import BenchmarkResult, run_benchmark
from engine.scene_graph.spatial_index import SpatialIndex
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR


def _random_world(n: int, spread: float, seed: int) -> WorldIR:
    rng = random.Random(seed)
    world = WorldIR()
    for i in range(n):
        x, y, z = (rng.uniform(-spread, spread) for _ in range(3))
        world.entities[f"e{i}"] = Entity(id=f"e{i}", transform={"position": {"x": x, "y": y, "z": z}})
    return world


def _brute_nearest(world: WorldIR, point, k: int):
    candidates = []
    for entity in world.entities.values():
        p = entity.transform["position"]
        d2 = (p["x"] - point[0]) ** 2 + (p["y"] - point[1]) ** 2 + (p["z"] - point[2]) ** 2
        candidates.append((entity.id, d2))
    candidates.sort(key=lambda pair: (pair[1], pair[0]))
    return candidates[:k]


def bench_grid_nearest(n: int, iterations: int = 20) -> BenchmarkResult:
    world = _random_world(n, spread=100.0, seed=n)
    index = SpatialIndex(world)  # build cost excluded -- steady-state query cost is what scales with n

    def run_once():
        index.nearest((0.0, 0.0, 0.0), k=10)

    return run_benchmark(f"grid_nearest(n={n})", iterations, run_once, metadata={"n": n, "k": 10})


def bench_brute_nearest(n: int, iterations: int = 20) -> BenchmarkResult:
    world = _random_world(n, spread=100.0, seed=n)

    def run_once():
        _brute_nearest(world, (0.0, 0.0, 0.0), k=10)

    return run_benchmark(f"brute_nearest(n={n})", iterations, run_once, metadata={"n": n, "k": 10})


def bench_grid_build(n: int, iterations: int = 5) -> BenchmarkResult:
    world = _random_world(n, spread=100.0, seed=n)

    def run_once():
        SpatialIndex(world)

    return run_benchmark(f"grid_build(n={n})", iterations, run_once, metadata={"n": n})


_SIZES = (100, 1_000, 10_000, 50_000)

BENCHMARKS = (
    [lambda n=n: bench_grid_build(n) for n in _SIZES]
    + [lambda n=n: bench_grid_nearest(n) for n in _SIZES]
    + [lambda n=n: bench_brute_nearest(n) for n in _SIZES]
)


def run() -> List[BenchmarkResult]:
    return [bench() for bench in BENCHMARKS]


def _print_table(results: List[BenchmarkResult]) -> None:
    header = f"{'name':<28} {'iterations':>10} {'total_s':>12} {'s/iter':>14} {'metadata'}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.name:<28} {r.iterations:>10} {r.total_seconds:>12.6f} {r.seconds_per_iteration:>14.8f} {r.metadata}")


if __name__ == "__main__":
    _print_table(run())
