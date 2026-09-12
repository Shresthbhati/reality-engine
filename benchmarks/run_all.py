"""Runs every real benchmark and prints a plain-text table."""

from __future__ import annotations

from typing import List

from benchmarks.harness import BenchmarkResult
from benchmarks.physics_bench import BENCHMARKS as PHYSICS_BENCHMARKS
from benchmarks.quality_bench import BENCHMARKS as QUALITY_BENCHMARKS


def run_all() -> List[BenchmarkResult]:
    results = []
    for bench_fn in PHYSICS_BENCHMARKS + QUALITY_BENCHMARKS:
        results.append(bench_fn())
    return results


def _print_table(results: List[BenchmarkResult]) -> None:
    header = f"{'name':<32} {'iterations':>10} {'total_s':>12} {'s/iter':>14} {'metadata'}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(f"{r.name:<32} {r.iterations:>10} {r.total_seconds:>12.6f} {r.seconds_per_iteration:>14.8f} {r.metadata}")


if __name__ == "__main__":
    _print_table(run_all())
