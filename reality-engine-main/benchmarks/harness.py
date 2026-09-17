"""Real timing harness. stdlib only -- time.perf_counter(), no profiling deps.

Determinism note: "deterministic" here means the operation count executed by
`fn` is fixed for given inputs, not that wall-clock seconds repeat across
runs/machines -- that's inherently non-deterministic.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    iterations: int
    total_seconds: float
    seconds_per_iteration: float
    metadata: dict = field(default_factory=dict)


def run_benchmark(
    name: str,
    iterations: int,
    fn: Callable[[], None],
    metadata: Optional[dict] = None,
) -> BenchmarkResult:
    """Runs fn() `iterations` times and times it for real. Does not catch
    exceptions from fn -- a crashing benchmark is not a passing benchmark.
    """
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    total = time.perf_counter() - start
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        total_seconds=total,
        seconds_per_iteration=total / iterations,
        metadata=metadata or {},
    )
