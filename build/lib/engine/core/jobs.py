"""Deterministic job system (spec sec 90 DETERMINISM: stable ordering;
sec 108 build order step 6, ahead of any solver).

A dependency-graph executor. Given the same set of jobs and dependencies,
execution order is always identical (topological sort with a stable,
id-based tie-break) -- this is what lets solvers built on top of it be
deterministic without each one reimplementing scheduling.

Deliberately single-threaded for now: correctness and determinism first,
parallel execution is a later optimization behind the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


class JobCycleError(ValueError):
    """Raised when the dependency graph contains a cycle."""


class DuplicateJobError(ValueError):
    """Raised when a job id is registered twice."""


class UnknownDependencyError(ValueError):
    """Raised when a job depends on an id that was never added."""


@dataclass
class Job:
    id: str
    fn: Callable[[], object]
    depends_on: tuple[str, ...] = field(default_factory=tuple)


class JobSystem:
    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def add_job(self, id: str, fn: Callable[[], object], depends_on: Optional[list[str]] = None) -> None:
        if id in self._jobs:
            raise DuplicateJobError(f"job '{id}' already registered")
        self._jobs[id] = Job(id=id, fn=fn, depends_on=tuple(depends_on or ()))

    def clear(self) -> None:
        self._jobs.clear()

    def topological_order(self) -> list[str]:
        """Stable topological order: at each step, among all jobs whose
        dependencies are satisfied, pick the smallest id. This makes the
        order a pure function of (job ids, dependency edges) -- not of
        insertion order or hash/set iteration.
        """
        for job in self._jobs.values():
            for dep in job.depends_on:
                if dep not in self._jobs:
                    raise UnknownDependencyError(
                        f"job '{job.id}' depends on unknown job '{dep}'"
                    )

        remaining = dict(self._jobs)
        done: set[str] = set()
        order: list[str] = []

        while remaining:
            ready = sorted(
                jid for jid, job in remaining.items()
                if all(dep in done for dep in job.depends_on)
            )
            if not ready:
                raise JobCycleError(
                    f"cycle detected among jobs: {sorted(remaining)}"
                )
            for jid in ready:
                order.append(jid)
                done.add(jid)
                del remaining[jid]

        return order

    def run(self) -> dict[str, object]:
        """Execute all jobs in deterministic order, return {id: result}."""
        results: dict[str, object] = {}
        for jid in self.topological_order():
            results[jid] = self._jobs[jid].fn()
        return results
