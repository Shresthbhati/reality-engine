"""Benchmarks engine/quality/report.py's compute_quality_report() against a
WorldIR with a meaningful number of entities and relationships, to check it
scales reasonably (it's an O(entities) scan today; this benchmark exists to
catch a future regression, not because current performance is in doubt).
"""

from __future__ import annotations

from engine.quality.report import compute_quality_report
from provenance import Provenance
from world_ir import Entity, Relationship, RelationshipKind, WorldIR

from .harness import BenchmarkResult, run_benchmark


def _build_world(entity_count: int) -> WorldIR:
    world = WorldIR(id="bench-quality")
    provenances = [Provenance.OBSERVED, Provenance.RECONSTRUCTED, Provenance.INFERRED, Provenance.GENERATED]
    for i in range(entity_count):
        entity = Entity(
            id=f"e{i}",
            provenance=provenances[i % len(provenances)],
            confidence=0.3 + (i % 7) * 0.1,
        )
        if i % 5 == 0 and i > 0:
            entity.relationships.append(Relationship(kind=RelationshipKind.ADJACENT_TO, target_id=f"e{i - 1}"))
        world.entities[entity.id] = entity
    return world


def bench_quality_report(entity_count: int = 500, iterations: int = 20) -> BenchmarkResult:
    world = _build_world(entity_count)
    return run_benchmark(
        f"quality_report_n{entity_count}",
        iterations,
        lambda: compute_quality_report(world),
        metadata={"entity_count": entity_count},
    )


BENCHMARKS = [bench_quality_report]
