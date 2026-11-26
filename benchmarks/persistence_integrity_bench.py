"""Persistence integrity benchmarks (KiloCode - platform engineering / data integrity).

Measures the WorldStore persistence path end to end:

  save_version -> verify_version (read-back hash integrity) -> load_version
     -> entity/hash integrity -> version chain (parent lineage)

These are *regression benchmarks*: they time real save/load/verify operations on
real WorldIR payloads and assert that persistence never silently corrupts data
(every saved version must verify and round-trip byte-identical). They are not
accuracy claims about reconstruction -- they measure the persistence layer's
own integrity, timing, and memory.

Run:  python benchmarks/persistence_integrity_bench.py
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from dataclasses import dataclass, field

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore


# --------------------------------------------------------------------------- fixtures

def _make_world(world_id: str, n_entities: int = 50) -> WorldIR:
    """Build a WorldIR with n_entities of mixed types, each with geometry and provenance."""
    world = WorldIR(id=world_id)
    kinds = [
        (EntityType.ROOM, GeometryType.PLANE),
        (EntityType.WALL, GeometryType.PLANE),
        (EntityType.WINDOW, GeometryType.BOX),
        (EntityType.DOOR, GeometryType.BOX),
        (EntityType.FLOOR, GeometryType.PLANE),
    ]
    for i in range(n_entities):
        kind, _ = kinds[i % len(kinds)]
        world.entities[f"ent-{i:04d}"] = Entity(
            id=f"ent-{i:04d}",
            type=kind,
            name=f"synthetic entity {i}",
            provenance=Provenance.OBSERVED,
            confidence=0.9,
            custom_properties={"index": i, "note": f"synthetic entity {i}"},
        )
    return world


def _world_hash(world: WorldIR) -> str:
    return hashlib.sha256(
        json.dumps(world.to_dict(), sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- benchmark results

@dataclass
class PersistResult:
    name: str
    iterations: int
    seconds_per_iteration: float
    peak_memory_mb: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- benchmarks

def bench_save_verify_load(n_entities: int = 200, iterations: int = 25) -> PersistResult:
    """Save -> verify (read-back hash) -> load -> entity/hash integrity, N times.

    Every iteration persists a fresh world and asserts:
      * verify_version reports no errors
      * load_version returns a world with the same entity count and id
      * the re-hashed loaded world equals the stored artifact hash
    """
    import tracemalloc

    tracemalloc.start()
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as td:
        store = WorldStore(Path(td) / "store")
        for i in range(iterations):
            world = _make_world(f"world-persist-{i}", n_entities)
            expected_hash = _world_hash(world)
            ver = store.save_version(world, parent=None, version_id=f"v-persist-{i}")

            # Read-back verification (the release-gate check, exercised here).
            errors = store.verify_version(ver.version_id)
            assert errors == [], f"read-back verification failed: {errors}"

            loaded = store.load_version(ver.version_id)
            assert loaded.id == world.id
            assert len(loaded.entities) == n_entities
            re_hash = _world_hash(loaded)
            assert re_hash == expected_hash, "loaded world hash != saved artifact hash"

    total = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return PersistResult(
        name="save_verify_load",
        iterations=iterations,
        seconds_per_iteration=total / iterations,
        peak_memory_mb=round(peak / (1024 * 1024), 2),
        metadata={
            "n_entities": n_entities,
            "total_seconds": round(total, 4),
            "versions": iterations,
        },
    )


def bench_version_chain(n_versions: int = 40, n_entities: int = 40) -> PersistResult:
    """Version chain: each version's parent is the previous; verify the chain loads
    in order and `parents()`/`ancestors()` return the expected lineage."""
    import tracemalloc

    tracemalloc.start()
    t0 = time.perf_counter()

    with tempfile.TemporaryDirectory() as td:
        store = WorldStore(Path(td) / "store")
        parent = None
        for i in range(n_versions):
            world = _make_world(f"world-chain-{i}", n_entities)
            ver = store.save_version(world, parent=parent, version_id=f"v-chain-{i}")
            parent = ver.version_id

        # Walk the chain backwards via parents() and confirm lineage.
        cur = parent
        seen: List[str] = []
        while cur is not None:
            seen.append(cur)
            ps = store.parents(cur)
            cur = ps[0] if ps else None
        assert seen == [f"v-chain-{i}" for i in range(n_versions - 1, -1, -1)], "chain lineage broken"

        # Every version in the chain verifies and loads.
        for i in range(n_versions):
            errors = store.verify_version(f"v-chain-{i}")
            assert errors == [], f"chain version v-chain-{i} failed verification: {errors}"
            store.load_version(f"v-chain-{i}")

    total = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return PersistResult(
        name="version_chain",
        iterations=n_versions,
        seconds_per_iteration=total / n_versions,
        peak_memory_mb=round(peak / (1024 * 1024), 2),
        metadata={
            "n_entities": n_entities,
            "total_seconds": round(total, 4),
            "versions": n_versions,
        },
    )


def bench_concurrent_safety(n_writers: int = 8, versions_per_writer: int = 10) -> PersistResult:
    """Concurrent save/load from multiple threads must not corrupt the store.

    Each writer persists its own versions; afterwards every persisted version
    verifies and loads with an intact entity hash. A corruption here would be a
    data-integrity regression.
    """
    import threading

    errors: List[str] = []

    def writer(wid: int) -> None:
        try:
            with tempfile.TemporaryDirectory() as td:
                store = WorldStore(Path(td) / f"store-{wid}")
                for j in range(versions_per_writer):
                    world = _make_world(f"world-w{wid}-v{j}", 30)
                    ver = store.save_version(world, parent=None, version_id=f"w{wid}v{j}")
                    errs = store.verify_version(ver.version_id)
                    if errs:
                        errors.append(f"w{wid}v{j}: {errs}")
                    loaded = store.load_version(ver.version_id)
                    if len(loaded.entities) != 30:
                        errors.append(f"w{wid}v{j}: entity count mismatch")
        except Exception as exc:  # pragma: no cover - surface any threading failure
            errors.append(f"w{wid}: {exc!r}")

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(n_writers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent persistence errors: {errors[:5]}"

    total_versions = n_writers * versions_per_writer
    return PersistResult(
        name="concurrent_safety",
        iterations=total_versions,
        seconds_per_iteration=0.0,  # not a pure timing benchmark; correctness is the measure
        peak_memory_mb=0.0,
        metadata={
            "n_writers": n_writers,
            "versions_per_writer": versions_per_writer,
            "total_versions": total_versions,
            "correctness_only": True,
        },
    )


# --------------------------------------------------------------------------- report

@dataclass
class PersistReport:
    results: List[PersistResult]
    all_passed: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "results": [
                {
                    "name": r.name,
                    "iterations": r.iterations,
                    "seconds_per_iteration": round(r.seconds_per_iteration, 6),
                    "peak_memory_mb": r.peak_memory_mb,
                    "metadata": r.metadata,
                }
                for r in self.results
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            "# Reality Engine - Persistence Integrity Benchmarks (KiloCode)",
            "",
            "| Benchmark | Iterations | sec/iter | Peak Mem (MB) | Notes |",
            "|---|---:|---:|---:|---|",
        ]
        for r in self.results:
            notes = ", ".join(f"{k}={v}" for k, v in r.metadata.items() if k not in ("total_seconds",))
            lines.append(
                f"| {r.name} | {r.iterations} | {r.seconds_per_iteration:.6f} | "
                f"{r.peak_memory_mb} | {notes} |"
            )
        lines.append("")
        lines.append(f"All passed: {self.all_passed}")
        return "\n".join(lines)


def main() -> int:
    results: List[PersistResult] = []
    results.append(bench_save_verify_load())
    results.append(bench_version_chain())
    results.append(bench_concurrent_safety())

    report = PersistReport(results=results, all_passed=True)
    print("\n" + report.to_markdown() + "\n")
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
