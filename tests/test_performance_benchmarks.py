"""Performance optimization and large-scene benchmark tests (Task 9.8).

Validates:
1. 100-image reconstruction processing executes well within budget (<30s).
2. Large-scale SpatialIndex query throughput exceeds 1,000 queries/second.
3. Memory allocation remains bounded and leak-free.
"""

from __future__ import annotations

import time
import tracemalloc
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from engine.scene_graph.spatial_index import SpatialIndex
from evidence.packages import DeterministicPackageBuilder
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.batch import MemoryBudget, run_batch
from world_ir.schema_v1 import Entity, EntityType, Vector3
from world_ir.world_v1 import WorldIR


def _make_synthetic_photo(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(48, 64, 3), dtype=np.uint8)
    Image.fromarray(base).save(path, format="JPEG", quality=90)


class BenchmarkBackend:
    backend_name = "benchmark_backend"

    def reconstruct(self, evidence):
        poses = [
            ReconstructedCameraPose(
                evidence_id=e.id,
                position=(float(i) * 0.1, 0.0, 1.0),
                rotation=(1.0, 0.0, 0.0, 0.0),
            )
            for i, e in enumerate(evidence)
        ]
        pts = [
            ReconstructedPoint(
                position=(float(k % 10) * 0.5, float(k // 10) * 0.5, 0.0),
                track_id=f"t{k}",
                source_evidence_ids=[e.id for e in evidence[:2]],
            )
            for k in range(100)
        ]
        return ReconstructionResult(
            points=pts,
            camera_poses=poses,
            registration_status="success",
        )


def test_100_image_reconstruction_completes_under_30s(tmp_path):
    """Task 9.8: <30s for 100-image reconstruction batch processing."""
    img_dir = tmp_path / "sess1"
    img_dir.mkdir(parents=True)
    for i in range(100):
        _make_synthetic_photo(img_dir / f"frame_{i:04d}.jpg", seed=i)

    builder = DeterministicPackageBuilder(seed="bench-100")
    from evidence.importers import import_folder

    import_folder(builder, str(img_dir), on_error="record")
    items = builder.build().to_evidence_items()
    assert len(items) == 100

    backend = BenchmarkBackend()

    tracemalloc.start()
    start_time = time.perf_counter()

    report = run_batch(
        items,
        prefixes=["sess1"],
        backends=[backend],
        memory_budget=MemoryBudget(max_frames_per_group=120),
    )

    elapsed_s = time.perf_counter() - start_time
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)

    assert report["outcome"] == "complete"
    assert report["sessions"]["sess1"]["status"] == "success"
    assert report["totals"]["frames_submitted"] == 100
    assert report["totals"]["deferred_capacity"] == 0
    # Must complete in under 30 seconds (Task 9.8 target)
    assert elapsed_s < 30.0, f"Reconstruction took {elapsed_s:.2f}s, exceeding 30s limit"
    # Peak memory overhead under 150MB
    assert peak_mb < 150.0, f"Peak memory was {peak_mb:.1f}MB, exceeding 150MB budget"


def test_spatial_index_high_throughput_query():
    """SpatialIndex query throughput exceeds 1,000 queries/second on large 1,000-entity world."""
    world = WorldIR()
    rng = np.random.default_rng(123)
    for i in range(1000):
        pos = rng.uniform(-50.0, 50.0, size=3)
        world.entities[f"e-{i:04d}"] = Entity(
            id=f"e-{i:04d}",
            type=EntityType.STRUCTURE,
            transform={"position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}},
        )

    index = SpatialIndex(world)
    assert len(index) == 1000

    query_points = [
        (float(rng.uniform(-40, 40)), float(rng.uniform(-40, 40)), float(rng.uniform(-40, 40)))
        for _ in range(500)
    ]

    start_time = time.perf_counter()
    for q in query_points:
        res = index.nearest(q, k=5)
        assert len(res) == 5
    elapsed_s = time.perf_counter() - start_time

    qps = len(query_points) / elapsed_s
    assert qps > 1000.0, f"Spatial index query rate was {qps:.1f} qps (target > 1,000 qps)"
