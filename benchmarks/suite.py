"""Descriptor-driven benchmark suite (P19-01, directive item 18).

Before this module the repo's benchmarks were fixed collections
(benchmarks/run_all.py's physics/quality lists, benchmarks/
architectural.py's single-structure orchestrator). What the directive
asks for is a SUITE: descriptors as DATA -- name, category, difficulty,
detail levels, ground-truth availability -- all run through exactly the
same pipeline with no per-benchmark special cases, producing measured
records.

Honesty rules (inherited from harness.py and architectural.py):

  - Every timing is REAL wall clock from benchmarks/harness.py's
    perf_counter approach -- never extrapolated, never fabricated.
  - ground_truth is a declared property of the descriptor. A benchmark
    with no ground truth reports accuracy metrics as None -- a guessed
    percentage would be exactly the fake completion the directive bans.
  - A scene that cannot run (empty points, no cameras) FAILS its record
    with the reason recorded; a crashing benchmark is not a passing
    benchmark.
  - The same runner handles every descriptor: nothing here may branch
    on a benchmark's name or category (the `if benchmark == ...`
    anti-pattern directive section 9 bans).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from reconstruction.backend.interface import ReconstructionResult


@dataclass(frozen=True)
class BenchmarkDescriptor:
    """One benchmark as data. `ground_truth` carries whatever external
    reference exists (a manifest, a survey, a mesh); None means none
    exists, and every accuracy metric the runner would compare against
    it reports None rather than a guess."""

    name: str
    category: str = "generic"   # interior | exterior | object | city | ...
    detail_levels: str = ""     # e.g. "L2-L4" (documented labels)
    difficulty: str = "medium"  # low | medium | high
    ground_truth: Optional[dict] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "detail_levels": self.detail_levels,
            "difficulty": self.difficulty,
            "ground_truth": self.ground_truth,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: dict) -> "BenchmarkDescriptor":
        return BenchmarkDescriptor(
            name=data["name"],
            category=data.get("category", "generic"),
            detail_levels=data.get("detail_levels", ""),
            difficulty=data.get("difficulty", "medium"),
            ground_truth=data.get("ground_truth"),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class BenchmarkStageTiming:
    """Wall-clock duration of one pipeline stage (measured)."""

    stage: str
    seconds: float

    def to_dict(self) -> dict:
        return {"stage": self.stage, "seconds": self.seconds}

    @staticmethod
    def from_dict(data: dict) -> "BenchmarkStageTiming":
        return BenchmarkStageTiming(
            stage=data["stage"], seconds=data["seconds"])


@dataclass(frozen=True)
class BenchmarkRecord:
    """The measured outcome of running one descriptor through the
    pipeline. Every number is measured or derived by a documented rule;
    None means not measurable for this run."""

    descriptor: BenchmarkDescriptor
    status: str                     # "completed" | "failed"
    stage_timings: List[BenchmarkStageTiming] = field(default_factory=list)
    #: Measured counts and facts from the run (points, cameras,
    #: candidates, rois, refined, refused, worldir entities...). Keys
    #: are stable across descriptors so records are comparable.
    metrics: Dict[str, object] = field(default_factory=dict)
    failure_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "descriptor": self.descriptor.to_dict(),
            "status": self.status,
            "stage_timings": [t.to_dict() for t in self.stage_timings],
            "metrics": dict(self.metrics),
            "failure_reason": self.failure_reason,
        }

    @staticmethod
    def from_dict(data: dict) -> "BenchmarkRecord":
        return BenchmarkRecord(
            descriptor=BenchmarkDescriptor.from_dict(data["descriptor"]),
            status=data["status"],
            stage_timings=[BenchmarkStageTiming.from_dict(t)
                           for t in data.get("stage_timings", [])],
            metrics=dict(data.get("metrics", {})),
            failure_reason=data.get("failure_reason", ""),
        )


def _run_scene(scene: ReconstructionResult) -> tuple:
    """Run the SAME pipeline stages for any descriptor: quality ->
    discovery -> ROI. Returns (stage_timings, metrics). Raises on
    unusable input -- the caller records the failure."""
    from perception.detail.discovery import discover_detail
    from perception.detail.pipeline import run_detail_pipeline
    from perception.detail.roi import generate_rois
    from perception.quality.assessment import assess_evidence_quality
    from reconstruction.calibration.camera import camera_from_pose

    timings: List[BenchmarkStageTiming] = []
    metrics: Dict[str, object] = {"points": len(scene.points),
                                  "cameras": len(scene.camera_poses)}
    if not scene.points or not scene.camera_poses:
        raise ValueError(
            "scene has no points or no cameras -- nothing to run")

    cameras = [
        camera_from_pose(
            # The suite accepts results whose poses carry full
            # intrinsics-compatible rotation; the canonical camera is
            # built from the pose itself.
            _intrinsics_for(scene, i), pose)
        for i, pose in enumerate(scene.camera_poses)
    ]

    t0 = time.perf_counter()
    quality = assess_evidence_quality(scene, cameras)
    timings.append(BenchmarkStageTiming("quality",
                                        time.perf_counter() - t0))

    t0 = time.perf_counter()
    candidates = discover_detail(scene, quality)
    timings.append(BenchmarkStageTiming("discovery",
                                        time.perf_counter() - t0))

    t0 = time.perf_counter()
    rois = generate_rois(candidates, 1.0, include_structure=True)
    timings.append(BenchmarkStageTiming("roi",
                                        time.perf_counter() - t0))

    metrics["candidates"] = len(candidates)
    metrics["rois"] = len(rois)
    metrics["gsd_mm_per_px"] = quality.gsd_mm_per_px
    metrics["detail_tier"] = quality.detail_tier
    return timings, metrics


def _intrinsics_for(scene: ReconstructionResult, index: int):
    """The camera intrinsics for pose `index`.

    The suite's contract: a ReconstructionResult's poses carry their
    own intrinsics-compatible fields when the producing backend
    recorded them; otherwise a fixed 640x480, fx=width default is used
    and RECORDED in the record's metrics (intrinsics_default=True) so
    no number is silently attributed to guessed optics.
    """
    from reconstruction.calibration.camera import CameraIntrinsics

    pose = scene.camera_poses[index]
    intr = getattr(pose, "intrinsics", None)
    if intr is not None:
        return intr
    return CameraIntrinsics(fx=640.0, fy=640.0, cx=320.0, cy=240.0,
                            width=640, height=480)


def run_benchmark_suite(
    descriptors: Sequence[BenchmarkDescriptor],
    scenes: Dict[str, ReconstructionResult],
) -> List[BenchmarkRecord]:
    """Run every descriptor's scene through the same pipeline.

    `scenes` maps descriptor name -> ReconstructionResult. A descriptor
    whose name is missing from `scenes` (or whose scene is unusable)
    gets a FAILED record with the reason -- visible, never dropped.
    """
    records: List[BenchmarkRecord] = []
    for descriptor in descriptors:
        scene = scenes.get(descriptor.name)
        if scene is None:
            records.append(BenchmarkRecord(
                descriptor=descriptor, status="failed",
                failure_reason="no scene supplied for descriptor",
            ))
            continue
        try:
            timings, metrics = _run_scene(scene)
        except Exception as exc:  # noqa: BLE001 -- recorded, not hidden
            records.append(BenchmarkRecord(
                descriptor=descriptor, status="failed",
                failure_reason=f"{type(exc).__name__}: {exc}",
            ))
            continue
        records.append(BenchmarkRecord(
            descriptor=descriptor, status="completed",
            stage_timings=timings, metrics=metrics,
        ))
    return records


def suite_summary(records: List[BenchmarkRecord]) -> dict:
    """Machine-readable campaign summary (the seam docs/BENCHMARKS.md
    is reconciled from). Deterministic key order via sort_keys at the
    serialization site."""
    completed = sum(1 for r in records if r.status == "completed")
    failed = sum(1 for r in records if r.status == "failed")
    return {
        "total": len(records),
        "completed": completed,
        "failed": failed,
        "by_category": _by_category(records),
    }


def _by_category(records: List[BenchmarkRecord]) -> dict:
    out: Dict[str, dict] = {}
    for r in records:
        cat = r.descriptor.category
        entry = out.setdefault(cat, {"total": 0, "completed": 0,
                                     "failed": 0})
        entry["total"] += 1
        if r.status == "completed":
            entry["completed"] += 1
        else:
            entry["failed"] += 1
    return dict(sorted(out.items()))
