"""Tests for benchmarks/suite.py (P19-01: descriptor-driven benchmark
suite -- directive item 18: "Don't hard-code individual buildings ...
Run Victoria Memorial, Fort, Skyscraper, Bridge ... using exactly the
same pipeline").

Contract under test:

  - A BenchmarkDescriptor (name, category, detail_levels, difficulty,
    notes) is DATA, not code: the suite runs every descriptor through
    the SAME runner with no per-benchmark special cases (the
    anti-pattern the directive bans: `if benchmark == ...`).
  - A run produces MEASURED records: stage timings from the real
    harness, entity/ROI counts, tier facts -- no hard-coded results,
    no fabricated accuracy numbers for captures that don't exist
    (ground_truth=None descriptors report accuracy as None, never a
    guessed percentage).
  - Records serialize deterministically to JSON (campaign-report
    seam) and round-trip.
  - A descriptor whose scene is empty/invalid FAILS its run with the
    reason recorded -- a crashing benchmark is not a passing benchmark
    (harness.py's rule, inherited).
  - docs/BENCHMARKS.md reconciliation is testable: the suite exposes
    the machine-readable summary the doc is generated from, so the doc
    cannot silently diverge from the code.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.suite import (
    BenchmarkDescriptor,
    BenchmarkRecord,
    BenchmarkStageTiming,
    run_benchmark_suite,
)
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)


def _scene(points: int = 400) -> ReconstructionResult:
    """A small deterministic synthetic scene: a z=2 wall of points seen
    by two cameras -- the minimal input every stage of the detail spine
    accepts."""
    pts = [
        ReconstructedPoint(
            position=(0.05 * (i % 20) - 0.5, 0.05 * (i // 20) - 0.5, 2.0),
            track_id=f"p{i}",
            source_evidence_ids=["c0", "c1"],
        )
        for i in range(points)
    ]
    poses = [
        ReconstructedCameraPose(evidence_id="c0", position=(0.0, 0.0, 0.0),
                                rotation=(1.0, 0.0, 0.0, 0.0)),
        ReconstructedCameraPose(evidence_id="c1", position=(0.2, 0.0, 0.0),
                                rotation=(1.0, 0.0, 0.0, 0.0)),
    ]
    return ReconstructionResult(points=pts, camera_poses=poses,
                                registration_status="success")


class TestDescriptors:
    def test_descriptor_is_data(self):
        d = BenchmarkDescriptor(
            name="synthetic-room", category="interior",
            detail_levels="L2-L4", difficulty="medium",
        )
        assert d.ground_truth is None  # no fabricated accuracy seam
        assert d.notes == ""

    def test_same_pipeline_for_all_descriptors(self):
        # Two descriptors of different categories run through the same
        # runner; the records must be structurally identical (same
        # stage names, same metric keys) -- the generalization proof.
        d1 = BenchmarkDescriptor(name="room", category="interior")
        d2 = BenchmarkDescriptor(name="facade", category="exterior")
        scene = _scene()
        r1, r2 = run_benchmark_suite([d1, d2], {"room": scene,
                                                "facade": scene})
        assert ([t.stage for t in r1.stage_timings]
                == [t.stage for t in r2.stage_timings])
        assert r1.metrics.keys() == r2.metrics.keys()


class TestMeasuredRecords:
    def test_record_reports_measured_facts(self):
        d = BenchmarkDescriptor(name="synthetic-room", category="interior",
                                detail_levels="L1-L3", difficulty="low")
        record = run_benchmark_suite([d], {"synthetic-room": _scene()})[0]
        assert record.status == "completed"
        assert record.descriptor.name == "synthetic-room"
        # Stage timings are real BenchmarkStageTiming entries with
        # positive measured durations (wall clock, per harness.py).
        stages = {t.stage: t for t in record.stage_timings}
        assert {"quality", "discovery", "roi"} <= set(stages)
        for t in record.stage_timings:
            assert t.seconds >= 0.0
        # Point/entity counts come from the scene, not guesses.
        assert record.metrics["points"] == 400

    def test_empty_scene_fails_with_reason(self):
        d = BenchmarkDescriptor(name="void", category="interior")
        record = run_benchmark_suite([d], {"void": ReconstructionResult(
            points=[], camera_poses=[], registration_status="success")})[0]
        assert record.status == "failed"
        assert record.failure_reason

    def test_record_roundtrips_through_json(self):
        d = BenchmarkDescriptor(name="synthetic-room", category="interior")
        record = run_benchmark_suite([d], {"synthetic-room": _scene()})[0]
        data = json.loads(json.dumps(record.to_dict()))
        restored = BenchmarkRecord.from_dict(data)
        assert restored.to_dict() == record.to_dict()


class TestSuiteSummary:
    def test_summary_is_machine_readable(self):
        ds = [BenchmarkDescriptor(name="room", category="interior"),
              BenchmarkDescriptor(name="void", category="interior")]
        scenes = {"room": _scene(),
                  "void": ReconstructionResult(points=[], camera_poses=[],
                                               registration_status="success")}
        records = run_benchmark_suite(ds, scenes)
        from benchmarks.suite import suite_summary
        summary = suite_summary(records)
        assert summary["total"] == 2
        assert summary["completed"] == 1
        assert summary["failed"] == 1
        assert json.dumps(summary, sort_keys=True)  # serializable
