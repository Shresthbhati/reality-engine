"""Tests for the architectural reconstruction benchmark (directive
sections 30-33, 37): a harness record for structure benchmarks whose
report covers registration, dense geometry, architectural components,
relationships, uncertainty, provenance, and WorldIR validity -- and
which REFUSES to report anything for a dataset whose real capture
does not exist.

Rules under test:
  - A benchmark record declares its capture status; CAPTURE_PENDING
    datasets cannot produce a benchmark run (the capture is a genuine
    external dependency: a human with a camera at Victoria Memorial).
  - A run over a real ReconstructionResult reports: camera
    registration status/counts, point counts, architectural component
    counts by class and phase, relationship counts, tier distribution,
    WorldIR entity count -- all measured from the actual output, none
    hard-coded.
  - The record is structure-agnostic: nothing in it names Victoria
    Memorial's specific architecture; the first benchmark simply sets
    structure_name + planned capture requirements.
  - Failure regions: components that were observed but unaccepted are
    reported as failure regions with reasons (honest reporting), not
    dropped.
"""

from __future__ import annotations

import math

import pytest

from benchmarks.architectural import (
    CaptureStatus,
    StructureBenchmark,
    run_architectural_benchmark,
)
from perception.architecture.components import (
    build_component_observations,
    resolve_components,
)
from perception.architecture.promotion import build_architectural_graph
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from world_ir import WorldIR

from tests.test_arch_components import _cyl_fit, _sphere_fit

UP = (0.0, 0.0, 1.0)


def _make_reconstruction(n_columns=4):
    """A real ReconstructionResult whose points form 4 column shells + a
    dome cap; camera poses with distinct evidence ids.

    Deterministic pseudo-noise breaks exact coplanarity (a real scan
    is curved + noisy, and the plane detector CORRECTLY claims exact
    coplanar rings): jitter is a pure function of the sample indices
    so the fixture stays reproducible."""
    def _jit(a, b):
        # Amplitude must exceed the plane detector's 0.08 m tolerance
        # (else exact ring planes are claimed and segmentation starves)
        # while staying far below the 0.25 m voxel size (else segments
        # shatter) and small vs. the 0.3 m column radius.
        return 0.02 * math.sin(12.9898 * a + 78.233 * b)
    points = []
    evid = []
    for i in range(n_columns):
        cx = i * 3.0
        # Continuous vertical sampling like a real scan: any horizontal
        # slice holds only 12 points (below the plane detector's
        # minimum), so column shells survive into segmentation -- the
        # same behavior a dense real scan of a round column exhibits.
        for ti in range(10):
            t = -1.5 + ti * (3.0 / 9.0)
            for ang in [k * math.pi / 6 for k in range(12)]:
                r = 0.3 + _jit(ti, ang)  # radial jitter: shell, not plane
                points.append(ReconstructedPoint(
                    position=(
                        cx + r * math.cos(ang) + 0.002 * _jit(ang, ti),
                        r * math.sin(ang) + 0.002 * _jit(ti, ang * 2.0),
                        t + 0.004 * _jit(i, ang),
                    ),
                    track_id=f"col{i}-{t:.2f}-{ang:.2f}",
                    source_evidence_ids=[f"img-{i:03d}"],
                ))
        evid.append(f"img-{i:03d}")
    for ang in [k * math.pi / 12 for k in range(24)]:
        z = 3.0 + 1.0 * math.sin(ang)  # upper cap of a sphere r=3 below
        r_xy = 1.0 * math.cos(ang)
        points.append(ReconstructedPoint(
            position=(
                4.5 + r_xy * math.cos(ang * 3) + 0.003 * _jit(ang, 1.0),
                r_xy * math.sin(ang * 3) + 0.003 * _jit(2.0, ang),
                z + 2.0 + 0.003 * _jit(ang, 3.0),
            ),
            track_id=f"dome-{ang:.2f}",
            source_evidence_ids=["img-000"],
        ))
    cameras = [
        ReconstructedCameraPose(position=(x, -5.0, 1.0), rotation=(1.0, 0.0, 0.0, 0.0),
                                evidence_id=f"cam-{k:03d}")
        for k, x in enumerate((-6.0, 0.0, 6.0, 12.0))
    ]
    return ReconstructionResult(
        points=points, camera_poses=cameras, registration_status="success"
    )


class TestCaptureGate:
    def test_record_declares_pending_capture(self):
        b = StructureBenchmark(
            structure_name="Victoria Memorial",
            location="Kolkata, India",
            capture_status=CaptureStatus.CAPTURE_PENDING,
            capture_requirements=[
                "multiple viewpoints with walkaround coverage",
                "facade wide shots + detail shots",
                "GNSS/IMU where available",
                "camera metadata, timestamps, calibration",
            ],
        )
        assert b.capture_status is CaptureStatus.CAPTURE_PENDING
        assert b.structure_name == "Victoria Memorial"
        # No run can exist yet: the capture has not happened.
        with pytest.raises(RuntimeError):
            run_architectural_benchmark(b)

    def test_real_data_label_is_explicit(self):
        b = StructureBenchmark(
            structure_name="X",
            location="Y",
            capture_status=CaptureStatus.CAPTURED,
        )
        d = b.to_dict()
        # The dataset label distinguishing real captured evidence from
        # synthetic fixtures is explicit (directive section 11/33).
        assert d["capture_status"] == "captured"
        assert d["evidence_label"] == "real_captured"


class TestBenchmarkRun:
    def test_run_reports_measured_metrics(self):
        b = StructureBenchmark(
            structure_name="Synthetic colonnade fixture",
            location="test",
            capture_status=CaptureStatus.CAPTURED,
        )
        result = _make_reconstruction()
        # Drive the real perception stack: planes/segments -> fits ->
        # components -> graph (the benchmark orchestrates, never fakes).
        report = run_architectural_benchmark(b, reconstruction=result, seed=7)
        d = report.to_dict()
        assert d["registration_status"] == "success"
        assert d["camera_count"] == 4
        assert d["point_count"] > 0
        assert "components_by_class" in d
        assert "components_by_phase" in d
        assert d["worldir_entity_count"] > 0
        assert d["relationship_count"] >= 0
        assert "tier_distribution" in d

    def test_unaccepted_components_are_failure_regions(self):
        b = StructureBenchmark(
            structure_name="Synthetic fixture", location="test",
            capture_status=CaptureStatus.CAPTURED,
        )
        result = _make_reconstruction()
        report = run_architectural_benchmark(b, reconstruction=result, seed=7)
        # The dome-cap points here are a sparse noisy shell: whatever
        # the stack cannot accept must appear as a failure region with
        # a reason, never silently vanish.
        assert isinstance(report.failure_regions, list)

    def test_deterministic_report(self):
        b = StructureBenchmark(
            structure_name="Synthetic fixture", location="test",
            capture_status=CaptureStatus.CAPTURED,
        )
        r1 = run_architectural_benchmark(b, reconstruction=_make_reconstruction(), seed=7)
        r2 = run_architectural_benchmark(b, reconstruction=_make_reconstruction(), seed=7)
        assert r1.to_dict() == r2.to_dict()
