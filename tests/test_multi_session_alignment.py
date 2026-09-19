"""Vertical slice: multi-session reconstruction -> cross-session
alignment (reliability priorities: cross-session registration + honest
statuses, wired into the existing backend contract -- not an isolated
module).

`align_reconstructed_sessions` consumes ReconstructionResults straight
from IReconstructionBackend runs (via the orchestrator) and resolves
every session into the reference session's frame. Sessions that share
no structure stay "unresolved"; nothing assumes identity alignment.

Fixtures are deterministic unit scenes (explicitly synthetic per repo
rules); real-data execution is recorded separately in the ledgers.
"""

from __future__ import annotations

import math

import pytest

from engine.math import Quat, Vec3
from provenance import Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.orchestrator import ReconstructionOrchestrator
from registration.cross_session import (
    align_reconstructed_sessions,
    result_to_points,
)
from evidence.session import EvidenceItem, EvidenceKind


def _grid_points(prefix, offset=(0.0, 0.0, 0.0), yaw_deg=0.0):
    """A 5x5 grid on z=offset.z, rotated by yaw about z. Deterministic."""
    half = math.radians(yaw_deg) / 2.0
    quat = Quat(w=math.cos(half), x=0.0, y=0.0, z=math.sin(half)).normalized()
    points = []
    for x in range(5):
        for y in range(5):
            p = quat.rotate(Vec3(float(x), float(y), 0.0))
            points.append(
                ReconstructedPoint(
                    position=(
                        p.x + offset[0], p.y + offset[1], p.z + offset[2]
                    ),
                    track_id=f"{prefix}-{x}-{y}",
                    source_evidence_ids=[f"{prefix}-img-{x}"],
                    uncertainty=Uncertainty(confidence=0.9),
                )
            )
    return points


def _poses(prefix, count=2):
    return [
        ReconstructedCameraPose(
            evidence_id=f"{prefix}-img-{i}",
            position=(float(i), 0.0, 0.0),
            rotation=(0.0, 0.0, 0.0, 1.0),
            uncertainty=Uncertainty(confidence=0.9),
        )
        for i in range(count)
    ]


def _evidence(prefix, count=2):
    return [
        EvidenceItem(
            id=f"{prefix}-img-{i}",
            kind=EvidenceKind.PHOTO,
            source_uri=f"{prefix}/img{i}.jpg",
        )
        for i in range(count)
    ]


def test_result_to_points_preserves_positions():
    result = ReconstructionResult(
        points=_grid_points("a"),
        camera_poses=_poses("a"),
        registration_status="success",
    )
    pts = result_to_points(result)
    assert len(pts) == 25
    assert pts[0].x == pytest.approx(0.0)
    assert pts[-1].z == pytest.approx(0.0)


def test_aligned_session_result_maps_into_reference_frame():
    """The flagship path: session-b's ReconstructionResult, produced in
    its own frame, comes back with points transformed INTO session-a's
    frame and a recorded real transform."""
    ref = ReconstructionResult(
        points=_grid_points("a"), camera_poses=_poses("a"),
        registration_status="success",
    )
    other = ReconstructionResult(
        points=_grid_points("b", offset=(3.0, -1.5, 0.2), yaw_deg=30.0),
        camera_poses=_poses("b"),
        registration_status="success",
    )
    report = align_reconstructed_sessions(
        {"session-a": ref, "session-b": other}, reference_session="session-a",
    )
    assert report.status_by_session["session-a"] == "reference"
    assert report.status_by_session["session-b"] == "aligned"
    t = report.transforms["session-b"]
    assert t is not None
    # Round-trip: b's grid must land on a's grid (same structure).
    a_pts = {(round(p.position[0], 1), round(p.position[1], 1)) for p in ref.points}
    mapped = 0
    for p in other.points:
        q = t.rotation.rotate(Vec3(*p.position)) + t.translation
        if (round(q.x, 1), round(q.y, 1)) in a_pts:
            mapped += 1
    assert mapped >= 20  # the vast majority land on real counterparts


def test_disjoint_session_is_unresolved_not_identity_placed():
    ref = ReconstructionResult(
        points=_grid_points("a"), camera_poses=_poses("a"),
        registration_status="success",
    )
    far = ReconstructionResult(
        points=_grid_points("c", offset=(100.0, 100.0, 0.0)),
        camera_poses=_poses("c"),
        registration_status="success",
    )
    report = align_reconstructed_sessions(
        {"session-a": ref, "session-c": far}, reference_session="session-a",
    )
    assert report.status_by_session["session-c"] == "unresolved"
    assert report.transforms["session-c"] is None
    assert "session-c" in report.reasons["session-c"]


def test_failed_session_is_skipped_with_reason():
    ref = ReconstructionResult(
        points=_grid_points("a"), camera_poses=_poses("a"),
        registration_status="success",
    )
    dead = ReconstructionResult(
        points=[], camera_poses=[], registration_status="failed",
    )
    report = align_reconstructed_sessions(
        {"session-a": ref, "session-d": dead}, reference_session="session-a",
    )
    assert report.status_by_session["session-d"] == "unresolved"
    assert "failed" in report.reasons["session-d"].lower()


def test_alignment_is_deterministic():
    ref = ReconstructionResult(
        points=_grid_points("a"), camera_poses=_poses("a"),
        registration_status="success",
    )
    other = ReconstructionResult(
        points=_grid_points("b", offset=(3.0, -1.5, 0.2), yaw_deg=30.0),
        camera_poses=_poses("b"),
        registration_status="success",
    )
    r1 = align_reconstructed_sessions(
        {"session-a": ref, "session-b": other}, reference_session="session-a",
    )
    r2 = align_reconstructed_sessions(
        {"session-a": ref, "session-b": other}, reference_session="session-a",
    )
    assert r1.to_dict() == r2.to_dict()


def test_unknown_reference_session_refused():
    ref = ReconstructionResult(
        points=_grid_points("a"), camera_poses=_poses("a"),
        registration_status="success",
    )
    with pytest.raises(ValueError):
        align_reconstructed_sessions(
            {"session-a": ref}, reference_session="not-there",
        )


def test_orchestrator_multi_batch_vertical_slice():
    """Full slice: two orchestrator runs (separate capture batches) ->
    align_reconstructed_sessions resolves batch B into batch A's frame.
    This is the integration the reliability mission asks for: multiple
    sessions entering one world frame through REAL registration."""
    backend_a = _CannedBackend(_grid_points("a"), _poses("a"))
    backend_b = _CannedBackend(
        _grid_points("b", offset=(3.0, -1.5, 0.2), yaw_deg=30.0), _poses("b"),
    )
    orch = ReconstructionOrchestrator([backend_a])
    run_a = orch.run(_evidence("a"))
    orch_b = ReconstructionOrchestrator([backend_b])
    run_b = orch_b.run(_evidence("b"))

    report = align_reconstructed_sessions(
        {"batch-a": run_a.result, "batch-b": run_b.result},
        reference_session="batch-a",
    )
    assert run_a.result.registration_status == "success"
    assert run_b.result.registration_status == "success"
    assert report.status_by_session["batch-b"] == "aligned"
    assert report.transforms["batch-b"] is not None


class _CannedBackend:
    """Minimal IReconstructionBackend-compatible canned backend (the
    repo's FakeReconstructionBackend is import-path private to tests;
    this mirrors it without inventing geometry)."""

    def __init__(self, points, poses):
        from reconstruction.backend.fake import FakeReconstructionBackend

        self._inner = FakeReconstructionBackend(points, poses)
        self.backend_name = "canned"

    def reconstruct(self, evidence):
        return self._inner.reconstruct(evidence)
