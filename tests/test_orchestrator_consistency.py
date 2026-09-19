"""Orchestrator x consistency wiring (reliability priority: detect
contradictory evidence between depth and poses INSIDE the orchestrated
run, so the diagnostics answer "did the depth evidence agree with
itself" for every run that carries metric depth maps).

`run_with_depth_consistency` is the additive entry: it runs the normal
orchestrated reconstruction, then -- only when the caller supplies the
depth maps the pipeline actually produced -- checks cross-view
consistency of those maps against the run's own camera poses and sparse
cloud, and records the verdict in the run diagnostics. A run without
depth maps is untouched. A contradictory verdict does NOT fabricate a
better result: the contradiction is reported, preserving both
measurements.
"""

from __future__ import annotations

import math

import pytest

from engine.physics.math3 import Vec3
from perception.depth.interface import DepthMap
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.orchestrator import (
    ReconstructionOrchestrator,
    ReconstructionRunDiagnostics,
)
from reconstruction.orchestrator_consistency import (
    DepthConsistencyDiagnostics,
    run_with_depth_consistency,
)
from evidence.session import EvidenceItem, EvidenceKind

from tests.test_depth_consistency import (
    _camera,
    _depth_map,
    _two_camera_fixture,
)


def _evidence(*ids):
    return [
        EvidenceItem(
            id=eid,
            kind=EvidenceKind.PHOTO,
            source_uri=f"{eid}.jpg",
        )
        for eid in ids
    ]


def _canned(points, poses):
    from reconstruction.backend.fake import FakeReconstructionBackend

    class _B:
        backend_name = "canned"

        def __init__(self, pts, ps):
            self._inner = FakeReconstructionBackend(pts, ps)

        def reconstruct(self, evidence):
            return self._inner.reconstruct(evidence)

    return _B(points, poses)


def test_run_without_depth_maps_has_no_consistency_block():
    """No depth maps supplied -> diagnostics carry no consistency data
    (honest absence, not a fabricated 'consistent')."""
    orch = ReconstructionOrchestrator([_canned(
        [ReconstructedPoint(position=(0, 0, 5), track_id="t",
                            source_evidence_ids=["a-img-0", "a-img-1"])],
        [ReconstructedCameraPose(evidence_id="a-img-0", position=(0, 0, 0),
                                 rotation=(0, 0, 0, 1)),
         ReconstructedCameraPose(evidence_id="a-img-1", position=(6, 0, 0),
                                 rotation=(0, 0, 0, 1))],
    )])
    run = run_with_depth_consistency(orch, _evidence("a-img-0", "a-img-1"), depth_maps=[])
    assert run.result.registration_status == "success"
    assert run.diagnostics.depth_consistency is None


def test_consistent_depth_maps_report_consistent():
    maps, cameras = _two_camera_fixture()
    # Sparse points = the shared scene; poses = the fixture cameras.
    # Evidence ids must match the fixture's camera ids ("view-a"/"view-b")
    # so the fake backend's canned lookup keeps points and poses.
    from tests.test_depth_consistency import SCENE
    points = [
        ReconstructedPoint(position=p, track_id=f"t{i}",
                           source_evidence_ids=["view-a", "view-b"])
        for i, p in enumerate(SCENE)
    ]
    poses = [
        ReconstructedCameraPose(
            evidence_id=eid,
            position=(cam.extrinsics.position.x, cam.extrinsics.position.y,
                      cam.extrinsics.position.z),
            rotation=(cam.extrinsics.rotation.w, cam.extrinsics.rotation.x,
                      cam.extrinsics.rotation.y, cam.extrinsics.rotation.z),
        )
        for eid, cam in cameras.items()
    ]
    orch = ReconstructionOrchestrator([_canned(points, poses)])
    run = run_with_depth_consistency(
        orch, _evidence("view-a", "view-b"), depth_maps=maps,
        cameras_by_evidence=cameras,
    )
    dc = run.diagnostics.depth_consistency
    assert dc is not None
    assert dc.report.status == "consistent"
    assert dc.status == "consistent"


def test_contradictory_depth_maps_surface_in_diagnostics():
    maps, cameras = _two_camera_fixture(bias_b=1.0)
    from tests.test_depth_consistency import SCENE
    points = [
        ReconstructedPoint(position=p, track_id=f"t{i}",
                           source_evidence_ids=["view-a", "view-b"])
        for i, p in enumerate(SCENE)
    ]
    poses = [
        ReconstructedCameraPose(
            evidence_id=eid,
            position=(cam.extrinsics.position.x, cam.extrinsics.position.y,
                      cam.extrinsics.position.z),
            rotation=(cam.extrinsics.rotation.w, cam.extrinsics.rotation.x,
                      cam.extrinsics.rotation.y, cam.extrinsics.rotation.z),
        )
        for eid, cam in cameras.items()
    ]
    orch = ReconstructionOrchestrator([_canned(points, poses)])
    run = run_with_depth_consistency(
        orch, _evidence("view-a", "view-b"), depth_maps=maps,
        cameras_by_evidence=cameras,
    )
    dc = run.diagnostics.depth_consistency
    assert dc is not None
    assert dc.status == "contradictions"
    assert len(dc.report.contradictions) > 0
    # The verdict is recorded, not swallowed -- and the run result is
    # still the backend's own (no fabricated cleanup).
    assert run.result.registration_status == "success"
    d = run.diagnostics.to_dict()["depth_consistency"]
    assert d["status"] == "contradictions"
    assert d["contradiction_count"] > 0


def test_depth_consistency_to_dict_roundtrip():
    dc = DepthConsistencyDiagnostics(status="consistent", report=None)
    d = dc.to_dict()
    assert d["status"] == "consistent"
