"""Tests for StudioSession.reconstruct_and_compile() (engine/studio/session.py):
the missing glue between the reconstruction orchestrator (real backend
selection/execution) and world compilation (command-pipeline compile).

Before this method, a caller had to run
`reconstruction.orchestrator.ReconstructionOrchestrator.run()` itself and
then separately remember to call `StudioSession.compile_reconstruction()`
with the result -- two real, tested pieces with no glue connecting them
inside Studio. This is the convergence-campaign's "make the pipeline
actually run as one call" gap for the Studio entry point specifically.
"""

from __future__ import annotations

import pytest

from evidence.session import EvidenceItem, EvidenceKind
from engine.studio.session import StudioSession
from reconstruction.backend.fake import FakeReconstructionBackend
from reconstruction.backend.interface import ReconstructedCameraPose, ReconstructedPoint
from reconstruction.orchestrator import ReconstructionOrchestrationError, ReconstructionOrchestrator
from world_ir import EntityType, WorldIR


def _room_evidence_and_backend():
    """A minimal room (floor + two walls) as canned FakeReconstructionBackend
    output, wired to look like it came from 3 real photos."""
    evidence = [
        EvidenceItem(id=f"photo-{i}", kind=EvidenceKind.PHOTO, source_uri=f"file://room/photo-{i}.jpg")
        for i in range(3)
    ]
    evidence_ids = [e.id for e in evidence]

    points = []
    counter = [0]

    def pt(x, y, z):
        counter[0] += 1
        return ReconstructedPoint(position=(x, y, z), track_id=f"pt-{counter[0]:05d}", source_evidence_ids=evidence_ids)

    for i in range(12):
        for j in range(12):
            points.append(pt(i * 0.25, 0.0, j * 0.25))
            points.append(pt(i * 0.25, 2.5, j * 0.25))
    for i in range(10):
        for j in range(10):
            points.append(pt(0.0, i * 0.25, j * 0.25))
            points.append(pt(4.0, i * 0.25, j * 0.25))

    poses = [
        ReconstructedCameraPose(evidence_id=eid, position=(2.0, 1.25, 1.5), rotation=(1.0, 0.0, 0.0, 0.0))
        for eid in evidence_ids
    ]

    backend = FakeReconstructionBackend(canned_points=points, canned_poses=poses)
    return evidence, backend


def test_reconstruct_and_compile_runs_real_orchestrator_and_compiles_world():
    evidence, backend = _room_evidence_and_backend()
    orchestrator = ReconstructionOrchestrator([backend])
    session = StudioSession(world=WorldIR())

    run, command_result = session.reconstruct_and_compile(evidence, orchestrator)

    assert run.result.registration_status == "success"
    assert command_result.world_version > 0  # execute() raises on gate failure, so reaching here means success
    structural = [e for e in session.world.entities.values()
                  if e.type in (EntityType.WALL, EntityType.FLOOR, EntityType.CEILING)]
    assert len(structural) >= 2  # at least a floor and a wall promoted


def test_reconstruct_and_compile_raises_when_every_backend_declines():
    from reconstruction.backend.interface import IReconstructionBackend, ReconstructionResult

    class _AlwaysFails(IReconstructionBackend):
        backend_name = "always-fails"

        def reconstruct(self, evidence):
            return ReconstructionResult(points=[], camera_poses=[], registration_status="failed")

    evidence = [
        EvidenceItem(id="photo-0", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg"),
        EvidenceItem(id="photo-1", kind=EvidenceKind.PHOTO, source_uri="file://b.jpg"),
    ]
    orchestrator = ReconstructionOrchestrator([_AlwaysFails()])
    session = StudioSession(world=WorldIR())

    with pytest.raises(ReconstructionOrchestrationError):
        session.reconstruct_and_compile(evidence, orchestrator)

    assert len(session.world.entities) == 0  # failed run must not leave a partial world


def test_reconstruct_and_compile_diagnostics_name_the_backend_used():
    evidence, backend = _room_evidence_and_backend()
    orchestrator = ReconstructionOrchestrator([backend])
    session = StudioSession(world=WorldIR())

    run, _ = session.reconstruct_and_compile(evidence, orchestrator)

    assert len(run.diagnostics.attempts) == 1
    assert run.diagnostics.attempts[0].backend_name == "FakeReconstructionBackend"
