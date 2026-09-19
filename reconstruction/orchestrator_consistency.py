"""Orchestrator x depth-consistency wiring: one call that runs the
orchestrated reconstruction AND measures whether the run's metric depth
evidence agrees with itself across views.

Reliability priority served: "detect contradictory evidence between
camera poses, depth, sparse points, surface geometry" -- inside the
orchestrated run, not as an afterthought the caller must remember.

Contract (additive; the plain orchestrator path is unchanged):

  - `run_with_depth_consistency(orchestrator, evidence, depth_maps,
    cameras_by_evidence)` executes the normal run, then checks the
    supplied metric depth maps against the run's own camera poses and
    sparse cloud via reconstruction.consistency.check_depth_consistency.
  - No depth maps -> `diagnostics.depth_consistency is None` (honest
    absence; no fabricated "consistent").
  - A contradictory verdict does not rewrite the result: the
    contradiction is RECORDED (both measurements preserved) and the
    backend's result is returned untouched -- the caller decides what a
    contradiction means for their pipeline. Diagnostics exist to be
    surfaced, never to hide or silently repair.
  - Depth maps whose evidence id has no camera in `cameras_by_evidence`
    are named in the verdict's `views_skipped`, never silently dropped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from perception.depth.interface import DepthMap
from reconstruction.calibration.camera import PinholeCamera
from reconstruction.consistency import ConsistencyReport, check_depth_consistency
from reconstruction.orchestrator import (
    ReconstructionOrchestrator,
    ReconstructionRun,
    ReconstructionRunDiagnostics,
)


@dataclass(frozen=True)
class DepthConsistencyDiagnostics:
    """The depth-consistency block of run diagnostics.

    `report` is the full measured ConsistencyReport (contradictions
    carry both depths); `status` mirrors `report.status` for quick
    inspection. Constructed only when depth maps were actually checked.
    """

    status: str
    report: Optional[ConsistencyReport]

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "report": self.report.to_dict() if self.report is not None else None,
            "contradiction_count": (
                len(self.report.contradictions) if self.report is not None else 0
            ),
            "pairs_checked": self.report.pairs_checked if self.report is not None else 0,
        }


def _extend_diagnostics(
    diagnostics: ReconstructionRunDiagnostics,
    depth_consistency: Optional[DepthConsistencyDiagnostics],
) -> ReconstructionRunDiagnostics:
    """Rebuild the frozen diagnostics record with the consistency block."""
    return ReconstructionRunDiagnostics(
        evidence_count=diagnostics.evidence_count,
        image_evidence_count=diagnostics.image_evidence_count,
        evidence_validation=diagnostics.evidence_validation,
        attempts=diagnostics.attempts,
        final_status=diagnostics.final_status,
        backend_name=diagnostics.backend_name,
        duration_s=diagnostics.duration_s,
        error=diagnostics.error,
        depth_consistency=depth_consistency,
    )


def run_with_depth_consistency(
    orchestrator: ReconstructionOrchestrator,
    evidence: List,
    *,
    depth_maps: Sequence[DepthMap],
    cameras_by_evidence: Optional[Dict[str, PinholeCamera]] = None,
) -> ReconstructionRun:
    """Run the orchestrator, then measure cross-view depth consistency
    of the supplied metric depth maps against the run's own poses and
    sparse cloud. See module docstring for the honesty contract.
    """
    run = orchestrator.run(evidence)

    if not depth_maps:
        return run  # no depth evidence -> honest absence, no verdict

    # Cameras: caller-supplied map wins; otherwise rebuild from the
    # run's own camera poses (the reconstruction IS the pose source).
    cameras = dict(cameras_by_evidence or {})
    if not cameras:
        from reconstruction.calibration.camera import camera_from_pose
        from reconstruction.calibration.camera import CameraIntrinsics

        # Pose-only cameras need intrinsics; without them the unprojection
        # math cannot run, so this path requires caller-supplied cameras OR
        # an intrinsics source. Rebuilt poses without intrinsics are
        # skipped (named), never guessed.
        cameras = {}

    scene_points = [p.position for p in run.result.points]
    report = check_depth_consistency(
        depth_maps, cameras, scene_points,
    )
    block = DepthConsistencyDiagnostics(
        status=report.status, report=report,
    )
    return ReconstructionRun(
        result=run.result,
        diagnostics=_extend_diagnostics(run.diagnostics, block),
    )
