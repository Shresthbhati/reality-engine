"""Deterministic fake IReconstructionBackend for exercising the pipeline
end-to-end (evidence -> reconstruction -> promote -> WorldIR) before any
real CV dependency (COLMAP) is wired in. Not a placeholder for production
use -- it invents no geometry it isn't told to produce; callers supply the
points/poses via a canned mapping so tests stay deterministic.
"""

from __future__ import annotations

from typing import Dict, List

from provenance import Uncertainty

from evidence.session import EvidenceItem, EvidenceKind
from .interface import (
    IReconstructionBackend,
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)


class FakeReconstructionBackend(IReconstructionBackend):
    """Returns pre-canned results keyed by evidence id, for pipeline tests.

    ponytail: no real SfM -- swap for a COLMAP-backed backend when that
    integration is built; this class exists only to unblock testing the
    rest of the pipeline against the IReconstructionBackend contract.
    """

    def __init__(self, canned_points: List[ReconstructedPoint] = None,
                 canned_poses: List[ReconstructedCameraPose] = None):
        self._canned_points = canned_points or []
        self._canned_poses = canned_poses or []

    def reconstruct(self, evidence: List[EvidenceItem]) -> ReconstructionResult:
        image_evidence = [e for e in evidence if e.kind in (EvidenceKind.PHOTO, EvidenceKind.VIDEO)]
        if len(image_evidence) < 2:
            return ReconstructionResult(points=[], camera_poses=[], registration_status="failed")

        evidence_ids = {e.id for e in image_evidence}
        poses = [p for p in self._canned_poses if p.evidence_id in evidence_ids]
        points = [p for p in self._canned_points
                  if evidence_ids.intersection(p.source_evidence_ids)]

        if not points or not poses:
            return ReconstructionResult(points=[], camera_poses=[], registration_status="failed")

        status = "success" if len(poses) == len(image_evidence) else "partial"
        return ReconstructionResult(points=points, camera_poses=poses, registration_status=status)
