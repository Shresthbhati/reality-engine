"""Reconstruction backend abstraction (Goal C, see docs/RECONSTRUCTION_BACKEND_DECISION.md).

Mirrors engine/physics/backend/interface.py's pattern: the canonical
architecture depends on this interface, never on a specific CV library.
COLMAP was the backend decided on (BSD license, RECONSTRUCTED-provenance
output maps onto Observation/Measurement) -- nothing here calls it yet.
This is the interface only; no backend implements it in this pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from evidence.session import EvidenceItem
from provenance import Uncertainty


@dataclass(frozen=True)
class ReconstructedPoint:
    """One 3D point recovered from imagery, track_id ties it back to source photos."""
    position: tuple  # (x, y, z)
    track_id: str
    source_evidence_ids: List[str]
    uncertainty: Uncertainty = field(default_factory=Uncertainty)


@dataclass(frozen=True)
class ReconstructedCameraPose:
    evidence_id: str  # the photo this pose belongs to
    position: tuple
    rotation: tuple  # quaternion (w, x, y, z)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)


@dataclass(frozen=True)
class ReconstructionResult:
    """Everything a backend recovers from one batch of photo/video evidence.

    Deliberately does not include a WorldIR Entity -- promoting this into
    WorldIR is a separate step (analogous to evidence/promote.py), so a
    reconstruction failure never leaves a half-built entity behind.
    """
    points: List[ReconstructedPoint]
    camera_poses: List[ReconstructedCameraPose]
    registration_status: str  # "success" | "partial" | "failed" -- never fabricated
    #: CrossSessionReport.to_dict() when this result was produced by
    #: MERGING multiple sub-models (diagnostics must survive: a false
    #: merge with a discarded report leaves no trace). None = single
    #: model, no merge happened.
    merge_report: Optional[dict] = None
    #: Fused dense-MVS cloud parsed from the backend's own fused.ply
    #: (ReconstructedPoint list, track ids "dense:<i>", provenance =
    #: the registered evidence ids). None = no dense run happened
    #: (default; dense is an explicit opt-in continuation).
    dense_points: Optional[List[ReconstructedPoint]] = None
    #: The dense run's observed facts: DenseMVSRun.to_dict() plus the
    #: sub-model it continued from ("sparse_model"), the parse facts
    #: ("parse_facts"), and provenance ("source_evidence_ids"). None =
    #: no dense run. A dense failure RAISES -- it never degrades into a
    #: sparse-only success, so this field's presence is proof of a
    #: completed, parsed dense chain.
    dense_report: Optional[dict] = None


class IReconstructionBackend(ABC):
    @abstractmethod
    def reconstruct(self, evidence: List[EvidenceItem]) -> ReconstructionResult:
        """Run structure-from-motion/MVS over a batch of PHOTO/VIDEO evidence.

        Must raise, not fabricate output, if the evidence is insufficient
        (too few images, no overlap) -- registration_status exists so
        partial results are reported honestly rather than silently.
        """
        ...
