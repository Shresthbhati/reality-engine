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


class IReconstructionBackend(ABC):
    @abstractmethod
    def reconstruct(self, evidence: List[EvidenceItem]) -> ReconstructionResult:
        """Run structure-from-motion/MVS over a batch of PHOTO/VIDEO evidence.

        Must raise, not fabricate output, if the evidence is insufficient
        (too few images, no overlap) -- registration_status exists so
        partial results are reported honestly rather than silently.
        """
        ...
