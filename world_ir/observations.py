"""Observation ledger and association-decision contracts (plan §2, Phase 1).

An `Observation` is one 2D detection lifted into the world frame, carrying the
image it came from, which model produced it (name@version + licence, so
non-commercial output stays traceable) and an honest position uncertainty:
monocular ("relative") depth widens that uncertainty a lot, metric/MVS depth
does not. An `AssociationDecision` is the recorded outcome of asking "is this
the same physical thing as an entity we already track?" -- including the cases
where the answer is "can't tell" (AMBIGUOUS) or "different".

Additive and standalone: nothing in WorldIR imports this yet (persisting the
ledger into WorldStore versions is Phase 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

Vec = Tuple[float, float, float]


class DepthSource(str, Enum):
    METRIC = "metric"      # calibrated / sensor depth
    MVS = "mvs"            # multi-view stereo depth
    RELATIVE = "relative"  # monocular, scale unknown


class AssociationState(str, Enum):
    SAME_ENTITY_CONFIRMED = "SAME_ENTITY_CONFIRMED"
    SAME_SURFACE_CONFIRMED = "SAME_SURFACE_CONFIRMED"
    DIFFERENT_ENTITY = "DIFFERENT_ENTITY"
    PHYSICAL_INTERSECTION = "PHYSICAL_INTERSECTION"
    IMAGE_OCCLUSION = "IMAGE_OCCLUSION"
    PARTIAL_OVERLAP = "PARTIAL_OVERLAP"
    AMBIGUOUS = "AMBIGUOUS"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


#: 1-sigma position error as a fraction of camera range, per depth source, with an absolute floor.
#: Starting points, NOT tuned on any dataset (Appendix B rule 11: no accuracy claim without ground truth).
SIGMA_FRACTION_OF_RANGE = {DepthSource.METRIC: 0.02, DepthSource.MVS: 0.03, DepthSource.RELATIVE: 0.25}
SIGMA_FLOOR_M = 0.05


def sigma_for(depth_source: DepthSource, range_m: float) -> float:
    """Isotropic 1-sigma position uncertainty (m) of a point seen at `range_m` with this depth source."""
    return max(SIGMA_FLOOR_M, SIGMA_FRACTION_OF_RANGE[depth_source] * max(range_m, 0.0))


@dataclass(frozen=True)
class Observation:
    observation_id: str
    image_id: str
    label: str
    world_pos: Vec
    sigma_m: float
    depth_source: DepthSource
    model: str = "unknown@unknown"
    license: str = "unknown"
    normal: Optional[Vec] = None
    bounds_min: Optional[Vec] = None
    bounds_max: Optional[Vec] = None
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "observation_id": self.observation_id,
            "image_id": self.image_id,
            "label": self.label,
            "world_pos": list(self.world_pos),
            "sigma_m": self.sigma_m,
            "depth_source": self.depth_source.value,
            "model": self.model,
            "license": self.license,
            "normal": list(self.normal) if self.normal else None,
            "bounds_min": list(self.bounds_min) if self.bounds_min else None,
            "bounds_max": list(self.bounds_max) if self.bounds_max else None,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class AssociationDecision:
    observation_id: str
    state: AssociationState
    entity_id: Optional[str]  # the entity the observation joined (None when it did not join one)
    probability: float  # likelihood-based weight of the chosen candidate, 0 when none passed the gates
    reason: str
    dist_m: Optional[float] = None
    mahalanobis_sq: Optional[float] = None
    normal_diff_deg: Optional[float] = None
    iou3d: Optional[float] = None
    geom_verified: Optional[bool] = None  # None = no epipolar check was possible (no cameras supplied)
    #: every entity that passed the gates, with its score -- AMBIGUOUS decisions list the tied ones
    candidates: Tuple[Tuple[str, float], ...] = ()

    def to_dict(self) -> dict:
        return {
            "observation_id": self.observation_id,
            "state": self.state.value,
            "entity_id": self.entity_id,
            "probability": self.probability,
            "reason": self.reason,
            "dist_m": self.dist_m,
            "mahalanobis_sq": self.mahalanobis_sq,
            "normal_diff_deg": self.normal_diff_deg,
            "iou3d": self.iou3d,
            "geom_verified": self.geom_verified,
            "candidates": [[eid, score] for eid, score in self.candidates],
        }
