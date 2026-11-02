"""Observation ledger and association-decision contracts (plan §2, Phase 1).

An `Observation` is one 2D detection lifted into the world frame, carrying the
image it came from, which model produced it (name@version + licence, so
non-commercial output stays traceable) and an honest position uncertainty:
monocular ("relative") depth widens that uncertainty a lot, metric/MVS depth
does not. An `AssociationDecision` is the recorded outcome of asking "is this
the same physical thing as an entity we already track?" -- including the cases
where the answer is "can't tell" (AMBIGUOUS) or "different".

The ledger (observations + every association decision) is saved INSIDE each WorldIR
version under ``world.metadata["observation_ledger"]`` (see `attach_ledger` / `ledger_of`):
metadata already round-trips through `WorldIR.to_dict/from_dict`, so no schema bump is
needed and a version written before the ledger existed simply has no such key and loads
with an empty ledger. Optional fields are None when not measured -- never a made-up value.
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


class Visibility(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    OCCLUDED = "occluded"


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
    #: 2D evidence the lift came from (doc 11): a stored mask artifact URI and/or a pixel box (x0, y0, x1, y1).
    mask_uri: Optional[str] = None
    bbox_px: Optional[Tuple[float, float, float, float]] = None
    visibility: Optional[Visibility] = None  # None = not assessed
    embedding_ref: Optional[str] = None  # reference to a stored appearance embedding, never the vector itself
    session_id: Optional[str] = None
    created_at: Optional[str] = None  # ISO-8601 UTC; None = not stamped (no invented clock)

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
            "mask_uri": self.mask_uri,
            "bbox_px": list(self.bbox_px) if self.bbox_px else None,
            "visibility": self.visibility.value if self.visibility else None,
            "embedding_ref": self.embedding_ref,
            "session_id": self.session_id,
            "created_at": self.created_at,
        }

    @staticmethod
    def from_dict(d: dict) -> "Observation":
        vec = lambda v: tuple(float(x) for x in v) if v else None
        vis = d.get("visibility")
        return Observation(
            observation_id=d["observation_id"], image_id=d["image_id"], label=d["label"],
            world_pos=vec(d["world_pos"]), sigma_m=float(d["sigma_m"]), depth_source=DepthSource(d["depth_source"]),
            model=d.get("model", "unknown@unknown"), license=d.get("license", "unknown"),
            normal=vec(d.get("normal")), bounds_min=vec(d.get("bounds_min")), bounds_max=vec(d.get("bounds_max")),
            confidence=d.get("confidence", 0.5), mask_uri=d.get("mask_uri"), bbox_px=vec(d.get("bbox_px")),
            visibility=Visibility(vis) if vis else None, embedding_ref=d.get("embedding_ref"),
            session_id=d.get("session_id"), created_at=d.get("created_at"),
        )


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

    @staticmethod
    def from_dict(d: dict) -> "AssociationDecision":
        return AssociationDecision(
            observation_id=d["observation_id"], state=AssociationState(d["state"]), entity_id=d.get("entity_id"),
            probability=d["probability"], reason=d.get("reason", ""), dist_m=d.get("dist_m"),
            mahalanobis_sq=d.get("mahalanobis_sq"), normal_diff_deg=d.get("normal_diff_deg"), iou3d=d.get("iou3d"),
            geom_verified=d.get("geom_verified"), candidates=tuple((c[0], c[1]) for c in d.get("candidates", ())),
        )


# ---- the ledger inside a WorldIR version --------------------------------------------------------------------------

LEDGER_KEY = "observation_ledger"


def attach_ledger(world, observations, decisions) -> None:
    """Store the observation ledger and every association decision inside `world` (its `metadata`), replacing any
    earlier ledger. Serialised with the world, so each saved version carries its own ledger."""
    world.metadata[LEDGER_KEY] = {
        "observations": [o.to_dict() for o in sorted(observations, key=lambda o: o.observation_id)],
        "decisions": [d.to_dict() for d in sorted(decisions, key=lambda d: d.observation_id)],
    }


def ledger_of(world) -> Tuple[Tuple[Observation, ...], Tuple[AssociationDecision, ...]]:
    """(observations, decisions) stored in `world`; both empty for a version written before the ledger existed."""
    raw = (getattr(world, "metadata", None) or {}).get(LEDGER_KEY) or {}
    return (tuple(Observation.from_dict(o) for o in raw.get("observations", ())),
            tuple(AssociationDecision.from_dict(d) for d in raw.get("decisions", ())))
