"""Spatial reasoning data model: structural observations and epistemically-typed entities.

Contract with perception (FreeBuff): a `StructuralObservation` is ONE detection of a wall / door / window
lifted into the world's plan view (x, y metres; z range) from ONE image, carrying that image's id, a stable
evidence id, an honest 1-sigma position error, and visibility. Nothing here is invented: a field that the
producer did not measure is None, never a default number.

Epistemic model: every attribute of a reconstructed entity (existence, position, dimensions, orientation,
geometry, topology, semantic class) is a `Belief` with a `Status` from the six-way taxonomy the compiler uses
everywhere, an optional numeric sigma, an optional confidence (None = not computed -- never 0.5, never 1.0),
and the observation ids that support it. UNKNOWN is a status, not a zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from world_ir.observations import DepthSource, Visibility

Vec2 = tuple[float, float]


class Kind(str, Enum):
    WALL = "wall"
    DOOR = "door"
    WINDOW = "window"
    OPENING = "opening"
    STAIR = "stair"


class Status(str, Enum):
    """Where a belief came from. CONFLICTING and UNKNOWN are not 'lower' than the others: they are different
    facts (the evidence disagrees / there is none)."""
    DIRECTLY_OBSERVED = "directly_observed"
    GEOMETRICALLY_INFERRED = "geometrically_inferred"
    SEMANTICALLY_INFERRED = "semantically_inferred"
    WEAKLY_SUPPORTED = "weakly_supported"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"


DIMENSIONS = ("existence", "position", "dimensions", "orientation", "geometry", "topology", "semantic")


@dataclass(frozen=True)
class StructuralObservation:
    """One lifted detection. `p0`/`p1` are the plan-view end points of the visible extent (for a door/window:
    the extent of the opening along its host line). `z0`/`z1` is the vertical extent when measured, else None."""
    evidence_id: str            # globally unique, resolvable back to the producer's detection
    image_id: str
    kind: Kind
    p0: Vec2
    p1: Vec2
    sigma_m: float              # isotropic 1-sigma on each end point, metres
    z0: float | None = None
    z1: float | None = None
    visibility: Visibility = Visibility.FULL
    #: which end points were actually seen. False = the extent was cut by the image border or an occluder, so
    #: that end is a LOWER BOUND on the entity's extent, not a corner.
    end0_observed: bool = True
    end1_observed: bool = True
    label_confidence: float | None = None   # producer's classification confidence; None = not measured
    depth_source: DepthSource | None = None
    frame_id: str = "world"
    producer: str = "unknown@unknown"
    storey: str | None = None

    def __post_init__(self):
        if not (self.sigma_m > 0 and math.isfinite(self.sigma_m)):
            raise ValueError("sigma_m must be a positive finite number")
        if math.dist(self.p0, self.p1) <= 0:
            raise ValueError("observation extent has zero length")

    @property
    def length(self) -> float:
        return math.dist(self.p0, self.p1)

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id, "image_id": self.image_id, "kind": self.kind.value,
            "p0": list(self.p0), "p1": list(self.p1), "sigma_m": self.sigma_m, "z0": self.z0, "z1": self.z1,
            "visibility": self.visibility.value, "end0_observed": self.end0_observed,
            "end1_observed": self.end1_observed, "label_confidence": self.label_confidence,
            "depth_source": self.depth_source.value if self.depth_source else None,
            "frame_id": self.frame_id, "producer": self.producer, "storey": self.storey,
        }

    @staticmethod
    def from_dict(d: dict) -> StructuralObservation:
        return StructuralObservation(
            evidence_id=d["evidence_id"], image_id=d["image_id"], kind=Kind(d["kind"]),
            p0=tuple(d["p0"]), p1=tuple(d["p1"]), sigma_m=d["sigma_m"], z0=d.get("z0"), z1=d.get("z1"),
            visibility=Visibility(d.get("visibility", "full")), end0_observed=d.get("end0_observed", True),
            end1_observed=d.get("end1_observed", True), label_confidence=d.get("label_confidence"),
            depth_source=DepthSource(d["depth_source"]) if d.get("depth_source") else None,
            frame_id=d.get("frame_id", "world"), producer=d.get("producer", "unknown@unknown"),
            storey=d.get("storey"),
        )


@dataclass(frozen=True)
class Belief:
    status: Status
    support: tuple[str, ...] = ()           # evidence ids that justify this belief
    value: float | None = None           # scalar summary where one exists (e.g. length m, angle rad)
    sigma: float | None = None           # 1-sigma of `value` in its own unit; None = no estimate
    confidence: float | None = None      # [0,1]; None = not computed. Never defaulted.
    note: str = ""

    def to_dict(self) -> dict:
        return {"status": self.status.value, "support": list(self.support), "value": self.value,
                "sigma": self.sigma, "confidence": self.confidence, "note": self.note}

    @staticmethod
    def from_dict(d: dict) -> Belief:
        return Belief(Status(d["status"]), tuple(d.get("support", ())), d.get("value"), d.get("sigma"),
                      d.get("confidence"), d.get("note", ""))


UNKNOWN_BELIEF = Belief(Status.UNKNOWN, note="no evidence")


@dataclass
class SpatialEntity:
    """A canonical entity inferred from >=1 observations. `a`/`b` are the fitted end points (plan view)."""
    entity_id: str
    kind: Kind
    a: Vec2
    b: Vec2
    z0: float | None
    z1: float | None
    evidence_ids: tuple[str, ...]
    image_ids: tuple[str, ...]
    beliefs: dict[str, Belief]
    #: per-end sigma (m) of the fitted end points; None = unknown
    sigma_a: float | None = None
    sigma_b: float | None = None
    #: True only when an end was seen as a corner/termination, False when it is a lower bound (occlusion/border)
    end_a_observed: bool = False
    end_b_observed: bool = False
    host_id: str | None = None           # for doors/windows
    host_alternatives: tuple[str, ...] = ()  # other plausible hosts (ambiguous attachment)
    storey: str | None = None
    #: alternative interpretations kept side by side instead of averaged (ambiguous membership / conflict)
    alternatives: list[dict] = field(default_factory=list)
    partially_visible: bool = False
    occluded: bool = False
    #: posterior isotropic (per-axis) sigma of an end after the joint constraint solve; absent = not constrained
    end_post: dict[str, float] = field(default_factory=dict)
    #: separated position-uncertainty sources (m): measurement, optimization (pose/constraint), model (un-modelled floor);
    #: association is a count of ambiguous observations, not a sigma. Absent keys = not computed.
    sigma_parts: dict[str, float | None] = field(default_factory=dict)

    def end_sigma(self, which: str) -> float | None:
        """Per-axis (isotropic) 1-sigma of an end point, metres. After the constraint solve this is the posterior;
        before it, the along-wall sigma combined with the across-wall sigma of the fitted line at that end (offset
        sigma plus angle sigma times the half-length lever arm). None if the wall's sigmas are unknown."""
        if which in self.end_post:
            return self.end_post[which]
        along = self.sigma_a if which == "a" else self.sigma_b
        d = self.beliefs["position"].sigma
        th = self.beliefs["orientation"].sigma
        if along is None or d is None:
            return None
        across = math.sqrt(d * d + ((self.length / 2.0) * th) ** 2) if th is not None else d
        return math.sqrt((along * along + across * across) / 2.0)

    @property
    def length(self) -> float:
        return math.dist(self.a, self.b)

    @property
    def angle(self) -> float:
        return math.atan2(self.b[1] - self.a[1], self.b[0] - self.a[0])


@dataclass(frozen=True)
class AssociationRecord:
    """Why an observation belongs (or does not belong) to an entity -- the audit trail."""
    evidence_id: str
    entity_id: str | None
    state: str                              # reuses world_ir.observations.AssociationState values
    reason: str
    candidates: tuple[str, ...] = ()
    normalized_offset: float | None = None   # perpendicular offset / combined sigma
    normalized_angle: float | None = None    # angle difference / combined angle sigma

    def to_dict(self) -> dict:
        return {"evidence_id": self.evidence_id, "entity_id": self.entity_id, "state": self.state,
                "reason": self.reason, "candidates": list(self.candidates),
                "normalized_offset": self.normalized_offset, "normalized_angle": self.normalized_angle}


@dataclass(frozen=True)
class Conflict:
    code: str
    entities: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    message: str
    magnitude: float | None = None       # metres or radians, depending on code
    normalized: float | None = None      # magnitude in sigmas, when a sigma exists

    def to_dict(self) -> dict:
        return {"code": self.code, "entities": list(self.entities), "evidence_ids": list(self.evidence_ids),
                "message": self.message, "magnitude": self.magnitude, "normalized": self.normalized}


def line_params(a: Vec2, b: Vec2) -> tuple[Vec2, float]:
    """(unit direction, angle in [0, pi)) of an UNDIRECTED line."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = math.hypot(dx, dy)
    ux, uy = dx / n, dy / n
    return (ux, uy), math.atan2(uy, ux) % math.pi


def angle_diff(a: float, b: float) -> float:
    """Smallest difference between two undirected-line angles, in [0, pi/2]."""
    d = abs(a - b) % math.pi
    return min(d, math.pi - d)
