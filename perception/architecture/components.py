"""Architectural component hypotheses + multi-view entity resolution
(directive sections 5-11, 24, 28) and repetition/symmetry priors.

Position in the perception stack:

    parametric fits (parametric.py)
    -> COMPONENT OBSERVATIONS (this module: registry acceptance,
       confidence, evidence)
    -> CANDIDATE COMPONENTS (resolve_components: union-find across
       observations of one physical component)
    -> PATTERN PRIORS (detect_repetition / detect_symmetry:
       supporting evidence, never geometry mutation)
    -> WorldIR promotion (evidence/promote_planes.py pattern; separate)

Honesty rules (constitution + directive):
  - An observation that fails its class's registry gates is RECORDED
    as unaccepted with the reason -- never silently dropped, never
    reclassified by guesswork.
  - Confidence tiers (directive section 28) are documented bands over
    measured confidence: OBSERVED >= 0.95, STRONG >= 0.75,
    WEAK >= 0.4, else UNOBSERVED-for-world-purposes (still recorded).
  - Repetition/symmetry are SUPPORTING evidence: they may raise a
    component's confidence (capped by the best-supported sibling of
    its pattern -- the pattern cannot claim more than its class
    demonstrates) but NEVER move, resize, or clone geometry.
  - Deterministic: no RNG, no clocks, sorted iteration everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from perception.architecture.parametric import (
    CircleFit,
    CylinderFit,
    SphereFit,
)
from perception.architecture.registry import FitKind, get_default_registry

Point3 = Tuple[float, float, float]
Vec3 = Tuple[float, float, float]


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


class ConfidenceTier(str, Enum):
    """Directive section 28's distinction: measured confidence mapped
    to honest bands. Bands are code, documented here, not folklore:
      OBSERVED    >= 0.95  (directly measured, tight residuals)
      STRONG      >= 0.75  (clearly supported by geometry)
      WEAK        >= 0.40  (supported but noisy)
      UNOBSERVED  <  0.40  (recorded for world purposes as unobserved;
                          the observation still exists upstream)
    """

    OBSERVED = "observed"
    STRONG = "strong"
    WEAK = "weak"
    UNOBSERVED = "unobserved"

    @staticmethod
    def from_confidence(c: float) -> "ConfidenceTier":
        if c >= 0.95:
            return ConfidenceTier.OBSERVED
        if c >= 0.75:
            return ConfidenceTier.STRONG
        if c >= 0.40:
            return ConfidenceTier.WEAK
        return ConfidenceTier.UNOBSERVED


@dataclass(frozen=True)
class ComponentObservation:
    """One fitted segment proposed as one architectural component --
    a hypothesis carrying its real geometry, evidence, and measured
    confidence. Acceptance against the class's registry gates is
    explicit and recorded, never silent."""

    segment_id: str
    arch_class: str
    fit: object  # CylinderFit | SphereFit | CircleFit (real geometry)
    evidence_ids: Tuple[str, ...]
    confidence: float
    accepted: bool
    rejection_reason: str = ""
    #: Position of the component (fit-derived centroid of support).
    position: Point3 = (0.0, 0.0, 0.0)

    @property
    def tier(self) -> ConfidenceTier:
        if not self.accepted:
            return ConfidenceTier.UNOBSERVED
        return ConfidenceTier.from_confidence(self.confidence)

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "arch_class": self.arch_class,
            "evidence_ids": list(self.evidence_ids),
            "confidence": self.confidence,
            "tier": self.tier.value,
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "position": list(self.position),
            "fit": self.fit.to_dict() if self.fit is not None else None,
        }


#: Which registry classes each fit kind can propose. A cylinder
#: proposes column (and, for towers later, the same fit with different
#: gates); a sphere proposes dome; a circle proposes arch.
_KIND_TO_CLASSES: Dict[FitKind, Tuple[str, ...]] = {
    FitKind.CYLINDER: ("column",),
    FitKind.SPHERE: ("dome",),
    FitKind.CIRCLE: ("arch",),
}


def _position_of(fit) -> Point3:
    if isinstance(fit, CylinderFit):
        t_mid = 0.5 * (fit.height_min_m + fit.height_max_m)
        return (
            fit.axis_point[0] + fit.axis[0] * t_mid,
            fit.axis_point[1] + fit.axis[1] * t_mid,
            fit.axis_point[2] + fit.axis[2] * t_mid,
        )
    if isinstance(fit, SphereFit):
        return fit.center
    if isinstance(fit, CircleFit):
        return fit.center
    raise ValueError(f"unsupported fit type {type(fit).__name__}")


def _evaluate_gates(cls, fit) -> Tuple[bool, str]:
    """Evaluate the class's registry thresholds against a fit.
    Returns (accepted, reason). Only measures that exist are gated;
    a threshold of None means the class does not gate on it."""
    if cls.min_axis_up_dot is not None and isinstance(fit, CylinderFit):
        # The gate is on |axis . up| magnitude; sign canonicalization
        # already points axes into the +up hemisphere, but a caller
        # may pass a non-canonical fit -- measure magnitude honestly.
        # The caller supplies up via the registry gate evaluation in
        # build_component_observations (axis is pre-canonicalized).
        if fit.axis[2] < cls.min_axis_up_dot:
            return False, (
                f"fitted axis is not vertical enough for a {cls.name} "
                f"(up-dot {fit.axis[2]:.3f} < {cls.min_axis_up_dot})"
            )
    if isinstance(fit, CircleFit):
        span = fit.angular_span_rad
        if cls.min_angular_span_rad is not None and span < cls.min_angular_span_rad:
            return False, (
                f"measured angular span {span:.3f} rad below the "
                f"{cls.name} minimum {cls.min_angular_span_rad:.3f}"
            )
        if cls.max_angular_span_rad is not None and span > cls.max_angular_span_rad:
            return False, (
                f"measured angular span {span:.3f} rad above the "
                f"{cls.name} maximum {cls.max_angular_span_rad:.3f}"
            )
    return True, ""


def build_component_observations(
    fitted_segments: Sequence[Tuple[Tuple[str, ...], object, Tuple[str, ...]]],
    up: Vec3,
    registry=None,
) -> List[ComponentObservation]:
    """Turn fitted segments into component observations.

    `fitted_segments` yields (segment_ids, fit, evidence_ids) --
    segment_ids is a tuple because one segment may be split across
    sub-segments that were fitted together. Each fit is matched to
    every registry class its kind can propose; the observation records
    the FIRST class whose gates it passes, or the first proposed class
    with the rejection reason (never dropped).

    Note on `up`: cylinder fits are canonicalized to +up at fit time,
    so gating uses the fit's own axis. The `up` parameter exists for
    future classes whose gates need the world frame directly.
    """
    reg = registry or get_default_registry()
    observations: List[ComponentObservation] = []
    for segment_ids, fit, evidence_ids in fitted_segments:
        kind = _kind_of(fit)
        if kind is None:
            raise ValueError(f"unsupported fit type {type(fit).__name__}")
        seg_label = "+".join(segment_ids)
        proposed = _KIND_TO_CLASSES[kind]
        recorded = False
        for class_name in proposed:
            cls = reg.get(class_name)
            accepted, reason = _evaluate_gates(cls, fit)
            observations.append(ComponentObservation(
                segment_id=seg_label,
                arch_class=class_name,
                fit=fit,
                evidence_ids=tuple(sorted(set(evidence_ids))),
                confidence=fit.confidence,
                accepted=accepted,
                rejection_reason=reason,
                position=_position_of(fit),
            ))
            if accepted:
                recorded = True
                break
        if not recorded:
            # All proposed classes rejected: the LAST observation in
            # the loop already records the reason. (Loop always runs
            # at least once -- _KIND_TO_CLASSES values are non-empty.)
            pass
    return observations


def _kind_of(fit) -> Optional[FitKind]:
    if isinstance(fit, CylinderFit):
        return FitKind.CYLINDER
    if isinstance(fit, SphereFit):
        return FitKind.SPHERE
    if isinstance(fit, CircleFit):
        return FitKind.CIRCLE
    return None


# ==================================================================
# Multi-view entity resolution
# ==================================================================


class _UnionFind:
    def __init__(self, n: int):
        self._parent = list(range(n))

    def find(self, i: int) -> int:
        while self._parent[i] != i:
            self._parent[i] = self._parent[self._parent[i]]
            i = self._parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)


@dataclass(frozen=True)
class CandidateComponent:
    """One physical architectural component, resolved from N
    observations across views. Retains every source observation; the
    fused geometry is the members' (they are consistent by the merge
    criterion); confidence comes from agreement."""

    component_id: str
    arch_class: str
    position: Point3
    #: Fused confidence: agreement across independent observations
    #: raises it above the best single view, capped at 1.0.
    confidence: float
    #: Confidence AFTER pattern priors (repetition/symmetry). Equals
    #: `confidence` until priors are applied. Geometry is never moved
    #: by priors -- only this field changes.
    prior_adjusted_confidence: Optional[float] = None
    source_observations: Tuple[ComponentObservation, ...] = ()
    #: The winning (accepted) member's fit -- the candidate's geometry.
    fit: object = None

    @property
    def observation_count(self) -> int:
        return len(self.source_observations)

    @property
    def evidence_ids(self) -> Tuple[str, ...]:
        ids: List[str] = []
        for o in self.source_observations:
            for e in o.evidence_ids:
                if e not in ids:
                    ids.append(e)
        return tuple(sorted(ids))

    def effective_confidence(self) -> float:
        return (
            self.prior_adjusted_confidence
            if self.prior_adjusted_confidence is not None
            else self.confidence
        )

    def to_dict(self) -> dict:
        return {
            "component_id": self.component_id,
            "arch_class": self.arch_class,
            "position": list(self.position),
            "confidence": self.confidence,
            "prior_adjusted_confidence": self.prior_adjusted_confidence,
            "observation_count": self.observation_count,
            "evidence_ids": list(self.evidence_ids),
            "fit": self.fit.to_dict() if self.fit is not None else None,
            "source_segment_ids": [o.segment_id for o in self.source_observations],
        }


def resolve_components(
    observations: Sequence[ComponentObservation],
    merge_distance_m: float = 0.5,
) -> List[CandidateComponent]:
    """Cluster accepted, same-class, spatially-close observations into
    candidate components (union-find, deterministic order). Unaccepted
    observations never join a candidate -- they are returned separately
    inside the report via `unaccepted` on... no: this function returns
    candidates only; callers inspect `build_component_observations`
    output for the unaccepted records. Every accepted observation that
    joins a candidate is retained in `source_observations`."""
    if merge_distance_m < 0:
        raise ValueError("merge_distance_m must be non-negative")
    accepted = [o for o in observations if o.accepted]
    if not accepted:
        return []
    ordered = sorted(accepted, key=lambda o: (o.arch_class, o.segment_id, o.position))
    uf = _UnionFind(len(ordered))
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            if ordered[i].arch_class != ordered[j].arch_class:
                continue
            if _dist(ordered[i].position, ordered[j].position) <= merge_distance_m:
                uf.union(i, j)

    groups: Dict[int, List[ComponentObservation]] = {}
    for i, obs in enumerate(ordered):
        groups.setdefault(uf.find(i), []).append(obs)

    candidates: List[CandidateComponent] = []
    for gi, (root, members) in enumerate(sorted(groups.items())):
        best = max(members, key=lambda o: o.confidence)
        best_single = best.confidence
        agreement = 1.0 - (1.0 - best_single) ** len(members)
        confidence = max(best_single, min(1.0, agreement))
        cx = sum(m.position[0] for m in members) / len(members)
        cy = sum(m.position[1] for m in members) / len(members)
        cz = sum(m.position[2] for m in members) / len(members)
        candidates.append(CandidateComponent(
            component_id=f"arch-{best.arch_class}-{gi:04d}",
            arch_class=best.arch_class,
            position=(cx, cy, cz),
            confidence=confidence,
            source_observations=tuple(members),
            fit=best.fit,
        ))
    candidates.sort(key=lambda c: (c.arch_class, c.position))
    return candidates


def _dist(a: Point3, b: Point3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


# ==================================================================
# Repetition prior (directive section 10)
# ==================================================================


@dataclass(frozen=True)
class RepetitionPrior:
    """A detected repetition group: same-class components with
    consistent spacing. Supporting evidence -- geometry is never
    cloned from siblings."""

    arch_class: str
    member_ids: Tuple[str, ...]
    spacing_m: float
    #: Measured consistency of the spacing (max deviation / spacing).
    spacing_consistency: float
    count: int

    def to_dict(self) -> dict:
        return {
            "arch_class": self.arch_class,
            "member_ids": list(self.member_ids),
            "spacing_m": self.spacing_m,
            "spacing_consistency": self.spacing_consistency,
            "count": self.count,
        }


def detect_repetition(
    candidates: Sequence[CandidateComponent],
    min_count: int = 3,
    consistency_max: float = 0.15,
) -> List[RepetitionPrior]:
    """Detect same-class components forming an evenly-spaced group
    along their nearest-neighbor chain. Deterministic; a group with
    inconsistent spacing is not a repetition (no pattern forcing)."""
    priors: List[RepetitionPrior] = []
    by_class: Dict[str, List[CandidateComponent]] = {}
    for c in candidates:
        by_class.setdefault(c.arch_class, []).append(c)
    for class_name, members in sorted(by_class.items()):
        if len(members) < min_count:
            continue
        ms = sorted(members, key=lambda c: c.position)
        # Nearest-neighbor chain spacings (per axis-dominant ordering:
        # use full 3D NN chain, which for colonnades is the row order).
        # Try each coordinate axis for the dominant line; pick the one
        # with the most consistent spacing.
        best = None
        for axis in range(3):
            chain = sorted(ms, key=lambda c: c.position[axis])
            gaps = [
                chain[k + 1].position[axis] - chain[k].position[axis]
                for k in range(len(chain) - 1)
            ]
            positive = [g for g in gaps if g > 1e-9]
            if len(positive) < min_count - 1:
                continue
            mean_gap = sum(positive) / len(positive)
            max_dev = max(abs(g - mean_gap) for g in positive)
            consistency = max_dev / mean_gap if mean_gap > 0 else float("inf")
            if best is None or consistency < best[0]:
                best = (consistency, mean_gap, [c.component_id for c in chain])
        if best is None or best[0] > consistency_max:
            continue
        consistency, mean_gap, chain_ids = best
        priors.append(RepetitionPrior(
            arch_class=class_name,
            member_ids=tuple(chain_ids),
            spacing_m=mean_gap,
            spacing_consistency=consistency,
            count=len(chain_ids),
        ))
    return priors


def apply_repetition_priors(
    candidates: List[CandidateComponent],
    priors: Sequence[RepetitionPrior],
) -> List[CandidateComponent]:
    """Layer repetition priors onto candidates: a member of a
    consistent pattern gains confidence from the siblings' agreement,
    but never above the best-supported sibling (the pattern cannot
    claim more than its class demonstrates) and never above 1.0.
    Geometry is untouched."""
    by_id = {c.component_id: c for c in candidates}
    out = list(candidates)
    for prior in priors:
        members = [by_id[i] for i in prior.member_ids if i in by_id]
        if len(members) < 2:
            continue
        ceiling = min(
            1.0, max(m.confidence for m in members)
        )
        for idx, c in enumerate(out):
            if c.component_id not in prior.member_ids:
                continue
            # Single-observation members gain the most; the boost
            # decays with the member's own support.
            base = c.confidence
            # Weight: members with fewer observations get more boost.
            support = 1.0 / c.observation_count
            boosted = base + (ceiling - base) * support
            boosted = min(ceiling, boosted)
            out[idx] = CandidateComponent(
                component_id=c.component_id,
                arch_class=c.arch_class,
                position=c.position,
                confidence=c.confidence,
                prior_adjusted_confidence=boosted,
                source_observations=c.source_observations,
                fit=c.fit,
            )
    return out


# ==================================================================
# Symmetry prior (directive section 11)
# ==================================================================


@dataclass(frozen=True)
class SymmetryPrior:
    """A detected symmetry of the component set. Recorded evidence --
    it must never move, resize, or clone any component's geometry."""

    kind: str  # "bilateral" | "radial"
    axis_point: Point3
    axis_direction: Optional[Vec3]
    matched_pairs: Tuple[Tuple[str, str], ...]
    #: Mean pair distance from the mirror plane / rotation center.
    residual_m: float

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "axis_point": list(self.axis_point),
            "axis_direction": list(self.axis_direction) if self.axis_direction else None,
            "matched_pairs": [list(p) for p in self.matched_pairs],
            "residual_m": self.residual_m,
        }


def detect_symmetry(
    candidates: Sequence[CandidateComponent],
    tolerance_m: float = 0.25,
) -> List[SymmetryPrior]:
    """Detect bilateral symmetry of same-class components about a
    vertical plane. Planes tried: the three coordinate planes through
    the class centroid, plus the two 45-degree diagonals -- a
    deterministic candidate set (general plane fitting is later work;
    these cover the dominant architectural cases including
    front-symmetric facades). Only same-class, mutually-near pairs
    count; unmatched components do not break symmetry, they are just
    not evidence for it."""
    priors: List[SymmetryPrior] = []
    by_class: Dict[str, List[CandidateComponent]] = {}
    for c in candidates:
        by_class.setdefault(c.arch_class, []).append(c)
    for class_name, members in sorted(by_class.items()):
        if len(members) < 2:
            continue
        centroid = (
            sum(m.position[0] for m in members) / len(members),
            sum(m.position[1] for m in members) / len(members),
            sum(m.position[2] for m in members) / len(members),
        )
        # Candidate vertical plane normals (x-y plane mirrors).
        normals = [
            (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
            (math.sqrt(0.5), math.sqrt(0.5), 0.0),
            (math.sqrt(0.5), -math.sqrt(0.5), 0.0),
        ]
        best = None
        for n in normals:
            pairs: List[Tuple[str, str]] = []
            devs: List[float] = []
            used = set()
            # Mirror each member across the plane through centroid
            # with normal n; match to the nearest unmirrored member.
            def mirror(p: Point3) -> Point3:
                d = (p[0] - centroid[0], p[1] - centroid[1], p[2] - centroid[2])
                dd = _dot(d, n)
                m = (d[0] - 2 * dd * n[0], d[1] - 2 * dd * n[1], d[2] - 2 * dd * n[2])
                return (
                    centroid[0] + m[0], centroid[1] + m[1], centroid[2] + m[2],
                )
            order = sorted(range(len(members)), key=lambda i: members[i].component_id)
            for i in order:
                if i in used:
                    continue
                mirrored = mirror(members[i].position)
                best_j = None
                best_d = tolerance_m
                for j in order:
                    if j == i or j in used:
                        continue
                    d = _dist(mirrored, members[j].position)
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j is not None:
                    used.add(i)
                    used.add(best_j)
                    pairs.append((members[i].component_id, members[best_j].component_id))
                    devs.append(best_d)
            if len(pairs) >= 1:
                residual = sum(devs) / len(devs) if devs else 0.0
                if best is None or len(pairs) > len(best[0]):
                    best = (pairs, residual, n)
        if best and len(best[0]) >= 1:
            pairs, residual, n = best
            priors.append(SymmetryPrior(
                kind="bilateral",
                axis_point=centroid,
                axis_direction=n,
                matched_pairs=tuple(sorted(pairs)),
                residual_m=residual,
            ))
    return priors
