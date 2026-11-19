"""Consensus and error resolution: turn several engines' independent estimates of ONE quantity into the best-supported value,
or into an explicit conflict / insufficiency -- never into a manufactured answer.

Principles enforced in code:

* Frames: estimates must share a coordinate frame (``FrameMismatch`` otherwise); nothing is compared across frames.
* Independence: estimates are grouped by the engine's correlation ``group`` (its model family). Members of one group are treated as
  ONE source (total correlation assumed, the conservative choice); a group never counts as several votes.
* No averaging as default, no vote counting: with fewer than 3 independent groups outliers cannot be identified, so disagreement
  beyond the stated tolerance is a CONFLICT; with 3+ groups the result is the largest set of sources that mutually agree under the
  stated tolerance / covariances (a RANSAC-style consensus set). A tie between equally large sets is a CONFLICT, never broken
  arbitrarily, and the original-image evidence may still overrule the chosen set.
* Uncertainty is reported only when it is *derived*: from supplied covariances (inverse-variance fusion, shrunk only if the caller
  attests that groups are independent) or from the observed spread. Otherwise it is ``None``. Nothing is a made-up confidence.
* Agreement is not proof: unless a geometric check from the original image evidence (e.g. reprojection error) is supplied, the
  result is ``unverified``; when a check IS supplied it can overrule the statistical consensus in favour of a minority group.
* Categorical claims (object class) are reconciled without a vote: unanimity is reported as agreed-but-unverified, any
  disagreement is a CONFLICT with all candidates kept unless an evidence-based score separates them by a stated margin.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2

from engine.multiengine.alignment import FrameMismatch


class ConsensusError(ValueError):
    pass


@dataclass(frozen=True)
class Estimate:
    engine_id: str
    observation_id: str
    group: str                                   # correlation family; use the engine id when it has no family
    frame: str                                   # coordinate frame the value is expressed in
    value: Tuple[float, ...]
    covariance: Optional[Tuple[Tuple[float, ...], ...]] = None   # None = the engine did not state one

    def v(self) -> np.ndarray:
        return np.asarray(self.value, dtype=float)

    def cov(self) -> Optional[np.ndarray]:
        return None if self.covariance is None else np.asarray(self.covariance, dtype=float)


@dataclass(frozen=True)
class GroupEstimate:
    group: str
    value: np.ndarray
    covariance: Optional[np.ndarray]
    members: Tuple[str, ...]                     # observation ids collapsed into this one source


@dataclass(frozen=True)
class Outlier:
    group: str
    observation_ids: Tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ConsensusResult:
    status: str                                  # "resolved" | "conflict" | "insufficient"
    value: Optional[np.ndarray]
    uncertainty: Optional[Dict[str, object]]     # None unless derived (see module docstring)
    method: str
    verification: str                            # "unverified" | "geometry_supported" | "geometry_preferred_minority"
    groups: Tuple[GroupEstimate, ...]
    outliers: Tuple[Outlier, ...]
    supporting: Tuple[str, ...]                  # observation ids that support the final value
    notes: Tuple[str, ...] = ()
    disagreement: Optional[Dict[str, float]] = None   # max pairwise distance between group estimates, in the value's units
    independence: str = ""                       # what was assumed about independence between groups

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"


def _chi2_disagree(a: GroupEstimate, b: GroupEstimate, alpha: float) -> Optional[bool]:
    """True when the two estimates disagree beyond what their stated covariances allow; None if a covariance is missing."""
    if a.covariance is None or b.covariance is None:
        return None
    d = a.value - b.value
    s = a.covariance + b.covariance
    try:
        m2 = float(d @ np.linalg.solve(s, d))
    except np.linalg.LinAlgError:
        return None
    return m2 > float(chi2.ppf(1.0 - alpha, df=len(d)))


def _collapse(estimates: Sequence[Estimate]) -> List[GroupEstimate]:
    by: Dict[str, List[Estimate]] = {}
    for e in estimates:
        by.setdefault(e.group, []).append(e)
    out = []
    for group, members in sorted(by.items()):
        vals = np.array([m.v() for m in members])
        covs = [m.cov() for m in members]
        if len(members) == 1:
            value, cov = vals[0], covs[0]
        else:
            # one source: the componentwise median of its members; if every member states a covariance, the most certain one
            value = np.median(vals, axis=0)
            cov = min((c for c in covs if c is not None), key=np.trace) if all(c is not None for c in covs) else None
        out.append(GroupEstimate(group, value, cov, tuple(sorted(m.observation_id for m in members))))
    return out


def resolve_estimates(
    estimates: Sequence[Estimate], *, tolerance: Optional[float] = None, alpha: float = 0.01,
    assume_independent_groups: bool = False,
    geometric_residual: Optional[Callable[[np.ndarray], Optional[float]]] = None, geometric_margin: float = 0.0,
) -> ConsensusResult:
    """Combine estimates of one vector quantity.

    ``tolerance``: the largest distance (value units) at which two independent groups still count as agreeing; REQUIRED unless every
    group supplies a covariance (then a chi-square test at ``alpha`` decides). ``assume_independent_groups``: the caller attests
    that the groups' errors are independent; only then may inverse-variance fusion shrink the uncertainty.
    ``geometric_residual``: maps a candidate value to a residual computed from the original image evidence (lower is better), or None
    if it cannot be evaluated; ``geometric_margin``: how much better (in residual units) a candidate must be to overrule the others.
    """
    if not estimates:
        raise ConsensusError("no estimates to resolve")
    frames = {e.frame for e in estimates}
    if len(frames) != 1:
        raise FrameMismatch(f"estimates are in different coordinate frames {sorted(frames)}; align them before resolving")
    if len({len(e.value) for e in estimates}) != 1:
        raise ConsensusError("estimates have different dimensionality")
    groups = _collapse(estimates)
    notes: List[str] = []
    independence = ("groups assumed independent by the caller" if assume_independent_groups
                    else "independence between groups is NOT established; uncertainty is not shrunk by the number of groups")
    all_obs = tuple(sorted(e.observation_id for e in estimates))
    if len(groups) < len(estimates):
        notes.append(f"{len(estimates)} estimates collapsed into {len(groups)} independent source(s) by correlation group")

    def disagree(a: GroupEstimate, b: GroupEstimate) -> Optional[bool]:
        verdict = _chi2_disagree(a, b, alpha)
        if verdict is not None:
            return verdict
        if tolerance is None:
            return None
        return float(np.linalg.norm(a.value - b.value)) > tolerance

    pair_dist = [float(np.linalg.norm(a.value - b.value)) for i, a in enumerate(groups) for b in groups[i + 1:]]
    spread = {"max_pairwise_distance": max(pair_dist)} if pair_dist else None

    if len(groups) == 1:
        g = groups[0]
        return _finish(g.value, "single_source", groups, (), all_obs,
                       notes + ["only one independent source: nothing was corroborated"], spread, independence,
                       geometric_residual, geometric_margin, uncertainty=_group_unc(g))

    verdicts = [disagree(a, b) for i, a in enumerate(groups) for b in groups[i + 1:]]
    if any(v is None for v in verdicts):
        return ConsensusResult("insufficient", None, None, "none", "unverified", tuple(groups), (), (),
                               tuple(notes + ["agreement cannot be judged: no covariances and no tolerance were supplied"]),
                               spread, independence)

    if len(groups) == 2:
        if verdicts[0]:
            pick = _geometry_pick(groups, geometric_residual, geometric_margin)
            if pick is not None:
                return _finish(pick.value, "geometric_residual", groups, (), pick.members,
                               notes + [f"two sources disagree; the original-image evidence prefers {pick.group!r}"],
                               spread, independence, geometric_residual, geometric_margin, uncertainty=_group_unc(pick),
                               verification="geometry_preferred_minority")
            return ConsensusResult("conflict", None, None, "none", "unverified", tuple(groups), (), (),
                                   tuple(notes + ["two independent sources disagree beyond tolerance and nothing in the original "
                                                  "evidence separates them"]), spread, independence)
        value, unc, method = _fuse(groups, assume_independent_groups)
        return _finish(value, method, groups, (), all_obs, notes, spread, independence, geometric_residual, geometric_margin,
                       uncertainty=unc)

    # ---- >= 3 independent sources: RANSAC-style consensus set = the largest set of sources that mutually agree under the stated
    # tolerance / covariances. Sources outside it are outliers. A tie between equally large sets is NOT broken arbitrarily.
    agree = {(i, j): not disagree(groups[i], groups[j]) for i in range(len(groups)) for j in range(i + 1, len(groups))}
    best: List[Tuple[int, ...]] = []
    for size in range(len(groups), 1, -1):
        found = [c for c in combinations(range(len(groups)), size) if all(agree[(a, b)] for a, b in combinations(c, 2))]
        if found:
            best = found
            break
    if len(best) != 1:
        why = ("no two independent sources agree under the stated tolerance" if not best
               else "two or more equally large sets of sources agree internally but not with each other")
        return ConsensusResult("conflict", None, None, "consensus_set", "unverified", tuple(groups), (), (),
                               tuple(notes + [why]), spread, independence)
    chosen = set(best[0])
    keep = [groups[i] for i in sorted(chosen)]
    outliers = [Outlier(groups[i].group, groups[i].members, "disagrees with the largest mutually consistent set of sources")
                for i in range(len(groups)) if i not in chosen]
    value, unc, method = _fuse(keep, assume_independent_groups)
    supporting = tuple(sorted(m for g in keep for m in g.members))
    # an evidence-based check may prefer a rejected minority over the statistical consensus
    pick = _geometry_pick(groups, geometric_residual, geometric_margin, incumbent=value)
    if pick is not None and pick.group in {o.group for o in outliers}:
        kept = tuple(o for o in outliers if o.group != pick.group)
        return _finish(pick.value, "geometric_residual", groups, kept, pick.members,
                       notes + [f"the statistical consensus was overruled: the original-image evidence prefers {pick.group!r}"],
                       spread, independence, geometric_residual, geometric_margin, uncertainty=_group_unc(pick),
                       verification="geometry_preferred_minority")
    return _finish(value, "consensus_set_" + method, groups, tuple(outliers), supporting, notes, spread, independence,
                   geometric_residual, geometric_margin, uncertainty=unc)


# ------------------------------------------------------------------------------------------------ internals

def _group_unc(g: GroupEstimate) -> Optional[Dict[str, object]]:
    return None if g.covariance is None else {"method": "source_covariance", "covariance": g.covariance.tolist()}


def _fuse(groups: Sequence[GroupEstimate], independent: bool) -> Tuple[np.ndarray, Optional[Dict[str, object]], str]:
    vals = np.array([g.value for g in groups])
    covs = [g.covariance for g in groups]
    if all(c is not None for c in covs):
        infos = [np.linalg.inv(c) for c in covs]
        total = sum(infos)
        fused = np.linalg.solve(total, sum(i @ v for i, v in zip(infos, vals)))
        if independent:
            return fused, {"method": "inverse_variance_independent", "covariance": np.linalg.inv(total).tolist()}, "inverse_variance"
        # independence not attested: keep the weights but report the most uncertain input's covariance (no shrinkage)
        worst = max(covs, key=np.trace)
        return fused, {"method": "no_shrinkage_independence_not_established", "covariance": worst.tolist()}, "inverse_variance"
    centre = np.median(vals, axis=0)
    return centre, {"method": "observed_spread", "half_range": ((vals.max(axis=0) - vals.min(axis=0)) / 2.0).tolist()}, "median"


def _geometry_pick(groups: Sequence[GroupEstimate], residual: Optional[Callable[[np.ndarray], Optional[float]]], margin: float,
                   incumbent: Optional[np.ndarray] = None) -> Optional[GroupEstimate]:
    """The group whose value the original-image evidence supports best, if every candidate could be evaluated and the best beats
    the next-best by more than ``margin`` (the incumbent statistical consensus competes too)."""
    if residual is None:
        return None
    scored: List[Tuple[float, GroupEstimate]] = []
    for g in groups:
        r = residual(g.value)
        if r is None:
            return None
        scored.append((r, g))
    if incumbent is not None:
        ri = residual(incumbent)
        if ri is None:
            return None
        scored.append((ri, GroupEstimate("__consensus__", incumbent, None, ())))
    scored.sort(key=lambda x: x[0])
    if len(scored) < 2 or not (scored[1][0] - scored[0][0] > margin):
        return None
    best = scored[0][1]
    return None if best.group == "__consensus__" else best


def _finish(value, method, groups, outliers, supporting, notes, spread, independence, residual, margin, *, uncertainty,
            verification: Optional[str] = None) -> ConsensusResult:
    if verification is None:
        verification = "unverified"
        if residual is not None:
            r = residual(np.asarray(value, dtype=float))
            others = [residual(g.value) for g in groups]
            if r is not None and all(o is not None for o in others) and all(r <= o + margin for o in others):
                verification = "geometry_supported"
    return ConsensusResult("resolved", np.asarray(value, dtype=float), uncertainty, method, verification, tuple(groups),
                           tuple(outliers), tuple(supporting), tuple(notes), spread, independence)


# ------------------------------------------------------------------------------------------------ categorical reconciliation

@dataclass(frozen=True)
class LabelClaim:
    engine_id: str
    observation_id: str
    group: str
    label: str


@dataclass(frozen=True)
class LabelResult:
    status: str                                  # "agreed_unverified" | "resolved_by_evidence" | "conflict"
    label: Optional[str]
    candidates: Tuple[str, ...]
    sources: Dict[str, Tuple[str, ...]]          # label -> independent groups claiming it
    notes: Tuple[str, ...] = ()


def reconcile_labels(claims: Sequence[LabelClaim], *, evidence_score: Optional[Callable[[str], Optional[float]]] = None,
                     margin: float = 0.0) -> LabelResult:
    """No voting. Unanimity among the independent groups is reported as agreed-but-unverified; any disagreement stays a conflict
    unless ``evidence_score`` (a likelihood from the original image evidence, higher = better) separates the best candidate from
    the next by more than ``margin``."""
    if not claims:
        raise ConsensusError("no label claims")
    sources: Dict[str, set] = {}
    for c in claims:
        sources.setdefault(c.label, set()).add(c.group)
    labels = tuple(sorted(sources))
    view = {k: tuple(sorted(v)) for k, v in sources.items()}
    if len(labels) == 1:
        return LabelResult("agreed_unverified", labels[0], labels, view,
                           ("agreement between models is not proof of correctness; no image-evidence check was applied",))
    if evidence_score is not None:
        scores = [(evidence_score(lab), lab) for lab in labels]
        if all(s is not None for s, _ in scores):
            scores.sort(key=lambda x: -x[0])
            if scores[0][0] - scores[1][0] > margin:
                return LabelResult("resolved_by_evidence", scores[0][1], labels, view,
                                   (f"image evidence scored {scores[0][1]!r} above {scores[1][1]!r} by more than the margin",))
    return LabelResult("conflict", None, labels, view, ("sources disagree and no evidence separates them; all candidates are kept",))


# ------------------------------------------------------------------------------------------------ geometric checks from image evidence

def reprojection_error(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray, uv: np.ndarray) -> float:
    """Pixel distance between the observed ``uv`` and the projection of world point ``X`` into the camera (R, t map world->camera).
    Returns +inf when the point is behind the camera: such a candidate has no valid projection."""
    Xc = np.asarray(R, float) @ np.asarray(X, float) + np.asarray(t, float)
    if Xc[2] <= 0:
        return float("inf")
    p = np.asarray(K, float) @ Xc
    return float(np.linalg.norm(p[:2] / p[2] - np.asarray(uv, float)))


def make_reprojection_check(cameras: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], observations: Sequence[np.ndarray]
                            ) -> Callable[[np.ndarray], Optional[float]]:
    """Residual function for ``resolve_estimates``: RMS reprojection error (pixels) of a candidate 3D point over the views that
    observed it. Needs at least two views, otherwise the candidate cannot be judged (None)."""
    if len(cameras) != len(observations):
        raise ConsensusError("one observation per camera is required")

    def residual(X: np.ndarray) -> Optional[float]:
        if len(cameras) < 2:
            return None
        errs = [reprojection_error(K, R, t, X, uv) for (K, R, t), uv in zip(cameras, observations)]
        return float(np.sqrt(np.mean(np.square(errs))))

    return residual
