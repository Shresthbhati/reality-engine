"""Multi-view scene identity: which photographs observe the SAME physical scene, and did they register?

Scope (the capability this module adds on top of the existing COLMAP pipeline, which it reuses):

1. GROUPING   -- ``group_images`` links photographs ONLY through geometrically verified correspondences
                 (epipolar-RANSAC inliers). Visual similarity is not scene identity: a pair that matches
                 loosely but fails verification is recorded as an *uncertain link* and never merges groups.
                 Declared (user-selected) sessions are hypotheses; a declared session that verification
                 splits is reported as a conflict, not forced together.
2. DIAGNOSIS  -- ``diagnose`` turns a ``ReconstructionResult`` plus measured facts into
                 ``RegistrationDiagnostics`` with an explicit outcome:
                 success | partial | failed | geometrically_incompatible. Nothing is invented: a value the
                 backend did not measure is ``None`` and says so.
3. CORRECTION -- ``reconstruct_group`` is a bounded retry loop. A correction is kept only if measured quality
                 improves AND (for guided matching, which is documented to warp repetitive scenes) the new
                 model agrees with the previous one on shared cameras. The last valid result is never
                 overwritten by a worse or failed retry. Geometry is accepted by measurement only; no LLM
                 is involved in acceptance.
4. IDENTITY   -- ``SceneRegistry`` persists group membership with stable ids and the last valid
                 reconstruction per world, so a restart or a failed retry cannot change world identity.

Thresholds in ``GroupingThresholds`` / ``QualityThresholds`` are PROVISIONAL starting points; the values
measured on real runs are recorded in docs/implementation/SCENE_IDENTITY.md.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from reconstruction.backend.interface import IReconstructionBackend, ReconstructionResult

# --------------------------------------------------------------------------- pair verification


@dataclass(frozen=True)
class PairEvidence:
    a: str
    b: str
    matches: int       # ratio-test survivors
    inliers: int       # survivors consistent with ONE epipolar geometry
    inlier_ratio: float

    def to_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "matches": self.matches, "inliers": self.inliers,
                "inlier_ratio": round(self.inlier_ratio, 4)}


PairVerifier = Callable[[str, str], PairEvidence]


class SiftPairVerifier:
    """SIFT + Lowe ratio + fundamental-matrix RANSAC (OpenCV). Features are cached per image."""

    #: Defaults chosen by measurement (docs/implementation/SCENE_IDENTITY.md): the looser ratio 0.8 / 1 px /
    #: one-way variant let different-texture look-alike scenes reach 16 "inliers", overlapping real links;
    #: ratio 0.75 / 0.5 px / mutual-NN brought that ceiling to 8 while keeping the true scene connected.
    def __init__(self, paths: Dict[str, Path], max_side: int = 960, ratio: float = 0.75, ransac_px: float = 0.5):
        self.paths, self.max_side, self.ratio, self.ransac_px = dict(paths), max_side, ratio, ransac_px
        self._feat: Dict[str, tuple] = {}

    def _features(self, eid: str):
        if eid not in self._feat:
            import cv2
            img = cv2.imread(str(self.paths[eid]), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"unreadable image for evidence {eid}: {self.paths[eid]}")
            scale = self.max_side / max(img.shape)
            if scale < 1:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            kp, des = cv2.SIFT_create(nfeatures=4000).detectAndCompute(img, None)
            self._feat[eid] = (np.float32([k.pt for k in kp]), des)
        return self._feat[eid]

    def __call__(self, a: str, b: str) -> PairEvidence:
        import cv2
        (pa, da), (pb, db) = self._features(a), self._features(b)
        if da is None or db is None or len(da) < 8 or len(db) < 8:
            return PairEvidence(a, b, 0, 0, 0.0)
        bf = cv2.BFMatcher(cv2.NORM_L2)
        back = {m.queryIdx: m.trainIdx for m in bf.match(db, da)}
        good = [m[0] for m in bf.knnMatch(da, db, k=2)
                if len(m) == 2 and m[0].distance < self.ratio * m[1].distance and back.get(m[0].trainIdx) == m[0].queryIdx]
        if len(good) < 8:
            return PairEvidence(a, b, len(good), 0, 0.0)
        p1 = pa[[m.queryIdx for m in good]]
        p2 = pb[[m.trainIdx for m in good]]
        _, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, self.ransac_px, 0.999, 5000)
        inl = int(mask.sum()) if mask is not None else 0
        return PairEvidence(a, b, len(good), inl, inl / len(good))


# --------------------------------------------------------------------------- grouping


@dataclass(frozen=True)
class GroupingThresholds:
    #: A pair is a VERIFIED link at/above both numbers. Provisional: calibrated in SCENE_IDENTITY.md.
    #: Measured on the synthetic room (A-A true pairs vs A-B different-texture pairs, tightened verifier):
    #: false-pair inliers max 8; scene A is one connected component at >=12. Margin is narrow -- re-measure on
    #: real indoor photographs before trusting it there.
    min_inliers: int = 12
    min_inlier_ratio: float = 0.2
    #: Pairs with at least this many inliers that miss the verified bar are recorded as uncertain.
    uncertain_inliers: int = 8


@dataclass
class SceneGroup:
    group_id: str
    members: List[str]
    verified_edges: List[dict]
    provenance: str = "verified_correspondence"
    declared_sessions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"group_id": self.group_id, "members": list(self.members), "verified_edges": list(self.verified_edges),
                "provenance": self.provenance, "declared_sessions": list(self.declared_sessions)}


@dataclass
class SceneGrouping:
    groups: List[SceneGroup]                 # connected components with >= 2 images, largest first
    unlinked: List[str]                      # images with no verified link to any other
    uncertain_links: List[dict]              # loose matches that did NOT pass verification (never merged)
    declared_conflicts: List[dict]           # a declared session that verification split
    pairs_checked: int
    verified_pairs: int
    thresholds: GroupingThresholds

    @property
    def primary(self) -> Optional[SceneGroup]:
        return self.groups[0] if self.groups else None

    def group_of(self, evidence_id: str) -> Optional[SceneGroup]:
        return next((g for g in self.groups if evidence_id in g.members), None)

    def to_dict(self) -> dict:
        return {"groups": [g.to_dict() for g in self.groups], "unlinked": list(self.unlinked),
                "uncertain_links": list(self.uncertain_links), "declared_conflicts": list(self.declared_conflicts),
                "pairs_checked": self.pairs_checked, "verified_pairs": self.verified_pairs,
                "thresholds": vars(self.thresholds)}


def _group_id(members: Iterable[str]) -> str:
    return "sg-" + hashlib.sha1("|".join(sorted(members)).encode()).hexdigest()[:10]


def group_images(ids: Sequence[str], verifier: PairVerifier,
                 thresholds: GroupingThresholds = GroupingThresholds(),
                 declared_sessions: Optional[Dict[str, Sequence[str]]] = None,
                 neighbour_window: int = 8, max_exhaustive_pairs: int = 2000) -> SceneGrouping:
    """Partition ``ids`` into scene groups using verified correspondences only."""
    ids = list(dict.fromkeys(ids))
    n = len(ids)
    if n * (n - 1) // 2 <= max_exhaustive_pairs:
        pairs = [(ids[i], ids[j]) for i in range(n) for j in range(i + 1, n)]
    else:
        # ponytail: ordered-neighbour window; misses links between far-apart ids -> they surface as
        # separate groups (conservative). Swap in a retrieval index (vocab tree) if captures exceed ~65 images.
        pairs = [(ids[i], ids[j]) for i in range(n) for j in range(i + 1, min(n, i + 1 + neighbour_window))]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges: List[dict] = []
    uncertain: List[dict] = []
    for a, b in pairs:
        ev = verifier(a, b)
        if ev.inliers >= thresholds.min_inliers and ev.inlier_ratio >= thresholds.min_inlier_ratio:
            edges.append(ev.to_dict())
            parent[find(a)] = find(b)
        elif ev.inliers >= thresholds.uncertain_inliers:
            uncertain.append(ev.to_dict())
    comps: Dict[str, List[str]] = {}
    for i in ids:
        comps.setdefault(find(i), []).append(i)
    session_of: Dict[str, List[str]] = {}
    for sid, members in (declared_sessions or {}).items():
        for m in members:
            session_of.setdefault(m, []).append(sid)
    groups = []
    for members in comps.values():
        if len(members) < 2:
            continue
        mset = set(members)
        groups.append(SceneGroup(
            group_id=_group_id(members), members=sorted(members),
            verified_edges=[e for e in edges if e["a"] in mset],
            declared_sessions=sorted({s for m in members for s in session_of.get(m, [])})))
    groups.sort(key=lambda g: (-len(g.members), g.group_id))
    unlinked = sorted(m[0] for m in comps.values() if len(m) == 1)
    conflicts = []
    for sid, members in (declared_sessions or {}).items():
        homes = {find(m) for m in members if m in parent}
        if len(homes) > 1:
            conflicts.append({"session": sid, "reason": "declared as one capture but verified correspondences "
                              f"split it into {len(homes)} scene groups (or unlinked images)",
                              "members": sorted(members)})
    return SceneGrouping(groups, unlinked, uncertain, conflicts, len(pairs), len(edges), thresholds)


# --------------------------------------------------------------------------- diagnostics


class ReconOutcome(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    INCOMPATIBLE = "geometrically_incompatible"


_VALID = (ReconOutcome.SUCCESS, ReconOutcome.PARTIAL)


@dataclass(frozen=True)
class QualityThresholds:
    #: Provisional; see module docstring.
    success_registered_fraction: float = 0.9
    partial_registered_fraction: float = 0.4
    max_reprojection_error_px: float = 2.0
    min_sparse_points: int = 30
    min_camera_coverage: float = 0.05


@dataclass
class RegistrationDiagnostics:
    outcome: ReconOutcome
    n_input: int
    registered_images: int
    registered_fraction: float
    verified_pairs: Optional[int]
    scene_groups: Optional[int]
    components: Optional[int]
    mean_reprojection_error_px: Optional[float]
    camera_coverage: Optional[float]
    sparse_points: int
    dense_points: Optional[int]
    mesh_valid: Optional[bool]
    runtime_s: float
    failure_reason: Optional[str]
    reasons: List[str]
    unmeasured: List[str]
    refused_components: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(vars(self))
        d["outcome"] = self.outcome.value
        return d


def check_poses(result: ReconstructionResult) -> List[str]:
    """Reasons the camera estimate is not physically usable (empty = usable)."""
    issues = []
    centers = []
    for p in result.camera_poses:
        pos, q = np.asarray(p.position, float), np.asarray(p.rotation, float)
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(q))):
            issues.append(f"non-finite pose for {p.evidence_id}")
        elif abs(np.linalg.norm(q) - 1.0) > 1e-3:
            issues.append(f"non-unit rotation quaternion for {p.evidence_id} (|q|={np.linalg.norm(q):.4f})")
        else:
            centers.append(pos)
    if len(centers) >= 2:
        c = np.array(centers)
        if float(np.ptp(c, axis=0).max()) < 1e-9:
            issues.append("all camera centres coincide: no baseline, depth is unobservable")
    pts = np.array([p.position for p in result.points], float) if result.points else np.zeros((0, 3))
    if pts.size and not np.all(np.isfinite(pts)):
        issues.append("non-finite 3D points")
    return issues


def camera_coverage(result: ReconstructionResult) -> Optional[float]:
    """1 - |mean unit direction from the point-cloud centroid to each camera|: 0 = all cameras on one side,
    ->1 = cameras surround the scene. Gauge-invariant (no assumed up axis or scale)."""
    if not result.points or len(result.camera_poses) < 2:
        return None
    centroid = np.median(np.array([p.position for p in result.points], float), axis=0)
    d = np.array([p.position for p in result.camera_poses], float) - centroid
    norms = np.linalg.norm(d, axis=1)
    d = d[norms > 1e-12] / norms[norms > 1e-12, None]
    return float(1.0 - np.linalg.norm(d.mean(axis=0))) if len(d) >= 2 else None


def diagnose(result: Optional[ReconstructionResult], n_input: int, *, runtime_s: float = 0.0,
             grouping: Optional[SceneGrouping] = None, error: Optional[str] = None,
             require_single_scene: bool = False, mesh_valid: Optional[bool] = None,
             thresholds: QualityThresholds = QualityThresholds()) -> RegistrationDiagnostics:
    unmeasured: List[str] = []
    reasons: List[str] = []
    refused: List[dict] = []
    if result is None:
        return RegistrationDiagnostics(
            ReconOutcome.FAILED, n_input, 0, 0.0, None, None, None, None, None, 0, None, mesh_valid,
            runtime_s, error or "backend raised", [error or "backend raised"], ["all"])
    stats = result.registration_stats or {}
    reg_ids = {p.evidence_id for p in result.camera_poses}
    registered = len(reg_ids)
    frac = registered / n_input if n_input else 0.0
    err = stats.get("mean_reprojection_error_px")
    if err is None:
        unmeasured.append("mean_reprojection_error_px")
    comps = len(stats["components"]) if stats.get("components") else (1 if stats else None)
    if comps is None:
        unmeasured.append("components")
    cov = camera_coverage(result)
    if cov is None:
        unmeasured.append("camera_coverage")
    mr = result.merge_report or {}
    for sid, status in (mr.get("status_by_session") or {}).items():
        if status not in ("aligned", "reference"):
            refused.append({"sub_model": sid, "status": status, "reason": (mr.get("reasons") or {}).get(sid)})
    pose_issues = check_poses(result)
    ngroups = len(grouping.groups) if grouping else None

    if pose_issues:
        outcome = ReconOutcome.FAILED
        reasons += pose_issues
    elif registered < 2 or len(result.points) < thresholds.min_sparse_points:
        outcome = ReconOutcome.FAILED
        reasons.append(f"only {registered}/{n_input} images registered with {len(result.points)} sparse points "
                       f"(need >=2 images and >={thresholds.min_sparse_points} points)")
    elif err is not None and err > thresholds.max_reprojection_error_px:
        outcome = ReconOutcome.FAILED
        reasons.append(f"mean reprojection error {err:.2f}px exceeds {thresholds.max_reprojection_error_px}px")
    elif refused or (require_single_scene and ngroups and ngroups > 1):
        outcome = ReconOutcome.INCOMPATIBLE
        if refused:
            reasons.append(f"{len(refused)} reconstructed component(s) could not be registered into the "
                           "reference frame and were excluded")
        if require_single_scene and ngroups and ngroups > 1:
            reasons.append(f"{ngroups} geometrically disconnected scene groups in a set declared as one scene")
    elif frac >= thresholds.success_registered_fraction and (cov is None or cov >= thresholds.min_camera_coverage):
        outcome = ReconOutcome.SUCCESS
    elif frac >= thresholds.partial_registered_fraction:
        outcome = ReconOutcome.PARTIAL
        reasons.append(f"{registered}/{n_input} images registered ({frac:.0%})"
                       + (f"; camera coverage {cov:.2f}" if cov is not None and cov < thresholds.min_camera_coverage else ""))
    else:
        outcome = ReconOutcome.FAILED
        reasons.append(f"only {registered}/{n_input} images registered ({frac:.0%}); below the partial floor")
    return RegistrationDiagnostics(
        outcome, n_input, registered, frac, grouping.verified_pairs if grouping else None, ngroups, comps,
        err, cov, len(result.points), len(result.dense_points) if result.dense_points is not None else None,
        mesh_valid, runtime_s, None if outcome in _VALID else "; ".join(reasons), reasons, unmeasured, refused)


def diagnose_counts(n_input: int, registered_ids: Sequence[str], n_points: int, *,
                    grouping: Optional[SceneGrouping] = None, runtime_s: float = 0.0,
                    thresholds: QualityThresholds = QualityThresholds()) -> RegistrationDiagnostics:
    """Outcome from counts alone, for callers (the job layer) that only hold the registered ids. Reprojection
    error, coverage and components were not measured at this layer and are reported as such."""
    registered = len(set(registered_ids))
    frac = registered / n_input if n_input else 0.0
    if registered < 2 or n_points < thresholds.min_sparse_points or frac < thresholds.partial_registered_fraction:
        outcome, reasons = ReconOutcome.FAILED, [f"{registered}/{n_input} images registered, {n_points} points"]
    elif frac >= thresholds.success_registered_fraction:
        outcome, reasons = ReconOutcome.SUCCESS, []
    else:
        outcome, reasons = ReconOutcome.PARTIAL, [f"{registered}/{n_input} images registered ({frac:.0%})"]
    return RegistrationDiagnostics(
        outcome, n_input, registered, frac, grouping.verified_pairs if grouping else None,
        len(grouping.groups) if grouping else None, None, None, None, n_points, None, None, runtime_s,
        None if outcome in _VALID else "; ".join(reasons), reasons,
        ["mean_reprojection_error_px", "components", "camera_coverage"])


# --------------------------------------------------------------------------- correction loop


def _rotmat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def pose_agreement_deg(a: ReconstructionResult, b: ReconstructionResult, min_shared: int = 3) -> Optional[float]:
    """Median gauge-invariant disagreement (degrees) of relative camera rotations on cameras both models
    registered. None when fewer than ``min_shared`` cameras are shared (agreement cannot be assessed)."""
    pa = {p.evidence_id: _rotmat(p.rotation) for p in a.camera_poses}
    pb = {p.evidence_id: _rotmat(p.rotation) for p in b.camera_poses}
    shared = sorted(set(pa) & set(pb))
    if len(shared) < min_shared:
        return None
    errs = []
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            ra = pa[shared[i]].T @ pa[shared[j]]
            rb = pb[shared[i]].T @ pb[shared[j]]
            c = (np.trace(ra.T @ rb) - 1) / 2
            errs.append(math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    return float(np.median(errs))


@dataclass
class Attempt:
    index: int
    config: dict
    diagnostics: RegistrationDiagnostics
    accepted: bool
    decision: str

    def to_dict(self) -> dict:
        return {"index": self.index, "config": self.config, "diagnostics": self.diagnostics.to_dict(),
                "accepted": self.accepted, "decision": self.decision}


@dataclass
class CorrectionTrace:
    attempts: List[Attempt]
    best_result: Optional[ReconstructionResult]
    best: Optional[RegistrationDiagnostics]
    stop_reason: str          # success | budget_exhausted | additional_evidence_required | no_correction_available
    additional_evidence_needed: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"attempts": [a.to_dict() for a in self.attempts], "stop_reason": self.stop_reason,
                "best": self.best.to_dict() if self.best else None,
                "additional_evidence_needed": list(self.additional_evidence_needed)}


#: Warping guard for guided matching: measured (colmap_backend.DEFAULT_GUIDED_MATCHING note) 24.6 deg median
#: rotation error on repetitive architecture while reprojection error stayed small.
MAX_GUIDED_DISAGREEMENT_DEG = 5.0
MAX_ERR_REGRESSION_PX = 0.5
#: A camera that only a RETRY registered cannot be cross-checked against the previous model, so it must carry its
#: own evidence: this many verified pair links (grouping) to cameras the previous model already placed. Measured:
#: the camera gained by guided matching on the 19-image subset had 1 strong link and a 23.6 deg / 1.0 m error.
MIN_GAIN_SUPPORT = 2
_DEFAULT_CFG = {"robust_sift": True, "guided_matching": False}
_RETRY_LADDER = [{"robust_sift": True, "guided_matching": True}, {"robust_sift": False, "guided_matching": False}]


def reconstruct_group(evidence: list, backend_factory: Callable[[dict], IReconstructionBackend], *,
                      grouping: Optional[SceneGrouping] = None, max_retries: int = 2,
                      thresholds: QualityThresholds = QualityThresholds(),
                      require_single_scene: bool = False) -> CorrectionTrace:
    """Reconstruct ``evidence`` (one scene group) with a bounded measured-improvement retry loop.

    ``backend_factory(config)`` builds a backend; config keys: ``robust_sift``, ``guided_matching``.
    The retry re-runs the whole backend (its workspace is temporary), changing only the matching configuration.
    """
    n = len(evidence)
    attempts: List[Attempt] = []
    best_res: Optional[ReconstructionResult] = None
    best: Optional[RegistrationDiagnostics] = None
    i = 0
    while True:
        cfg = _DEFAULT_CFG if i == 0 else _RETRY_LADDER[i - 1]
        t0 = time.monotonic()
        try:
            res, exc = backend_factory(cfg).reconstruct(evidence), None
        except Exception as e:                       # a raising retry must not erase the best result
            res, exc = None, f"{type(e).__name__}: {e}"
        diag = diagnose(res, n, runtime_s=time.monotonic() - t0, grouping=grouping, error=exc,
                        require_single_scene=require_single_scene, thresholds=thresholds)
        if best is None:
            accepted = diag.outcome in _VALID
            why = "first valid attempt" if accepted else f"first attempt {diag.outcome.value}; not adopted"
        else:
            accepted, why = _accept(diag, res, best, best_res, cfg, grouping)
        attempts.append(Attempt(i, dict(cfg), diag, accepted, why))
        if accepted:
            best_res, best = res, diag
        if best is not None and best.outcome == ReconOutcome.SUCCESS:
            return CorrectionTrace(attempts, best_res, best, "success")
        missing = _unregistered(evidence, best_res if best_res is not None else res)
        no_overlap = [m for m in missing if grouping is not None and grouping.group_of(m) is None]
        cur = best or diag
        reproj_bad = (cur.mean_reprojection_error_px or 0.0) > thresholds.max_reprojection_error_px
        if no_overlap or reproj_bad:
            return CorrectionTrace(attempts, best_res, best, "additional_evidence_required", no_overlap or missing)
        i += 1
        if i > max_retries:
            return CorrectionTrace(attempts, best_res, best, "budget_exhausted", missing)
        if i > len(_RETRY_LADDER):
            return CorrectionTrace(attempts, best_res, best, "no_correction_available", missing)


def _unregistered(evidence: list, result: Optional[ReconstructionResult]) -> List[str]:
    got = {p.evidence_id for p in result.camera_poses} if result else set()
    return sorted(e.id for e in evidence if e.id not in got)


def _accept(new: RegistrationDiagnostics, new_res, best: RegistrationDiagnostics, best_res, cfg: dict,
            grouping: Optional[SceneGrouping] = None):
    if new_res is None or new.outcome not in _VALID:
        return False, f"retry produced {new.outcome.value}; last valid result kept"
    if new.registered_images <= best.registered_images:
        return False, f"no gain ({new.registered_images} vs {best.registered_images} registered); kept previous"
    if (new.mean_reprojection_error_px is not None and best.mean_reprojection_error_px is not None
            and new.mean_reprojection_error_px > best.mean_reprojection_error_px + MAX_ERR_REGRESSION_PX):
        return False, "reprojection error regressed beyond tolerance; kept previous"
    placed = {p.evidence_id for p in best_res.camera_poses}
    gained = {p.evidence_id for p in new_res.camera_poses} - placed
    links: Dict[str, set] = {}
    for g in (grouping.groups if grouping else []):
        for e in g.verified_edges:
            links.setdefault(e["a"], set()).add(e["b"])
            links.setdefault(e["b"], set()).add(e["a"])
    weak = sorted(c for c in gained if len(links.get(c, set()) & placed) < MIN_GAIN_SUPPORT)
    if weak:
        return False, (f"gained camera(s) {weak} lack {MIN_GAIN_SUPPORT} independently verified links to the "
                       "previously placed cameras; gain not trusted, kept previous")
    if cfg.get("guided_matching"):
        dev = pose_agreement_deg(best_res, new_res)
        if dev is None:
            return False, "guided-matching gain cannot be cross-checked (<3 shared cameras); kept previous"
        if dev > MAX_GUIDED_DISAGREEMENT_DEG:
            return False, f"guided matching warped shared cameras by {dev:.1f} deg (> {MAX_GUIDED_DISAGREEMENT_DEG}); rejected"
    return True, f"registered {new.registered_images} vs {best.registered_images}, error within tolerance"


# --------------------------------------------------------------------------- persistence / identity

_RANK = {"success": 3, "partial": 2}


class SceneRegistry:
    """JSON-persisted scene groups + last valid reconstruction, one file per world."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def _path(self, world_id: str) -> Path:
        return self.root / f"{world_id}.json"

    def load(self, world_id: str) -> dict:
        try:
            return json.loads(self._path(world_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"world_id": world_id, "groups": [], "last_valid": None, "history": []}

    def _save(self, state: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self._path(state["world_id"])
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, p)

    def record_grouping(self, world_id: str, grouping: SceneGrouping) -> List[SceneGroup]:
        """Persist groups, KEEPING the id of any previously known group they overlap (stable identity)."""
        state = self.load(world_id)
        known = {g["group_id"]: set(g["members"]) for g in state["groups"]}
        out = []
        for g in grouping.groups:
            overlaps = sorted(((len(set(g.members) & m), gid) for gid, m in known.items() if set(g.members) & m),
                              reverse=True)
            gid = overlaps[0][1] if overlaps else g.group_id
            for _, absorbed in overlaps[1:]:
                known.pop(absorbed, None)
            known[gid] = set(g.members) | known.get(gid, set())
            out.append(replace(g, group_id=gid, members=sorted(known[gid])))
        state["groups"] = [g.to_dict() for g in out]
        state["unlinked"] = list(grouping.unlinked)
        state["uncertain_links"] = list(grouping.uncertain_links)
        state["declared_conflicts"] = list(grouping.declared_conflicts)
        self._save(state)
        return out

    def commit_reconstruction(self, world_id: str, group_id: str, diag: RegistrationDiagnostics,
                              version_id: Optional[str], registered_ids: Sequence[str]) -> Tuple[bool, str]:
        """Adopt as the world's last valid reconstruction only if valid and not worse than the current one."""
        state = self.load(world_id)
        entry = {"group_id": group_id, "version_id": version_id, "registered_ids": sorted(registered_ids),
                 "diagnostics": diag.to_dict()}
        prev = state.get("last_valid")
        if diag.outcome not in _VALID:
            ok, why = False, f"{diag.outcome.value}: last valid reconstruction kept"
        elif prev is not None and prev["group_id"] == group_id and (
                _RANK[diag.outcome.value], diag.registered_images) < (
                _RANK[prev["diagnostics"]["outcome"]], prev["diagnostics"]["registered_images"]):
            ok, why = False, "worse than the last valid reconstruction: kept previous"
        else:
            ok, why = True, "adopted"
        if ok:
            state["last_valid"] = entry
        state["history"].append(dict(entry, adopted=ok, decision=why, at=time.time()))
        self._save(state)
        return ok, why
