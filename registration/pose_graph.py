"""Robust SE(3) pose-graph optimisation (backend: scipy least-squares; GTSAM can slot in behind the same API).

Inputs are RAW measurements and are never mutated: ``optimize_pose_graph`` returns new poses next to the untouched
initial ones, so optimised estimates stay distinguishable from observations.

Conventions (checked, not assumed): a pose is the transform taking body coordinates to the world frame -- a position
``t`` in metres and a rotation ``R`` (unit quaternion ``[x, y, z, w]``). A relative-pose edge ``i -> j`` carries the
measured pose of j in i's frame: ``R_m = R_i^T R_j`` and ``t_m = R_i^T (t_j - t_i)``. Residual per edge is
``[rotvec(R_m^T R_i^T R_j) / sigma_r, (R_i^T (t_j - t_i) - t_m) / sigma_t]``. A GNSS/ENU prior ties a node's POSITION
to a world-frame point (metres) with an isotropic sigma.

Gauge: with no priors the first node (sorted id) is held fixed (the graph only defines poses relative to it). With
priors, at least 3 non-collinear priors are required to fix rotation; fewer raises, because a silently arbitrary
orientation would look like a measured one.

Robustness: ``loss`` is linear by default. Measured on synthetic trajectories: Huber smears a gross outlier's error
onto healthy odometry edges, and Cauchy started from a drifted guess rejects a LEGITIMATE loop closure. So outliers are
found by a likelihood-ratio test (see optimize_pose_graph), the offending constraint is REJECTED and reported, and the
graph re-solved. A disconnected graph (or one that becomes disconnected after a rejection) raises instead of being
optimised piecewise into inconsistent frames.

Costs reported are 0.5 * sum(normalised residual^2) over the constraints in force at that point.

Not here: inertial pre-integration / VIO. IMU data would have to be turned into relative-pose edges by a separate stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

CHI2_999_DOF6 = 22.458          # chi-square 0.999 quantile, 6 degrees of freedom (edge)
CHI2_999_DOF3 = 16.266          # 3 degrees of freedom (GNSS prior)
MAX_REJECTION_PASSES = 5
AMBIGUITY_RATIO = 0.8           # a rival whose removal explains >= 80% as much as the best makes the pick ambiguous


class PoseGraphError(ValueError):
    """Invalid or unsolvable pose graph (bad input, disconnected, under-constrained)."""


@dataclass(frozen=True)
class Pose:
    t: Tuple[float, float, float]
    q: Tuple[float, float, float, float]          # x, y, z, w

    def rot(self) -> Rotation:
        return Rotation.from_quat(self.q)


@dataclass(frozen=True)
class Edge:
    id: str
    i: str
    j: str
    t: Tuple[float, float, float]                 # measured position of j in i's frame (m)
    q: Tuple[float, float, float, float]          # measured rotation of j in i's frame
    sigma_t: float = 0.05                         # m
    sigma_r: float = 0.02                         # rad


@dataclass(frozen=True)
class GnssPrior:
    node: str
    xyz: Tuple[float, float, float]               # world/ENU metres
    sigma: float = 1.0                            # m, isotropic


@dataclass
class PoseGraphResult:
    backend: str
    poses: Dict[str, Pose]                        # OPTIMISED estimates
    initial: Dict[str, Pose]                      # untouched input
    converged: bool
    iterations: int
    initial_cost: float
    final_cost: float
    rejected_edges: List[dict] = field(default_factory=list)
    rejected_priors: List[dict] = field(default_factory=list)
    edge_chi2: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    #: ids of constraints that are inconsistent with the rest but NOT individually identifiable (removing any one of
    #: them restores consistency about equally well, e.g. a single loop). Nothing was rejected; poses are a compromise.
    ambiguous: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        p = lambda d: {k: {"t": list(v.t), "q": list(v.q)} for k, v in d.items()}   # noqa: E731
        return {"backend": self.backend, "converged": self.converged, "iterations": self.iterations,
                "initial_cost": self.initial_cost, "final_cost": self.final_cost, "optimized": p(self.poses),
                "initial": p(self.initial), "rejected_edges": self.rejected_edges,
                "rejected_priors": self.rejected_priors, "edge_chi2": self.edge_chi2, "warnings": self.warnings,
                "ambiguous": self.ambiguous}


def _components(nodes: Sequence[str], edges: Sequence[Edge]) -> List[List[str]]:
    parent = {n: n for n in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for e in edges:
        parent[find(e.i)] = find(e.j)
    groups: Dict[str, List[str]] = {}
    for n in nodes:
        groups.setdefault(find(n), []).append(n)
    return sorted(sorted(g) for g in groups.values())


def _validate(poses, edges, priors):
    if not poses:
        raise PoseGraphError("no poses")
    ids = [e.id for e in edges]
    if len(set(ids)) != len(ids):
        raise PoseGraphError("duplicate edge ids")
    for name, pose in poses.items():
        if not np.all(np.isfinite(pose.t)) or abs(np.linalg.norm(pose.q) - 1.0) > 1e-3:
            raise PoseGraphError(f"pose {name!r}: non-finite position or non-unit quaternion")
    for e in edges:
        if e.i not in poses or e.j not in poses or e.i == e.j:
            raise PoseGraphError(f"edge {e.id!r}: unknown or identical nodes {e.i!r}->{e.j!r}")
        if e.sigma_t <= 0 or e.sigma_r <= 0 or abs(np.linalg.norm(e.q) - 1.0) > 1e-3 or not np.all(np.isfinite(e.t)):
            raise PoseGraphError(f"edge {e.id!r}: sigma must be > 0, q unit-norm, t finite (metres)")
    for g in priors:
        if g.node not in poses or g.sigma <= 0 or not np.all(np.isfinite(g.xyz)):
            raise PoseGraphError(f"prior on {g.node!r}: unknown node, sigma <= 0 or non-finite position")


def _check_priors(priors):
    if priors:
        pts = np.array([g.xyz for g in priors], float)
        if len(priors) < 3 or np.linalg.matrix_rank(pts - pts.mean(0), tol=1e-6) < 2:
            raise PoseGraphError("GNSS priors cannot fix orientation: need >= 3 non-collinear anchors")


def _base(order, poses):
    """Linearisation point: batch rotation + (n,3) positions, in sorted-id order."""
    return (Rotation.from_quat([poses[n].q for n in order]), np.array([poses[n].t for n in order], float))


def _apply(base, d):
    """Perturbation d (n,6) = [body-frame rotation delta (rotvec), position delta] applied to ``base``."""
    return base[0] * Rotation.from_rotvec(d[:, :3]), base[1] + d[:, 3:]


def _to_poses(order, base):
    q, t = base[0].as_quat(), base[1]
    return {n: Pose(tuple(map(float, t[k])), tuple(map(float, q[k]))) for k, n in enumerate(order)}


def _residual_fn(order, edges, priors, base):
    """Vectorised residuals as a function of per-node perturbations from ``base`` (well-conditioned: no rotation
    wrap-around, unlike absolute rotation vectors, which are singular past pi)."""
    idx = {n: k for k, n in enumerate(order)}
    ii = np.array([idx[e.i] for e in edges], int)
    jj = np.array([idx[e.j] for e in edges], int)
    pn = np.array([idx[g.node] for g in priors], int)
    Rm = Rotation.from_quat([e.q for e in edges]) if edges else None
    tm = np.array([e.t for e in edges], float).reshape(-1, 3)
    st = np.array([e.sigma_t for e in edges], float)[:, None]
    sr = np.array([e.sigma_r for e in edges], float)[:, None]
    gx = np.array([g.xyz for g in priors], float).reshape(-1, 3)
    gs = np.array([g.sigma for g in priors], float)[:, None]

    def residual(d):
        R, T = _apply(base, d.reshape(-1, 6))
        parts = []
        if len(edges):
            ri = R[ii]
            parts.append(np.hstack([(Rm.inv() * ri.inv() * R[jj]).as_rotvec() / sr,
                                    (ri.inv().apply(T[jj] - T[ii]) - tm) / st]).ravel())
        if len(priors):
            parts.append(((T[pn] - gx) / gs).ravel())
        return np.concatenate(parts) if parts else np.zeros(0)

    return residual


def _sparsity(order, edges, priors, fix_first):
    """Each residual block touches only its own nodes: lets finite differencing group columns (orders faster)."""
    from scipy.sparse import lil_matrix
    idx = {n: k for k, n in enumerate(order)}
    off = 1 if fix_first else 0
    s = lil_matrix((6 * len(edges) + 3 * len(priors), 6 * (len(order) - off)), dtype=int)
    for r, e in enumerate(edges):
        for n in (e.i, e.j):
            if idx[n] >= off:
                s[6 * r:6 * r + 6, 6 * (idx[n] - off):6 * (idx[n] - off) + 6] = 1
    for r, g in enumerate(priors):
        if idx[g.node] >= off:
            row, col = 6 * len(edges) + 3 * r, 6 * (idx[g.node] - off) + 3
            s[row:row + 3, col:col + 3] = 1
    return s


def _solve(order, base, edges, priors, fix_first, loss):
    """One robust solve around ``base``; returns the re-based estimate, success, nfev and final residuals."""
    full = _residual_fn(order, edges, priors, base)
    n = len(order)
    if fix_first:
        fun = lambda xf: full(np.concatenate([np.zeros(6), xf]))      # noqa: E731
        size = 6 * (n - 1)
    else:
        fun, size = full, 6 * n
    ok, nfev, x = True, 0, np.zeros(size)
    if size and (len(edges) or len(priors)):
        sol = least_squares(fun, x, loss=loss, f_scale=1.0, method="trf", max_nfev=100, xtol=1e-9, ftol=1e-9,
                            gtol=1e-9, jac_sparsity=_sparsity(order, edges, priors, fix_first))
        x, ok, nfev = sol.x, bool(sol.success), int(sol.nfev)
    d = np.concatenate([np.zeros(6), x]) if fix_first else x
    new = _apply(base, d.reshape(-1, 6))
    return new, ok, nfev, _residual_fn(order, edges, priors, new)(np.zeros(6 * n))


def _kabsch(order, base, priors):
    """Rigid (no scale) alignment of the initial trajectory onto the GNSS anchors, so a local solver starts in the
    anchors' frame. Returns a new base; positions AND orientations are carried over."""
    idx = {n: k for k, n in enumerate(order)}
    src = np.array([base[1][idx[g.node]] for g in priors])
    dst = np.array([g.xyz for g in priors], float)
    cs, cd = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    sign = np.sign(np.linalg.det(vt.T @ u.T)) or 1.0
    rot = Rotation.from_matrix(vt.T @ np.diag([1, 1, sign]) @ u.T)
    return rot * base[0], rot.apply(base[1] - cs) + cd


def _chi2(res, edges, priors):
    chi, k = {}, 0
    for e in edges:
        chi[e.id] = float(np.sum(res[k:k + 6] ** 2)); k += 6
    pchi = {}
    for g in priors:
        pchi[g.node] = float(np.sum(res[k:k + 3] ** 2)); k += 3
    return chi, pchi


def optimize_pose_graph(poses: Dict[str, Pose], edges: Sequence[Edge], priors: Sequence[GnssPrior] = (),
                        loss: str = "linear", test_candidates: int = 6) -> PoseGraphResult:
    """Optimise ``poses`` against ``edges``/``priors``. Raises PoseGraphError rather than returning a guess.

    Outliers are found with a likelihood-ratio test, not by residual size: a candidate constraint is dropped, the
    graph re-solved, and the constraint is rejected only if 2 * (cost drop) exceeds the chi-square 99.9% bound for
    its degrees of freedom. A consistent loop closure removes about its own (small) residual; a gross outlier
    removes far more, because the rest of the graph had been bent to accommodate it."""
    edges, priors = list(edges), list(priors)
    _validate(poses, edges, priors)
    _check_priors(priors)
    order = sorted(poses)
    comps = _components(order, edges)
    if len(comps) > 1:
        raise PoseGraphError(f"graph is disconnected: {len(comps)} components {comps}")
    base = _base(order, poses)
    if priors:
        base = _kabsch(order, base, priors)       # initialise in the anchors' frame; optimisation then refines
    zero = np.zeros(6 * len(order))
    fix_first = not priors
    initial_cost = 0.5 * float(np.sum(_residual_fn(order, edges, priors, base)(zero) ** 2))
    active, active_priors = list(edges), list(priors)
    rejected_edges: List[dict] = []
    rejected_priors: List[dict] = []
    warnings: List[str] = []
    ok, iterations = True, 0
    ambiguous: List[str] = []

    for _ in range(MAX_REJECTION_PASSES + 1):
        base, ok, nfev, res = _solve(order, base, active, active_priors, not active_priors, loss)
        iterations += nfev
        cost0 = 0.5 * float(np.sum(res ** 2))
        chi, pchi = _chi2(res, active, active_priors)
        cands = [("e", e, chi[e.id]) for e in active] + [("p", g, pchi[g.node]) for g in active_priors]
        cands = sorted(cands, key=lambda c: -c[2])[:test_candidates]
        best, explaining = None, []
        for kind, c, _c in cands:
            e2 = [e for e in active if e is not c]
            p2 = [g for g in active_priors if g is not c]
            if kind == "e" and len(_components(order, e2)) > 1:
                continue                                             # removal would disconnect: untestable
            if kind == "p":
                try:
                    _check_priors(p2)
                except PoseGraphError:
                    continue
            _, _, _, r2 = _solve(order, base, e2, p2, not p2, loss)
            drop = 2.0 * (cost0 - 0.5 * float(np.sum(r2 ** 2)))
            bound = CHI2_999_DOF6 if kind == "e" else CHI2_999_DOF3
            if drop > bound:
                explaining.append((drop / bound, kind, c))
            if drop > bound and (best is None or drop / bound > best[3] / best[4]):
                best = (kind, c, _c, drop, bound)
        if best is None:
            break
        near = [x for x in explaining if x[0] >= AMBIGUITY_RATIO * best[3] / best[4]]
        if len(near) > 1:
            # several constraints each explain the inconsistency about equally well: the data cannot say which one
            # is wrong, so rejecting "the best" would be an arbitrary pick that may discard a good measurement
            ambiguous = sorted(c.id if k == "e" else f"gnss:{c.node}" for _, k, c in near)
            warnings.append(f"inconsistent but not identifiable: removing any of {ambiguous} restores consistency "
                            "about equally well; nothing rejected, poses are a compromise")
            break
        kind, c, own, drop, bound = best
        if len(rejected_edges) + len(rejected_priors) >= MAX_REJECTION_PASSES:
            warnings.append("rejection budget exhausted; remaining inconsistency (if any) is unresolved")
            break
        if kind == "e":
            rejected_edges.append({"id": c.id, "i": c.i, "j": c.j, "chi2": own, "lr_statistic": drop,
                                   "reason": f"removing it lowers 2*cost by {drop:.1f} > {bound} (chi2 99.9%)"})
            active = [e for e in active if e is not c]
        else:
            rejected_priors.append({"node": c.node, "chi2": own, "lr_statistic": drop,
                                    "reason": f"removing it lowers 2*cost by {drop:.1f} > {bound} (chi2 99.9%)"})
            active_priors = [g for g in active_priors if g is not c]

    res = _residual_fn(order, active, active_priors, base)(zero)
    chi, _ = _chi2(res, active, active_priors)
    if not active and not active_priors:
        warnings.append("no constraints: poses returned unchanged")
    return PoseGraphResult(
        backend="scipy-least-squares", poses=_to_poses(order, base), initial=dict(poses), converged=ok,
        iterations=iterations, initial_cost=initial_cost, final_cost=0.5 * float(np.sum(res ** 2)),
        rejected_edges=rejected_edges, rejected_priors=rejected_priors, edge_chi2=chi, warnings=warnings,
        ambiguous=ambiguous)
