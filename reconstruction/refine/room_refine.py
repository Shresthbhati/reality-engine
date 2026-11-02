"""Anytime robust refinement of a room shell (floor, ceiling, walls as planes) against a point cloud.

Why: ``detect_room_planes`` finds a room from histogram peaks. The planes are good enough to name a room but carry the
bias of a peak bin and of RANSAC-style tilt: a wall a few degrees off vertical or a few percent off in offset moves the
room's length / width / height by several percent. This module fits ALL planes jointly, with the architecture as a
constraint, and keeps the best result found so far until a time budget or convergence.

Model (all in a frame normalised by the cloud's robust extent, so no absolute-metre threshold exists anywhere):

  * floor and ceiling share ONE normal (parallel) and have their own offsets;
  * every wall is perpendicular to that normal (parameterised by an azimuth around it) and has an offset;
  * adjacent walls are *encouraged* to be parallel / perpendicular by a soft penalty sin(2*dphi): only pairs already
    within ``orth_gate_deg`` of that are included, and its weight is chosen from the DATA by held-out residual
    (``prior on`` vs ``off``), so a skewed room keeps its skew.

Estimator: EM with soft assignment of each point to planes (plus an outlier floor), Tukey re-descending weights on
point-to-plane residuals, noise scale re-estimated by MAD each iteration, and an exact weighted-least-squares M-step
(the weighted sufficient statistic of each plane is a 4x4 matrix, so the M-step does not touch the points).

The data-driven part: held-out model selection (prior on/off), Poisson-bootstrap restarts that are accepted only when
they lower a held-fixed truncated residual score on the points, and a bootstrap ensemble giving per-plane parameter
uncertainty; unstable or unsupported planes are rejected (returned unchanged and flagged).

Honest limits: it refines planes the detector already found; it does not discover missing walls. Opening rectangles are
excluded from the wall they sit on and re-projected onto the refined plane; they are not estimated.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import least_squares

TUKEY_C = 4.685                   # Tukey biweight constant (95% efficiency under Gaussian noise)
_KAPPA = math.exp(-4.5)           # outlier floor of the soft assignment: a point > ~3 sigma from all planes is an outlier
_ELL = 0.25                       # RMS lateral position of a wall point (normalised units): angle error -> displacement
_MIN_WEIGHT = 20.0                # a plane with less total weight than this is frozen (not enough evidence to move it)
_HORIZ_COS = math.cos(math.radians(20.0))


# --------------------------------------------------------------------------------------------------- result types
@dataclass
class RefinedPlane:
    role: str                       # floor | ceiling | wall
    normal: Tuple[float, float, float]
    d: float                        # n . x + d = 0, same units/frame as the input points
    inlier_count: int               # points whose nearest plane is this one and within 3 sigma
    inlier_rms: float               # RMS point-to-plane distance of those points (input units)
    angle_std_deg: float            # bootstrap std of the normal direction (nan if no ensemble)
    offset_std_rel: float           # bootstrap std of the offset, relative to the cloud scale (nan if no ensemble)
    status: str                     # refined | rejected_unstable | unsupported
    source_index: int               # index in the input plane list
    plane_id: str | None = None


@dataclass
class RoomRefinement:
    planes: List[RefinedPlane]
    ignored_planes: List[int]
    openings: List[dict]
    rms_before: float               # inlier RMS (input units) over the points the refined model calls inliers
    rms_after: float
    rms_before_rel: float
    rms_after_rel: float
    dims_before: Dict[str, float]
    dims_after: Dict[str, float]
    iterations: int
    restarts: int
    seconds: float
    converged: bool
    stop_reason: str
    orthogonality: Dict[str, float]
    scale: float
    diagnostics: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def plane_tuples(self) -> List[Tuple[str, Tuple[float, float, float], float]]:
        return [(p.role, p.normal, p.d) for p in self.planes]


# ------------------------------------------------------------------------------------------------ plane utilities
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= 0.0:
        raise ValueError("degenerate vector")
    return v / n


def _coerce(p, i: int) -> Tuple[str | None, np.ndarray, float, str | None]:
    if isinstance(p, dict):
        role, n, d, pid = p.get("role"), p["normal"], p["d"], p.get("plane_id")
    elif isinstance(p, (tuple, list)) and len(p) == 3 and isinstance(p[0], str):
        role, n, d = p
        pid = None
    else:
        role, n, d, pid = getattr(p, "role", None), p.normal, p.d, getattr(p, "plane_id", None)
    n = np.asarray(n, dtype=float)
    ln = float(np.linalg.norm(n))
    if not (math.isfinite(ln) and ln > 0 and math.isfinite(float(d))):
        raise ValueError(f"plane {i} is degenerate")
    return role, n / ln, float(d) / ln, pid


def _separation(a: Tuple[np.ndarray, float], b: Tuple[np.ndarray, float], c: np.ndarray) -> float:
    """Distance between two (near-)parallel planes measured around point ``c`` (symmetrised)."""
    (na, da), (nb, db) = a, b
    pb = c - (nb @ c + db) * nb
    pa = c - (na @ c + da) * na
    return 0.5 * (abs(na @ pb + da) + abs(nb @ pa + db))


def room_dimensions(planes: Sequence[Tuple[str, Sequence[float], float]], centre: Sequence[float],
                    oppose_deg: float = 30.0) -> Dict[str, float]:
    """length >= width (largest opposed-wall separations on the two dominant axes) and height (floor-ceiling).
    Works for any number of walls (an L-shape reports its bounding extent); a missing quantity is 0.0."""
    c = np.asarray(centre, dtype=float)
    hor = [(np.asarray(n, float), float(d)) for r, n, d in planes if r in ("floor", "ceiling")]
    walls = [(np.asarray(n, float), float(d)) for r, n, d in planes if r == "wall"]
    height = _separation(hor[0], hor[1], c) if len(hor) >= 2 else 0.0
    cos_op = -math.cos(math.radians(oppose_deg))
    groups: List[Tuple[np.ndarray, float]] = []        # (axis, max separation)
    for i in range(len(walls)):
        for j in range(i + 1, len(walls)):
            if walls[i][0] @ walls[j][0] < cos_op:
                axis = _unit(walls[i][0] - walls[j][0])
                s = _separation(walls[i], walls[j], c)
                for g, (ax, best) in enumerate(groups):
                    if abs(ax @ axis) > math.cos(math.radians(oppose_deg)):
                        groups[g] = (ax, max(best, s))
                        break
                else:
                    groups.append((axis, s))
    vals = sorted((s for _, s in groups), reverse=True)
    return {"length": vals[0] if vals else 0.0, "width": vals[1] if len(vals) > 1 else 0.0, "height": height}


# --------------------------------------------------------------------------------------------- model parameters
class _Layout:
    """theta = [a, b, d_horizontal..., phi_wall..., d_wall...]; planes ordered horizontals first then walls."""

    def __init__(self, n_horizontal: int, n_walls: int, n0: np.ndarray, ref: np.ndarray):
        self.nh, self.nw, self.K = n_horizontal, n_walls, n_horizontal + n_walls
        self.n0 = n0
        t = np.array([1.0, 0.0, 0.0]) if abs(n0[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        self.e1 = _unit(t - (t @ n0) * n0)
        self.e2 = np.cross(n0, self.e1)
        self.ref = ref
        self.size = 2 + n_horizontal + 2 * n_walls

    def frame(self, th: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        nh = _unit(self.n0 + th[0] * self.e1 + th[1] * self.e2)
        f1 = _unit(self.ref - (self.ref @ nh) * nh)
        return nh, f1, np.cross(nh, f1)

    def unpack(self, th: np.ndarray) -> np.ndarray:
        nh, f1, f2 = self.frame(th)
        V = np.empty((self.K, 4))
        V[: self.nh, :3] = nh
        V[: self.nh, 3] = th[2: 2 + self.nh]
        phi = th[2 + self.nh: 2 + self.nh + self.nw]
        V[self.nh:, :3] = np.cos(phi)[:, None] * f1 + np.sin(phi)[:, None] * f2
        V[self.nh:, 3] = th[2 + self.nh + self.nw:]
        return V

    def phi(self, th: np.ndarray) -> np.ndarray:
        return th[2 + self.nh: 2 + self.nh + self.nw]


def _estep(X: np.ndarray, V: np.ndarray, sigma: float, mask: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray]:
    A = np.abs(X @ V[:, :3].T + V[:, 3])
    g = np.exp(-0.5 * (A / sigma) ** 2)
    if mask is not None:
        g[mask] = 0.0
    w = g / (g.sum(1, keepdims=True) + _KAPPA)
    u = np.minimum(A / (TUKEY_C * sigma), 1.0)
    return w * (1.0 - u * u) ** 2, A


def _score(X: np.ndarray, V: np.ndarray, tau: float) -> float:
    dm = np.abs(X @ V[:, :3].T + V[:, 3]).min(1)
    return float(np.mean(np.minimum(1.0, (dm / tau) ** 2)))


def _mstep(lay: _Layout, th: np.ndarray, M: np.ndarray, wsum: np.ndarray, lam: float, gate: float) -> np.ndarray:
    M = np.where((wsum >= _MIN_WEIGHT)[:, None, None], M, 0.0)
    ev, Q = np.linalg.eigh(M)
    F = np.sqrt(np.clip(ev, 0.0, None))[:, :, None] * Q.transpose(0, 2, 1)
    pi = pj = coef = None
    if lam > 0 and lay.nw >= 2:
        phi = lay.phi(th)
        ws = wsum[lay.nh:]
        i, j = np.triu_indices(lay.nw, 1)
        dev = np.abs(((phi[i] - phi[j] + math.pi / 4) % (math.pi / 2)) - math.pi / 4)
        ok = (dev <= gate) & (ws[i] >= _MIN_WEIGHT) & (ws[j] >= _MIN_WEIGHT)
        if ok.any():
            pi, pj = i[ok], j[ok]
            coef = np.sqrt(lam * np.minimum(ws[pi], ws[pj])) * _ELL * 0.5

    def fun(x: np.ndarray) -> np.ndarray:
        r = np.einsum("kij,kj->ki", F, lay.unpack(x)).ravel()
        if pi is not None:
            ph = lay.phi(x)
            r = np.concatenate([r, coef * np.sin(2.0 * (ph[pi] - ph[pj]))])
        return r

    sol = least_squares(fun, th, method="trf", xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=40)
    return sol.x if np.all(np.isfinite(sol.x)) else th


def _em(X1: np.ndarray, th: np.ndarray, lay: _Layout, sigma: float, lam: float, gate: float, mask,
        *, rw=None, fix_sigma=False, max_iter=120, tol=1e-6, deadline=math.inf) -> Tuple[np.ndarray, float, int, bool]:
    """Returns (theta, sigma, iterations, converged). Stops early (converged=False) at the deadline."""
    X = X1[:, :3]
    for it in range(max_iter):
        if time.perf_counter() > deadline:
            return th, sigma, it, False
        V = lay.unpack(th)
        Wt, A = _estep(X, V, sigma, mask)
        if rw is not None:
            Wt = Wt * rw[:, None]
        sig_new = sigma
        if not fix_sigma:
            dm = A.min(1)
            inl = dm < TUKEY_C * sigma
            if inl.sum() >= 10:
                sig_new = max(1.4826 * float(np.median(dm[inl])), 1e-7, 0.4 * sigma)
        M = np.stack([X1.T @ (X1 * Wt[:, k: k + 1]) for k in range(lay.K)])
        th_new = _mstep(lay, th, M, Wt.sum(0), lam, gate)
        delta = float(np.max(np.abs(th_new - th)))
        th, ds, sigma = th_new, abs(sig_new - sigma) / sigma, sig_new
        if delta < tol and ds < 1e-3:
            return th, sigma, it + 1, True
    return th, sigma, max_iter, False


# ------------------------------------------------------------------------------------------------------ the driver
def refine_room(points, planes, cameras=None, *, up=(0.0, 1.0, 0.0), time_budget_s: float = 300.0, seed: int = 0,
                openings: Sequence[dict] | None = None, max_points: int = 150_000, patience: int = 5,
                n_bootstrap: int = 20, orth_lambda: float = 30.0, orth_gate_deg: float = 15.0,
                max_angle_std_deg: float = 1.0, max_offset_std_rel: float = 0.01,
                progress: Callable[[dict], None] | None = None) -> RoomRefinement:
    """Refine a room shell against ``points`` (N,3, any scale / frame). See the module docstring.

    ``planes``: objects with ``.normal`` / ``.d`` (e.g. ``DetectedPlane``), dicts, or ``(role, normal, d)`` tuples;
    ``n . x + d = 0``. Roles are inferred from ``up`` when absent. ``openings``: ``{"wall": input_index,
    "corners": 4x3}``. Always returns a valid result: the input planes if nothing better was found in the budget."""
    t_start = time.perf_counter()
    deadline = t_start + float(time_budget_s)
    rng = np.random.default_rng(seed)
    up = _unit(np.asarray(up, float))

    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must be (N,3)")
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) < 50:
        raise ValueError("need at least 50 finite points")

    # ---- scale-free normalisation: centre = median, scale = geometric mean of the 10-90 percentile spans
    centre = np.median(P, axis=0)
    lo, hi = np.percentile(P, 10, axis=0), np.percentile(P, 90, axis=0)
    scale = float(np.prod(np.maximum(hi - lo, 1e-300)) ** (1.0 / 3.0))
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError("degenerate point cloud")
    Pn_all = (P - centre) / scale

    # ---- parse planes, infer roles, order: horizontals (floor, ceiling) then walls
    parsed = [_coerce(p, i) for i, p in enumerate(planes)]
    roles: List[str | None] = [r for r, _, _, _ in parsed]
    h_of = [(-d / (n @ up) if abs(n @ up) > 1e-9 else math.nan) for _, n, d, _ in parsed]   # height along up
    for i, (r, n, d, _) in enumerate(parsed):
        if r in ("floor", "ceiling", "wall"):
            continue
        roles[i] = "H" if abs(n @ up) >= _HORIZ_COS else ("wall" if abs(n @ up) <= math.sin(math.radians(20)) else None)
    hz = sorted([i for i, r in enumerate(roles) if r == "H"], key=lambda i: h_of[i])
    cam_h = None
    if cameras is not None and len(cameras):
        cam_h = float(np.median(np.asarray(cameras, float) @ up))
    if len(hz) == 1:
        roles[hz[0]] = "ceiling" if (cam_h is not None and h_of[hz[0]] > cam_h) else "floor"
    elif len(hz) >= 2:
        roles[hz[0]], roles[hz[-1]] = "floor", "ceiling"
        for i in hz[1:-1]:
            roles[i] = None
    floor_i = next((i for i, r in enumerate(roles) if r == "floor"), None)
    ceil_i = next((i for i, r in enumerate(roles) if r == "ceiling"), None)
    wall_is = [i for i, r in enumerate(roles) if r == "wall"]
    horiz_is = [i for i in (floor_i, ceil_i) if i is not None]
    used = horiz_is + wall_is
    ignored = [i for i in range(len(parsed)) if i not in used]
    if not used:
        raise ValueError("no usable plane (need floor / ceiling / vertical walls)")

    # input planes in normalised coordinates (unit normals)
    Vin = np.array([[*parsed[i][1], (float(parsed[i][1] @ centre) + parsed[i][2]) / scale] for i in used])

    hn = [parsed[i][1] for i in horiz_is]
    n0 = _unit(sum((n if n @ up > 0 else -n) for n in hn)) if hn else up
    wn = [parsed[i][1] for i in wall_is]
    ref = _unit(wn[0] - (wn[0] @ n0) * n0) if wn else _unit(np.cross(n0, [1.0, 0.0, 0.0]) + 1e-9)
    lay = _Layout(len(horiz_is), len(wall_is), n0, ref)
    th0 = np.zeros(lay.size)
    sgn = []
    for k, i in enumerate(horiz_is):
        n, d = Vin[k, :3], Vin[k, 3]
        s = 1.0 if n @ n0 > 0 else -1.0
        sgn.append(s)
        th0[2 + k] = (s * d) / (s * n @ n0)
    _, f1, f2 = lay.frame(th0)
    for k in range(lay.nw):
        n, d = Vin[lay.nh + k, :3], Vin[lay.nh + k, 3]
        u = _unit(n - (n @ n0) * n0)
        th0[2 + lay.nh + k] = math.atan2(u @ f2, u @ f1)
        th0[2 + lay.nh + lay.nw + k] = d / (n @ u)

    # ---- working set
    sel = np.arange(len(Pn_all)) if len(Pn_all) <= max_points else np.sort(rng.choice(len(Pn_all), max_points, False))
    X = Pn_all[sel]
    X1 = np.concatenate([X, np.ones((len(X), 1))], axis=1)

    mask = None
    if openings:
        mask = np.zeros((len(X), lay.K), dtype=bool)
        for op in openings:
            k = lay.nh + wall_is.index(int(op["wall"]))
            C = (np.asarray(op["corners"], float) - centre) / scale
            n = Vin[k, :3]
            t0 = _unit(np.cross(n0, n))
            u, v = X @ t0, X @ n0
            cu, cv = C @ t0, C @ n0
            mask[:, k] |= (u >= cu.min()) & (u <= cu.max()) & (v >= cv.min()) & (v <= cv.max())

    gate = math.radians(orth_gate_deg)
    d0 = np.abs(X @ Vin[:, :3].T + Vin[:, 3]).min(1)
    sigma0 = float(np.clip(1.4826 * np.median(d0), 1e-4, 0.1))

    # ---- phase 1: primary fit with the prior off; commit only if it beats the input on the fixed score
    it_total = 0
    th1, sig1, n_it, conv1 = _em(X1, th0, lay, sigma0, 0.0, gate, mask, deadline=deadline)
    it_total += n_it
    tau = 3.0 * sig1
    best_th, best_score = None, _score(X, Vin, tau)
    s1 = _score(X, lay.unpack(th1), tau)
    if s1 < best_score:
        best_th, best_score = th1, s1
    stop = "converged" if conv1 else "time_budget"
    if progress:
        progress({"phase": "primary", "iterations": it_total, "score": best_score, "sigma": sig1})

    # ---- phase 2: held-out choice of the orthogonality prior (on vs off), 3 random 75/25 splits
    lam, ho = 0.0, {"off": math.nan, "on": math.nan}
    if best_th is not None and lay.nw >= 2 and time.perf_counter() < deadline:
        tot = {0.0: 0.0, orth_lambda: 0.0}
        done = 0
        for _ in range(3):
            perm = rng.permutation(len(X))
            cut = int(0.75 * len(X))
            tr, va = perm[:cut], perm[cut:]
            m_tr = mask[tr] if mask is not None else None
            for lm in tot:
                if time.perf_counter() > deadline:
                    break
                th_s, _, _, _ = _em(X1[tr], th0, lay, sigma0, lm, gate, m_tr, deadline=deadline, max_iter=60)
                tot[lm] += _score(X[va], lay.unpack(th_s), tau)
            else:
                done += 1
                continue
            break
        if done == 3:
            ho = {"off": tot[0.0] / 3, "on": tot[orth_lambda] / 3}
            lam = orth_lambda if ho["on"] <= ho["off"] * 1.005 else 0.0
    if lam > 0 and time.perf_counter() < deadline:
        th2, sig2, n_it, conv2 = _em(X1, best_th, lay, sig1, lam, gate, mask, deadline=deadline)
        it_total += n_it
        s2 = _score(X, lay.unpack(th2), tau)
        if s2 < best_score:
            best_th, best_score, sig1 = th2, s2, sig2

    # ---- phase 3: Poisson-bootstrap restarts from jittered best; accepted only if the fixed score improves
    restarts, fails = 0, 0
    if best_th is not None:
        while fails < patience and time.perf_counter() < deadline:
            jit = np.zeros(lay.size)
            jit[:2] = rng.normal(0, 0.02, 2)
            jit[2: 2 + lay.nh] = rng.normal(0, 0.02, lay.nh)
            jit[2 + lay.nh: 2 + lay.nh + lay.nw] = rng.normal(0, 0.03, lay.nw)
            jit[2 + lay.nh + lay.nw:] = rng.normal(0, 0.02, lay.nw)
            rw = rng.poisson(1.0, len(X)).astype(float)
            th_r, sg_r, n_it, _ = _em(X1, best_th + jit, lay, 2.0 * sig1, lam, gate, mask, rw=rw, deadline=deadline)
            it_total += n_it
            restarts += 1
            s_r = _score(X, lay.unpack(th_r), tau)
            if s_r < best_score - 1e-9:
                best_th, best_score, sig1 = th_r, s_r, sg_r
                fails = 0
            else:
                fails += 1
            if progress:
                progress({"phase": "restart", "restarts": restarts, "score": best_score})
        if fails >= patience and conv1:
            stop = "converged"
        elif time.perf_counter() > deadline:
            stop = "time_budget"

    # ---- phase 4: bootstrap ensemble -> per-plane uncertainty
    ens: List[np.ndarray] = []
    if best_th is not None:
        for _ in range(n_bootstrap):
            if time.perf_counter() > deadline:
                break
            rw = rng.poisson(1.0, len(X)).astype(float)
            th_b, _, n_it, _ = _em(X1, best_th, lay, sig1, lam, gate, mask, rw=rw, fix_sigma=True, max_iter=8,
                                   tol=1e-5, deadline=deadline)
            it_total += n_it
            ens.append(th_b)

    V_best = lay.unpack(best_th) if best_th is not None else Vin.copy()
    # keep the input orientation convention of horizontal normals
    V_out_n = V_best.copy()
    for k, s in enumerate(sgn):
        if s < 0 and best_th is not None:
            V_out_n[k] = -V_best[k]
    angle_std = np.full(lay.K, math.nan)
    off_std = np.full(lay.K, math.nan)
    if len(ens) >= 4:
        Vs = np.stack([lay.unpack(t) for t in ens])                                  # (B,K,4)
        for k in range(lay.K):
            m = _unit(Vs[:, k, :3].mean(0))
            ang = np.arccos(np.clip(Vs[:, k, :3] @ m, -1.0, 1.0))
            angle_std[k] = math.degrees(math.sqrt(float(np.mean(ang ** 2))))
            off_std[k] = float(np.std(Vs[:, k, 3]))

    # ---- final statistics on ALL points (chunked), against the (post-rejection) output planes
    sig_final = sig1
    n_in_pl = len(used)
    status = ["refined" if best_th is not None else "unsupported"] * n_in_pl
    V_final = V_out_n.copy()
    if best_th is not None:
        sup = _stats(Pn_all, V_best, 3.0 * sig_final)[0]
        for k in range(n_in_pl):
            if sup[k] < max(25, 0.0005 * len(Pn_all)):
                status[k] = "unsupported"
            elif not math.isnan(angle_std[k]) and (angle_std[k] > max_angle_std_deg or off_std[k] > max_offset_std_rel):
                status[k] = "rejected_unstable"
            if status[k] != "refined":
                V_final[k] = Vin[k]
    cnt, rms_pl, dm_final = _stats(Pn_all, V_final, 3.0 * sig_final)
    inl = dm_final <= 3.0 * sig_final
    dm_before = np.abs(Pn_all @ Vin[:, :3].T + Vin[:, 3]).min(1)
    rms_after = float(np.sqrt(np.mean(dm_final[inl] ** 2))) if inl.any() else 0.0
    rms_before = float(np.sqrt(np.mean(dm_before[inl] ** 2))) if inl.any() else 0.0

    def to_world(V):
        n = V[:3]
        return (float(n[0]), float(n[1]), float(n[2])), float(V[3] * scale - n @ centre)

    out_planes: List[RefinedPlane] = []
    names = [roles[i] for i in used]
    for k, i in enumerate(used):
        n, dw = to_world(V_final[k])
        out_planes.append(RefinedPlane(
            role=names[k], normal=n, d=dw, inlier_count=int(cnt[k]), inlier_rms=float(rms_pl[k] * scale),
            angle_std_deg=float(angle_std[k]), offset_std_rel=float(off_std[k]), status=status[k], source_index=i,
            plane_id=parsed[i][3]))

    before_t = [(names[k], tuple(parsed[i][1]), parsed[i][2]) for k, i in enumerate(used)]
    after_t = [(p.role, p.normal, p.d) for p in out_planes]
    dims_before = room_dimensions(before_t, centre)
    dims_after = room_dimensions(after_t, centre)

    out_open: List[dict] = []
    for op in openings or ():
        k = wall_is.index(int(op["wall"]))
        p = out_planes[lay.nh + k]
        n = np.asarray(p.normal)
        C = np.asarray(op["corners"], float)
        Cp = C - ((C @ n + p.d)[:, None]) * n
        out_open.append({"wall": int(op["wall"]), "corners": Cp.tolist(),
                         "mean_shift": float(np.mean(np.abs(C @ n + p.d)))})

    diag: Dict[str, object] = {"points_used": int(len(X)), "points_total": int(len(Pn_all)),
                               "sigma_rel": float(sig_final), "sigma": float(sig_final * scale),
                               "bootstrap_members": len(ens), "initial_sigma_rel": sigma0}
    if cameras is not None and len(cameras):
        cams = np.asarray(cameras, float)
        hs = [p for p in out_planes if p.role in ("floor", "ceiling")]
        if len(hs) == 2:
            hh = sorted(-p.d / (np.asarray(p.normal) @ up) for p in hs)
            ch = cams @ up
            diag["cameras_between_floor_and_ceiling"] = bool(np.all((ch > hh[0]) & (ch < hh[1])))

    return RoomRefinement(
        planes=out_planes, ignored_planes=ignored, openings=out_open, rms_before=rms_before * scale,
        rms_after=rms_after * scale, rms_before_rel=rms_before, rms_after_rel=rms_after, dims_before=dims_before,
        dims_after=dims_after, iterations=it_total, restarts=restarts, seconds=time.perf_counter() - t_start,
        converged=stop == "converged", stop_reason=stop,
        orthogonality={"selected": float(lam > 0), "lambda": lam, "heldout_off": ho["off"], "heldout_on": ho["on"]},
        scale=scale, diagnostics=diag)


def _stats(Pn: np.ndarray, V: np.ndarray, tau: float, chunk: int = 200_000):
    """(inlier count per plane, inlier RMS per plane, nearest-plane distance) of all points; nearest-plane assignment."""
    K = len(V)
    cnt, ss = np.zeros(K), np.zeros(K)
    dmin = np.empty(len(Pn))
    for a in range(0, len(Pn), chunk):
        A = np.abs(Pn[a: a + chunk] @ V[:, :3].T + V[:, 3])
        arg, dm = A.argmin(1), A.min(1)
        dmin[a: a + chunk] = dm
        ok = dm <= tau
        np.add.at(cnt, arg[ok], 1.0)
        np.add.at(ss, arg[ok], dm[ok] ** 2)
    return cnt, np.sqrt(ss / np.maximum(cnt, 1.0)), dmin
