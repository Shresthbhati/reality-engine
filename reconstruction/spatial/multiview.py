"""Joint multi-view optimization of camera poses and wall lines (plan view).

Problem: observations arrive in each IMAGE's frame; the poses that place images in the world (from SfM/COLMAP,
owned by the perception side) carry error, and that error is systematic per image -- it shifts every wall that
image saw by the same amount, so views disagree with each other and the association layer sees 'conflicts' that are
really pose error. This module consumes those poses (as priors with their sigmas) and the observations, and solves

    min over {pose corrections, wall lines}  sum_obs rho( (n_w . (R(theta_i) p + t_i) - d_w) / sigma_obs )
                                           + sum_img  || prior residual of the pose correction ||^2

with a robust loss rho (Cauchy by default). Only the dimension an observation actually measures is used: the
residual is the PERPENDICULAR distance of an end point to its wall line. Along-wall position is not measured by a
wall observation, so it is not optimised from it; where the data cannot constrain a pose parameter (e.g. all walls
parallel) the pose prior dominates and the posterior sigma stays at the prior's, which `observed_dims` reports.

Gauge: the pose priors fix the global frame. If NO image has a prior sigma, image 0 is held (a declared gauge fix).
Posterior covariance of every pose is (J^T J)^-1 at the solution (first order; robust-loss weights ignored), and the
corrected observations carry sigma = sqrt(sigma_obs^2 + posterior pose variance at the observation) -- uncertainty
propagated through the transform, never reset. Raw (image-frame) measurements and prior poses are preserved in
`MultiViewResult.raw`; corrected copies keep the SAME evidence ids.

Not done here (owned elsewhere / out of scope): feature matching, SfM, 3D bundle adjustment over image points, scale
estimation (poses are metric), tilted cameras (plan-view poses only).
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import splu

from .association import AssociationConfig, associate_structure
from .model import Kind, StructuralObservation
from .sparse_lm import _rho_and_weight, solve_sparse_lm


@dataclass(frozen=True)
class ImagePose:
    """Prior pose of an image frame in the world (plan view): p_world = R(theta) p_img + (tx, ty)."""
    image_id: str
    theta: float
    tx: float
    ty: float
    sigma_theta: float | None = None   # rad; None = the producer gave no uncertainty
    sigma_t: float | None = None       # m


@dataclass(frozen=True)
class MultiViewConfig:
    loss: str = "cauchy"                  # 'cauchy' | 'huber' | 'soft_l1' | 'linear'
    f_scale: float = 1.0
    n_rounds: int = 2                     # associate -> optimise -> re-associate
    min_views: int = 2
    outlier_k: float = 3.5
    max_nfev: int = 100
    observed_ratio: float = 0.7           # posterior/prior sigma below this => the data constrains that parameter
    solver: str = "auto"                  # 'auto' | 'dense' | 'sparse'; auto = dense up to `auto_sparse_above` images, sparse above
    auto_sparse_above: int = 16           # measured crossover: dense is faster below ~20 images, sparse wins on time and memory above
    association: AssociationConfig = field(default_factory=AssociationConfig)


@dataclass
class PoseResult:
    image_id: str
    theta: float
    tx: float
    ty: float
    cov: np.ndarray | None            # 3x3 over (theta, tx, ty); None when the image took no part
    prior: ImagePose
    observed_dims: dict[str, bool | None]   # theta / tx / ty: True = data-constrained, False = prior-dominated, None = unknown

    def sigma(self) -> tuple[float, float] | None:
        if self.cov is None:
            return None
        return math.sqrt(max(self.cov[0, 0], 0.0)), math.sqrt(max((self.cov[1, 1] + self.cov[2, 2]) / 2.0, 0.0))


@dataclass
class MultiViewResult:
    poses: dict[str, PoseResult]
    observations: list[StructuralObservation]          # corrected, world frame, same evidence ids
    raw: dict[str, dict]                               # evidence id -> original image-frame measurement + prior pose
    residual_z: dict[str, float]                       # max |normalised perpendicular residual| per wall observation
    outliers: list[str]
    n_clusters: int
    cost_before: float
    cost_after: float
    nfev: int
    rounds: int
    seconds: float
    converged: bool
    solver: str = "dense"
    jac_seconds: float = 0.0
    n_params: int = 0
    n_residuals: int = 0
    jac_nnz: int = 0
    assoc_cache: dict = field(default_factory=dict)   # summed association fit-cache hits/misses over the optimisation rounds
    diagnostics: list = field(default_factory=list)   # one dict per round: solver, termination status/message, nfev/njev, optimality, cost

    def to_dict(self) -> dict:
        return {"poses": {k: {"theta": p.theta, "tx": p.tx, "ty": p.ty, "cov": None if p.cov is None else p.cov.tolist(),
                              "prior": {"theta": p.prior.theta, "tx": p.prior.tx, "ty": p.prior.ty,
                                        "sigma_theta": p.prior.sigma_theta, "sigma_t": p.prior.sigma_t},
                              "observed_dims": p.observed_dims} for k, p in sorted(self.poses.items())},
                "raw": self.raw, "outliers": self.outliers, "n_clusters": self.n_clusters,
                "cost_before": self.cost_before, "cost_after": self.cost_after, "nfev": self.nfev,
                "rounds": self.rounds, "converged": self.converged}


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return c, s


def apply_pose(o: StructuralObservation, theta: float, tx: float, ty: float, sigma: float | None = None) -> StructuralObservation:
    c, s = math.cos(theta), math.sin(theta)

    def f(p):
        return (c * p[0] - s * p[1] + tx, s * p[0] + c * p[1] + ty)

    return replace(o, p0=f(o.p0), p1=f(o.p1), frame_id="world", sigma_m=sigma if sigma is not None else o.sigma_m)


def _inflated_sigma(o: StructuralObservation, pose_sigma: tuple[float, float] | None) -> float:
    """First-order sigma of a world-frame point from the measurement plus the pose: sqrt(s^2 + s_t^2 + (s_theta * r)^2),
    r = distance of the point from the image origin (the lever arm of a rotation error). Unknown pose sigma adds nothing
    here (it is unknown, not zero) -- the caller then knows the sigma is measurement-only."""
    if pose_sigma is None:
        return o.sigma_m
    st, sl = pose_sigma
    r = max(math.hypot(*o.p0), math.hypot(*o.p1))
    return math.sqrt(o.sigma_m ** 2 + sl ** 2 + (st * r) ** 2)


def optimize_multiview(
    observations: Sequence[StructuralObservation], poses: Mapping[str, ImagePose],
    cfg: MultiViewConfig = MultiViewConfig(), *, warm_start: MultiViewResult | None = None,
) -> MultiViewResult:
    t0 = time.perf_counter()
    for o in observations:
        if o.frame_id not in poses:
            raise ValueError(f"observation {o.evidence_id!r} is in frame {o.frame_id!r} with no pose: refusing to guess")
        if o.frame_id != o.image_id:
            raise ValueError(f"observation {o.evidence_id!r}: frame_id must be its image id for multi-view optimization")
    img_ids = sorted(poses)
    idx = {k: i for i, k in enumerate(img_ids)}
    prior = {k: poses[k] for k in img_ids}
    cur = {k: (warm_start.poses[k].theta, warm_start.poses[k].tx, warm_start.poses[k].ty)
           if warm_start and k in warm_start.poses else (prior[k].theta, prior[k].tx, prior[k].ty) for k in img_ids}
    raw = {o.evidence_id: {"image_id": o.image_id, "p0": list(o.p0), "p1": list(o.p1), "sigma_m": o.sigma_m,
                           "prior_pose": [prior[o.image_id].theta, prior[o.image_id].tx, prior[o.image_id].ty]}
           for o in observations}
    walls = [o for o in observations if o.kind is Kind.WALL]
    post_sigma: dict[str, tuple[float, float] | None] = {
        k: ((prior[k].sigma_theta, prior[k].sigma_t) if prior[k].sigma_theta is not None and prior[k].sigma_t is not None else None)
        for k in img_ids}
    cov: dict[str, np.ndarray | None] = {k: None for k in img_ids}
    cost0 = cost1 = 0.0
    nfev_total, rounds_done, converged, in_clusters = 0, 0, True, set()
    jac_seconds, shape_info, solver_used = 0.0, (0, 0, 0), "dense"
    diag: list = []
    cache_sum: dict = {}
    nI = len(img_ids)
    n_clusters, z_obs = 0, {}
    for rnd in range(cfg.n_rounds):
        world = [apply_pose(o, *cur[o.image_id], sigma=_inflated_sigma(o, post_sigma[o.image_id])) for o in walls]
        assoc = associate_structure(world, cfg.association)
        for k_ in ("hits", "misses"):
            cache_sum[k_] = cache_sum.get(k_, 0) + assoc.cache_stats.get(k_, 0)
        clusters = [e for e in assoc.entities if e.kind is Kind.WALL and len(e.image_ids) >= cfg.min_views]
        if not clusters:
            break
        by_ev = {o.evidence_id: o for o in walls}
        rows = []          # (image idx, local x, local y, cluster idx, sigma)
        for ci, e in enumerate(clusters):
            for ev in e.evidence_ids:
                o = by_ev[ev]
                for p in (o.p0, o.p1):
                    rows.append((idx[o.image_id], p[0], p[1], ci, o.sigma_m))
        R = np.array(rows, float)
        im = R[:, 0].astype(int)
        cl = R[:, 3].astype(int)
        X, Y, SG = R[:, 1], R[:, 2], R[:, 4]
        used_imgs = sorted(set(im.tolist()))
        n_clusters = len(clusters)
        a0, r0 = [], []
        for e in clusters:
            dx, dy = e.b[0] - e.a[0], e.b[1] - e.a[1]
            L = math.hypot(dx, dy)
            nx, ny = -dy / L, dx / L
            a0.append(math.atan2(ny, nx))
            r0.append(nx * e.a[0] + ny * e.a[1])
        nI = len(img_ids)
        th0 = np.array([cur[k][0] for k in img_ids])
        tx0 = np.array([cur[k][1] for k in img_ids])
        ty0 = np.array([cur[k][2] for k in img_ids])
        pth = np.array([prior[k].theta for k in img_ids])
        ptx = np.array([prior[k].tx for k in img_ids])
        pty = np.array([prior[k].ty for k in img_ids])
        s_th = np.array([prior[k].sigma_theta if prior[k].sigma_theta else np.nan for k in img_ids])
        s_t = np.array([prior[k].sigma_t if prior[k].sigma_t else np.nan for k in img_ids])
        has_prior = ~np.isnan(s_th) & ~np.isnan(s_t)
        gauge = None if has_prior.any() else used_imgs[0]
        free = [i for i in used_imgs]
        fpos = {i: k for k, i in enumerate(free)}
        nP = 3 * len(free)

        def unpack(x):
            d = x[:nP].reshape(-1, 3)
            th, tx, ty = th0.copy(), tx0.copy(), ty0.copy()
            for i, k in fpos.items():
                th[i] = th0[i] + d[k, 0]
                tx[i] = tx0[i] + d[k, 1]
                ty[i] = ty0[i] + d[k, 2]
            return th, tx, ty, x[nP:nP + n_clusters], x[nP + n_clusters:]

        def resid(x):
            th, tx, ty, al, rho = unpack(x)
            c, s = np.cos(th[im]), np.sin(th[im])
            wx = c * X - s * Y + tx[im]
            wy = s * X + c * Y + ty[im]
            r = (np.cos(al[cl]) * wx + np.sin(al[cl]) * wy - rho[cl]) / SG
            pr = []
            for i in free:
                if has_prior[i]:
                    pr += [(th[i] - pth[i]) / s_th[i], (tx[i] - ptx[i]) / s_t[i], (ty[i] - pty[i]) / s_t[i]]
                elif gauge == i:
                    pr += [(th[i] - pth[i]) / 1e-4, (tx[i] - ptx[i]) / 1e-4, (ty[i] - pty[i]) / 1e-4]
            return np.concatenate([r, np.array(pr, float)])

        x0 = np.concatenate([np.zeros(nP), np.array(a0), np.array(r0)])
        f0 = resid(x0)
        cost0 = cost0 if rnd else float(0.5 * f0 @ f0)
        m = len(X)
        prior_rows = [i for i in free if has_prior[i] or gauge == i]
        prior_sig = {i: ((s_th[i], s_t[i], s_t[i]) if has_prior[i] else (1e-4, 1e-4, 1e-4)) for i in prior_rows}

        def jac(x):
            """Analytic Jacobian of `resid` (dense: the parameter count is ~3 per image + 2 per wall)."""
            th, tx, ty, al, rho = unpack(x)
            c, s_ = np.cos(th[im]), np.sin(th[im])
            wx, wy = c * X - s_ * Y + tx[im], s_ * X + c * Y + ty[im]
            ca, sa = np.cos(al[cl]), np.sin(al[cl])
            Jm = np.zeros((m + 3 * len(prior_rows), len(x)))
            rows = np.arange(m)
            k = np.array([fpos[i] for i in im])
            Jm[rows, 3 * k] = (ca * (-s_ * X - c * Y) + sa * (c * X - s_ * Y)) / SG
            Jm[rows, 3 * k + 1] = ca / SG
            Jm[rows, 3 * k + 2] = sa / SG
            Jm[rows, nP + cl] = (-sa * wx + ca * wy) / SG
            Jm[rows, nP + n_clusters + cl] = -1.0 / SG
            for r_i, i in enumerate(prior_rows):
                for j in range(3):
                    Jm[m + 3 * r_i + j, 3 * fpos[i] + j] = 1.0 / prior_sig[i][j]
            return Jm

        # Sparsity structure: row r (an observation end point) touches its image's 3 pose parameters and its wall's 2 line
        # parameters; a prior row touches one pose parameter. The (row, col) index arrays are built ONCE per round and reused
        # on every iteration; only the values are recomputed.
        k_of = np.array([fpos[i] for i in im])
        obs_rows = np.repeat(np.arange(m), 5)
        obs_cols = np.stack([3 * k_of, 3 * k_of + 1, 3 * k_of + 2, nP + cl, nP + n_clusters + cl], axis=1).ravel()
        pr_rows = m + np.arange(3 * len(prior_rows))
        pr_cols = np.array([3 * fpos[i] + j for i in prior_rows for j in range(3)], dtype=int)
        pr_inv = np.array([1.0 / prior_sig[i][j] for i in prior_rows for j in range(3)])
        all_rows, all_cols = np.concatenate([obs_rows, pr_rows]), np.concatenate([obs_cols, pr_cols])
        shape = (m + 3 * len(prior_rows), len(x0))

        def jac_sparse(x):
            th, tx, ty, al, rho = unpack(x)
            c, s_ = np.cos(th[im]), np.sin(th[im])
            wx, wy = c * X - s_ * Y + tx[im], s_ * X + c * Y + ty[im]
            ca, sa = np.cos(al[cl]), np.sin(al[cl])
            vals = np.stack([(ca * (-s_ * X - c * Y) + sa * (c * X - s_ * Y)) / SG, ca / SG, sa / SG,
                             (-sa * wx + ca * wy) / SG, -1.0 / SG], axis=1).ravel()
            return csr_matrix((np.concatenate([vals, pr_inv]), (all_rows, all_cols)), shape=shape)

        use_sparse = cfg.solver == "sparse" or (cfg.solver == "auto" and nI > cfg.auto_sparse_above)
        solver_used = "sparse" if use_sparse else "dense"
        jt = [0.0]
        jfun = jac_sparse if use_sparse else jac

        def timed_jac(x):
            t_ = time.perf_counter()
            out = jfun(x)
            jt[0] += time.perf_counter() - t_
            return out

        if use_sparse:   # exact sparse-LU Levenberg-Marquardt (see sparse_lm.py for why LSMR trust-region steps were replaced)
            # LM counts every TRIAL step (accepted or rejected) as an evaluation, TRF only accepted iterations, so the budget is scaled
            # 3x; measured on development seeds: converged runs need 12-47 evaluations, the cap only binds in flat 2-4 view valleys
            sol = solve_sparse_lm(resid, timed_jac, x0, loss=cfg.loss, f_scale=cfg.f_scale, max_nfev=3 * cfg.max_nfev)
        else:
            sol = least_squares(resid, x0, jac=timed_jac, loss=cfg.loss, f_scale=cfg.f_scale, method="trf",
                                max_nfev=cfg.max_nfev, x_scale="jac")
        jac_seconds += jt[0]
        diag.append({"round": rnd, "solver": solver_used, "status": int(sol.status), "message": str(sol.message), "success": bool(sol.success),
                     "nfev": int(sol.nfev), "njev": int(sol.njev if sol.njev is not None else -1), "optimality": float(sol.optimality),
                     "cost": float(sol.cost), "n_params": int(len(x0)), "n_residuals": int(shape[0])})
        shape_info = (len(x0), shape[0], int(len(all_rows)))
        nfev_total += int(sol.nfev)
        converged = converged and bool(sol.success)
        th, tx, ty, al, rho = unpack(sol.x)
        for i in used_imgs:
            cur[img_ids[i]] = (float(th[i]), float(tx[i]), float(ty[i]))
        # ONE covariance definition for both solvers: (J^T W J)^-1 with W the IRLS weights rho'(z) of the robust loss at the
        # solution. (SciPy's `sol.jac` for robust losses carries second-order terms the sparse LM does not, so using it would make
        # the two solvers report different covariances for the same solution.) First-order; ignores the weight derivative.
        _, wts = _rho_and_weight(resid(sol.x), cfg.loss, cfg.f_scale)
        Jfin = jac_sparse(sol.x) if use_sparse else csr_matrix(jac(sol.x))
        Jw = diags(np.sqrt(wts)) @ Jfin
        if use_sparse:
            H = (Jw.T @ Jw).tocsc() + 1e-12 * identity(Jw.shape[1], format="csc")
            E = np.zeros((Jw.shape[1], nP))
            E[np.arange(nP), np.arange(nP)] = 1.0
            C = splu(H).solve(E)
        else:
            Jd = Jw.toarray()
            C = np.linalg.pinv(Jd.T @ Jd)
        for i, k in fpos.items():
            cov[img_ids[i]] = C[3 * k:3 * k + 3, 3 * k:3 * k + 3].copy()
            post_sigma[img_ids[i]] = (math.sqrt(max(cov[img_ids[i]][0, 0], 0.0)),
                                      math.sqrt(max((cov[img_ids[i]][1, 1] + cov[img_ids[i]][2, 2]) / 2.0, 0.0)))
        c, s = np.cos(th[im]), np.sin(th[im])
        wx, wy = c * X - s * Y + tx[im], s * X + c * Y + ty[im]
        z = np.abs((np.cos(al[cl]) * wx + np.sin(al[cl]) * wy - rho[cl]) / SG)
        z_obs = {}
        for ci, e in enumerate(clusters):
            for ev in e.evidence_ids:
                z_obs[ev] = 0.0
        k = 0
        for ci, e in enumerate(clusters):
            for ev in e.evidence_ids:
                z_obs[ev] = float(max(z[k], z[k + 1]))
                k += 2
        in_clusters |= {img_ids[i] for i in used_imgs}
        cost1 = float(sol.cost)
        rounds_done = rnd + 1

    out_obs = [apply_pose(o, *cur[o.image_id], sigma=_inflated_sigma(o, post_sigma[o.image_id])) for o in observations]
    pres: dict[str, PoseResult] = {}
    for k in img_ids:
        pr = prior[k]
        obs_dims: dict[str, bool | None] = {"theta": None, "tx": None, "ty": None}
        if cov[k] is not None and pr.sigma_theta and pr.sigma_t:
            ratio = (math.sqrt(max(cov[k][0, 0], 0)) / pr.sigma_theta, math.sqrt(max(cov[k][1, 1], 0)) / pr.sigma_t,
                     math.sqrt(max(cov[k][2, 2], 0)) / pr.sigma_t)
            obs_dims = {n: bool(v < cfg.observed_ratio) for n, v in zip(("theta", "tx", "ty"), ratio)}
        pres[k] = PoseResult(k, *cur[k], cov[k], pr, obs_dims)
    outliers = sorted(ev for ev, z in z_obs.items() if z > cfg.outlier_k)
    return MultiViewResult(pres, out_obs, raw, z_obs, outliers, n_clusters, cost0, cost1, nfev_total, rounds_done,
                           time.perf_counter() - t0, converged, solver_used, jac_seconds, *shape_info, cache_sum, diag)


def attach_optimization(world, result: MultiViewResult) -> None:
    """Store the raw measurements, prior poses, refined poses and covariances with the world (never dropped)."""
    world.metadata["spatial_optimization"] = result.to_dict()
