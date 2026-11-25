"""Exact sparse Levenberg-Marquardt with iteratively reweighted robust losses.

Why this exists (diagnosis, not a speed trick): SciPy's `least_squares(..., tr_solver="lsmr")` solves each trust-region step
INEXACTLY with LSMR. On the multi-view problem (pose angles in radians, translations and wall offsets in metres, a 1e-4 gauge
prior) the system is badly scaled, and in a reproduced 96-view case (irregular room, seed 101) the first round stalled:
optimality 39.5 -> 26.6 after 400 evaluations at cost ~317, while the dense solver reached cost 293.5 in 38 iterations.
Raising `max_nfev` did not help, so the cause is the inexact linear solves, not the iteration limit.

This solver forms the sparse normal equations (J^T W J + lambda diag(J^T W J)) and solves them EXACTLY with sparse LU
(`scipy.sparse.linalg.splu`): the same linear algebra as the dense path, so the same iterates up to rounding, with Marquardt
diagonal scaling absorbing the unit mismatch. W holds the IRLS weights rho'(z) of the robust loss (Cauchy / Huber / soft-L1).

Termination is reported, never hidden: status 1 gradient, 2 cost change, 3 step size, 0 evaluation limit; `success` is False
only for 0. The returned Jacobian is sqrt(W) J, the matrix whose J^T J approximates the robust Hessian (used for covariance).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu


def _rho_and_weight(r: np.ndarray, loss: str, f: float):
    z = (r / f) ** 2
    if loss == "linear":
        return z, np.ones_like(z)
    if loss == "cauchy":
        return np.log1p(z), 1.0 / (1.0 + z)
    if loss == "soft_l1":
        s = np.sqrt(1.0 + z)
        return 2.0 * (s - 1.0), 1.0 / s
    if loss == "huber":
        small = z <= 1.0
        sz = np.sqrt(np.where(small, 1.0, z))
        return np.where(small, z, 2.0 * sz - 1.0), np.where(small, 1.0, 1.0 / sz)
    raise ValueError(f"unknown loss {loss!r}")


@dataclass
class LMResult:
    x: np.ndarray
    cost: float
    status: int
    message: str
    success: bool
    nfev: int
    njev: int
    optimality: float
    jac: object            # sqrt(W) J, scipy sparse
    lam: float


def solve_sparse_lm(resid, jac, x0, *, loss: str = "cauchy", f_scale: float = 1.0, max_nfev: int = 100,
                    gtol: float = 1e-8, ftol: float = 1e-8, xtol: float = 1e-8) -> LMResult:
    def robust_cost(r):
        rho, _ = _rho_and_weight(r, loss, f_scale)
        return 0.5 * f_scale ** 2 * float(rho.sum())

    x = np.asarray(x0, float).copy()
    r = resid(x)
    nfev, njev = 1, 0
    cost = robust_cost(r)
    lam = 1e-3
    nu = 2.0
    J = jac(x)
    njev += 1
    status, msg = 0, "maximum number of function evaluations reached"
    opt = float("inf")
    while nfev < max_nfev:
        _, w = _rho_and_weight(r, loss, f_scale)
        sw = np.sqrt(w)
        Jw = diags(sw) @ J
        g = Jw.T @ (sw * r)
        opt = float(np.abs(g).max()) if g.size else 0.0
        if opt <= gtol:
            status, msg = 1, "gradient tolerance satisfied"
            break
        A = (Jw.T @ Jw).tocsc()
        d = A.diagonal()
        d = np.where(d > 0, d, 1.0)
        improved = False
        while nfev < max_nfev:
            H = A + lam * diags(d, format="csc")
            try:
                step = -splu(csc_matrix(H)).solve(g)
            except RuntimeError:           # singular at this damping: damp harder (recorded through the iteration count)
                lam *= 10.0
                nfev += 1
                continue
            xn = x + step
            rn = resid(xn)
            nfev += 1
            cn = robust_cost(rn)
            if np.isfinite(cn) and cn < cost:
                dc = cost - cn
                # gain ratio (actual / predicted reduction of the quadratic model): Nielsen damping update
                pred = 0.5 * float(step @ (lam * d * step - g)) + 1e-300
                rho_gain = dc / pred
                lam = lam * max(1.0 / 3.0, 1.0 - (2.0 * rho_gain - 1.0) ** 3)
                lam = max(lam, 1e-12)
                nu = 2.0
                x, r, cost = xn, rn, cn
                improved = True
                if np.linalg.norm(step) <= xtol * (np.linalg.norm(x) + xtol):
                    status, msg = 3, "step size tolerance satisfied"
                elif dc <= ftol * cost:
                    status, msg = 2, "cost change tolerance satisfied"
                break
            lam *= nu
            nu *= 2.0
            if lam > 1e16:
                status, msg = 3, "damping saturated: no descent step"
                break
        if status in (1, 2, 3) and (improved or lam > 1e16):
            break
        if not improved and nfev >= max_nfev:
            break
        J = jac(x)
        njev += 1
    _, w = _rho_and_weight(r, loss, f_scale)
    Jw = diags(np.sqrt(w)) @ J
    return LMResult(x, cost, status, msg, status != 0, nfev, njev, opt, Jw.tocsr(), lam)
