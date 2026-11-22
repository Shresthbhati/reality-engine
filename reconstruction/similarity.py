"""Similarity (scale + rotation + translation) between two point sets."""

from __future__ import annotations

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity transform ``dst ~ s * R @ src + t`` minimising squared error (Umeyama 1991). Needs >= 3 non-collinear points;
    ``R`` is always a proper rotation (a mirror image is never fitted by a reflection)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError("need matching Nx3 arrays with N >= 3")
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var_s = (xs ** 2).sum() / len(src)
    if var_s < 1e-12:
        raise ValueError("source points are degenerate")
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / var_s)
    return s, R, mu_d - s * R @ mu_s
