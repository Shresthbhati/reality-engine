"""Statistical outlier removal for sparse / dense point clouds: Open3D when installed, an equivalent SciPy implementation otherwise.

A point is an outlier when its mean distance to its ``nb_neighbors`` nearest neighbours exceeds the cloud-wide mean of that
distance by more than ``std_ratio`` standard deviations (the definition both Open3D and PCL use). The backend that did the work is
returned in the facts, so a missing library is a recorded fallback, never a silent change of behaviour.

    keep, facts = remove_statistical_outliers(xyz)       # keep: bool mask, facts: {backend, n_in, n_kept, ...}
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

DEFAULT_NEIGHBORS = 20
DEFAULT_STD_RATIO = 2.0


def _scipy_mask(xyz: np.ndarray, nb_neighbors: int, std_ratio: float) -> np.ndarray:
    from scipy.spatial import cKDTree

    k = min(nb_neighbors + 1, len(xyz))                      # +1: the query point is its own nearest neighbour
    dist, _ = cKDTree(xyz).query(xyz, k=k)
    mean_d = dist[:, 1:].mean(axis=1)
    threshold = mean_d.mean() + std_ratio * mean_d.std()
    return mean_d <= threshold


def remove_statistical_outliers(points, nb_neighbors: int = DEFAULT_NEIGHBORS, std_ratio: float = DEFAULT_STD_RATIO,
                                prefer: str = "open3d") -> Tuple[np.ndarray, dict]:
    """``(keep_mask, facts)`` for an ``(N, 3)`` array-like. Fewer than ``nb_neighbors + 2`` points are returned unchanged
    (a neighbourhood statistic over a handful of points is noise), and the facts say so."""
    xyz = np.asarray(points, dtype=float).reshape(-1, 3)
    facts = {"n_in": int(len(xyz)), "nb_neighbors": nb_neighbors, "std_ratio": std_ratio}
    finite = np.isfinite(xyz).all(axis=1)
    if int(finite.sum()) < nb_neighbors + 2:
        return finite, {**facts, "backend": "none", "n_kept": int(finite.sum()),
                        "note": "too few finite points for a neighbourhood statistic; only non-finite points removed"}
    keep = np.zeros(len(xyz), dtype=bool)
    backend = "scipy"
    clean = xyz[finite]
    if prefer == "open3d":
        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(clean))
            _, idx = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            sub = np.zeros(len(clean), dtype=bool)
            sub[np.asarray(idx, dtype=int)] = True
            backend = "open3d"
        except ImportError:
            sub = _scipy_mask(clean, nb_neighbors, std_ratio)
    else:
        sub = _scipy_mask(clean, nb_neighbors, std_ratio)
    keep[np.nonzero(finite)[0][sub]] = True
    return keep, {**facts, "backend": backend, "n_kept": int(keep.sum())}
