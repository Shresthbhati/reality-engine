"""Deterministic point-cloud preprocessing before surface reconstruction.

Three filters, all deterministic (no RNG, no timing, stable tie-breaking):

  - voxel downsample (one representative point per voxel, the FIRST in
    stable point order -- order preservation beats centroid averaging
    because depth-fused points are not i.i.d. samples)
  - statistical outlier removal (kNN mean-distance percentile gate,
    the classic Rusu et al. filter)
  - camera-oriented normals (local PCA over a kNN patch, oriented toward
    the camera centers that observed each point -- the orientation
    Poisson reconstruction needs and cannot infer by itself)

Normals come from scipy.spatial.cKDTree (already a repo dependency via
COLMAP tooling); no Open3D required.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

Vert = Tuple[float, float, float]


class PreprocessError(ValueError):
    pass


def _points_array(points: Sequence[Vert]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise PreprocessError(f"expected (N, 3) points, got shape {arr.shape}")
    return arr


def voxel_downsample(
    points: Sequence[Vert],
    voxel_size_m: float,
    normals: Optional[Sequence[Vert]] = None,
) -> Tuple[List[Vert], Optional[List[Vert]]]:
    """One point per voxel (side `voxel_size_m`): the first point in the
    input order that falls in each voxel. Deterministic: voxel keys are
    floor-divided indices; ties resolved by input order."""
    if voxel_size_m <= 0:
        raise PreprocessError(f"voxel_size_m must be > 0, got {voxel_size_m}")
    arr = _points_array(points)
    keep: List[int] = []
    seen: Dict[Tuple[int, int, int], bool] = {}
    keys = np.floor(arr / voxel_size_m).astype(np.int64)
    for i, key in enumerate(map(tuple, keys)):
        if key not in seen:
            seen[key] = True
            keep.append(i)
    kept_normals = None
    if normals is not None:
        if len(normals) != len(points):
            raise PreprocessError(
                f"normals ({len(normals)}) != points ({len(points)})"
            )
        kept_normals = [normals[i] for i in keep]
    return [tuple(arr[i]) for i in keep], kept_normals


def statistical_outlier_filter(
    points: Sequence[Vert],
    k_neighbors: int = 16,
    stddev_ratio: float = 2.0,
) -> Tuple[List[Vert], Dict[str, float]]:
    """Keep points whose mean distance to their `k_neighbors` nearest
    neighbours is within `mean + stddev_ratio * std` of that statistic
    across the cloud. Deterministic: cKDTree with stable worker count and
    a fixed std computed over all points; boundary (exactly-at-threshold)
    points are KEPT (strict >)."""
    if not points:
        return [], {"kept": 0, "removed": 0, "threshold_m": 0.0}
    if k_neighbors < 1 or stddev_ratio < 0:
        raise PreprocessError("k_neighbors must be >= 1 and stddev_ratio >= 0")
    arr = _points_array(points)
    k = min(k_neighbors + 1, len(arr))  # +1 because the point itself is included
    tree = cKDTree(arr)
    dists, _ = tree.query(arr, k=k, workers=1)
    mean_d = dists.mean(axis=1)
    mu = float(mean_d.mean())
    sigma = float(mean_d.std(ddof=0))
    threshold = mu + stddev_ratio * sigma
    mask = mean_d <= threshold
    kept = [tuple(p) for p, m in zip(arr, mask) if m]
    return kept, {
        "kept": int(mask.sum()),
        "removed": int((~mask).sum()),
        "threshold_m": round(threshold, 6),
        "mean_neighbor_dist_m": round(mu, 6),
    }


def _deterministic_pca_normal(patch: np.ndarray) -> Vert:
    """Smallest-eigenvector of the patch covariance. scipy/numpy eigh
    returns eigenvalues ascending, so the LAST column of eigenvectors is
    the normal direction. Symmetric-input determinism is guaranteed by
    eigh's algorithm; degenerate patches (all points identical) produce a
    zero normal that the caller must skip."""
    centered = patch - patch.mean(axis=0)
    cov = centered.T @ centered
    _, vecs = np.linalg.eigh(cov)
    return tuple(float(c) for c in vecs[:, 0])


def estimate_oriented_normals(
    points: Sequence[Vert],
    camera_centers: Sequence[Vert],
    k_neighbors: int = 16,
) -> List[Vert]:
    """Per-point normal from local PCA, oriented toward the NEAREST camera
    center that observed the point (dot(normal, camera - point) >= 0;
    flip when negative). A point with no finite camera center or a
    degenerate patch raises PreprocessError -- silently emitting an
    unoriented normal would poison Poisson reconstruction."""
    if not points:
        return []
    if not camera_centers:
        raise PreprocessError("camera_centers is empty -- cannot orient normals")
    arr = _points_array(points)
    cams = np.asarray(camera_centers, dtype=np.float64)
    if cams.ndim != 2 or cams.shape[1] != 3:
        raise PreprocessError(f"expected (M, 3) camera centers, got {cams.shape}")
    if not np.isfinite(cams).all():
        raise PreprocessError("camera_centers contain non-finite values")
    if not np.isfinite(arr).all():
        raise PreprocessError("points contain non-finite values")

    cam_tree = cKDTree(cams)
    tree = cKDTree(arr)
    k = min(k_neighbors, len(arr))
    _, idx = tree.query(arr, k=k, workers=1)
    if k == 1:
        idx = idx[:, None]

    normals: List[Vert] = []
    for i in range(len(arr)):
        patch = arr[idx[i]]
        n = np.asarray(_deterministic_pca_normal(patch), dtype=np.float64)
        if not np.isfinite(n).all() or np.linalg.norm(n) < 1e-9:
            raise PreprocessError(
                f"degenerate patch at point {i} -- cannot compute a normal; "
                "filter degenerate points first"
            )
        # orient toward the nearest camera center
        _, cam_i = cam_tree.query(arr[i], k=1, workers=1)
        to_cam = cams[cam_i] - arr[i]
        if float(np.dot(n, to_cam)) < 0:
            n = -n
        norm = float(np.linalg.norm(n))
        normals.append((n[0] / norm, n[1] / norm, n[2] / norm))
    return normals
