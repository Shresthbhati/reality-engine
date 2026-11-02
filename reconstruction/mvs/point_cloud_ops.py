"""Point cloud processing operations (Master Development Plan Tasks 4.5 & 4.8).

Pure, deterministic geometric point cloud operations using NumPy and SciPy KDTree:
1. Voxel Grid Downsampling:
   - Aggregates 3D points, colors, and normals within uniform voxel grids.
   - Deterministic: sorted iteration by integer voxel coordinates.
2. Statistical Outlier Removal (SOR):
   - Computes mean distance to k-nearest neighbors via KDTree.
   - Filters noise points exceeding $\\mu + \\alpha \\cdot \\sigma$.
3. Radius Outlier Removal (ROR):
   - Filters isolated points with fewer than min_neighbors within search radius.
4. Normal Estimation:
   - Fits local surface tangent planes via covariance eigenvectors of k-neighborhoods.
   - Canonicalizes orientation towards camera viewpoints or world up.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.spatial import KDTree


def voxel_downsample(
    points: np.ndarray,
    voxel_size_m: float,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Downsample points by spatial hash voxel grid decimation.

    Args:
        points: (N, 3) float array of 3D coordinates.
        voxel_size_m: Voxel edge length in meters (e.g., 0.02 for 2cm resolution).
        colors: Optional (N, 3) or (N, 4) array of colors.
        normals: Optional (N, 3) array of surface normals.

    Returns:
        (downsampled_points, downsampled_colors, downsampled_normals)
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return pts, colors, normals
    if voxel_size_m <= 0.0:
        raise ValueError(f"voxel_size_m must be > 0, got {voxel_size_m}")

    has_colors = colors is not None and len(colors) == len(pts)
    has_normals = normals is not None and len(normals) == len(pts)

    cols = np.asarray(colors, dtype=np.float64) if has_colors else None
    norms = np.asarray(normals, dtype=np.float64) if has_normals else None

    # Compute integer voxel grid coordinates
    voxel_indices = np.floor(pts / voxel_size_m).astype(np.int64)

    # Group point indices by voxel coordinate tuple
    voxel_dict: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, (vx, vy, vz) in enumerate(voxel_indices):
        key = (int(vx), int(vy), int(vz))
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(idx)

    # Deterministic output order: sort keys
    sorted_keys = sorted(voxel_dict.keys())

    out_pts = np.zeros((len(sorted_keys), 3), dtype=np.float64)
    out_cols = np.zeros((len(sorted_keys), cols.shape[1]), dtype=np.float64) if has_colors else None
    out_norms = np.zeros((len(sorted_keys), 3), dtype=np.float64) if has_normals else None

    for i, key in enumerate(sorted_keys):
        member_indices = voxel_dict[key]
        # Average position
        out_pts[i] = np.mean(pts[member_indices], axis=0)

        # Average color
        if has_colors and out_cols is not None and cols is not None:
            out_cols[i] = np.mean(cols[member_indices], axis=0)

        # Average normal (normalized to unit length)
        if has_normals and out_norms is not None and norms is not None:
            avg_n = np.mean(norms[member_indices], axis=0)
            norm_len = float(np.linalg.norm(avg_n))
            out_norms[i] = avg_n / norm_len if norm_len > 1e-12 else np.array([0.0, 0.0, 1.0])

    if has_colors and out_cols is not None:
        if np.issubdtype(np.asarray(colors).dtype, np.integer):
            out_cols = np.round(out_cols).astype(np.asarray(colors).dtype)

    return out_pts, out_cols, out_norms


def statistical_outlier_removal(
    points: np.ndarray,
    k_neighbors: int = 20,
    std_ratio: float = 2.0,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Filter noise points whose average distance to k-nearest neighbors exceeds mean + std_ratio * std.

    Returns:
        (inlier_points, inlier_colors, inlier_normals, inlier_mask)
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n <= k_neighbors:
        mask = np.ones(n, dtype=bool)
        return pts, colors, normals, mask

    tree = KDTree(pts)
    # Query k_neighbors + 1 (first neighbor is self at distance 0)
    dists, _ = tree.query(pts, k=k_neighbors + 1)
    mean_dists = np.mean(dists[:, 1:], axis=1)

    mu = float(np.mean(mean_dists))
    sigma = float(np.std(mean_dists))
    threshold = mu + std_ratio * sigma

    mask = mean_dists <= threshold

    inlier_pts = pts[mask]
    inlier_cols = None
    if colors is not None and len(colors) == n:
        inlier_cols = np.asarray(colors)[mask]

    inlier_norms = None
    if normals is not None and len(normals) == n:
        inlier_norms = np.asarray(normals)[mask]

    return inlier_pts, inlier_cols, inlier_norms, mask


def radius_outlier_removal(
    points: np.ndarray,
    radius_m: float,
    min_neighbors: int = 5,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    """Filter isolated noise points that have fewer than min_neighbors within radius_m.

    Returns:
        (inlier_points, inlier_colors, inlier_normals, inlier_mask)
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0 or radius_m <= 0.0:
        mask = np.ones(n, dtype=bool)
        return pts, colors, normals, mask

    tree = KDTree(pts)
    neighbor_lists = tree.query_ball_point(pts, r=radius_m)

    # Each point includes itself in query_ball_point, so require >= min_neighbors + 1
    mask = np.array([len(neighbors) - 1 >= min_neighbors for neighbors in neighbor_lists], dtype=bool)

    inlier_pts = pts[mask]
    inlier_cols = None
    if colors is not None and len(colors) == n:
        inlier_cols = np.asarray(colors)[mask]

    inlier_norms = None
    if normals is not None and len(normals) == n:
        inlier_norms = np.asarray(normals)[mask]

    return inlier_pts, inlier_cols, inlier_norms, mask


def estimate_point_normals(
    points: np.ndarray,
    k_neighbors: int = 15,
    view_point: Optional[Sequence[float]] = None,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Estimate surface normal vectors via local covariance eigenvectors.

    Args:
        points: (N, 3) point coordinates.
        k_neighbors: Neighborhood size for PCA plane fit.
        view_point: Optional camera 3D center (x, y, z) to orient normals toward.
        world_up: Fallback orientation reference if view_point is omitted.

    Returns:
        (N, 3) array of unit normal vectors.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    k = min(n, max(3, k_neighbors))
    tree = KDTree(pts)
    _, idxs = tree.query(pts, k=k)

    normals = np.zeros((n, 3), dtype=np.float64)
    v_pt = np.asarray(view_point, dtype=np.float64) if view_point is not None else None
    up = np.asarray(world_up, dtype=np.float64)

    for i in range(n):
        neighbors = pts[idxs[i]]
        centered = neighbors - np.mean(neighbors, axis=0)
        cov = centered.T @ centered

        # Eigenvalues and eigenvectors (eigh returns sorted ascending)
        vals, vecs = np.linalg.eigh(cov)
        normal = vecs[:, 0]  # Eigenvector for smallest eigenvalue

        norm_len = float(np.linalg.norm(normal))
        if norm_len > 1e-12:
            normal = normal / norm_len
        else:
            normal = up

        # Canonicalize orientation
        if v_pt is not None:
            # Point toward camera: normal . (camera - point) >= 0
            to_cam = v_pt - pts[i]
            if np.dot(normal, to_cam) < 0.0:
                normal = -normal
        else:
            # Default towards +world_up
            if np.dot(normal, up) < 0.0:
                normal = -normal

        normals[i] = normal

    return normals
