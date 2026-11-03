"""Point cloud processing operations (Master Development Plan Tasks 4.5 & 4.8).

Pure, deterministic geometric point cloud operations using NumPy and SciPy KDTree:
1. Voxel Grid Downsampling:
   - Aggregates 3D points, colors, and normals within uniform voxel grids.
   - Deterministic: sorted iteration by integer voxel coordinates.
2. Statistical Outlier Removal (SOR):
   - Computes mean distance to k-nearest neighbors via KDTree.
   - Filters noise points exceeding mu + alpha * sigma.
3. Radius Outlier Removal (ROR):
   - Filters isolated points with fewer than min_neighbors within search radius.
4. Normal Estimation:
   - Fits local surface tangent planes via covariance eigenvectors of k-neighborhoods.
   - Canonicalizes orientation towards camera viewpoints or world up.
5. Chamfer Distance & F-Score Evaluation:
   - Computes bidirectional Chamfer distance CD(S1, S2).
   - Evaluates precision, recall, and harmonic F-Score at distance threshold tau.
6. PLY Point Cloud Serialization:
   - Exports points, colors, and normals to standard ASCII or binary little-endian PLY.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
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
    view_point: Optional[Union[Sequence[float], np.ndarray]] = None,
    world_up: Sequence[float] = (0.0, 0.0, 1.0),
) -> np.ndarray:
    """Estimate surface normal vectors via local covariance eigenvectors.

    Args:
        points: (N, 3) point coordinates.
        k_neighbors: Neighborhood size for PCA plane fit.
        view_point: Optional camera 3D center (x, y, z) or (N, 3) per-point viewpoints to orient normals toward.
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
    v_pts = None
    if view_point is not None:
        v_arr = np.asarray(view_point, dtype=np.float64)
        if v_arr.ndim == 1:
            v_pts = np.tile(v_arr, (n, 1))
        elif v_arr.shape == (n, 3):
            v_pts = v_arr
        else:
            raise ValueError(f"view_point must have shape (3,) or ({n}, 3), got {v_arr.shape}")

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
        if v_pts is not None:
            # Point toward camera: normal . (camera - point) >= 0
            to_cam = v_pts[i] - pts[i]
            if np.dot(normal, to_cam) < 0.0:
                normal = -normal
        else:
            # Default towards +world_up
            if np.dot(normal, up) < 0.0:
                normal = -normal

        normals[i] = normal

    return normals


def chamfer_distance(
    points_a: np.ndarray,
    points_b: np.ndarray,
    return_components: bool = False,
) -> Union[float, Tuple[float, float, float]]:
    """Compute the bidirectional Chamfer Distance between two point clouds:

        CD(A, B) = (1 / |A|) * sum_{x in A} min_{y in B} ||x - y||_2
                 + (1 / |B|) * sum_{y in B} min_{x in A} ||x - y||_2

    Args:
        points_a: (N, 3) float array of 3D points.
        points_b: (M, 3) float array of 3D points.
        return_components: If True, returns (cd_total, d_a_to_b, d_b_to_a).

    Returns:
        Chamfer distance (in same units as input, e.g. meters), or tuple of components.
    """
    pts_a = np.asarray(points_a, dtype=np.float64)
    pts_b = np.asarray(points_b, dtype=np.float64)

    if len(pts_a) == 0 or len(pts_b) == 0:
        val = float("inf")
        return (val, val, val) if return_components else val

    tree_b = KDTree(pts_b)
    dists_a_to_b, _ = tree_b.query(pts_a, k=1)
    mean_a_to_b = float(np.mean(dists_a_to_b))

    tree_a = KDTree(pts_a)
    dists_b_to_a, _ = tree_a.query(pts_b, k=1)
    mean_b_to_a = float(np.mean(dists_b_to_a))

    cd_total = mean_a_to_b + mean_b_to_a
    if return_components:
        return cd_total, mean_a_to_b, mean_b_to_a
    return cd_total


def f_score_tau(
    points_pred: np.ndarray,
    points_gt: np.ndarray,
    tau: float = 0.01,
) -> Tuple[float, float, float]:
    """Compute Precision, Recall, and harmonic F-Score at distance threshold tau.

    Args:
        points_pred: (N, 3) predicted point coordinates.
        points_gt: (M, 3) ground truth point coordinates.
        tau: Distance threshold in meters (default 0.01m = 10mm).

    Returns:
        (precision, recall, f_score) in range [0.0, 1.0].
    """
    pts_p = np.asarray(points_pred, dtype=np.float64)
    pts_g = np.asarray(points_gt, dtype=np.float64)

    if len(pts_p) == 0 or len(pts_g) == 0:
        return 0.0, 0.0, 0.0

    tree_g = KDTree(pts_g)
    dists_p, _ = tree_g.query(pts_p, k=1)
    precision = float(np.mean(dists_p <= tau))

    tree_p = KDTree(pts_p)
    dists_g, _ = tree_p.query(pts_g, k=1)
    recall = float(np.mean(dists_g <= tau))

    if precision + recall > 1e-12:
        f_score = 2.0 * (precision * recall) / (precision + recall)
    else:
        f_score = 0.0

    return precision, recall, f_score


def compute_point_cloud_density(
    points: np.ndarray,
    bounding_volume_m3: Optional[float] = None,
) -> float:
    """Compute point cloud density in points per cubic meter.

    Args:
        points: (N, 3) point coordinates.
        bounding_volume_m3: Optional explicit volume in m^3. If omitted, computes from AABB.

    Returns:
        Density in points / m^3.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return 0.0

    if bounding_volume_m3 is not None and bounding_volume_m3 > 0:
        return float(len(pts)) / float(bounding_volume_m3)

    bmin = np.min(pts, axis=0)
    bmax = np.max(pts, axis=0)
    extents = np.maximum(bmax - bmin, 0.01)
    volume = float(extents[0] * extents[1] * extents[2])
    return float(len(pts)) / volume if volume > 0 else 0.0


def export_point_cloud_ply(
    path: Union[str, Path],
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    binary: bool = True,
) -> None:
    """Export points, colors, and normals to standard PLY format.

    Args:
        path: Output file path.
        points: (N, 3) point coordinates float32 or float64.
        colors: Optional (N, 3) or (N, 4) uint8 colors.
        normals: Optional (N, 3) float32/float64 normals.
        binary: If True, writes binary_little_endian PLY; else writes ASCII PLY.
    """
    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)
    has_colors = colors is not None and len(colors) == n
    has_normals = normals is not None and len(normals) == n

    cols = np.asarray(colors, dtype=np.uint8) if has_colors else None
    norms = np.asarray(normals, dtype=np.float32) if has_normals else None

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0" if binary else "format ascii 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_normals:
        header_lines.extend([
            "property float nx",
            "property float ny",
            "property float nz",
        ])
    if has_colors:
        header_lines.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    header_lines.append("end_header\n")
    header = "\n".join(header_lines).encode("ascii")

    if binary:
        with open(target_path, "wb") as f:
            f.write(header)
            # Build structured record array
            dtype_fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
            if has_normals:
                dtype_fields.extend([("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")])
            if has_colors:
                dtype_fields.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])

            vertex_data = np.empty(n, dtype=dtype_fields)
            vertex_data["x"] = pts[:, 0]
            vertex_data["y"] = pts[:, 1]
            vertex_data["z"] = pts[:, 2]
            if has_normals and norms is not None:
                vertex_data["nx"] = norms[:, 0]
                vertex_data["ny"] = norms[:, 1]
                vertex_data["nz"] = norms[:, 2]
            if has_colors and cols is not None:
                vertex_data["red"] = cols[:, 0]
                vertex_data["green"] = cols[:, 1]
                vertex_data["blue"] = cols[:, 2]

            vertex_data.tofile(f)
    else:
        with open(target_path, "w", encoding="ascii") as f:
            f.write("\n".join(header_lines))
            for i in range(n):
                line_parts = [f"{pts[i, 0]:.6f}", f"{pts[i, 1]:.6f}", f"{pts[i, 2]:.6f}"]
                if has_normals and norms is not None:
                    line_parts.extend([f"{norms[i, 0]:.6f}", f"{norms[i, 1]:.6f}", f"{norms[i, 2]:.6f}"])
                if has_colors and cols is not None:
                    line_parts.extend([str(int(cols[i, 0])), str(int(cols[i, 1])), str(int(cols[i, 2]))])
                f.write(" ".join(line_parts) + "\n")
