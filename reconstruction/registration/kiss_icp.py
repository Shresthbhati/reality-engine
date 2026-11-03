"""KISS-ICP algorithmic extraction: Robust LiDAR Odometry with Voxel Hashing.

Implements an efficient point-to-point and point-to-plane registration pipeline:
1. Spatial voxel downsampling using hash-based integer grid coordinates.
2. Local voxel map storing dynamic point centroids and covariances.
3. Adaptive thresholding: rejects correspondences whose residual exceeds adaptive robust threshold.
4. Gauss-Newton or SVD Kabsch rigid pose update.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from reconstruction.geometry.lie_algebra import SE3, SO3


@dataclass
class KissIcpResult:
    """Output of KISS-ICP registration."""
    pose: SE3                           # Estimated relative rigid pose
    converged: bool
    iterations: int
    residual_rmse_m: float
    num_correspondences: int
    inlier_ratio: float


class VoxelHashMap:
    """Dynamic spatial hash map storing points organized into 3D voxel cells."""

    def __init__(self, voxel_size: float = 0.2, max_points_per_voxel: int = 20):
        self.voxel_size = max(0.01, float(voxel_size))
        self.max_points_per_voxel = max_points_per_voxel
        self.grid: Dict[Tuple[int, int, int], List[np.ndarray]] = {}

    def _point_to_voxel_coord(self, pt: np.ndarray) -> Tuple[int, int, int]:
        coord = np.floor(pt / self.voxel_size).astype(np.int64)
        return (int(coord[0]), int(coord[1]), int(coord[2]))

    def insert(self, points: np.ndarray) -> None:
        """Insert points into voxel grid, maintaining per-voxel capacity."""
        for pt in points:
            vc = self._point_to_voxel_coord(pt)
            if vc not in self.grid:
                self.grid[vc] = []
            if len(self.grid[vc]) < self.max_points_per_voxel:
                self.grid[vc].append(pt.copy())

    def get_all_points(self) -> np.ndarray:
        """Return all points currently in the voxel map."""
        pts = []
        for vpts in self.grid.values():
            pts.extend(vpts)
        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return np.array(pts, dtype=np.float64)

    def find_nearest_neighbor(
        self,
        query: np.ndarray,
        max_search_dist: float,
    ) -> Tuple[Optional[np.ndarray], float]:
        """Find closest point in neighboring voxels within max_search_dist."""
        center_vc = self._point_to_voxel_coord(query)
        r_voxels = max(1, int(math.ceil(max_search_dist / self.voxel_size)))

        best_pt = None
        best_dist_sq = max_search_dist * max_search_dist

        for dx in range(-r_voxels, r_voxels + 1):
            for dy in range(-r_voxels, r_voxels + 1):
                for dz in range(-r_voxels, r_voxels + 1):
                    vc = (center_vc[0] + dx, center_vc[1] + dy, center_vc[2] + dz)
                    if vc in self.grid:
                        for candidate in self.grid[vc]:
                            diff = query - candidate
                            dist_sq = float(np.dot(diff, diff))
                            if dist_sq < best_dist_sq:
                                best_dist_sq = dist_sq
                                best_pt = candidate

        return best_pt, math.sqrt(best_dist_sq) if best_pt is not None else float("inf")


class KissIcpOdometry:
    """KISS-ICP registration engine with voxel hashing and adaptive residual thresholding."""

    def __init__(
        self,
        voxel_size: float = 0.2,
        initial_threshold: float = 2.0,
        min_threshold: float = 0.15,
        max_iterations: int = 30,
        convergence_tol: float = 1e-4,
    ):
        self.voxel_size = max(0.02, float(voxel_size))
        self.initial_threshold = max(initial_threshold, self.voxel_size * 2.0)
        self.min_threshold = max(min_threshold, self.voxel_size * 1.5)
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.map = VoxelHashMap(voxel_size=self.voxel_size)

    def voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """Downsample point cloud to one representative point per voxel cell."""
        if len(points) <= 1:
            return points.copy()
        coords = np.floor(points / self.voxel_size).astype(np.int64)
        _, indices = np.unique(coords, axis=0, return_index=True)
        return points[indices]

    def register(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        initial_pose: Optional[SE3] = None,
    ) -> KissIcpResult:
        """Register source points to target points."""
        current_pose = initial_pose if initial_pose is not None else SE3.identity()

        # Build target voxel map
        target_down = self.voxel_downsample(target_points)
        target_map = VoxelHashMap(voxel_size=self.voxel_size)
        target_map.insert(target_down)

        source_down = self.voxel_downsample(source_points)
        if len(source_down) == 0 or len(target_down) == 0:
            return KissIcpResult(
                pose=current_pose,
                converged=False,
                iterations=0,
                residual_rmse_m=0.0,
                num_correspondences=0,
                inlier_ratio=0.0,
            )

        threshold = self.initial_threshold
        converged = False
        iteration = 0
        final_rmse = 0.0
        final_corrs = 0

        for it in range(self.max_iterations):
            iteration = it + 1
            # Transform source by current estimated pose
            transformed_source = current_pose.transform_points(source_down)

            src_matches = []
            tgt_matches = []
            residuals = []

            for i, p_src_trans in enumerate(transformed_source):
                closest_tgt, dist = target_map.find_nearest_neighbor(p_src_trans, max_search_dist=threshold)
                if closest_tgt is not None and dist <= threshold:
                    src_matches.append(p_src_trans)
                    tgt_matches.append(closest_tgt)
                    residuals.append(dist)

            if len(src_matches) < 4:
                break

            final_corrs = len(src_matches)
            residuals_arr = np.array(residuals, dtype=np.float64)
            final_rmse = float(np.sqrt(np.mean(residuals_arr ** 2)))

            # Update adaptive threshold based on residual dispersion (3 * sigma rule)
            median_residual = float(np.median(residuals_arr))
            sigma = 1.4826 * float(np.median(np.abs(residuals_arr - median_residual)))
            threshold = max(self.min_threshold, 3.0 * max(sigma, 0.01))

            # Solve incremental rigid alignment (Delta T) aligning transformed source to target
            src_arr = np.array(src_matches, dtype=np.float64)
            tgt_arr = np.array(tgt_matches, dtype=np.float64)

            src_centroid = np.mean(src_arr, axis=0)
            tgt_centroid = np.mean(tgt_arr, axis=0)

            src_centered = src_arr - src_centroid
            tgt_centered = tgt_arr - tgt_centroid

            h_mat = src_centered.T @ tgt_centered
            u, s, vt = np.linalg.svd(h_mat)
            r_opt = vt.T @ u.T

            # Reflection correction
            if np.linalg.det(r_opt) < 0:
                vt[2, :] *= -1.0
                r_opt = vt.T @ u.T

            t_opt = tgt_centroid - (r_opt @ src_centroid)
            delta_pose = SE3(SO3(r_opt), t_opt)

            # Update accumulated pose: T_curr = Delta T @ T_curr
            current_pose = delta_pose @ current_pose

            # Check convergence step norm
            step_norm = float(np.linalg.norm(delta_pose.log()))
            if step_norm < self.convergence_tol:
                converged = True
                break

        inlier_ratio = float(final_corrs / max(1, len(source_down)))
        return KissIcpResult(
            pose=current_pose,
            converged=converged,
            iterations=iteration,
            residual_rmse_m=final_rmse,
            num_correspondences=final_corrs,
            inlier_ratio=inlier_ratio,
        )
