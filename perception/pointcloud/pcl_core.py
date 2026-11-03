"""Self-contained PCL Classical Point-Cloud Processing Core (Block 28 of 50-tech evaluation).

Implements:
1. Statistical Outlier Removal (SOR) filtering
2. Local Covariance PCA surface normal estimation
3. Planar RANSAC decomposition for architectural surfaces (floors, walls, ceilings)
Based on the Point Cloud Library (PCL) algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PclFilteredCloud:
    """Result of statistical filtering on raw 3D point cloud."""

    filtered_points: np.ndarray  # (M, 3) inliers
    outlier_points: np.ndarray  # (N - M, 3) rejected outliers
    mean_distance_m: float
    std_dev_m: float
    inlier_ratio: float


@dataclass(frozen=True)
class PclPlanarSegment:
    """Architectural plane extracted via RANSAC."""

    plane_normal: np.ndarray  # (3,) unit normal
    plane_distance_d: float  # scalar d in ax + by + cz + d = 0
    inlier_indices: np.ndarray  # (K,) indices
    inlier_points: np.ndarray  # (K, 3)
    inlier_rmse_m: float
    classification: str  # "FLOOR" | "CEILING" | "WALL" | "OTHER"


class PclGeometricProcessor:
    """Classical geometric perception toolkit for 3D point clouds."""

    @staticmethod
    def statistical_outlier_removal(
        points: np.ndarray,
        k_neighbors: int = 20,
        std_ratio: float = 1.0,
    ) -> PclFilteredCloud:
        """Statistical Outlier Removal (SOR) filter.

        Filters sparse sensor noise and flying artifacts where mean neighbor
        distance exceeds mu + (std_ratio * sigma).
        """
        pts = np.asarray(points, dtype=np.float64)
        N = len(pts)
        if N <= k_neighbors:
            return PclFilteredCloud(
                filtered_points=pts,
                outlier_points=np.zeros((0, 3)),
                mean_distance_m=0.0,
                std_dev_m=0.0,
                inlier_ratio=1.0,
            )

        k = min(k_neighbors, N - 1)
        mean_dists = np.zeros(N, dtype=np.float64)

        # Vectorized / chunked nearest neighbor distance calculation
        # For evaluation, computes distance to k closest points
        # If cloud is large, subsamples for speed
        step = 1 if N < 2000 else int(N / 1500)
        sample_indices = np.arange(0, N, step)

        for idx in range(N):
            # Euclidean distance from pts[idx] to all sample points
            dists = np.linalg.norm(pts[idx] - pts[sample_indices], axis=1)
            dists.sort()
            # Average distance to k nearest (excluding self distance 0)
            mean_dists[idx] = np.mean(dists[1 : min(len(dists), k + 1)])

        mu = float(np.mean(mean_dists))
        sigma = float(np.std(mean_dists))
        threshold = mu + (std_ratio * sigma)

        inlier_mask = mean_dists <= threshold
        inliers = pts[inlier_mask]
        outliers = pts[~inlier_mask]

        return PclFilteredCloud(
            filtered_points=inliers,
            outlier_points=outliers,
            mean_distance_m=mu,
            std_dev_m=sigma,
            inlier_ratio=float(len(inliers) / N),
        )

    @staticmethod
    def estimate_surface_normals(
        points: np.ndarray,
        k_neighbors: int = 15,
    ) -> np.ndarray:
        """Estimate 3D surface normals using PCA on local neighborhood covariance matrices.

        Returns: (N, 3) unit normal vectors.
        """
        pts = np.asarray(points, dtype=np.float64)
        N = len(pts)
        normals = np.zeros((N, 3), dtype=np.float64)

        if N < 3:
            return normals

        k = min(k_neighbors, N)

        for i in range(N):
            # Compute distances to all points and find k nearest
            dists = np.linalg.norm(pts - pts[i], axis=1)
            k_idx = np.argpartition(dists, k)[:k]
            neighbors = pts[k_idx]

            # Covariance matrix C = 1/k * sum((p - centroid)(p - centroid)^T)
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            cov = (centered.T @ centered) / k

            # Eigenvalues and eigenvectors (eigenvalues sorted ascending)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # Normal is the eigenvector of the smallest eigenvalue
            n = eigvecs[:, 0]
            norm = np.linalg.norm(n)
            normals[i] = n / norm if norm > 1e-8 else np.array([0.0, 0.0, 1.0])

        return normals

    @staticmethod
    def extract_dominant_plane(
        points: np.ndarray,
        distance_threshold: float = 0.03,  # 3cm threshold
        max_iterations: int = 200,
    ) -> Optional[PclPlanarSegment]:
        """Extract the largest dominant plane via RANSAC and classify as FLOOR, CEILING, or WALL."""
        pts = np.asarray(points, dtype=np.float64)
        N = len(pts)
        if N < 3:
            return None

        best_inliers: Optional[np.ndarray] = None
        best_normal: Optional[np.ndarray] = None
        best_d: float = 0.0
        max_inlier_count = 0

        rng = np.random.RandomState(42)

        for _ in range(max_iterations):
            idx = rng.choice(N, size=3, replace=False)
            p1, p2, p3 = pts[idx]

            # Normal: cross product
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-7:
                continue
            normal /= norm
            d = -np.dot(normal, p1)

            # Distances: |n . p + d|
            dists = np.abs(np.dot(pts, normal) + d)
            inliers = np.where(dists <= distance_threshold)[0]

            if len(inliers) > max_inlier_count:
                max_inlier_count = len(inliers)
                best_inliers = inliers
                best_normal = normal
                best_d = d

        if best_inliers is None or max_inlier_count < 3:
            return None

        # Least-squares refit on inliers
        inlier_pts = pts[best_inliers]
        centroid = np.mean(inlier_pts, axis=0)
        centered = inlier_pts - centroid
        cov = centered.T @ centered
        _, eigvecs = np.linalg.eigh(cov)
        refined_normal = eigvecs[:, 0]
        refined_normal /= np.linalg.norm(refined_normal)
        refined_d = -float(np.dot(refined_normal, centroid))

        # Architectural classification based on normal direction
        # +Z is up in metric ENU / architectural conventions
        abs_nz = abs(refined_normal[2])
        if abs_nz > 0.85:
            # Horizontal plane
            classification = "FLOOR" if centroid[2] < 0.5 else "CEILING"
        else:
            # Vertical plane
            classification = "WALL"

        rmse = float(np.sqrt(np.mean((np.dot(inlier_pts, refined_normal) + refined_d) ** 2)))

        return PclPlanarSegment(
            plane_normal=refined_normal,
            plane_distance_d=refined_d,
            inlier_indices=best_inliers,
            inlier_points=inlier_pts,
            inlier_rmse_m=rmse,
            classification=classification,
        )
