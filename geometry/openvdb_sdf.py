"""OpenVDB algorithmic extraction: Hierarchical Sparse Signed Distance Field (SDF) & CSG.

Provides narrow-band volumetric signed distance field (SDF) primitives,
exact analytic distance functions, and Constructive Solid Geometry (CSG) operations:
- Sphere, Box, and Half-Space analytic distance functions.
- Exact CSG Boolean Union, Intersection, and Difference in SDF space.
- Narrow-band sparse voxel grid storage for scalable volumetric representation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


def sdf_sphere(points: np.ndarray, radius: float = 1.0, center: Optional[np.ndarray] = None) -> np.ndarray:
    """Exact signed distance function for a sphere: Phi(x) = ||x - c|| - r."""
    pts = np.asarray(points, dtype=np.float64)
    if center is not None:
        pts = pts - np.asarray(center, dtype=np.float64)
    dist = np.linalg.norm(pts, axis=-1)
    return dist - radius


def sdf_box(points: np.ndarray, extents: np.ndarray, center: Optional[np.ndarray] = None) -> np.ndarray:
    """Exact signed distance function for an axis-aligned box with half-extents b:
    Phi(x) = ||max(|x| - b, 0)|| + min(max(dx, dy, dz), 0).
    """
    pts = np.asarray(points, dtype=np.float64)
    if center is not None:
        pts = pts - np.asarray(center, dtype=np.float64)

    b = np.asarray(extents, dtype=np.float64)
    d = np.abs(pts) - b
    outside_dist = np.linalg.norm(np.maximum(d, 0.0), axis=-1)
    inside_dist = np.minimum(np.max(d, axis=-1), 0.0)
    return outside_dist + inside_dist


def csg_union(sdf_a: np.ndarray, sdf_b: np.ndarray) -> np.ndarray:
    """CSG Boolean Union in SDF space: min(A, B)."""
    return np.minimum(sdf_a, sdf_b)


def csg_intersection(sdf_a: np.ndarray, sdf_b: np.ndarray) -> np.ndarray:
    """CSG Boolean Intersection in SDF space: max(A, B)."""
    return np.maximum(sdf_a, sdf_b)


def csg_difference(sdf_a: np.ndarray, sdf_b: np.ndarray) -> np.ndarray:
    """CSG Boolean Difference (A \\ B) in SDF space: max(A, -B)."""
    return np.maximum(sdf_a, -sdf_b)


@dataclass
class SparseVoxelGridSDF:
    """Narrow-band volumetric grid storing signed distances in sparse active blocks."""
    voxel_size: float = 0.1
    half_bandwidth: float = 0.3          # Narrow-band boundary limit
    # Sparse storage: (ix, iy, iz) -> distance value
    voxels: Dict[Tuple[int, int, int], float] = field(default_factory=dict)

    def discretize_primitive(
        self,
        sdf_func: Callable[[np.ndarray], np.ndarray],
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
    ) -> int:
        """Rasterize an analytic SDF function into narrow-band voxels within bounding box."""
        min_b = np.asarray(bounds_min, dtype=np.float64)
        max_b = np.asarray(bounds_max, dtype=np.float64)

        i_min = np.floor(min_b / self.voxel_size).astype(np.int64)
        i_max = np.ceil(max_b / self.voxel_size).astype(np.int64)

        xs = np.arange(i_min[0], i_max[0] + 1) * self.voxel_size
        ys = np.arange(i_min[1], i_max[1] + 1) * self.voxel_size
        zs = np.arange(i_min[2], i_max[2] + 1) * self.voxel_size

        grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")
        query_pts = np.stack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()], axis=-1)

        distances = sdf_func(query_pts)

        # Store only active narrow-band voxels
        active_count = 0
        for idx, dist in enumerate(distances):
            if abs(dist) <= self.half_bandwidth:
                pt = query_pts[idx]
                vc = (
                    int(round(pt[0] / self.voxel_size)),
                    int(round(pt[1] / self.voxel_size)),
                    int(round(pt[2] / self.voxel_size)),
                )
                self.voxels[vc] = float(dist)
                active_count += 1

        return active_count

    def sample_at_point(self, pt: np.ndarray) -> Optional[float]:
        """Query signed distance at point, returning None if outside narrow band."""
        vc = (
            int(round(pt[0] / self.voxel_size)),
            int(round(pt[1] / self.voxel_size)),
            int(round(pt[2] / self.voxel_size)),
        )
        return self.voxels.get(vc)
