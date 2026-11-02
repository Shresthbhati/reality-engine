"""Multi-view dense point cloud generation and fusion (Master Development Plan Task 4.4).

Fuses multi-view metric depth maps and calibrated pinhole cameras into a clean,
dense, colored 3D point cloud:
1. Multi-view depth consistency verification:
   - Reprojects 3D points into co-visible views to check geometric agreement.
   - Eliminates occlusion artifacts and single-view depth estimation hallucinations.
2. Color sampling from calibrated source imagery.
3. Statistical outlier filtering and voxel decimation.
4. Consistent surface normal estimation directed toward observing cameras.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.depth.interface import DepthMap
from reconstruction.calibration.camera import PinholeCamera
from reconstruction.mvs.point_cloud_ops import (
    estimate_point_normals,
    statistical_outlier_removal,
    voxel_downsample,
)


@dataclass(frozen=True)
class DenseCloudResult:
    """Outcome of multi-view dense point cloud fusion."""

    points: np.ndarray  # (N, 3) float64
    colors: Optional[np.ndarray]  # (N, 3) uint8
    normals: Optional[np.ndarray]  # (N, 3) float64
    point_count: int
    view_count: int
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    density_points_per_m3: float
    diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_count": self.point_count,
            "view_count": self.view_count,
            "bounds_min": [round(v, 4) for v in self.bounds_min],
            "bounds_max": [round(v, 4) for v in self.bounds_max],
            "density_points_per_m3": round(self.density_points_per_m3, 2),
            "diagnostics": self.diagnostics,
        }


class MultiViewDenseCloudBuilder:
    """Builds a dense 3D point cloud from multi-view cameras and metric depth maps."""

    def __init__(
        self,
        stride: int = 2,
        consistency_check: bool = True,
        depth_tolerance: float = 0.08,
        min_consistent_views: int = 1,
        voxel_size_m: Optional[float] = 0.02,
        sor_std_ratio: Optional[float] = 2.0,
    ):
        self.stride = max(1, stride)
        self.consistency_check = consistency_check
        self.depth_tolerance = depth_tolerance
        self.min_consistent_views = min_consistent_views
        self.voxel_size_m = voxel_size_m
        self.sor_std_ratio = sor_std_ratio

    def fuse(
        self,
        cameras: Dict[str, PinholeCamera],
        depth_maps: Dict[str, DepthMap],
        images_rgb: Optional[Dict[str, np.ndarray]] = None,
    ) -> DenseCloudResult:
        """Fuse metric depth maps from multiple views into a single consistent point cloud."""
        common_ids = sorted(set(cameras.keys()) & set(depth_maps.keys()))
        if not common_ids:
            empty_pts = np.empty((0, 3), dtype=np.float64)
            return DenseCloudResult(
                points=empty_pts,
                colors=None,
                normals=None,
                point_count=0,
                view_count=0,
                bounds_min=(0.0, 0.0, 0.0),
                bounds_max=(0.0, 0.0, 0.0),
                density_points_per_m3=0.0,
                diagnostics={"status": "no_matching_views", "cameras": len(cameras), "depth_maps": len(depth_maps)},
            )

        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        all_cam_centers: List[np.ndarray] = []

        total_pixels_evaluated = 0
        consistent_pixels_accepted = 0

        # Pre-extract camera optical centers
        cam_centers: Dict[str, np.ndarray] = {
            ev_id: np.array([
                cameras[ev_id].extrinsics.position.x,
                cameras[ev_id].extrinsics.position.y,
                cameras[ev_id].extrinsics.position.z,
            ], dtype=np.float64)
            for ev_id in common_ids
        }

        for ref_id in common_ids:
            ref_cam = cameras[ref_id]
            ref_depth = depth_maps[ref_id]

            if ref_depth.unit != "meters":
                continue

            ref_img = images_rgb.get(ref_id) if images_rgb else None
            h_scale, w_scale = 1.0, 1.0
            if ref_img is not None:
                h_scale = ref_img.shape[0] / float(ref_depth.height)
                w_scale = ref_img.shape[1] / float(ref_depth.width)

            # Subsampled grid
            rows = range(0, ref_depth.height, self.stride)
            cols = range(0, ref_depth.width, self.stride)

            for r in rows:
                depth_row = ref_depth.values[r]
                for c in cols:
                    d = depth_row[c]
                    if not math.isfinite(d) or d <= 0.0:
                        continue

                    total_pixels_evaluated += 1

                    # Unproject point into 3D world space
                    world_pt = ref_cam.unproject(c + 0.5, r + 0.5, d)
                    p_vec = np.array([world_pt.x, world_pt.y, world_pt.z], dtype=np.float64)

                    # Multi-view depth consistency check
                    if self.consistency_check and len(common_ids) > 1 and self.min_consistent_views > 1:
                        consistent_views = 1  # reference view is consistent with itself
                        for other_id in common_ids:
                            if other_id == ref_id:
                                continue
                            other_cam = cameras[other_id]
                            other_depth = depth_maps[other_id]

                            # Project world_pt into other_cam
                            uv = other_cam.project(world_pt)
                            if uv is None:
                                continue
                            u_other, v_other = uv
                            if 0 <= u_other < other_depth.width and 0 <= v_other < other_depth.height:
                                p_cam_other = other_cam.extrinsics.world_to_camera(world_pt)
                                d_proj = p_cam_other.z
                                if d_proj <= 0:
                                    continue
                                d_meas = other_depth.values[int(v_other)][int(u_other)]
                                if math.isfinite(d_meas) and d_meas > 0:
                                    rel_diff = abs(d_proj - d_meas) / d_meas
                                    if rel_diff <= self.depth_tolerance:
                                        consistent_views += 1

                        if consistent_views < self.min_consistent_views:
                            continue

                    consistent_pixels_accepted += 1
                    all_points.append(p_vec)
                    all_cam_centers.append(cam_centers[ref_id])

                    # Sample color
                    if ref_img is not None:
                        pr = min(int((r + 0.5) * h_scale), ref_img.shape[0] - 1)
                        pc = min(int((c + 0.5) * w_scale), ref_img.shape[1] - 1)
                        rgb_px = ref_img[pr, pc][:3]
                        all_colors.append(np.asarray(rgb_px, dtype=np.uint8))

        if not all_points:
            empty_pts = np.empty((0, 3), dtype=np.float64)
            return DenseCloudResult(
                points=empty_pts,
                colors=None,
                normals=None,
                point_count=0,
                view_count=len(common_ids),
                bounds_min=(0.0, 0.0, 0.0),
                bounds_max=(0.0, 0.0, 0.0),
                density_points_per_m3=0.0,
                diagnostics={"status": "zero_points_after_consistency", "evaluated": total_pixels_evaluated},
            )

        pts_arr = np.vstack(all_points)
        cols_arr = np.vstack(all_colors) if all_colors else None

        # 1. Statistical Outlier Removal
        if self.sor_std_ratio is not None and len(pts_arr) > 30:
            pts_arr, cols_arr, _, _ = statistical_outlier_removal(
                points=pts_arr,
                k_neighbors=min(20, len(pts_arr) - 1),
                std_ratio=self.sor_std_ratio,
                colors=cols_arr,
            )

        # 2. Voxel Grid Downsampling
        if self.voxel_size_m is not None and self.voxel_size_m > 0.0 and len(pts_arr) > 0:
            pts_arr, cols_arr, _ = voxel_downsample(
                points=pts_arr,
                voxel_size_m=self.voxel_size_m,
                colors=cols_arr,
            )

        # 3. Normal Estimation
        normals_arr = None
        if len(pts_arr) >= 4:
            # Estimate normals oriented towards centroid of observing cameras
            mean_cam = np.mean(all_cam_centers, axis=0) if all_cam_centers else None
            normals_arr = estimate_point_normals(pts_arr, k_neighbors=15, view_point=mean_cam)

        # Bounding box & Density
        bmin = np.min(pts_arr, axis=0)
        bmax = np.max(pts_arr, axis=0)
        extent = np.maximum(bmax - bmin, 0.01)
        volume_m3 = float(extent[0] * extent[1] * extent[2])
        density = len(pts_arr) / volume_m3 if volume_m3 > 0 else 0.0

        return DenseCloudResult(
            points=pts_arr,
            colors=cols_arr,
            normals=normals_arr,
            point_count=len(pts_arr),
            view_count=len(common_ids),
            bounds_min=(float(bmin[0]), float(bmin[1]), float(bmin[2])),
            bounds_max=(float(bmax[0]), float(bmax[1]), float(bmax[2])),
            density_points_per_m3=density,
            diagnostics={
                "status": "success",
                "pixels_evaluated": total_pixels_evaluated,
                "pixels_accepted": consistent_pixels_accepted,
                "final_point_count": len(pts_arr),
                "stride": self.stride,
                "voxel_size_m": self.voxel_size_m,
            },
        )
