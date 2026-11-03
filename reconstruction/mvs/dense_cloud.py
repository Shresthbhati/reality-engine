"""Multi-view dense point cloud generation and fusion (Master Development Plan Task 4.4, Epic E03-1).

Fuses multi-view metric depth maps and calibrated pinhole cameras into a clean,
dense, colored 3D point cloud:
1. Multi-view depth consistency verification:
   - Vectorized ray unprojection into 3D world space.
   - Vectorized reprojection into co-visible views with sub-pixel depth sampling.
   - Co-visibility consistency voting: rejects single-view depth hallucinations,
     edge bleed, and occluded surfaces.
2. Multi-view radiance color blending:
   - Averages radiance across consistent observing views.
3. Statistical outlier filtering and voxel grid decimation:
   - Removes floating depth noise and ensures uniform metric spatial density.
4. Consistent surface normal estimation:
   - Local surface tangent plane fitting via covariance PCA.
   - Canonicalized orientation towards observing camera optical centers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from perception.depth.interface import DepthMap
from provenance import Uncertainty
from reconstruction.backend.interface import ReconstructedPoint
from reconstruction.calibration.camera import (
    CameraIntrinsics,
    PinholeCamera,
    _UNDISTORT_ITERATIONS,
    quat_to_matrix,
)
from reconstruction.mvs.point_cloud_ops import (
    compute_point_cloud_density,
    estimate_point_normals,
    export_point_cloud_ply,
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
    confidences: Optional[np.ndarray] = None  # (N,) float64

    def to_dict(self) -> Dict[str, Any]:
        return {
            "point_count": self.point_count,
            "view_count": self.view_count,
            "bounds_min": [round(float(v), 4) for v in self.bounds_min],
            "bounds_max": [round(float(v), 4) for v in self.bounds_max],
            "density_points_per_m3": round(float(self.density_points_per_m3), 2),
            "diagnostics": self.diagnostics,
        }

    def to_reconstructed_points(
        self,
        default_track_prefix: str = "dense",
        source_evidence_ids: Optional[List[str]] = None,
    ) -> List[ReconstructedPoint]:
        """Convert fused dense cloud into standard ReconstructedPoint instances."""
        ev_ids = list(source_evidence_ids or self.diagnostics.get("registered_evidence_ids", []))
        out: List[ReconstructedPoint] = []
        has_colors = self.colors is not None and len(self.colors) == self.point_count
        has_conf = self.confidences is not None and len(self.confidences) == self.point_count

        for i in range(self.point_count):
            p = (float(self.points[i, 0]), float(self.points[i, 1]), float(self.points[i, 2]))
            col = (int(self.colors[i, 0]), int(self.colors[i, 1]), int(self.colors[i, 2])) if has_colors else None
            conf = float(self.confidences[i]) if has_conf else 0.85
            out.append(
                ReconstructedPoint(
                    position=p,
                    track_id=f"{default_track_prefix}:{i}",
                    source_evidence_ids=ev_ids,
                    uncertainty=Uncertainty(confidence=max(0.0, min(1.0, conf))),
                    color=col,
                )
            )
        return out

    def save_ply(self, path: Union[str, Path], binary: bool = True) -> None:
        """Export dense point cloud to standard PLY format."""
        export_point_cloud_ply(
            path=path,
            points=self.points,
            colors=self.colors,
            normals=self.normals,
            binary=binary,
        )


class MultiViewDenseCloudBuilder:
    """Builds a dense 3D point cloud from multi-view cameras and metric depth maps."""

    def __init__(
        self,
        stride: Optional[int] = 2,
        consistency_check: bool = True,
        depth_tolerance: float = 0.08,
        min_consistent_views: Optional[int] = 1,
        voxel_size_m: Optional[float] = 0.02,
        sor_std_ratio: Optional[float] = 2.0,
        sor_k_neighbors: int = 20,
        bilinear_depth_sample: bool = True,
        radiance_blending: bool = True,
        min_depth_m: float = 0.05,
        max_depth_m: float = 100.0,
    ):
        self.stride = stride
        self.consistency_check = consistency_check
        self.depth_tolerance = depth_tolerance
        self.min_consistent_views = min_consistent_views
        self.voxel_size_m = voxel_size_m
        self.sor_std_ratio = sor_std_ratio
        self.sor_k_neighbors = max(3, sor_k_neighbors)
        self.bilinear_depth_sample = bilinear_depth_sample
        self.radiance_blending = radiance_blending
        self.min_depth_m = max(0.001, min_depth_m)
        self.max_depth_m = max(self.min_depth_m + 0.1, max_depth_m)

    def _resolve_stride(self, depth_maps: Dict[str, DepthMap]) -> int:
        """Automatically resolve optimal ray stride if unspecified (Results Over Technologies)."""
        if self.stride is not None and self.stride >= 1:
            return int(self.stride)
        max_dim = max((max(dm.width, dm.height) for dm in depth_maps.values()), default=480)
        if max_dim <= 400:
            return 1
        elif max_dim <= 1080:
            return 2
        else:
            return 4

    def _resolve_min_views(self, view_count: int) -> int:
        """Automatically resolve consistency view threshold if unspecified."""
        if self.min_consistent_views is not None and self.min_consistent_views >= 1:
            return int(self.min_consistent_views)
        return 2 if view_count >= 2 else 1

    def fuse(
        self,
        cameras: Dict[str, PinholeCamera],
        depth_maps: Dict[str, DepthMap],
        images_rgb: Optional[Dict[str, np.ndarray]] = None,
    ) -> DenseCloudResult:
        """Fuse metric depth maps from multiple calibrated views into a consistent point cloud."""
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

        effective_stride = self._resolve_stride(depth_maps)
        effective_min_views = self._resolve_min_views(len(common_ids))

        all_points: List[np.ndarray] = []
        all_colors: List[np.ndarray] = []
        all_viewpoints: List[np.ndarray] = []
        all_confidences: List[np.ndarray] = []

        total_pixels_evaluated = 0
        consistent_pixels_accepted = 0

        # Pre-extract camera optical centers and rotation matrices
        cam_rotations: Dict[str, np.ndarray] = {}
        cam_centers: Dict[str, np.ndarray] = {}
        for ev_id in common_ids:
            cam = cameras[ev_id]
            cam_rotations[ev_id] = np.array(quat_to_matrix(cam.extrinsics.rotation), dtype=np.float64)
            cam_centers[ev_id] = np.array([
                cam.extrinsics.position.x,
                cam.extrinsics.position.y,
                cam.extrinsics.position.z,
            ], dtype=np.float64)

        for ref_id in common_ids:
            ref_cam = cameras[ref_id]
            ref_depth = depth_maps[ref_id]

            if ref_depth.unit != "meters":
                continue

            ref_d_arr = np.asarray(ref_depth.values, dtype=np.float64)
            h, w = ref_depth.height, ref_depth.width

            ref_img = images_rgb.get(ref_id) if images_rgb else None
            h_scale = (ref_img.shape[0] / float(h)) if ref_img is not None else 1.0
            w_scale = (ref_img.shape[1] / float(w)) if ref_img is not None else 1.0

            # Subsampled grid
            rows = np.arange(0, h, effective_stride)
            cols = np.arange(0, w, effective_stride)
            grid_c, grid_r = np.meshgrid(cols, rows)

            flat_r = grid_r.ravel()
            flat_c = grid_c.ravel()
            d_vals = ref_d_arr[flat_r, flat_c]

            # Valid depth mask
            valid_mask = np.isfinite(d_vals) & (d_vals >= self.min_depth_m) & (d_vals <= self.max_depth_m)
            num_valid = int(np.sum(valid_mask))
            total_pixels_evaluated += len(d_vals)

            if num_valid == 0:
                continue

            u_valid = flat_c[valid_mask].astype(np.float64) + 0.5
            v_valid = flat_r[valid_mask].astype(np.float64) + 0.5
            d_valid = d_vals[valid_mask]
            r_valid = flat_r[valid_mask]
            c_valid = flat_c[valid_mask]

            # Vectorized unprojection to camera coordinates
            intr = ref_cam.intrinsics
            x_d = (u_valid - intr.cx) / intr.fx
            y_d = (v_valid - intr.cy) / intr.fy

            # Handle Brown-Conrady undistortion if non-zero
            if intr.k1 != 0.0 or intr.k2 != 0.0 or intr.p1 != 0.0 or intr.p2 != 0.0 or intr.k3 != 0.0:
                x = x_d.copy()
                y = y_d.copy()
                for _ in range(_UNDISTORT_ITERATIONS):
                    r2 = x * x + y * y
                    radial = 1.0 + intr.k1 * r2 + intr.k2 * (r2 ** 2) + intr.k3 * (r2 ** 3)
                    tx = 2.0 * intr.p1 * x * y + intr.p2 * (r2 + 2.0 * x * x)
                    ty = intr.p1 * (r2 + 2.0 * y * y) + 2.0 * intr.p2 * x * y
                    x = (x_d - tx) / radial
                    y = (y_d - ty) / radial
            else:
                x = x_d
                y = y_d

            pts_cam = np.stack([x * d_valid, y * d_valid, d_valid], axis=1)  # (M, 3)

            # Camera-to-world transform
            R_c2w = cam_rotations[ref_id]
            t_c2w = cam_centers[ref_id]
            pts_world = (R_c2w @ pts_cam.T).T + t_c2w  # (M, 3)

            m_pts = len(pts_world)
            consistent_views = np.ones(m_pts, dtype=np.int32)

            accum_colors = np.zeros((m_pts, 3), dtype=np.float64)
            accum_weights = np.zeros(m_pts, dtype=np.float64)

            # Sample reference color
            if ref_img is not None:
                pr = np.clip((r_valid * h_scale).astype(np.int64), 0, ref_img.shape[0] - 1)
                pc = np.clip((c_valid * w_scale).astype(np.int64), 0, ref_img.shape[1] - 1)
                ref_rgb = ref_img[pr, pc, :3].astype(np.float64)
                accum_colors += ref_rgb
                accum_weights += 1.0

            # Multi-view depth consistency check
            if self.consistency_check and len(common_ids) > 1 and effective_min_views > 1:
                for other_id in common_ids:
                    if other_id == ref_id:
                        continue

                    other_cam = cameras[other_id]
                    other_depth = depth_maps[other_id]
                    other_d_arr = np.asarray(other_depth.values, dtype=np.float64)
                    h_o, w_o = other_depth.height, other_depth.width

                    R_o_c2w = cam_rotations[other_id]
                    t_o_c2w = cam_centers[other_id]

                    # World to other camera frame
                    # R_w2c = R_c2w^T
                    pts_other_cam = (pts_world - t_o_c2w) @ R_o_c2w  # (M, 3)
                    z_other = pts_other_cam[:, 2]

                    # Points in front of other camera
                    in_front = z_other > self.min_depth_m
                    if not np.any(in_front):
                        continue

                    sub_idx = np.where(in_front)[0]
                    p_sub = pts_other_cam[sub_idx]
                    z_sub = z_other[sub_idx]

                    x_n = p_sub[:, 0] / z_sub
                    y_n = p_sub[:, 1] / z_sub

                    intr_o = other_cam.intrinsics
                    if intr_o.k1 != 0.0 or intr_o.k2 != 0.0 or intr_o.p1 != 0.0 or intr_o.p2 != 0.0 or intr_o.k3 != 0.0:
                        r2_o = x_n * x_n + y_n * y_n
                        rad_o = 1.0 + intr_o.k1 * r2_o + intr_o.k2 * (r2_o ** 2) + intr_o.k3 * (r2_o ** 3)
                        x_dist = x_n * rad_o + 2.0 * intr_o.p1 * x_n * y_n + intr_o.p2 * (r2_o + 2.0 * x_n * x_n)
                        y_dist = y_n * rad_o + intr_o.p1 * (r2_o + 2.0 * y_n * y_n) + 2.0 * intr_o.p2 * x_n * y_n
                    else:
                        x_dist = x_n
                        y_dist = y_n

                    u_proj = intr_o.fx * x_dist + intr_o.cx
                    v_proj = intr_o.fy * y_dist + intr_o.cy

                    # Bounds check
                    in_bounds = (u_proj >= 0.0) & (u_proj <= float(w_o - 1.0)) & (v_proj >= 0.0) & (v_proj <= float(h_o - 1.0))
                    if not np.any(in_bounds):
                        continue

                    valid_idx = sub_idx[in_bounds]
                    u_b = u_proj[in_bounds]
                    v_b = v_proj[in_bounds]
                    z_b = z_sub[in_bounds]

                    # Depth sampling (bilinear or nearest)
                    if self.bilinear_depth_sample:
                        u0 = np.floor(u_b).astype(np.int64)
                        v0 = np.floor(v_b).astype(np.int64)
                        u1 = np.clip(u0 + 1, 0, w_o - 1)
                        v1 = np.clip(v0 + 1, 0, h_o - 1)

                        du = u_b - u0
                        dv = v_b - v0

                        d00 = other_d_arr[v0, u0]
                        d01 = other_d_arr[v0, u1]
                        d10 = other_d_arr[v1, u0]
                        d11 = other_d_arr[v1, u1]

                        interp_valid = (
                            np.isfinite(d00) & (d00 > 0) &
                            np.isfinite(d01) & (d01 > 0) &
                            np.isfinite(d10) & (d10 > 0) &
                            np.isfinite(d11) & (d11 > 0)
                        )
                        d_meas = (1.0 - du) * (1.0 - dv) * d00 + du * (1.0 - dv) * d01 + (1.0 - du) * dv * d10 + du * dv * d11
                        # Fallback for boundary/isolated pixels
                        d_meas = np.where(interp_valid, d_meas, other_d_arr[np.round(v_b).astype(np.int64), np.round(u_b).astype(np.int64)])
                    else:
                        u_near = np.clip(np.round(u_b).astype(np.int64), 0, w_o - 1)
                        v_near = np.clip(np.round(v_b).astype(np.int64), 0, h_o - 1)
                        d_meas = other_d_arr[v_near, u_near]

                    meas_valid = np.isfinite(d_meas) & (d_meas > 0.0)
                    rel_diff = np.abs(z_b - d_meas) / np.maximum(d_meas, 1e-6)
                    is_consistent = meas_valid & (rel_diff <= self.depth_tolerance)

                    if np.any(is_consistent):
                        matched_indices = valid_idx[is_consistent]
                        consistent_views[matched_indices] += 1

                        # Radiance color blending
                        if self.radiance_blending and images_rgb and other_id in images_rgb:
                            o_img = images_rgb[other_id]
                            h_sc = o_img.shape[0] / float(h_o)
                            w_sc = o_img.shape[1] / float(w_o)
                            pr_o = np.clip((v_b[is_consistent] * h_sc).astype(np.int64), 0, o_img.shape[0] - 1)
                            pc_o = np.clip((u_b[is_consistent] * w_sc).astype(np.int64), 0, o_img.shape[1] - 1)
                            o_rgb = o_img[pr_o, pc_o, :3].astype(np.float64)
                            accum_colors[matched_indices] += o_rgb
                            accum_weights[matched_indices] += 1.0

            # Filter points satisfying minimum view consistency
            accepted_mask = consistent_views >= effective_min_views
            num_accepted = int(np.sum(accepted_mask))
            if num_accepted == 0:
                continue

            consistent_pixels_accepted += num_accepted
            pts_accepted = pts_world[accepted_mask]
            all_points.append(pts_accepted)

            # Store camera centers for normal orientation
            c_rep = np.tile(t_c2w, (num_accepted, 1))
            all_viewpoints.append(c_rep)

            # Normalized consistency confidence
            conf_norm = np.clip(consistent_views[accepted_mask] / float(max(1, len(common_ids))), 0.2, 1.0)
            all_confidences.append(conf_norm)

            if accum_weights.max() > 0:
                weights = np.maximum(accum_weights[accepted_mask, None], 1.0)
                blended_rgb = np.clip(np.round(accum_colors[accepted_mask] / weights), 0, 255).astype(np.uint8)
                all_colors.append(blended_rgb)

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
                diagnostics={
                    "status": "zero_points_after_consistency",
                    "pixels_evaluated": total_pixels_evaluated,
                    "registered_evidence_ids": common_ids,
                },
            )

        pts_arr = np.vstack(all_points)
        cols_arr = np.vstack(all_colors) if all_colors else None
        view_arr = np.vstack(all_viewpoints) if all_viewpoints else None
        conf_arr = np.concatenate(all_confidences) if all_confidences else None

        # 1. Statistical Outlier Removal
        if self.sor_std_ratio is not None and len(pts_arr) > self.sor_k_neighbors:
            pts_arr, cols_arr, _, inlier_mask = statistical_outlier_removal(
                points=pts_arr,
                k_neighbors=min(self.sor_k_neighbors, len(pts_arr) - 1),
                std_ratio=self.sor_std_ratio,
                colors=cols_arr,
            )
            if view_arr is not None:
                view_arr = view_arr[inlier_mask]
            if conf_arr is not None:
                conf_arr = conf_arr[inlier_mask]

        # 2. Voxel Grid Downsampling
        if self.voxel_size_m is not None and self.voxel_size_m > 0.0 and len(pts_arr) > 0:
            pts_arr, cols_arr, _ = voxel_downsample(
                points=pts_arr,
                voxel_size_m=self.voxel_size_m,
                colors=cols_arr,
            )
            # Normal viewpoints downsampled to centroid
            view_arr = None

        # 3. Surface Normal Estimation
        normals_arr = None
        if len(pts_arr) >= 4:
            # Orient normals toward camera optical center or mean observation point
            mean_cam = np.mean(list(cam_centers.values()), axis=0) if cam_centers else None
            normals_arr = estimate_point_normals(pts_arr, k_neighbors=15, view_point=mean_cam)

        # 4. Density & Bounds
        bmin = np.min(pts_arr, axis=0)
        bmax = np.max(pts_arr, axis=0)
        density = compute_point_cloud_density(pts_arr)

        return DenseCloudResult(
            points=pts_arr,
            colors=cols_arr,
            normals=normals_arr,
            point_count=len(pts_arr),
            view_count=len(common_ids),
            bounds_min=(float(bmin[0]), float(bmin[1]), float(bmin[2])),
            bounds_max=(float(bmax[0]), float(bmax[1]), float(bmax[2])),
            density_points_per_m3=density,
            confidences=conf_arr[:len(pts_arr)] if conf_arr is not None else None,
            diagnostics={
                "status": "success",
                "pixels_evaluated": total_pixels_evaluated,
                "pixels_accepted": consistent_pixels_accepted,
                "final_point_count": len(pts_arr),
                "stride": effective_stride,
                "voxel_size_m": self.voxel_size_m,
                "min_consistent_views": effective_min_views,
                "registered_evidence_ids": common_ids,
            },
        )
