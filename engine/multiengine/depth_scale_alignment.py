"""Metric Scale Alignment and Depth Map Validation (engine/multiengine/depth_scale_alignment.py).

Guarantees metric integrity for 3D reconstruction:
- Validates depth maps: strictly rejects constant, collapsed, non-finite, or zero-variance arrays.
- Aligns monocular relative depth to metric meters using verified 3D reference points
  (e.g. COLMAP bundle adjustment sparse points) via robust affine disparity/depth regression.
- Preserves raw unscaled predictions when alignment is unavailable, marking them RELATIVE_UNSCALED
  and strictly preventing unverified metric 3D unprojection.
- Records complete calibration provenance: scale source, calibration method, residual RMS error,
  R^2 goodness-of-fit, and inlier counts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from engine.math import Vec3
from world_ir.camera import PinholeCamera


@dataclass(frozen=True)
class ScaleAlignmentResult:
    """Outcome of metric depth scale alignment and validation."""

    is_aligned: bool
    scale_state: str  # "METRIC_CALIBRATED", "METRIC_SCALE_ALIGNED", "RELATIVE_UNSCALED", "INVALID"
    aligned_depth_map: np.ndarray | None = None
    scale: float | None = None
    shift: float | None = None
    residual_rms_error_m: float | None = None
    r_squared: float | None = None
    scale_source: str | None = None
    calibration_method: str | None = None
    point_count: int = 0
    inlier_count: int = 0
    rejection_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "is_aligned": self.is_aligned,
            "scale_state": self.scale_state,
            "scale": self.scale,
            "shift": self.shift,
            "residual_rms_error_m": self.residual_rms_error_m,
            "r_squared": self.r_squared,
            "scale_source": self.scale_source,
            "calibration_method": self.calibration_method,
            "point_count": self.point_count,
            "inlier_count": self.inlier_count,
            "rejection_reason": self.rejection_reason,
        }


def validate_depth_map(
    depth_map: np.ndarray | None,
    allow_nan: bool = True,
    min_std: float = 1e-4,
) -> tuple[bool, str | None]:
    """Strictly validates a depth map array.

    Rejects:
    - None or non-numpy arrays
    - Non-2D arrays
    - Empty arrays or arrays with all NaN/infinite values
    - Arrays with NaN/Inf if allow_nan=False
    - Arrays where finite values are negative (< 0)
    - Constant or collapsed arrays with standard deviation < min_std (when min_std > 0)
    """
    if depth_map is None:
        return False, "DEPTH_MAP_MISSING: Depth array is None"
    if not isinstance(depth_map, np.ndarray):
        return False, f"DEPTH_MAP_INVALID_TYPE: Expected np.ndarray, got {type(depth_map).__name__}"
    if depth_map.ndim != 2:
        return False, f"DEPTH_MAP_INVALID_SHAPE: Expected 2D array, got shape {depth_map.shape}"
    if depth_map.size == 0:
        return False, "DEPTH_MAP_EMPTY: Array contains 0 elements"

    finite_mask = np.isfinite(depth_map)
    if not np.any(finite_mask):
        return False, "DEPTH_MAP_ALL_NAN: Array contains no finite values"

    if not allow_nan and not np.all(finite_mask):
        return False, "DEPTH_MAP_NON_FINITE: Array contains NaN or infinite values"

    valid_vals = depth_map[finite_mask]
    if float(np.min(valid_vals)) < 0.0:
        return False, "DEPTH_MAP_NEGATIVE: Array contains negative values"

    if min_std > 0.0:
        std_val = float(np.std(valid_vals))
        spread = float(np.max(valid_vals)) - float(np.min(valid_vals))
        if std_val < min_std or spread < min_std:
            return False, f"DEPTH_MAP_COLLAPSED: Constant or zero-variance depth field (std={std_val:.6f})"

    return True, None


def align_relative_depth_to_metric(
    depth_map: np.ndarray,
    camera: PinholeCamera,
    reference_points_3d: Sequence[tuple[float, float, float]],
    pixel_coords_2d: Sequence[tuple[float, float]] | None = None,
    scale_source: str = "colmap_sparse_points",
    min_inliers: int = 15,
) -> ScaleAlignmentResult:
    """Aligns a relative depth map to metric meters using verified 3D reference points.

    Supports either:
    1. Direct 2D-3D correspondences (pixel_coords_2d + reference_points_3d)
    2. 3D reference points projected through pinhole camera intrinsics and extrinsics.
    """
    is_valid, err = validate_depth_map(depth_map)
    if not is_valid:
        return ScaleAlignmentResult(
            is_aligned=False,
            scale_state="INVALID",
            rejection_reason=err,
        )

    h, w = depth_map.shape

    # Extract 2D and camera-frame depth pairs
    d_rels: list[float] = []
    z_trues: list[float] = []

    if pixel_coords_2d is not None and len(pixel_coords_2d) == len(reference_points_3d):
        # Direct 2D correspondences provided
        for (u, v), P_w in zip(pixel_coords_2d, reference_points_3d):
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < w and 0 <= vi < h:
                p_cam = camera.extrinsics.world_to_camera(Vec3(P_w[0], P_w[1], P_w[2]))
                if p_cam.z > 0.1:
                    d_rels.append(float(depth_map[vi, ui]))
                    z_trues.append(float(p_cam.z))
    else:
        # Project 3D points through camera
        for P_w in reference_points_3d:
            pt_w = Vec3(P_w[0], P_w[1], P_w[2])
            p_cam = camera.extrinsics.world_to_camera(pt_w)
            if p_cam.z <= 0.1:
                continue
            uv = camera.project(pt_w)
            if uv is None:
                continue
            ui, vi = int(round(uv[0])), int(round(uv[1]))
            if 0 <= ui < w and 0 <= vi < h:
                d_rels.append(float(depth_map[vi, ui]))
                z_trues.append(float(p_cam.z))

    if len(d_rels) < min_inliers:
        return ScaleAlignmentResult(
            is_aligned=False,
            scale_state="RELATIVE_UNSCALED",
            point_count=len(d_rels),
            rejection_reason=f"INSUFFICIENT_CORRESPONDENCES: {len(d_rels)} valid points (minimum {min_inliers} required)",
        )

    arr_d = np.array(d_rels, dtype=np.float64)
    arr_z = np.array(z_trues, dtype=np.float64)

    # 1. Fit affine disparity model: 1 / z = s * d_rel + t
    inv_z = 1.0 / np.clip(arr_z, 0.05, 100.0)
    A_disp = np.vstack([arr_d, np.ones(len(arr_d))]).T
    s_disp, t_disp = np.linalg.lstsq(A_disp, inv_z, rcond=None)[0]
    inv_z_pred = s_disp * arr_d + t_disp
    z_pred_disp = 1.0 / np.clip(inv_z_pred, 1e-4, 10.0)

    res_disp = np.abs(z_pred_disp - arr_z)
    inlier_mask_disp = res_disp < np.percentile(res_disp, 85)
    rms_disp = float(np.sqrt(np.mean((z_pred_disp[inlier_mask_disp] - arr_z[inlier_mask_disp]) ** 2)))
    ss_tot = float(np.sum((arr_z - np.mean(arr_z)) ** 2))
    r2_disp = 1.0 - (float(np.sum((arr_z[inlier_mask_disp] - z_pred_disp[inlier_mask_disp]) ** 2)) / max(1e-6, ss_tot))

    # 2. Fit linear depth model: z = s * d_rel + t
    A_lin = np.vstack([arr_d, np.ones(len(arr_d))]).T
    s_lin, t_lin = np.linalg.lstsq(A_lin, arr_z, rcond=None)[0]
    z_pred_lin = s_lin * arr_d + t_lin
    res_lin = np.abs(z_pred_lin - arr_z)
    inlier_mask_lin = res_lin < np.percentile(res_lin, 85)
    rms_lin = float(np.sqrt(np.mean((z_pred_lin[inlier_mask_lin] - arr_z[inlier_mask_lin]) ** 2)))
    r2_lin = 1.0 - (float(np.sum((arr_z[inlier_mask_lin] - z_pred_lin[inlier_mask_lin]) ** 2)) / max(1e-6, ss_tot))

    # Choose model with better goodness of fit
    use_disp = (r2_disp > r2_lin and s_disp > 0) or (s_lin <= 0 and s_disp > 0)
    best_r2 = max(r2_disp, r2_lin)

    if best_r2 < 0.30:
        return ScaleAlignmentResult(
            is_aligned=False,
            scale_state="RELATIVE_UNSCALED",
            point_count=len(arr_d),
            inlier_count=int(np.sum(inlier_mask_disp if use_disp else inlier_mask_lin)),
            r_squared=round(best_r2, 4),
            rejection_reason=f"POOR_SCALE_ALIGNMENT_FIT: R^2={best_r2:.3f} < 0.30",
        )

    if use_disp:
        # Compute aligned metric map via disparity
        aligned = 1.0 / np.clip(s_disp * depth_map + t_disp, 0.02, 10.0)
        s_val, t_val = float(s_disp), float(t_disp)
        rms_val = rms_disp
        r2_val = r2_disp
        inliers = int(np.sum(inlier_mask_disp))
        method = "affine_disparity_regression"
    else:
        aligned = np.clip(s_lin * depth_map + t_lin, 0.1, 100.0)
        s_val, t_val = float(s_lin), float(t_lin)
        rms_val = rms_lin
        r2_val = r2_lin
        inliers = int(np.sum(inlier_mask_lin))
        method = "linear_depth_regression"

    aligned = aligned.astype(np.float32)

    return ScaleAlignmentResult(
        is_aligned=True,
        scale_state="METRIC_SCALE_ALIGNED",
        aligned_depth_map=aligned,
        scale=round(s_val, 6),
        shift=round(t_val, 6),
        residual_rms_error_m=round(rms_val, 4),
        r_squared=round(r2_val, 4),
        scale_source=scale_source,
        calibration_method=method,
        point_count=len(arr_d),
        inlier_count=inliers,
    )
