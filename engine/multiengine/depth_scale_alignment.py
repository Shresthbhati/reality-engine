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
class MetricAnchor:
    """Independent ground-truth physical metric anchor.

    Without a verified metric anchor with valid reference points and measured distance,
    reconstructions derived from SfM (e.g. COLMAP) remain in arbitrary gauge coordinates
    (COLMAP_SFM_UNITS) and MUST NOT be labeled as METRES.
    """

    anchor_type: str  # "known_distance", "laser_survey", "lidar_ground_truth", "synthetic_ground_truth"
    measured_distance_m: float  # Physical measured distance in meters
    uncertainty_m: float = 0.01  # Measurement uncertainty in meters
    reference_point_a: tuple[float, float, float] | None = None  # 3D coordinate in unscaled world frame
    reference_point_b: tuple[float, float, float] | None = None  # 3D coordinate in unscaled world frame
    reference_frame: str = "world:colmap"  # Coordinate frame of reference points
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "anchor_type": self.anchor_type,
            "measured_distance_m": self.measured_distance_m,
            "uncertainty_m": self.uncertainty_m,
            "reference_point_a": list(self.reference_point_a) if self.reference_point_a else None,
            "reference_point_b": list(self.reference_point_b) if self.reference_point_b else None,
            "reference_frame": self.reference_frame,
            "notes": self.notes,
        }


def validate_metric_anchor(
    anchor: MetricAnchor | None,
    allow_synthetic: bool = False,
) -> tuple[bool, str | None, float | None]:
    """Strictly validates an independent physical metric anchor.

    Returns:
        (is_valid, rejection_reason, scale_to_metres)
    """
    if anchor is None:
        return True, None, None

    if anchor.anchor_type == "synthetic_ground_truth" and not allow_synthetic:
        return (
            False,
            "SYNTHETIC_ANCHOR_REJECTED: Synthetic ground truth cannot establish production physical accuracy",
            None,
        )

    # Validate measured distance
    if not isinstance(anchor.measured_distance_m, (int, float)) or not np.isfinite(anchor.measured_distance_m):
        return False, "INVALID_METRIC_ANCHOR: measured_distance_m must be a finite number", None
    if float(anchor.measured_distance_m) <= 1e-4:
        return False, "INVALID_METRIC_ANCHOR: measured_distance_m must be strictly positive (> 1e-4 m)", None

    # Validate measurement uncertainty
    if not isinstance(anchor.uncertainty_m, (int, float)) or not np.isfinite(anchor.uncertainty_m):
        return False, "INVALID_METRIC_ANCHOR: uncertainty_m must be a finite number", None
    if float(anchor.uncertainty_m) < 0.0:
        return False, "INVALID_METRIC_ANCHOR: uncertainty_m cannot be negative", None

    # Require valid physical reference points in the reconstructed coordinate frame
    if anchor.reference_point_a is None or anchor.reference_point_b is None:
        return (
            False,
            "INVALID_METRIC_ANCHOR: Missing reference points (both reference_point_a and reference_point_b required)",
            None,
        )

    for name, pt in (("reference_point_a", anchor.reference_point_a), ("reference_point_b", anchor.reference_point_b)):
        if not isinstance(pt, (tuple, list)) or len(pt) != 3:
            return False, f"INVALID_METRIC_ANCHOR: {name} must be a 3-element coordinate tuple", None
        if not all(isinstance(c, (int, float)) and np.isfinite(c) for c in pt):
            return False, f"INVALID_METRIC_ANCHOR: {name} contains non-finite or non-numeric values", None

    p_a = np.array(anchor.reference_point_a, dtype=np.float64)
    p_b = np.array(anchor.reference_point_b, dtype=np.float64)
    dist_sfm = float(np.linalg.norm(p_a - p_b))

    if dist_sfm <= 1e-4:
        return (
            False,
            "INVALID_METRIC_ANCHOR: Distance between reference points is zero or coincident",
            None,
        )

    scale_to_metres = float(anchor.measured_distance_m) / dist_sfm
    if not np.isfinite(scale_to_metres) or scale_to_metres <= 0.0:
        return (
            False,
            "INVALID_METRIC_ANCHOR: Derived scale factor to metres is non-finite or non-positive",
            None,
        )

    return True, None, scale_to_metres


@dataclass(frozen=True)
class ScaleAlignmentResult:
    """Outcome of metric depth scale alignment and validation."""

    is_aligned: bool
    scale_state: str  # "METRIC_SCALE_ALIGNED", "SFM_ALIGNED_UNMETRIC", "SYNTHETIC_SCALE_ALIGNED", "RELATIVE_UNSCALED", "INVALID"
    units: str = "UNSPECIFIED"  # "METRES", "COLMAP_SFM_UNITS", "RELATIVE_DISPARITY", "UNSPECIFIED"
    is_physically_metric: bool = False
    aligned_depth_map: np.ndarray | None = None
    scale: float | None = None
    shift: float | None = None
    scale_to_metres: float | None = None
    scale_uncertainty: float | None = None
    shift_uncertainty: float | None = None
    residual_units: str = "UNSPECIFIED"  # "METRES", "COLMAP_SFM_UNITS"
    residual_rms_error: float | None = None
    held_out_rms_error: float | None = None
    spatial_residual_std: float | None = None
    residual_rms_error_m: float | None = None  # Backward-compatible alias
    held_out_rms_error_m: float | None = None  # Backward-compatible alias
    spatial_residual_std_m: float | None = None  # Backward-compatible alias
    r_squared: float | None = None
    scale_source: str | None = None
    calibration_method: str | None = None
    point_count: int = 0
    inlier_count: int = 0
    held_out_count: int = 0
    metric_anchor: MetricAnchor | None = None
    rejection_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "is_aligned": self.is_aligned,
            "scale_state": self.scale_state,
            "units": self.units,
            "is_physically_metric": self.is_physically_metric,
            "scale": self.scale,
            "shift": self.shift,
            "scale_to_metres": self.scale_to_metres,
            "scale_uncertainty": self.scale_uncertainty,
            "shift_uncertainty": self.shift_uncertainty,
            "residual_units": self.residual_units,
            "residual_rms_error": self.residual_rms_error,
            "held_out_rms_error": self.held_out_rms_error,
            "spatial_residual_std": self.spatial_residual_std,
            "residual_rms_error_m": self.residual_rms_error_m,
            "held_out_rms_error_m": self.held_out_rms_error_m,
            "spatial_residual_std_m": self.spatial_residual_std_m,
            "r_squared": self.r_squared,
            "scale_source": self.scale_source,
            "calibration_method": self.calibration_method,
            "point_count": self.point_count,
            "inlier_count": self.inlier_count,
            "held_out_count": self.held_out_count,
            "metric_anchor": self.metric_anchor.to_dict() if self.metric_anchor else None,
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
    metric_anchor: MetricAnchor | None = None,
    min_inliers: int = 15,
    train_ratio: float = 0.7,
    allow_synthetic: bool = False,
) -> ScaleAlignmentResult:
    """Aligns a relative depth map to reference 3D coordinates with fit vs held-out evaluation.

    Strict Scale Contract:
    - If `metric_anchor` is provided with a valid physical measurement and reference points,
      the output is scaled to physical meters: units='METRES', scale_state='METRIC_SCALE_ALIGNED',
      is_physically_metric=True, scale_to_metres=lambda.
    - If `metric_anchor` has missing or malformed reference points or non-positive distance,
      the alignment is rejected with scale_state='INVALID'.
    - If `metric_anchor` is None, the output remains in arbitrary SfM gauge units:
      units='COLMAP_SFM_UNITS', scale_state='SFM_ALIGNED_UNMETRIC', is_physically_metric=False.
    - Synthetic anchors cannot establish physical metric accuracy and are rejected unless
      allow_synthetic=True, in which case is_physically_metric remains False.
    - Correspondences are split into fit (e.g. 70%) and held-out (30%) sets; held-out RMSE,
      held-out R^2, and quadrant-level spatial residual variation are evaluated. Weak held-out
      fits are rejected even if training R^2 is high.
    """
    is_valid, err = validate_depth_map(depth_map)
    if not is_valid:
        return ScaleAlignmentResult(
            is_aligned=False,
            scale_state="INVALID",
            units="RELATIVE_DISPARITY",
            is_physically_metric=False,
            rejection_reason=err,
        )

    # Validate metric anchor contract
    is_anchor_valid, anchor_err, scale_to_metres = validate_metric_anchor(
        metric_anchor,
        allow_synthetic=allow_synthetic,
    )
    if not is_anchor_valid:
        return ScaleAlignmentResult(
            is_aligned=False,
            scale_state="INVALID",
            units="RELATIVE_DISPARITY",
            is_physically_metric=False,
            metric_anchor=metric_anchor,
            rejection_reason=anchor_err,
        )

    h, w = depth_map.shape

    # Extract 2D and camera-frame depth pairs
    d_rels: list[float] = []
    z_trues: list[float] = []
    u_pts: list[float] = []
    v_pts: list[float] = []

    if pixel_coords_2d is not None and len(pixel_coords_2d) == len(reference_points_3d):
        # Direct 2D correspondences provided
        for (u, v), P_w in zip(pixel_coords_2d, reference_points_3d):
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < w and 0 <= vi < h:
                p_cam = camera.extrinsics.world_to_camera(Vec3(P_w[0], P_w[1], P_w[2]))
                if p_cam.z > 0.1:
                    d_rels.append(float(depth_map[vi, ui]))
                    z_trues.append(float(p_cam.z))
                    u_pts.append(float(u))
                    v_pts.append(float(v))
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
                u_pts.append(float(uv[0]))
                v_pts.append(float(uv[1]))

    if len(d_rels) < min_inliers:
        return ScaleAlignmentResult(
            is_aligned=False,
            scale_state="RELATIVE_UNSCALED",
            units="RELATIVE_DISPARITY",
            is_physically_metric=False,
            point_count=len(d_rels),
            metric_anchor=metric_anchor,
            rejection_reason=f"INSUFFICIENT_CORRESPONDENCES: {len(d_rels)} valid points (minimum {min_inliers} required)",
        )

    arr_d = np.array(d_rels, dtype=np.float64)
    arr_z = np.array(z_trues, dtype=np.float64)
    arr_u = np.array(u_pts, dtype=np.float64)
    arr_v = np.array(v_pts, dtype=np.float64)

    # Deterministic train/test split for held-out evaluation
    n_total = len(arr_d)
    rng = np.random.RandomState(42)
    shuffled_idx = rng.permutation(n_total)
    n_train = max(int(round(n_total * train_ratio)), min(10, n_total - 1))
    train_idx = shuffled_idx[:n_train]
    test_idx = shuffled_idx[n_train:]

    d_train, z_train = arr_d[train_idx], arr_z[train_idx]
    d_test, z_test = arr_d[test_idx], arr_z[test_idx]

    # 1. Fit affine disparity model on train set: 1 / z = s * d_rel + t
    inv_z_train = 1.0 / np.clip(z_train, 0.05, 100.0)
    A_disp_train = np.vstack([d_train, np.ones(len(d_train))]).T
    s_disp, t_disp = np.linalg.lstsq(A_disp_train, inv_z_train, rcond=None)[0]
    inv_z_pred_train = s_disp * d_train + t_disp
    z_pred_disp_train = 1.0 / np.clip(inv_z_pred_train, 1e-4, 10.0)

    res_disp_train = np.abs(z_pred_disp_train - z_train)
    inlier_mask_disp_train = res_disp_train < np.percentile(res_disp_train, 85)
    rms_disp_train = float(np.sqrt(np.mean((z_pred_disp_train[inlier_mask_disp_train] - z_train[inlier_mask_disp_train]) ** 2)))
    ss_tot_train = float(np.sum((z_train - np.mean(z_train)) ** 2))
    ss_res_disp = float(np.sum((z_train - z_pred_disp_train) ** 2))
    r2_disp = (1.0 - (ss_res_disp / max(1e-6, ss_tot_train))) if s_disp > 0 else -1.0

    # 2. Fit linear depth model on train set: z = s * d_rel + t
    A_lin_train = np.vstack([d_train, np.ones(len(d_train))]).T
    s_lin, t_lin = np.linalg.lstsq(A_lin_train, z_train, rcond=None)[0]
    z_pred_lin_train = s_lin * d_train + t_lin
    res_lin_train = np.abs(z_pred_lin_train - z_train)
    inlier_mask_lin_train = res_lin_train < np.percentile(res_lin_train, 85)
    rms_lin_train = float(np.sqrt(np.mean((z_pred_lin_train[inlier_mask_lin_train] - z_train[inlier_mask_lin_train]) ** 2)))
    ss_res_lin = float(np.sum((z_train - z_pred_lin_train) ** 2))
    r2_lin = (1.0 - (ss_res_lin / max(1e-6, ss_tot_train))) if s_lin > 0 else -1.0

    # Choose model with better goodness of fit
    use_disp = (r2_disp > r2_lin and s_disp > 0) or (s_lin <= 0 and s_disp > 0)
    best_r2 = max(r2_disp, r2_lin)

    if best_r2 < 0.30:
        return ScaleAlignmentResult(
            is_aligned=False,
            scale_state="RELATIVE_UNSCALED",
            units="RELATIVE_DISPARITY",
            is_physically_metric=False,
            point_count=n_total,
            inlier_count=int(np.sum(inlier_mask_disp_train if use_disp else inlier_mask_lin_train)),
            held_out_count=len(test_idx),
            r_squared=round(best_r2, 4),
            metric_anchor=metric_anchor,
            rejection_reason=f"POOR_SCALE_ALIGNMENT_FIT: R^2={best_r2:.3f} < 0.30",
        )

    # Compute parameter uncertainties via OLS covariance
    if use_disp:
        aligned_raw = 1.0 / np.clip(s_disp * depth_map + t_disp, 0.02, 10.0)
        s_val, t_val = float(s_disp), float(t_disp)
        rms_val = rms_disp_train
        r2_val = r2_disp
        inliers = int(np.sum(inlier_mask_disp_train))
        method = "affine_disparity_regression"
        A_used = A_disp_train
        var_res = rms_disp_train ** 2
    else:
        aligned_raw = np.clip(s_lin * depth_map + t_lin, 0.1, 100.0)
        s_val, t_val = float(s_lin), float(t_lin)
        rms_val = rms_lin_train
        r2_val = r2_lin
        inliers = int(np.sum(inlier_mask_lin_train))
        method = "linear_depth_regression"
        A_used = A_lin_train
        var_res = rms_lin_train ** 2

    try:
        cov_params = np.linalg.pinv(A_used.T @ A_used) * var_res
        s_unc = float(np.sqrt(max(0.0, cov_params[0, 0])))
        t_unc = float(np.sqrt(max(0.0, cov_params[1, 1])))
    except Exception:
        s_unc, t_unc = None, None

    # Evaluate on held-out validation set
    if len(d_test) > 0:
        if use_disp:
            z_pred_test = 1.0 / np.clip(s_disp * d_test + t_disp, 1e-4, 10.0)
        else:
            z_pred_test = s_lin * d_test + t_lin
        held_out_rms = float(np.sqrt(np.mean((z_pred_test - z_test) ** 2)))
        ss_tot_test = float(np.sum((z_test - np.mean(z_test)) ** 2))
        ss_res_test = float(np.sum((z_test - z_pred_test) ** 2))
        held_out_r2 = (1.0 - (ss_res_test / max(1e-6, ss_tot_test))) if len(d_test) >= 3 else 1.0
    else:
        held_out_rms = rms_val
        held_out_r2 = best_r2

    # Reject weak fits using held-out performance, not training R^2 alone
    if len(d_test) >= 3 and (held_out_r2 < 0.10 or (held_out_rms > 2.5 * rms_val and held_out_rms > 0.5)):
        return ScaleAlignmentResult(
            is_aligned=False,
            scale_state="RELATIVE_UNSCALED",
            units="RELATIVE_DISPARITY",
            is_physically_metric=False,
            point_count=n_total,
            inlier_count=inliers,
            held_out_count=len(test_idx),
            r_squared=round(best_r2, 4),
            residual_rms_error=round(rms_val, 4),
            residual_rms_error_m=round(rms_val, 4),
            held_out_rms_error=round(held_out_rms, 4),
            held_out_rms_error_m=round(held_out_rms, 4),
            metric_anchor=metric_anchor,
            rejection_reason=f"POOR_HELD_OUT_EVALUATION: Held-out R^2={held_out_r2:.3f} or held-out RMS={held_out_rms:.3f} indicates severe overfit/generalization failure",
        )

    # Spatial residual variation across 4 quadrants
    quadrant_residuals: list[float] = []
    if use_disp:
        z_pred_all = 1.0 / np.clip(s_disp * arr_d + t_disp, 1e-4, 10.0)
    else:
        z_pred_all = s_lin * arr_d + t_lin
    res_all = np.abs(z_pred_all - arr_z)

    mid_u, mid_v = w / 2.0, h / 2.0
    q_masks = [
        (arr_u < mid_u) & (arr_v < mid_v),
        (arr_u >= mid_u) & (arr_v < mid_v),
        (arr_u < mid_u) & (arr_v >= mid_v),
        (arr_u >= mid_u) & (arr_v >= mid_v),
    ]
    for qm in q_masks:
        if np.sum(qm) >= 2:
            quadrant_residuals.append(float(np.mean(res_all[qm])))
    spatial_std = float(np.std(quadrant_residuals)) if len(quadrant_residuals) >= 2 else 0.0

    # Metric Anchor Scaling Check
    if metric_anchor is not None:
        assert scale_to_metres is not None
        metric_scale_factor = scale_to_metres
        aligned_map = (aligned_raw * metric_scale_factor).astype(np.float32)
        out_units = "METRES"
        is_metric = (metric_anchor.anchor_type != "synthetic_ground_truth")
        out_state = "METRIC_SCALE_ALIGNED" if is_metric else "SYNTHETIC_SCALE_ALIGNED"
        res_units = "METRES"
        final_scale = round(s_val * metric_scale_factor, 6)
        final_shift = round(t_val * metric_scale_factor, 6)
        final_rms = round(rms_val * metric_scale_factor, 4)
        final_held_out = round(held_out_rms * metric_scale_factor, 4)
        final_spatial = round(spatial_std * metric_scale_factor, 4)
        final_s_unc = round(s_unc * metric_scale_factor, 6) if s_unc is not None else None
        final_t_unc = round(t_unc * metric_scale_factor, 6) if t_unc is not None else None
    else:
        metric_scale_factor = 1.0
        aligned_map = aligned_raw.astype(np.float32)
        out_units = "COLMAP_SFM_UNITS"
        out_state = "SFM_ALIGNED_UNMETRIC"
        is_metric = False
        res_units = "COLMAP_SFM_UNITS"
        final_scale = round(s_val, 6)
        final_shift = round(t_val, 6)
        final_rms = round(rms_val, 4)
        final_held_out = round(held_out_rms, 4)
        final_spatial = round(spatial_std, 4)
        final_s_unc = round(s_unc, 6) if s_unc is not None else None
        final_t_unc = round(t_unc, 6) if t_unc is not None else None
        scale_to_metres = None

    return ScaleAlignmentResult(
        is_aligned=True,
        scale_state=out_state,
        units=out_units,
        is_physically_metric=is_metric,
        aligned_depth_map=aligned_map,
        scale=final_scale,
        shift=final_shift,
        scale_to_metres=round(scale_to_metres, 6) if scale_to_metres is not None else None,
        scale_uncertainty=final_s_unc,
        shift_uncertainty=final_t_unc,
        residual_units=res_units,
        residual_rms_error=final_rms,
        held_out_rms_error=final_held_out,
        spatial_residual_std=final_spatial,
        residual_rms_error_m=final_rms,
        held_out_rms_error_m=final_held_out,
        spatial_residual_std_m=final_spatial,
        r_squared=round(r2_val, 4),
        scale_source=scale_source,
        calibration_method=method,
        point_count=n_total,
        inlier_count=inliers,
        held_out_count=len(test_idx),
        metric_anchor=metric_anchor,
    )
