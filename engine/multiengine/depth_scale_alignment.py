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

    Without a verified metric anchor, reconstructions derived from SfM (e.g. COLMAP)
    remain in arbitrary gauge coordinates (COLMAP_SFM_UNITS) and MUST NOT be labeled as METRES.
    """

    anchor_type: str  # "known_distance", "laser_survey", "lidar_ground_truth", "synthetic_ground_truth"
    measured_distance_m: float  # Physical measured distance in meters
    uncertainty_m: float = 0.01  # Measurement uncertainty in meters
    reference_point_a: tuple[float, float, float] | None = None  # 3D coordinate in unscaled world frame
    reference_point_b: tuple[float, float, float] | None = None  # 3D coordinate in unscaled world frame
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "anchor_type": self.anchor_type,
            "measured_distance_m": self.measured_distance_m,
            "uncertainty_m": self.uncertainty_m,
            "reference_point_a": list(self.reference_point_a) if self.reference_point_a else None,
            "reference_point_b": list(self.reference_point_b) if self.reference_point_b else None,
            "notes": self.notes,
        }


#: Anchor kinds that are an INDEPENDENT physical measurement. Anything else -- in particular synthetic ground truth -- can never
#: establish metres.
PHYSICAL_ANCHOR_TYPES = (
    "known_distance", "laser_survey", "lidar_ground_truth", "laser", "laser_distance_meter", "total_station", "tape_measure",
    "calibration_target",
)
#: Conservative acceptance gates for the held-out check. NOT benchmark-established: they are explicit parameters of
#: ``align_relative_depth_to_metric`` so a caller (and any report) sees the value that gated the result.
DEFAULT_MAX_HELD_OUT_REL_RMSE = 0.25
DEFAULT_MIN_HELD_OUT = 5


def validate_metric_anchor(
    anchor: MetricAnchor, reference_points_3d: Sequence[tuple[float, float, float]] = ()
) -> str | None:
    """Reason the anchor cannot establish physical metres, or None when it can.

    A metric anchor needs: a positive finite measured distance, a finite non-negative uncertainty smaller than the distance, an
    independent physical measurement type (never synthetic), TWO distinct finite reference points in the reconstruction's own
    coordinate frame, and each of those points must resolve to the reconstruction (lie on one of its 3D points) -- a point that
    corresponds to nothing in the reconstructed frame is a typo, not a measurement of that frame.
    """
    d = anchor.measured_distance_m
    if not (isinstance(d, (int, float)) and np.isfinite(d)) or d <= 0.0:
        return "INVALID_METRIC_ANCHOR: measured_distance_m must be > 0 and finite"
    u = anchor.uncertainty_m
    if not (isinstance(u, (int, float)) and np.isfinite(u)) or u < 0.0 or u >= d:
        return "INVALID_METRIC_ANCHOR: uncertainty_m must be finite, >= 0 and smaller than the measured distance"
    if "synthetic" in str(anchor.anchor_type).lower():
        return "SYNTHETIC_ANCHOR: synthetic ground truth cannot establish physical metres"
    pa, pb = anchor.reference_point_a, anchor.reference_point_b
    if pa is None or pb is None:
        return "ANCHOR_REFERENCE_POINTS_MISSING: both reference points are required to tie the measurement to the reconstruction"
    try:
        a, b = np.asarray(pa, dtype=np.float64), np.asarray(pb, dtype=np.float64)
    except (TypeError, ValueError):
        return "INVALID_METRIC_ANCHOR: reference points are not numeric"
    if a.shape != (3,) or b.shape != (3,) or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return "INVALID_METRIC_ANCHOR: reference points must be finite 3-vectors"
    if anchor.anchor_type not in PHYSICAL_ANCHOR_TYPES:
        return f"UNVERIFIED_ANCHOR_TYPE: {anchor.anchor_type!r} is not an independent physical measurement type {PHYSICAL_ANCHOR_TYPES}"
    cloud = np.asarray(reference_points_3d, dtype=np.float64).reshape(-1, 3) if len(reference_points_3d) else np.zeros((0, 3))
    extent = float(np.linalg.norm(cloud.max(axis=0) - cloud.min(axis=0))) if len(cloud) else 0.0
    if float(np.linalg.norm(a - b)) <= max(1e-9, 1e-6 * extent):
        return "INVALID_METRIC_ANCHOR: Distance between reference points is zero"
    if len(cloud):
        tol = max(1e-9, 0.02 * extent)
        for name, p in (("a", a), ("b", b)):
            if float(np.min(np.linalg.norm(cloud - p, axis=1))) > tol:
                return f"ANCHOR_POINT_NOT_IN_RECONSTRUCTION: reference point {name} does not coincide with any reconstructed 3D point"
    return None


@dataclass(frozen=True)
class ScaleAlignmentResult:
    """Outcome of metric depth scale alignment and validation."""

    is_aligned: bool
    scale_state: str  # "METRIC_SCALE_ALIGNED", "SFM_ALIGNED_UNMETRIC", "RELATIVE_UNSCALED", "INVALID"
    units: str = "UNSPECIFIED"  # "METRES", "COLMAP_SFM_UNITS", "RELATIVE_DISPARITY", "UNSPECIFIED"
    is_physically_metric: bool = False
    aligned_depth_map: np.ndarray | None = None
    scale: float | None = None
    shift: float | None = None
    scale_uncertainty: float | None = None
    shift_uncertainty: float | None = None
    # Errors are in ``error_units`` (COLMAP_SFM_UNITS unless a validated physical anchor made them METRES) -- never silently metres.
    residual_rms_error: float | None = None
    held_out_rms_error: float | None = None
    spatial_residual_std: float | None = None
    error_units: str | None = None
    held_out_relative_rmse: float | None = None  # unit-free: sqrt(mean(((pred - true) / true)^2)) on held-out points
    max_held_out_relative_rmse: float | None = None  # the gate that was applied (a parameter, not a benchmark result)
    region_residuals: tuple[dict, ...] = ()  # per image quadrant: {"region", "n", "mean_abs_residual" | None, "status"}
    anchor_relative_uncertainty: float | None = None  # uncertainty_m / measured_distance_m, folded into the scale uncertainty
    metric_scale_factor: float | None = None  # SfM units -> metres (None unless a validated anchor was applied)
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
            "scale_uncertainty": self.scale_uncertainty,
            "shift_uncertainty": self.shift_uncertainty,
            "residual_rms_error": self.residual_rms_error,
            "held_out_rms_error": self.held_out_rms_error,
            "spatial_residual_std": self.spatial_residual_std,
            "error_units": self.error_units,
            "held_out_relative_rmse": self.held_out_relative_rmse,
            "max_held_out_relative_rmse": self.max_held_out_relative_rmse,
            "region_residuals": [dict(r) for r in self.region_residuals],
            "anchor_relative_uncertainty": self.anchor_relative_uncertainty,
            "metric_scale_factor": self.metric_scale_factor,
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
    max_held_out_rel_rmse: float = DEFAULT_MAX_HELD_OUT_REL_RMSE,
    min_held_out: int = DEFAULT_MIN_HELD_OUT,
) -> ScaleAlignmentResult:
    """Aligns a relative depth map to reference 3D coordinates with fit vs held-out evaluation.

    Strict Scale Contract:
    - If `metric_anchor` is provided with a valid physical measurement, the output is scaled
      to physical meters: units='METRES', scale_state='METRIC_SCALE_ALIGNED', is_physically_metric=True.
    - If `metric_anchor` is None, the output remains in arbitrary SfM gauge units:
      units='COLMAP_SFM_UNITS', scale_state='SFM_ALIGNED_UNMETRIC', is_physically_metric=False.
    - Correspondences are split into fit (e.g. 70%) and held-out (30%) sets; held-out RMSE
      and quadrant-level spatial residual variation are recorded.
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

    if metric_anchor is not None:
        anchor_problem = validate_metric_anchor(metric_anchor, reference_points_3d)
        if anchor_problem is not None:  # an unverified, synthetic or incomplete anchor never becomes metres
            return ScaleAlignmentResult(
                is_aligned=False,
                scale_state="INVALID",
                units="RELATIVE_DISPARITY",
                is_physically_metric=False,
                metric_anchor=metric_anchor,
                rejection_reason=anchor_problem,
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

    # Held-out validation is an ACCEPTANCE GATE: a fit that only looks good on the points it was trained on is refused, and a fit
    # that cannot be validated (too few held-out points) is refused too -- training error is never substituted for held-out error.
    if len(d_test) < min_held_out:
        return ScaleAlignmentResult(
            is_aligned=False, scale_state="RELATIVE_UNSCALED", units="RELATIVE_DISPARITY", is_physically_metric=False,
            point_count=n_total, inlier_count=inliers, held_out_count=len(d_test), r_squared=round(r2_val, 4),
            metric_anchor=metric_anchor,
            rejection_reason=f"INSUFFICIENT_HELD_OUT: {len(d_test)} held-out point(s) (minimum {min_held_out}); the fit cannot be validated",
        )
    if use_disp:
        z_pred_test = 1.0 / np.clip(s_disp * d_test + t_disp, 1e-4, 10.0)
    else:
        z_pred_test = s_lin * d_test + t_lin
    held_out_rms = float(np.sqrt(np.mean((z_pred_test - z_test) ** 2)))
    held_out_rel = float(np.sqrt(np.mean(((z_pred_test - z_test) / np.clip(z_test, 0.05, None)) ** 2)))
    if not np.isfinite(held_out_rel) or held_out_rel > max_held_out_rel_rmse:
        return ScaleAlignmentResult(
            is_aligned=False, scale_state="RELATIVE_UNSCALED", units="RELATIVE_DISPARITY", is_physically_metric=False,
            point_count=n_total, inlier_count=inliers, held_out_count=len(d_test), r_squared=round(r2_val, 4),
            held_out_relative_rmse=None if not np.isfinite(held_out_rel) else round(held_out_rel, 4),
            max_held_out_relative_rmse=max_held_out_rel_rmse, metric_anchor=metric_anchor,
            rejection_reason=(
                f"POOR_HELD_OUT_FIT: held-out relative RMSE {held_out_rel:.3f} > {max_held_out_rel_rmse} "
                f"(training R^2={r2_val:.3f} does not validate the fit)"
            ),
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
    def _fit_predict(d_fit: np.ndarray, z_fit: np.ndarray, d_eval: np.ndarray) -> np.ndarray:
        """Refit the SAME model family that was selected, on the given points only, and predict depth for ``d_eval``."""
        if use_disp:
            s, t = np.linalg.lstsq(np.vstack([d_fit, np.ones(len(d_fit))]).T, 1.0 / np.clip(z_fit, 0.05, 100.0), rcond=None)[0]
            return 1.0 / np.clip(s * d_eval + t, 1e-4, 10.0)
        s, t = np.linalg.lstsq(np.vstack([d_fit, np.ones(len(d_fit))]).T, z_fit, rcond=None)[0]
        return s * d_eval + t

    # Spatially separated validation: each image quadrant is held out in turn (fit on every point OUTSIDE it, score the points
    # INSIDE it), so a fit that is good on average but wrong in one part of the image cannot hide behind the global statistics.
    region_records: list[dict] = []
    for name, qm in zip(("top_left", "top_right", "bottom_left", "bottom_right"), q_masks, strict=True):
        n_q, n_out = int(np.sum(qm)), int(np.sum(~qm))
        if n_q < 2:  # a region with (almost) no correspondences is not validated; that is recorded, not hidden
            region_records.append({"region": name, "n": n_q, "mean_abs_residual": None, "leave_region_out_rel_rmse": None,
                                   "status": "unvalidated_too_few_points"})
            continue
        mean_res = float(np.mean(res_all[qm]))
        quadrant_residuals.append(mean_res)
        loro: float | None = None
        status = "validated"
        if n_out >= 4:
            pred = _fit_predict(arr_d[~qm], arr_z[~qm], arr_d[qm])
            loro = float(np.sqrt(np.mean(((pred - arr_z[qm]) / np.clip(arr_z[qm], 0.05, None)) ** 2)))
            if not np.isfinite(loro) or loro > max_held_out_rel_rmse:
                status = "local_fit_failure"
        else:
            status = "unvalidated_too_few_points"
        region_records.append({"region": name, "n": n_q, "mean_abs_residual": mean_res,
                               "leave_region_out_rel_rmse": None if loro is None or not np.isfinite(loro) else round(loro, 4),
                               "status": status})
    spatial_std = float(np.std(quadrant_residuals)) if len(quadrant_residuals) >= 2 else 0.0
    failed_regions = [r["region"] for r in region_records if r["status"] == "local_fit_failure"]
    if failed_regions:  # good on average, wrong somewhere: refused, with the per-region evidence attached
        return ScaleAlignmentResult(
            is_aligned=False, scale_state="RELATIVE_UNSCALED", units="RELATIVE_DISPARITY", is_physically_metric=False,
            point_count=n_total, inlier_count=inliers, held_out_count=len(d_test), r_squared=round(r2_val, 4),
            held_out_relative_rmse=round(held_out_rel, 4), max_held_out_relative_rmse=max_held_out_rel_rmse,
            region_residuals=tuple(region_records), metric_anchor=metric_anchor,
            rejection_reason=f"LOCAL_FIT_FAILURE: leave-region-out error exceeds {max_held_out_rel_rmse} in {failed_regions}",
        )

    # Metric anchor: reached ONLY with a validated anchor (two distinct finite reference points that resolve to the reconstruction,
    # a finite positive distance, an independent physical measurement type). The factor is measured / reconstructed separation.
    metric_scale_factor: float | None = None
    anchor_rel_unc: float | None = None
    scale_unc_out = s_unc
    if metric_anchor is not None:
        dist_sfm = float(np.linalg.norm(np.asarray(metric_anchor.reference_point_a, dtype=np.float64)
                                        - np.asarray(metric_anchor.reference_point_b, dtype=np.float64)))
        metric_scale_factor = metric_anchor.measured_distance_m / dist_sfm
        anchor_rel_unc = metric_anchor.uncertainty_m / metric_anchor.measured_distance_m
        if s_unc is not None:  # fit uncertainty and the anchor's own relative uncertainty combine in quadrature
            scale_unc_out = float(np.sqrt((s_unc * metric_scale_factor) ** 2 + (abs(s_val) * metric_scale_factor * anchor_rel_unc) ** 2))
        k = metric_scale_factor
        aligned_map = (aligned_raw * k).astype(np.float32)
        out_units, out_state, is_metric, err_units = "METRES", "METRIC_SCALE_ALIGNED", True, "METRES"
        final_scale, final_shift = round(s_val * k, 6), round(t_val * k, 6)
        final_rms, final_held_out, final_spatial = round(rms_val * k, 4), round(held_out_rms * k, 4), round(spatial_std * k, 4)
        region_records = [
            {**r, "mean_abs_residual": None if r["mean_abs_residual"] is None else round(r["mean_abs_residual"] * k, 6)}
            for r in region_records
        ]
    else:
        aligned_map = aligned_raw.astype(np.float32)
        out_units, out_state, is_metric, err_units = "COLMAP_SFM_UNITS", "SFM_ALIGNED_UNMETRIC", False, "COLMAP_SFM_UNITS"
        final_scale, final_shift = round(s_val, 6), round(t_val, 6)
        final_rms, final_held_out, final_spatial = round(rms_val, 4), round(held_out_rms, 4), round(spatial_std, 4)
        region_records = [
            {**r, "mean_abs_residual": None if r["mean_abs_residual"] is None else round(r["mean_abs_residual"], 6)}
            for r in region_records
        ]

    return ScaleAlignmentResult(
        is_aligned=True,
        scale_state=out_state,
        units=out_units,
        is_physically_metric=is_metric,
        aligned_depth_map=aligned_map,
        scale=final_scale,
        shift=final_shift,
        scale_uncertainty=round(scale_unc_out, 6) if scale_unc_out is not None else None,
        shift_uncertainty=round(t_unc * (metric_scale_factor or 1.0), 6) if t_unc is not None else None,
        residual_rms_error=final_rms,
        held_out_rms_error=final_held_out,
        spatial_residual_std=final_spatial,
        error_units=err_units,
        held_out_relative_rmse=round(held_out_rel, 4),
        max_held_out_relative_rmse=max_held_out_rel_rmse,
        region_residuals=tuple(region_records),
        anchor_relative_uncertainty=anchor_rel_unc,
        metric_scale_factor=metric_scale_factor,
        r_squared=round(r2_val, 4),
        scale_source=scale_source,
        calibration_method=method,
        point_count=n_total,
        inlier_count=inliers,
        held_out_count=len(test_idx),
        metric_anchor=metric_anchor,
    )
