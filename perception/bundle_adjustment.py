"""Bundle adjustment over 3D points and camera poses.

Jointly optimises camera poses (rotation + translation) and 3D point
positions to minimise reprojection error across all views. Built on the
exact sparse Levenberg-Marquardt solver (reconstruction/spatial/sparse_lm.py)
and the camera model (world_ir.camera.PinholeCamera).

Conventions:

* Camera pose is world->camera: ``x_cam = R @ X_world + t``.
* A 3D point ``X`` projects to pixel ``(u, v)`` via
  ``u = fx * X_cam.x / X_cam.z + cx, v = fy * X_cam.y / X_cam.z + cy``.
* Each observation touches one camera's 6 DOF (3 rotation + 3 translation)
  and one 3D point's 3 DOF. The Jacobian is sparse: each row has at most 9
  nonzeros.
* Gauge freedom: camera 0's pose is FIXED (not optimised). All other cameras
  and points are optimised relative to it.
* Points behind a camera (Z <= 0) are excluded from that view's residual.
* Robust loss (Cauchy) defaults; outliers are down-weighted, never dropped.

The result is a refined camera pose graph + 3D point cloud with per-observation
reprojection errors and per-camera posterior covariances. No accuracy claim is
made -- this is an internal consistency optimisation, not a survey result.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import splu

from perception.camera_pose_graph import rotation_exp, rotation_log
from reconstruction.spatial.sparse_lm import solve_sparse_lm, LMResult
from world_ir.camera import CameraIntrinsics, PinholeCamera, quat_to_matrix


class BundleAdjustmentError(ValueError):
    pass


@dataclass(frozen=True)
class CameraObservation:
    """One 2D observation of one 3D point in one camera."""
    camera_id: str
    point_id: str
    u: float          # observed pixel column
    v: float          # observed pixel row
    sigma_px: float   # pixel noise estimate, > 0


@dataclass(frozen=True)
class CameraParams:
    """One camera's intrinsics and current pose estimate."""
    camera_id: str
    intrinsics: CameraIntrinsics
    # world->camera rotation (3x3) and translation (3,)
    R: np.ndarray
    t: np.ndarray


@dataclass(frozen=True)
class Point3D:
    """One 3D point estimate."""
    point_id: str
    position: np.ndarray   # (3,) in world coordinates


@dataclass
class CameraResult:
    """Refined result for one camera."""
    camera_id: str
    R: np.ndarray
    t: np.ndarray
    cov: np.ndarray | None      # 6x6 over (rot_log(3), t(3)); None for camera 0 (fixed gauge)
    n_observations: int
    median_reprojection_px: float


@dataclass
class PointResult:
    """Refined result for one 3D point."""
    point_id: str
    position: np.ndarray
    cov: np.ndarray | None      # 3x3
    n_observations: int
    median_reprojection_px: float


@dataclass
class BundleAdjustmentResult:
    """Full BA result."""
    status: str                 # "optimised" | "failed" | "unavailable"
    cameras: dict[str, CameraResult]
    points: dict[str, PointResult]
    observations: dict[str, float]   # (camera_id, point_id) -> reprojection_px
    median_reprojection_px: float
    n_cameras: int
    n_points: int
    n_observations: int
    n_kept: int               # observations kept after visibility filtering
    seconds: float
    nfev: int
    optimality: float
    message: str
    lm_result: LMResult | None


def _project(K: CameraIntrinsics, X_cam: np.ndarray) -> tuple[float, float] | None:
    """Project a camera-frame point to pixels. None when behind the camera."""
    x, y, z = float(X_cam[0]), float(X_cam[1]), float(X_cam[2])
    if z <= 1e-6:
        return None
    return (K.fx * x / z + K.cx, K.fy * y / z + K.cy)


def _rot_params_from_R(R: np.ndarray) -> np.ndarray:
    """Rotation vector (so(3)) from a rotation matrix."""
    return rotation_log(R)


def _R_from_rot_params(w: np.ndarray) -> np.ndarray:
    """Rotation matrix from a rotation vector."""
    return rotation_exp(w)


def build_bundle_problem(
    cameras: Mapping[str, CameraParams],
    points: Mapping[str, Point3D],
    observations: Sequence[CameraObservation],
) -> dict[str, Any]:
    """Assemble the sparse bundle adjustment problem.

    Camera 0 (first in sorted order) is the gauge: its pose is fixed and not
    part of the optimisation vector. All other cameras have 6 DOF each.
    """
    cam_ids = sorted(cameras)
    pt_ids = sorted(points)
    cam_idx = {cid: i for i, cid in enumerate(cam_ids)}
    pt_idx = {pid: i for i, pid in enumerate(pt_ids)}

    # Camera 0 is the gauge (fixed). Optimisable cameras: 1..N-1.
    opt_cam_ids = cam_ids[1:]  # exclude camera 0
    opt_cam_idx = {cid: i for i, cid in enumerate(opt_cam_ids)}
    n_opt_cam = len(opt_cam_ids)
    n_pt = len(pt_ids)

    # Parameters: 6 per optimisable camera + 3 per point
    cam_param_start = 0
    pt_param_start = 6 * n_opt_cam
    n_params = pt_param_start + 3 * n_pt

    # Initial parameter vector
    x0 = np.zeros(n_params)
    for cid, i in opt_cam_idx.items():
        cp = cameras[cid]
        x0[cam_param_start + 6 * i: cam_param_start + 6 * i + 3] = _rot_params_from_R(cp.R)
        x0[cam_param_start + 6 * i + 3: cam_param_start + 6 * i + 6] = cp.t
    for pid, i in pt_idx.items():
        x0[pt_param_start + 3 * i: pt_param_start + 3 * i + 3] = points[pid].position

    # Filter observations: both camera and point must exist
    kept: list[CameraObservation] = []
    for obs in observations:
        if obs.camera_id in cam_idx and obs.point_id in pt_idx:
            kept.append(obs)
    n_obs = len(kept)

    # Sparsity pattern: each obs produces 2 residual rows (u, v), each touching
    # 6 camera params + 3 point params = 9 cols. Camera 0 is fixed (no columns).
    n_residuals = 2 * n_obs
    obs_rows = np.empty(n_obs * 18, dtype=np.int64)
    obs_cols = np.empty(n_obs * 18, dtype=np.int64)
    for k, obs in enumerate(kept):
        pi = pt_idx[obs.point_id]
        pt_cols = pt_param_start + 3 * pi + np.arange(3)
        if obs.camera_id in opt_cam_idx:
            ci = opt_cam_idx[obs.camera_id]
            cam_cols = cam_param_start + 6 * ci + np.arange(6)
            has_cam_cols = True
        else:
            cam_cols = np.array([], dtype=np.int64)
            has_cam_cols = False
        # Row u (2*k)
        if has_cam_cols:
            obs_rows[18 * k:18 * k + 9] = 2 * k
            obs_cols[18 * k:18 * k + 6] = cam_cols
            obs_cols[18 * k + 6:18 * k + 9] = pt_cols
        else:
            # Camera 0 is fixed: only point columns
            obs_rows[18 * k:18 * k + 3] = 2 * k
            obs_cols[18 * k:18 * k + 3] = pt_cols
            obs_rows[18 * k + 3:18 * k + 9] = -1  # padding, unused
            obs_cols[18 * k + 3:18 * k + 9] = -1
        # Row v (2*k+1)
        if has_cam_cols:
            obs_rows[18 * k + 9:18 * k + 18] = 2 * k + 1
            obs_cols[18 * k + 9:18 * k + 15] = cam_cols
            obs_cols[18 * k + 15:18 * k + 18] = pt_cols
        else:
            obs_rows[18 * k + 9:18 * k + 12] = 2 * k + 1
            obs_cols[18 * k + 9:18 * k + 12] = pt_cols
            obs_rows[18 * k + 12:18 * k + 18] = -1
            obs_cols[18 * k + 12:18 * k + 18] = -1

    # Remove padding entries (-1 rows)
    valid = obs_rows >= 0
    obs_rows = obs_rows[valid]
    obs_cols = obs_cols[valid]

    shape = (n_residuals, n_params)

    def residuals(x: np.ndarray) -> np.ndarray:
        # Unpack optimisable camera poses
        Rs: dict[str, np.ndarray] = {}
        ts: dict[str, np.ndarray] = {}
        for cid, i in opt_cam_idx.items():
            off = cam_param_start + 6 * i
            Rs[cid] = _R_from_rot_params(x[off:off + 3])
            ts[cid] = x[off + 3:off + 6]
        # Camera 0 uses its fixed initial pose
        R0 = cameras[cam_ids[0]].R
        t0 = cameras[cam_ids[0]].t
        Rs[cam_ids[0]] = R0
        ts[cam_ids[0]] = t0
        # Unpack points
        Ps: dict[str, np.ndarray] = {}
        for pid, i in pt_idx.items():
            off = pt_param_start + 3 * i
            Ps[pid] = x[off:off + 3]

        r = np.zeros(n_residuals)
        for k, obs in enumerate(kept):
            R = Rs[obs.camera_id]
            t = ts[obs.camera_id]
            P = Ps[obs.point_id]
            X_cam = R @ P + t
            proj = _project(cameras[obs.camera_id].intrinsics, X_cam)
            if proj is None:
                r[2 * k] = 0.0
                r[2 * k + 1] = 0.0
                continue
            r[2 * k] = (proj[0] - obs.u) / obs.sigma_px
            r[2 * k + 1] = (proj[1] - obs.v) / obs.sigma_px
        return r

    def jacobian(x: np.ndarray) -> csr_matrix:
        Rs: dict[str, np.ndarray] = {}
        ts: dict[str, np.ndarray] = {}
        for cid, i in opt_cam_idx.items():
            off = cam_param_start + 6 * i
            Rs[cid] = _R_from_rot_params(x[off:off + 3])
            ts[cid] = x[off + 3:off + 6]
        R0 = cameras[cam_ids[0]].R
        t0 = cameras[cam_ids[0]].t
        Rs[cam_ids[0]] = R0
        ts[cam_ids[0]] = t0
        Ps: dict[str, np.ndarray] = {}
        for pid, i in pt_idx.items():
            off = pt_param_start + 3 * i
            Ps[pid] = x[off:off + 3]

        n_valid = len(obs_rows)
        vals = np.zeros(n_valid)
        fill = 0
        for k, obs in enumerate(kept):
            R = Rs[obs.camera_id]
            t = ts[obs.camera_id]
            P = Ps[obs.point_id]
            K = cameras[obs.camera_id].intrinsics
            X_cam = R @ P + t
            x_c, y_c, z_c = float(X_cam[0]), float(X_cam[1]), float(X_cam[2])
            if z_c <= 1e-6:
                fill += 18 if obs.camera_id in opt_cam_idx else 6
                continue
            inv_z = 1.0 / z_c
            inv_z2 = inv_z * inv_z
            du_dx = K.fx * inv_z
            du_dz = -K.fx * x_c * inv_z2
            dv_dy = K.fy * inv_z
            dv_dz = -K.fy * y_c * inv_z2

            skew_P = np.array([[0.0, -P[2], P[1]],
                               [P[2], 0.0, -P[0]],
                               [-P[1], P[0], 0.0]])
            dXc_drot = R @ skew_P
            dXc_dt = R
            dXc_dP = R

            dproj_dXc_u = np.array([[du_dx, 0.0, du_dz]])
            dproj_dXc_v = np.array([[0.0, dv_dy, dv_dz]])
            sig = obs.sigma_px

            if obs.camera_id in opt_cam_idx:
                # Row u (2*k): 9 values
                vals[fill] = float((dproj_dXc_u @ dXc_drot)[0, 0]) / sig
                vals[fill + 1] = float((dproj_dXc_u @ dXc_drot)[0, 1]) / sig
                vals[fill + 2] = float((dproj_dXc_u @ dXc_drot)[0, 2]) / sig
                vals[fill + 3] = float((dproj_dXc_u @ dXc_dt)[0, 0]) / sig
                vals[fill + 4] = float((dproj_dXc_u @ dXc_dt)[0, 1]) / sig
                vals[fill + 5] = float((dproj_dXc_u @ dXc_dt)[0, 2]) / sig
                vals[fill + 6] = float((dproj_dXc_u @ dXc_dP)[0, 0]) / sig
                vals[fill + 7] = float((dproj_dXc_u @ dXc_dP)[0, 1]) / sig
                vals[fill + 8] = float((dproj_dXc_u @ dXc_dP)[0, 2]) / sig
                # Row v (2*k+1): 9 values
                vals[fill + 9] = float((dproj_dXc_v @ dXc_drot)[0, 0]) / sig
                vals[fill + 10] = float((dproj_dXc_v @ dXc_drot)[0, 1]) / sig
                vals[fill + 11] = float((dproj_dXc_v @ dXc_drot)[0, 2]) / sig
                vals[fill + 12] = float((dproj_dXc_v @ dXc_dt)[0, 0]) / sig
                vals[fill + 13] = float((dproj_dXc_v @ dXc_dt)[0, 1]) / sig
                vals[fill + 14] = float((dproj_dXc_v @ dXc_dt)[0, 2]) / sig
                vals[fill + 15] = float((dproj_dXc_v @ dXc_dP)[0, 0]) / sig
                vals[fill + 16] = float((dproj_dXc_v @ dXc_dP)[0, 1]) / sig
                vals[fill + 17] = float((dproj_dXc_v @ dXc_dP)[0, 2]) / sig
                fill += 18
            else:
                # Camera 0 fixed: only point columns (3 per row)
                vals[fill] = float((dproj_dXc_u @ dXc_dP)[0, 0]) / sig
                vals[fill + 1] = float((dproj_dXc_u @ dXc_dP)[0, 1]) / sig
                vals[fill + 2] = float((dproj_dXc_u @ dXc_dP)[0, 2]) / sig
                vals[fill + 3] = float((dproj_dXc_v @ dXc_dP)[0, 0]) / sig
                vals[fill + 4] = float((dproj_dXc_v @ dXc_dP)[0, 1]) / sig
                vals[fill + 5] = float((dproj_dXc_v @ dXc_dP)[0, 2]) / sig
                fill += 6

        return csr_matrix((vals, (obs_rows, obs_cols)), shape=shape)

    return {
        "x0": x0,
        "n_params": n_params,
        "residuals": residuals,
        "jacobian": jacobian,
        "shape": shape,
        "kept": kept,
        "cam_ids": cam_ids,
        "pt_ids": pt_ids,
        "cam_idx": cam_idx,
        "pt_idx": pt_idx,
        "opt_cam_idx": opt_cam_idx,
        "cam_param_start": cam_param_start,
        "pt_param_start": pt_param_start,
        "n_obs": n_obs,
        "n_residuals": n_residuals,
        "gauge_camera_id": cam_ids[0],
    }


def run_bundle_adjustment(
    cameras: Mapping[str, CameraParams],
    points: Mapping[str, Point3D],
    observations: Sequence[CameraObservation],
    *,
    loss: str = "cauchy",
    f_scale: float = 1.0,
    max_nfev: int = 100,
) -> BundleAdjustmentResult:
    """Run bundle adjustment: optimise camera poses and 3D point positions.

    Camera 0 (first in sorted order) is fixed as the gauge. Returns a
    BundleAdjustmentResult with refined poses, points, and per-observation
    reprojection errors.
    """
    t0 = time.perf_counter()

    if not cameras or not points or not observations:
        return BundleAdjustmentResult(
            status="unavailable",
            cameras={}, points={}, observations={},
            median_reprojection_px=0.0,
            n_cameras=len(cameras), n_points=len(points), n_observations=len(observations),
            n_kept=0, seconds=0.0, nfev=0, optimality=0.0,
            message="no cameras, points, or observations", lm_result=None,
        )

    prob = build_bundle_problem(cameras, points, observations)
    kept = prob["kept"]
    n_kept = len(kept)

    if n_kept == 0:
        return BundleAdjustmentResult(
            status="unavailable",
            cameras={}, points={}, observations={},
            median_reprojection_px=0.0,
            n_cameras=len(cameras), n_points=len(points), n_observations=len(observations),
            n_kept=0, seconds=0.0, nfev=0, optimality=0.0,
            message="no valid observations after filtering", lm_result=None,
        )

    sol = solve_sparse_lm(
        prob["residuals"], prob["jacobian"], prob["x0"],
        loss=loss, f_scale=f_scale, max_nfev=max_nfev,
    )

    # Unpack refined parameters
    x = sol.x
    refined_R: dict[str, np.ndarray] = {}
    refined_t: dict[str, np.ndarray] = {}
    # Camera 0 stays at its initial pose (gauge)
    refined_R[prob["gauge_camera_id"]] = cameras[prob["gauge_camera_id"]].R
    refined_t[prob["gauge_camera_id"]] = cameras[prob["gauge_camera_id"]].t
    for cid, i in prob["opt_cam_idx"].items():
        off = prob["cam_param_start"] + 6 * i
        refined_R[cid] = _R_from_rot_params(x[off:off + 3])
        refined_t[cid] = x[off + 3:off + 6]
    refined_P: dict[str, np.ndarray] = {}
    for pid, i in prob["pt_idx"].items():
        off = prob["pt_param_start"] + 3 * i
        refined_P[pid] = x[off:off + 3]

    # Per-observation reprojection errors
    obs_errors: dict[str, float] = {}
    all_errs: list[float] = []
    cam_errs: dict[str, list[float]] = {cid: [] for cid in prob["cam_ids"]}
    pt_errs: dict[str, list[float]] = {pid: [] for pid in prob["pt_ids"]}
    for obs in kept:
        R = refined_R[obs.camera_id]
        t = refined_t[obs.camera_id]
        P = refined_P[obs.point_id]
        K = cameras[obs.camera_id].intrinsics
        X_cam = R @ P + t
        proj = _project(K, X_cam)
        if proj is None:
            err = 0.0
        else:
            err = math.hypot(proj[0] - obs.u, proj[1] - obs.v)
        key = f"{obs.camera_id}|{obs.point_id}"
        obs_errors[key] = round(err, 4)
        all_errs.append(err)
        cam_errs[obs.camera_id].append(err)
        pt_errs[obs.point_id].append(err)

    median_err = float(np.median(all_errs)) if all_errs else 0.0

    # Covariances: (J^T J)^-1 at the solution, per parameter block
    J = sol.jac  # sqrt(W) J, sparse
    cam_covs: dict[str, np.ndarray | None] = {}
    pt_covs: dict[str, np.ndarray | None] = {}
    if J is not None:
        H = (J.T @ J).tocsc() + 1e-12 * identity(J.shape[1], format="csc")
        try:
            H_dense = H.toarray()
            Hinv = np.linalg.inv(H_dense)
        except (RuntimeError, np.linalg.LinAlgError):
            Hinv = None
        if Hinv is not None:
            for cid, i in prob["opt_cam_idx"].items():
                off = prob["cam_param_start"] + 6 * i
                cam_covs[cid] = Hinv[off:off + 6, off:off + 6].copy()
            for pid, j in prob["pt_idx"].items():
                off = prob["pt_param_start"] + 3 * j
                pt_covs[pid] = Hinv[off:off + 3, off:off + 3].copy()
        else:
            cam_covs = {cid: None for cid in prob["opt_cam_idx"]}
            pt_covs = {pid: None for pid in prob["pt_ids"]}
    else:
        cam_covs = {cid: None for cid in prob["opt_cam_idx"]}
        pt_covs = {pid: None for pid in prob["pt_ids"]}
    # Camera 0 (gauge) has no covariance
    cam_covs[prob["gauge_camera_id"]] = None

    cam_results: dict[str, CameraResult] = {}
    for cid in prob["cam_ids"]:
        n_obs_cam = len(cam_errs[cid])
        cam_results[cid] = CameraResult(
            camera_id=cid,
            R=refined_R[cid],
            t=refined_t[cid],
            cov=cam_covs.get(cid),
            n_observations=n_obs_cam,
            median_reprojection_px=round(float(np.median(cam_errs[cid])), 4) if cam_errs[cid] else 0.0,
        )

    pt_results: dict[str, PointResult] = {}
    for pid in prob["pt_ids"]:
        i = prob["pt_idx"][pid]
        n_obs_pt = len(pt_errs[pid])
        pt_results[pid] = PointResult(
            point_id=pid,
            position=refined_P[pid],
            cov=pt_covs[pid],
            n_observations=n_obs_pt,
            median_reprojection_px=round(float(np.median(pt_errs[pid])), 4) if pt_errs[pid] else 0.0,
        )

    return BundleAdjustmentResult(
        status="optimised" if sol.success else "failed",
        cameras=cam_results,
        points=pt_results,
        observations=obs_errors,
        median_reprojection_px=round(median_err, 4),
        n_cameras=len(prob["cam_ids"]),
        n_points=len(prob["pt_ids"]),
        n_observations=n_kept,
        n_kept=n_kept,
        seconds=round(time.perf_counter() - t0, 4),
        nfev=int(sol.nfev),
        optimality=float(sol.optimality),
        message=str(sol.message),
        lm_result=sol,
    )


__all__ = [
    "BundleAdjustmentError",
    "CameraObservation",
    "CameraParams",
    "Point3D",
    "CameraResult",
    "PointResult",
    "BundleAdjustmentResult",
    "build_bundle_problem",
    "run_bundle_adjustment",
]
