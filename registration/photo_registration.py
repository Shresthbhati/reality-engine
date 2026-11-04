"""Incremental photo registration against an existing world model (Blueprint Task 6.2, Pipeline S3).

Registers new images into an existing sparse point cloud or WorldIR model without
re-running full SfM from scratch:
- Computes 2D-3D correspondences between image features and existing 3D points.
- Estimates camera pose via robust PnP with RANSAC outlier rejection.
- Enforces strict inlier and reprojection error gates.
- Refuses non-matching or disjoint photos with an explicit, honest reason (Scenarios A11, A19),
  preventing force-merging of unrelated imagery.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from engine.math import Quat, Vec3
def _matrix_to_quat(R) -> Quat:
    """Convert 3x3 rotation matrix to unit Quat(w, x, y, z)."""
    r00, r01, r02 = float(R[0][0]), float(R[0][1]), float(R[0][2])
    r10, r11, r12 = float(R[1][0]), float(R[1][1]), float(R[1][2])
    r20, r21, r22 = float(R[2][0]), float(R[2][1]), float(R[2][2])

    trace = r00 + r11 + r22
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r21 - r12) * s
        y = (r02 - r20) * s
        z = (r10 - r01) * s
    elif r00 > r11 and r00 > r22:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + r00 - r11 - r22))
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + r11 - r00 - r22))
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = 2.0 * math.sqrt(max(1e-12, 1.0 + r22 - r00 - r11))
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s

    return Quat(w, x, y, z).normalized()


@dataclass
class PhotoRegistrationResult:
    """Outcome of attempting to register one photo into an existing world model."""

    photo_id: str
    status: str  # "registered" | "refused"
    position: Optional[Tuple[float, float, float]] = None
    rotation_quat: Optional[Tuple[float, float, float, float]] = None  # (w, x, y, z)
    inlier_count: int = 0
    total_correspondences: int = 0
    reprojection_error_px: float = 0.0
    reason: Optional[str] = None

    @property
    def refusal_reason(self) -> Optional[str]:
        return self.reason

    @property
    def rotation_quaternion(self) -> Optional[Tuple[float, float, float, float]]:
        return self.rotation_quat


@dataclass
class IncrementalRegistrationBatch:
    """Batch results for incremental photo registration."""

    registered: Dict[str, PhotoRegistrationResult] = field(default_factory=dict)
    refused: Dict[str, PhotoRegistrationResult] = field(default_factory=dict)

    @property
    def results(self) -> List[PhotoRegistrationResult]:
        return list(self.registered.values()) + list(self.refused.values())

    @property
    def total_count(self) -> int:
        return len(self.registered) + len(self.refused)

    @property
    def registered_count(self) -> int:
        return len(self.registered)

    @property
    def refused_count(self) -> int:
        return len(self.refused)

    @property
    def success_rate(self) -> float:
        return self.registered_count / max(1, self.total_count)


def _to_camera_matrix(k_or_intrinsics: Any) -> np.ndarray:
    if k_or_intrinsics is None:
        return np.array([
            [1000.0, 0.0, 500.0],
            [0.0, 1000.0, 500.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    if hasattr(k_or_intrinsics, "fx"):
        return np.array([
            [float(k_or_intrinsics.fx), 0.0, float(k_or_intrinsics.cx)],
            [0.0, float(k_or_intrinsics.fy), float(k_or_intrinsics.cy)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    return np.ascontiguousarray(k_or_intrinsics, dtype=np.float64).reshape(3, 3)


def register_photo_to_sparse_model(
    photo_id: Optional[str] = None,
    points_2d: Sequence[Sequence[float]] = (),
    points_3d: Sequence[Sequence[float]] = (),
    camera_matrix: Optional[Any] = None,
    distortion_coeffs: Optional[Sequence[float]] = None,
    *,
    image_id: Optional[str] = None,
    intrinsics: Optional[Any] = None,
    min_inliers: int = 15,
    max_reprojection_error_px: float = 4.0,
    reprojection_error_threshold_px: Optional[float] = None,
) -> PhotoRegistrationResult:
    """Register one new photograph against existing 3D world points via PnP RANSAC.

    Parameters:
        photo_id: Unique identifier for the photograph (or image_id)
        points_2d: Nx2 observed 2D feature coordinates in pixels
        points_3d: Nx3 corresponding 3D world coordinates in meters
        camera_matrix: 3x3 intrinsic matrix or CameraIntrinsics instance
        distortion_coeffs: Lens distortion coefficients (optional)
        min_inliers: Minimum number of geometric inliers required for acceptance
        max_reprojection_error_px: Inlier reprojection threshold for RANSAC

    Returns:
        PhotoRegistrationResult with status='registered' or 'refused' with explicit reason.
    """
    effective_id = photo_id or image_id or "photo_unknown"
    effective_k = camera_matrix if camera_matrix is not None else intrinsics
    effective_max_err = (
        reprojection_error_threshold_px
        if reprojection_error_threshold_px is not None
        else max_reprojection_error_px
    )

    n_pts = len(points_2d)
    if n_pts != len(points_3d):
        raise ValueError(f"points_2d ({n_pts}) and points_3d ({len(points_3d)}) must have identical lengths")

    if n_pts < min_inliers:
        return PhotoRegistrationResult(
            photo_id=effective_id,
            status="refused",
            total_correspondences=n_pts,
            inlier_count=0,
            reason=f"this photo doesn't match the world: too few 2D-3D correspondences ({n_pts} < {min_inliers} required)",
        )

    pts_2d_arr = np.ascontiguousarray(points_2d, dtype=np.float64).reshape(-1, 2)
    pts_3d_arr = np.ascontiguousarray(points_3d, dtype=np.float64).reshape(-1, 3)
    k_mat = _to_camera_matrix(effective_k)
    dist_mat = (
        np.ascontiguousarray(distortion_coeffs, dtype=np.float64).reshape(-1, 1)
        if distortion_coeffs is not None
        else np.zeros((4, 1), dtype=np.float64)
    )

    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pts_3d_arr,
            imagePoints=pts_2d_arr,
            cameraMatrix=k_mat,
            distCoeffs=dist_mat,
            reprojectionError=effective_max_err,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except Exception as exc:
        return PhotoRegistrationResult(
            photo_id=effective_id,
            status="refused",
            total_correspondences=n_pts,
            inlier_count=0,
            reason=f"PnP solver exception: {exc}",
        )

    if not success or inliers is None:
        return PhotoRegistrationResult(
            photo_id=effective_id,
            status="refused",
            total_correspondences=n_pts,
            inlier_count=0,
            reason="this photo doesn't match the world: PnP RANSAC found no geometrically valid pose",
        )

    inlier_count = len(inliers)
    if inlier_count < min_inliers:
        return PhotoRegistrationResult(
            photo_id=effective_id,
            status="refused",
            total_correspondences=n_pts,
            inlier_count=inlier_count,
            reason=f"this photo doesn't match the world: insufficient inlier support ({inlier_count} < {min_inliers} required)",
        )

    # Compute mean reprojection error over inliers
    proj_pts, _ = cv2.projectPoints(pts_3d_arr[inliers[:, 0]], rvec, tvec, k_mat, dist_mat)
    reproj_errors = np.linalg.norm(pts_2d_arr[inliers[:, 0]] - proj_pts.reshape(-1, 2), axis=1)
    mean_reproj_err = float(np.mean(reproj_errors))

    # Convert camera pose to world coordinates
    # PnP gives camera-from-world: x_cam = R * x_world + t
    # World pose: x_world = -R^T * t
    r_mat, _ = cv2.Rodrigues(rvec)
    cam_pos_world = -r_mat.T @ tvec.flatten()

    # World-to-camera rotation matrix -> quaternion (w, x, y, z)
    # R_cam_to_world is R^T
    r_world_mat = r_mat.T
    quat_w = _matrix_to_quat(r_world_mat)  # Quat(w, x, y, z)

    return PhotoRegistrationResult(
        photo_id=effective_id,
        status="registered",
        position=(float(cam_pos_world[0]), float(cam_pos_world[1]), float(cam_pos_world[2])),
        rotation_quat=(float(quat_w.w), float(quat_w.x), float(quat_w.y), float(quat_w.z)),
        inlier_count=inlier_count,
        total_correspondences=n_pts,
        reprojection_error_px=mean_reproj_err,
    )


def register_incremental_photos(
    world_or_photos: Any,
    new_photos: Optional[Sequence[dict]] = None,
    default_camera_matrix: Optional[Sequence[Sequence[float]]] = None,
    min_inliers: int = 15,
    max_reproj_error_px: float = 4.0,
) -> IncrementalRegistrationBatch:
    """Register a batch of newly added photos against existing WorldIR points and geometry.

    Parameters:
        world_or_photos: WorldIR instance or direct sequence of photo candidate dicts.
        new_photos: Sequence of dicts with:
            - 'photo_id' or 'image_id': str
            - 'points_2d': Nx2 list of pixel coords
            - 'points_3d': Nx3 list of 3D coords
            - 'camera_matrix' or 'intrinsics' (optional): 3x3 intrinsics
        default_camera_matrix: 3x3 fallback intrinsic matrix
    """
    if new_photos is None and isinstance(world_or_photos, (list, tuple)):
        photos_list = world_or_photos
    else:
        photos_list = new_photos or []

    batch = IncrementalRegistrationBatch()
    default_k = default_camera_matrix or [
        [1000.0, 0.0, 500.0],
        [0.0, 1000.0, 500.0],
        [0.0, 0.0, 1.0],
    ]

    for photo in photos_list:
        pid = photo.get("photo_id") or photo.get("image_id") or "photo_unknown"
        pts2d = photo.get("points_2d", [])
        pts3d = photo.get("points_3d", [])
        k = photo.get("camera_matrix") or photo.get("intrinsics") or default_k

        res = register_photo_to_sparse_model(
            photo_id=pid,
            points_2d=pts2d,
            points_3d=pts3d,
            camera_matrix=k,
            min_inliers=min_inliers,
            max_reprojection_error_px=max_reproj_error_px,
        )

        if res.status == "registered":
            batch.registered[pid] = res
        else:
            batch.refused[pid] = res

    return batch
