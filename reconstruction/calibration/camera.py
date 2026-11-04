"""Camera calibration module.

Camera types (CameraIntrinsics, CameraExtrinsics, PinholeCamera) now live
in ``world_ir.camera`` (the neutral contract layer). This module re-exports
them for backward compatibility and provides reconstruction-specific
helpers (``camera_from_pose``) that convert reconstruction poses to
cameras.
"""

from __future__ import annotations

from engine.math import Quat, Vec3
from world_ir.camera import (
    CameraExtrinsics,
    CameraIntrinsics,
    CameraIntrinsicsError,
    PinholeCamera,
    _MIN_PROJECTABLE_DEPTH_M,
    _UNDISTORT_ITERATIONS,
    _distort,
    _undistort,
    quat_to_matrix,
    matrix_to_quat,
)

__all__ = [
    "CameraIntrinsics",
    "CameraExtrinsics",
    "PinholeCamera",
    "CameraIntrinsicsError",
    "quat_to_matrix",
    "matrix_to_quat",
    "camera_from_pose",
    "_MIN_PROJECTABLE_DEPTH_M",
    "_UNDISTORT_ITERATIONS",
    "_distort",
    "_undistort",
]


def camera_from_pose(intrinsics: CameraIntrinsics, pose) -> PinholeCamera:
    """Build a PinholeCamera from a backend pose
    (``reconstruction.backend.interface.ReconstructedCameraPose``: tuple
    position + (w, x, y, z) quaternion, camera-to-world -- the same
    convention CameraExtrinsics documents). This is the seam where
    camera-reconstruction output becomes a usable camera model."""
    w, x, y, z = (float(c) for c in pose.rotation)
    quat = Quat(w=w, x=x, y=y, z=z).normalized()
    px, py, pz = (float(c) for c in pose.position)
    return PinholeCamera(
        intrinsics=intrinsics,
        extrinsics=CameraExtrinsics(position=Vec3(px, py, pz), rotation=quat),
    )
