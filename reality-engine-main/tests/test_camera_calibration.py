"""Numerical tests for the pinhole camera model
(reconstruction/calibration/camera.py): intrinsics validation,
projection, unprojection, round trips (with and without distortion),
ray casting, and coordinate-frame conversion."""

from __future__ import annotations

import math

import pytest

from engine.physics.math3 import Quat, Vec3
from reconstruction.calibration.camera import (
    CameraExtrinsics,
    CameraIntrinsics,
    CameraIntrinsicsError,
    PinholeCamera,
)


def _identity_camera(**intrinsics_overrides) -> PinholeCamera:
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480, **intrinsics_overrides)
    extrinsics = CameraExtrinsics(position=Vec3.zero(), rotation=Quat.identity())
    return PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics)


# ------------------------------------------------------------- intrinsics


def test_intrinsics_rejects_non_positive_focal_length():
    with pytest.raises(CameraIntrinsicsError):
        CameraIntrinsics(fx=0.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)


def test_intrinsics_rejects_non_positive_dimensions():
    with pytest.raises(CameraIntrinsicsError):
        CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=0, height=480)


def test_intrinsics_rejects_non_finite_values():
    with pytest.raises(CameraIntrinsicsError):
        CameraIntrinsics(fx=500.0, fy=500.0, cx=float("nan"), cy=240.0, width=640, height=480)


def test_intrinsics_roundtrip_dict():
    intrinsics = CameraIntrinsics(fx=500.0, fy=510.0, cx=320.0, cy=240.0, width=640, height=480, k1=0.1, p1=0.01)
    restored = CameraIntrinsics.from_dict(intrinsics.to_dict())
    assert restored == intrinsics


# ------------------------------------------------------------- projection


def test_point_on_optical_axis_projects_to_principal_point():
    camera = _identity_camera()
    pixel = camera.project(Vec3(0.0, 0.0, 5.0))
    assert pixel is not None
    u, v = pixel
    assert math.isclose(u, 320.0, abs_tol=1e-9)
    assert math.isclose(v, 240.0, abs_tol=1e-9)


def test_point_behind_camera_does_not_project():
    camera = _identity_camera()
    assert camera.project(Vec3(0.0, 0.0, -5.0)) is None


def test_point_at_camera_plane_does_not_project():
    camera = _identity_camera()
    assert camera.project(Vec3(1.0, 0.0, 0.0)) is None


def test_projection_scales_with_focal_length():
    camera = _identity_camera()
    u, v = camera.project(Vec3(1.0, 0.0, 5.0))
    # x/z = 0.2, fx=500 -> u = 320 + 100
    assert math.isclose(u, 420.0, abs_tol=1e-9)
    assert math.isclose(v, 240.0, abs_tol=1e-9)


# ------------------------------------------------------------ unprojection


def test_unproject_rejects_non_positive_depth():
    camera = _identity_camera()
    with pytest.raises(ValueError):
        camera.unproject(320.0, 240.0, depth=0.0)


def test_unproject_principal_point_lies_on_optical_axis():
    camera = _identity_camera()
    point = camera.unproject(320.0, 240.0, depth=5.0)
    assert math.isclose(point.x, 0.0, abs_tol=1e-9)
    assert math.isclose(point.y, 0.0, abs_tol=1e-9)
    assert math.isclose(point.z, 5.0, abs_tol=1e-9)


# ------------------------------------------------------------- round trips


@pytest.mark.parametrize("point", [
    Vec3(0.3, -0.2, 4.0),
    Vec3(1.5, 0.9, 8.0),
    Vec3(-2.0, 1.0, 3.0),
])
def test_project_then_unproject_recovers_original_point_no_distortion(point):
    camera = _identity_camera()
    pixel = camera.project(point)
    assert pixel is not None
    recovered = camera.unproject(pixel[0], pixel[1], depth=point.z)
    assert math.isclose(recovered.x, point.x, abs_tol=1e-6)
    assert math.isclose(recovered.y, point.y, abs_tol=1e-6)
    assert math.isclose(recovered.z, point.z, abs_tol=1e-6)


@pytest.mark.parametrize("point", [
    Vec3(0.3, -0.2, 4.0),
    Vec3(1.0, 0.6, 6.0),
])
def test_project_then_unproject_recovers_original_point_with_distortion(point):
    camera = _identity_camera(k1=0.08, k2=-0.01, p1=0.002, p2=-0.001)
    pixel = camera.project(point)
    assert pixel is not None
    recovered = camera.unproject(pixel[0], pixel[1], depth=point.z)
    assert math.isclose(recovered.x, point.x, abs_tol=1e-4)
    assert math.isclose(recovered.y, point.y, abs_tol=1e-4)
    assert math.isclose(recovered.z, point.z, abs_tol=1e-9)


def test_unproject_then_project_recovers_pixel():
    camera = _identity_camera(k1=0.05)
    world_point = camera.unproject(400.0, 260.0, depth=3.0)
    recovered_pixel = camera.project(world_point)
    assert recovered_pixel is not None
    assert math.isclose(recovered_pixel[0], 400.0, abs_tol=1e-4)
    assert math.isclose(recovered_pixel[1], 260.0, abs_tol=1e-4)


# --------------------------------------------------------------- ray cast


def test_ray_through_principal_point_points_along_optical_axis():
    camera = _identity_camera()
    origin, direction = camera.ray(320.0, 240.0)
    assert origin.to_dict() == Vec3.zero().to_dict()
    assert math.isclose(direction.x, 0.0, abs_tol=1e-9)
    assert math.isclose(direction.y, 0.0, abs_tol=1e-9)
    assert math.isclose(direction.z, 1.0, abs_tol=1e-9)


def test_ray_direction_is_unit_length():
    camera = _identity_camera()
    _, direction = camera.ray(450.0, 200.0)
    assert math.isclose(direction.length(), 1.0, abs_tol=1e-9)


# --------------------------------------------------------- coordinate frame


def test_camera_translated_along_z_shifts_projection():
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    extrinsics = CameraExtrinsics(position=Vec3(0.0, 0.0, -2.0), rotation=Quat.identity())
    camera = PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics)

    # World point at z=3 is now only 5m in front of the camera (camera at z=-2).
    pixel_moved = camera.project(Vec3(0.0, 0.0, 3.0))
    pixel_at_origin = _identity_camera().project(Vec3(0.0, 0.0, 5.0))
    assert pixel_moved == pixel_at_origin


def test_camera_rotated_180_about_y_flips_forward_direction():
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480)
    # 180 deg rotation about Y: (w, x, y, z) = (0, 0, 1, 0)
    extrinsics = CameraExtrinsics(position=Vec3.zero(), rotation=Quat(0.0, 0.0, 1.0, 0.0))
    camera = PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics)

    # A point in front of the identity camera (+z) is now behind this one.
    assert camera.project(Vec3(0.0, 0.0, 5.0)) is None
    # A point behind the identity camera (-z) is now in front of this one.
    assert camera.project(Vec3(0.0, 0.0, -5.0)) is not None


def test_camera_to_dict_roundtrip():
    camera = _identity_camera(k1=0.1)
    restored = PinholeCamera.from_dict(camera.to_dict())
    assert restored.intrinsics == camera.intrinsics
    assert restored.extrinsics.position.to_dict() == camera.extrinsics.position.to_dict()
    assert restored.extrinsics.rotation.to_dict() == camera.extrinsics.rotation.to_dict()
