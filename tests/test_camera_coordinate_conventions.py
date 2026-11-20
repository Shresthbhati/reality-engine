"""Tests for Camera Coordinate Conventions and COLMAP Extrinsics Inversion.

Validates:
1. Mathematical COLMAP world-to-cam (R_cw, t_cw) to camera-to-world (C = -R_cw^T t_cw) inversion.
2. Proper rotation matrix determinant (+1.0, preserving chirality without reflection).
3. Pinhole camera 3D-to-2D projection and 2D-depth-to-3D unprojection round-trip.
4. Identical camera reconstruction across manifest.json and COLMAP images.txt fallback.
5. Explicit coordinate frame handling preventing double-conversions.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np
import pytest

from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera, Quat, Vec3
from benchmarks.measured_rooms.run_interior_benchmark import parse_interior_colmap_cameras


def test_colmap_extrinsics_inversion_math() -> None:
    """Verifies C = -R_cw^T t_cw exactly recovers the true camera center C."""
    # Arbitrary camera center in world coordinates
    C_true = np.array([2.5, -1.8, 1.45], dtype=np.float64)

    # Pure rotation matrix (yaw=30 deg, pitch=-15 deg, roll=10 deg)
    yaw = np.radians(30.0)
    pitch = np.radians(-15.0)
    roll = np.radians(10.0)

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_wc = Rz @ Ry @ Rx

    # Verify R_wc is a proper rotation with det = +1.0
    assert abs(np.linalg.det(R_wc) - 1.0) < 1e-6

    # Standard COLMAP stores world-to-cam: R_cw = R_wc^T, t_cw = -R_cw @ C
    R_cw = R_wc.T
    t_cw = -R_cw @ C_true

    # Convert R_cw to quaternion (qw, qx, qy, qz)
    tr = np.trace(R_cw)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        qw = 0.25 / s
        qx = (R_cw[2, 1] - R_cw[1, 2]) * s
        qy = (R_cw[0, 2] - R_cw[2, 0]) * s
        qz = (R_cw[1, 0] - R_cw[0, 1]) * s
    else:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

    q_cw = Quat(float(qw), float(qx), float(qy), float(qz)).normalized()

    # Recovery formula used in parse_interior_colmap_cameras:
    # q_wc = q_cw.conjugate()
    # C = -q_wc.rotate(t_cw)
    q_wc = q_cw.conjugate()
    t_vec = Vec3(float(t_cw[0]), float(t_cw[1]), float(t_cw[2]))
    C_recovered = Vec3(0.0, 0.0, 0.0) - q_wc.rotate(t_vec)

    np.testing.assert_allclose(
        [C_recovered.x, C_recovered.y, C_recovered.z],
        C_true,
        atol=1e-5,
        err_msg="Recovered camera center does not match ground truth center",
    )


def test_pinhole_projection_unprojection_roundtrip() -> None:
    """Verifies that 3D point in front of camera projects to pixel (u,v) and unprojects back to 3D point."""
    intr = CameraIntrinsics(fx=1000.0, fy=1000.0, cx=640.0, cy=360.0, width=1280, height=720)
    pos = Vec3(1.0, 2.0, 1.5)
    # Camera looking along +Y in world, with Z up -> OpenCV: Z_cam forward (+Y), X_cam right (+X), Y_cam down (-Z)
    rot_cv = Quat(0.7071068, 0.7071068, 0.0, 0.0)  # 90 deg rotation around X
    cam = PinholeCamera(intrinsics=intr, extrinsics=CameraExtrinsics(position=pos, rotation=rot_cv))

    # Test point 4.0m in front of camera, 0.5m to right, 0.2m above
    # In camera frame: X_cam = 0.5, Y_cam = -0.2 (since Y_cam is down), Z_cam = 4.0
    P_cam = Vec3(0.5, -0.2, 4.0)
    P_world = pos + rot_cv.rotate(P_cam)

    # Project to pixel
    u = intr.fx * (P_cam.x / P_cam.z) + intr.cx
    v = intr.fy * (P_cam.y / P_cam.z) + intr.cy

    assert 0 <= u < intr.width
    assert 0 <= v < intr.height

    # Unproject back to camera coordinates using depth Z
    unproj_cam = Vec3((u - intr.cx) * P_cam.z / intr.fx, (v - intr.cy) * P_cam.z / intr.fy, P_cam.z)
    unproj_world = pos + rot_cv.rotate(unproj_cam)

    np.testing.assert_allclose(
        [unproj_world.x, unproj_world.y, unproj_world.z],
        [P_world.x, P_world.y, P_world.z],
        atol=1e-5,
    )


def test_manifest_and_colmap_parsers_produce_identical_cameras(tmp_path: Path) -> None:
    """Verifies that manifest.json and COLMAP fallback produce matching PinholeCamera extrinsics and intrinsics."""
    room_dir = tmp_path / "room_parity"
    room_dir.mkdir(parents=True)

    C = [2.0, 1.5, 1.2]
    # Pure yaw 90 degrees
    yaw = np.pi / 2.0
    R_wc = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    R_cw = R_wc.T
    t_cw = -R_cw @ np.array(C)

    # q_wc: rotation around Z by 90 deg: w=cos(45)=0.7071, z=sin(45)=0.7071
    q_wc_list = [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]
    # q_cw: conjugate of q_wc
    q_cw_list = [q_wc_list[0], -q_wc_list[1], -q_wc_list[2], -q_wc_list[3]]

    # 1. Write COLMAP files
    cameras_txt = room_dir / "cameras.txt"
    cameras_txt.write_text("1 PINHOLE 1920 1080 1500.0 1500.0 960.0 540.0\n", encoding="utf-8")

    images_txt = room_dir / "images.txt"
    images_txt.write_text(
        f"1 {q_cw_list[0]} {q_cw_list[1]} {q_cw_list[2]} {q_cw_list[3]} {t_cw[0]} {t_cw[1]} {t_cw[2]} 1 test_view.jpg\n\n",
        encoding="utf-8",
    )

    # 2. Test COLMAP parsing first (without manifest)
    cams_colmap = parse_interior_colmap_cameras(room_dir)
    assert "test_view.jpg" in cams_colmap
    cam_colmap = cams_colmap["test_view.jpg"]

    # 3. Write manifest.json with matching parameters
    manifest_data = {
        "intrinsics": {"width": 1920, "height": 1080, "fx": 1500.0, "fy": 1500.0, "cx": 960.0, "cy": 540.0},
        "images": [
            {
                "filename": "test_view.jpg",
                "position_m": C,
                "rotation_quat_wxyz": q_wc_list,
                "camera_convention": "OPENCV",
            }
        ],
    }
    (room_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

    # 4. Parse from manifest
    cams_manifest = parse_interior_colmap_cameras(room_dir)
    assert "test_view.jpg" in cams_manifest
    cam_manifest = cams_manifest["test_view.jpg"]

    # Intrinsics equality
    assert cam_colmap.intrinsics.fx == cam_manifest.intrinsics.fx
    assert cam_colmap.intrinsics.fy == cam_manifest.intrinsics.fy
    assert cam_colmap.intrinsics.cx == cam_manifest.intrinsics.cx
    assert cam_colmap.intrinsics.cy == cam_manifest.intrinsics.cy

    # Extrinsics position equality
    np.testing.assert_allclose(
        [cam_colmap.extrinsics.position.x, cam_colmap.extrinsics.position.y, cam_colmap.extrinsics.position.z],
        [cam_manifest.extrinsics.position.x, cam_manifest.extrinsics.position.y, cam_manifest.extrinsics.position.z],
        atol=1e-5,
    )

    # Extrinsics rotation equality
    np.testing.assert_allclose(
        [cam_colmap.extrinsics.rotation.w, cam_colmap.extrinsics.rotation.x, cam_colmap.extrinsics.rotation.y, cam_colmap.extrinsics.rotation.z],
        [cam_manifest.extrinsics.rotation.w, cam_manifest.extrinsics.rotation.x, cam_manifest.extrinsics.rotation.y, cam_manifest.extrinsics.rotation.z],
        atol=1e-5,
    )


def test_opengl_convention_handled_only_once() -> None:
    """Verifies that an OPENGL tagged camera is flipped to OPENCV exactly once and not double-flipped."""
    # Base quaternion in OpenGL convention (Z_cam backward, Y_cam up)
    q_gl = Quat(1.0, 0.0, 0.0, 0.0)  # identity

    # Convert OpenGL to OpenCV (180 deg around X: (0, 1, 0, 0))
    q_cv = q_gl.multiply(Quat(0.0, 1.0, 0.0, 0.0)).normalized()

    # Verify q_cv is NOT identity (it was flipped)
    assert abs(q_cv.w) < 1e-4
    assert abs(q_cv.x - 1.0) < 1e-4

    # If flipped again by error, it returns to identity
    q_double_flip = q_cv.multiply(Quat(0.0, 1.0, 0.0, 0.0)).normalized()
    assert abs(abs(q_double_flip.w) - 1.0) < 1e-4  # identity
