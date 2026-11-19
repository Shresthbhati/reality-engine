"""Tests for Image-Backed Reprojection and Visibility Validation."""

from __future__ import annotations

import numpy as np
import pytest

from engine.math import Quat, Vec3
from engine.multiengine.visibility_reprojection import (
    VisibilityReprojectionValidator,
)
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera


@pytest.fixture
def test_camera() -> PinholeCamera:
    """Standard 640x480 pinhole camera at origin looking down +Z axis."""
    intr = CameraIntrinsics(
        fx=500.0, fy=500.0, cx=320.0, cy=240.0,
        width=640, height=480,
    )
    extr = CameraExtrinsics(
        position=Vec3(0.0, 0.0, 0.0),
        rotation=Quat.identity(),
    )
    return PinholeCamera(intrinsics=intr, extrinsics=extr)


def test_point_visibility_exact_pinhole_projection(test_camera: PinholeCamera) -> None:
    """Point at (0, 0, 2) should land directly on principal point (320, 240)."""
    vis = VisibilityReprojectionValidator.check_point_visibility(test_camera, (0.0, 0.0, 2.0))

    assert vis.is_visible is True
    assert vis.is_in_front is True
    assert vis.is_in_frustum is True
    assert vis.is_unoccluded is True
    assert vis.pixel_coord is not None
    u, v = vis.pixel_coord
    assert pytest.approx(u, abs=1e-3) == 320.0
    assert pytest.approx(v, abs=1e-3) == 240.0


def test_point_visibility_behind_camera(test_camera: PinholeCamera) -> None:
    """Point behind camera (Z = -1.0) must report not in front and not visible."""
    vis = VisibilityReprojectionValidator.check_point_visibility(test_camera, (0.0, 0.0, -1.0))

    assert vis.is_visible is False
    assert vis.is_in_front is False
    assert vis.pixel_coord is None
    assert "behind" in vis.reason


def test_point_visibility_outside_frustum(test_camera: PinholeCamera) -> None:
    """Point far to the right (X = 5.0, Z = 2.0) projects outside the 640px image width."""
    # u = 500 * (5.0 / 2.0) + 320 = 1250 + 320 = 1570 px > 640
    vis = VisibilityReprojectionValidator.check_point_visibility(test_camera, (5.0, 0.0, 2.0))

    assert vis.is_visible is False
    assert vis.is_in_front is True
    assert vis.is_in_frustum is False
    assert vis.pixel_coord is not None
    assert vis.pixel_coord[0] > 640.0
    assert "outside image bounds" in vis.reason


def test_depth_occlusion_detection(test_camera: PinholeCamera) -> None:
    """When a foreground obstacle is recorded in the depth map, occluded points report false."""
    depth_map = np.full((480, 640), 1.0, dtype=np.float32)  # Wall at 1.0 meter

    # Target point is at Z = 3.0 meters (behind the 1.0m wall)
    vis = VisibilityReprojectionValidator.check_point_visibility(
        test_camera,
        point_world=(0.0, 0.0, 3.0),
        depth_map=depth_map,
        depth_tolerance_m=0.05,
    )

    assert vis.is_visible is False
    assert vis.is_in_front is True
    assert vis.is_in_frustum is True
    assert vis.is_unoccluded is False
    assert vis.occlusion_depth_m == 1.0
    assert "Occluded" in vis.reason


def test_evaluate_reprojection_consistency_with_known_box(test_camera: PinholeCamera) -> None:
    """Evaluates 3D box projection against matching 2D detection bounding box."""
    # 3D box at center (0, 0, 2.0) with size (0.4, 0.4, 0.4)
    # Z=2.0 -> corners at X = [-0.2, +0.2], Y = [-0.2, +0.2]
    # u_min = 500 * (-0.2 / 2) + 320 = -50 + 320 = 270
    # u_max = 500 * (+0.2 / 2) + 320 = +50 + 320 = 370
    # v_min = 500 * (-0.2 / 2) + 240 = -50 + 240 = 190
    # v_max = 500 * (+0.2 / 2) + 240 = +50 + 240 = 290
    observed_box = (270.0, 190.0, 370.0, 290.0)

    eval_res = VisibilityReprojectionValidator.evaluate_reprojection(
        camera=test_camera,
        center_3d=(0.0, 0.0, 2.0),
        size_3d=(0.4, 0.4, 0.4),
        observed_box_2d=observed_box,
    )

    assert eval_res.is_visible is True
    assert eval_res.reprojection_error_px is not None
    assert pytest.approx(eval_res.reprojection_error_px, abs=1.0) == 0.0
    assert eval_res.iou_2d is not None
    assert eval_res.iou_2d > 0.80
    assert eval_res.is_consistent is True
