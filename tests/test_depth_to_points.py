"""Tests for depth map -> point cloud (reconstruction/depth_to_points.py)."""

from __future__ import annotations

import math

import pytest

from engine.physics.math3 import Quat, Vec3
from perception.depth.interface import DepthMap
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from reconstruction.depth_to_points import DepthToPointsError, depth_map_to_points

_WIDTH, _HEIGHT = 10, 8


def _camera() -> PinholeCamera:
    intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=_WIDTH / 2, cy=_HEIGHT / 2, width=_WIDTH, height=_HEIGHT)
    return PinholeCamera(intrinsics=intrinsics, extrinsics=CameraExtrinsics(position=Vec3.zero(), rotation=Quat.identity()))


def _flat_depth(value=4.0, unit="meters") -> DepthMap:
    return DepthMap(
        evidence_id="ev-1", width=_WIDTH, height=_HEIGHT,
        values=[[value for _ in range(_WIDTH)] for _ in range(_HEIGHT)], unit=unit,
    )


def test_every_valid_pixel_produces_one_point():
    points = depth_map_to_points(_flat_depth(), _camera())
    assert len(points) == _WIDTH * _HEIGHT


def test_points_carry_source_evidence_id_and_uncertainty():
    depth = _flat_depth()
    points = depth_map_to_points(depth, _camera())
    assert all(p.source_evidence_ids == ["ev-1"] for p in points)
    assert all(p.uncertainty == depth.uncertainty for p in points)


def test_point_depths_match_the_source_depth_value():
    points = depth_map_to_points(_flat_depth(value=6.0), _camera())
    assert all(math.isclose(p.position[2], 6.0, abs_tol=1e-9) for p in points)


def test_invalid_depth_pixels_are_skipped_not_fabricated():
    values = [[4.0 for _ in range(_WIDTH)] for _ in range(_HEIGHT)]
    values[0][0] = 0.0       # invalid: non-positive
    values[0][1] = float("nan")  # invalid: non-finite
    depth = DepthMap(evidence_id="ev-1", width=_WIDTH, height=_HEIGHT, values=values, unit="meters")

    points = depth_map_to_points(depth, _camera())
    assert len(points) == _WIDTH * _HEIGHT - 2


def test_relative_depth_is_refused():
    with pytest.raises(DepthToPointsError):
        depth_map_to_points(_flat_depth(unit="relative"), _camera())


def test_stride_decimates_deterministically():
    points_full = depth_map_to_points(_flat_depth(), _camera(), stride=1)
    points_strided = depth_map_to_points(_flat_depth(), _camera(), stride=2)
    expected_count = len(range(0, _HEIGHT, 2)) * len(range(0, _WIDTH, 2))
    assert len(points_strided) == expected_count
    assert len(points_strided) < len(points_full)


def test_invalid_stride_raises():
    with pytest.raises(DepthToPointsError):
        depth_map_to_points(_flat_depth(), _camera(), stride=0)


def test_track_ids_are_deterministic_and_unique():
    points_a = depth_map_to_points(_flat_depth(), _camera())
    points_b = depth_map_to_points(_flat_depth(), _camera())
    assert [p.track_id for p in points_a] == [p.track_id for p in points_b]
    assert len(points_a) == len(set(p.track_id for p in points_a))


def test_output_composes_with_real_ransac_plane_detection():
    """The output type (ReconstructedPoint) must be directly usable by
    the existing reconstruction pipeline -- prove it by running the real
    RANSAC plane detector over depth-derived points, same as it already
    runs over COLMAP sparse points."""
    from perception.geometry.planes import detect_planes
    from reconstruction.backend.interface import ReconstructionResult

    # A camera looking straight down (+y) at a flat floor 3m below it
    # produces a depth map that is one large planar surface once unprojected.
    intrinsics = CameraIntrinsics(fx=50.0, fy=50.0, cx=_WIDTH / 2, cy=_HEIGHT / 2, width=_WIDTH, height=_HEIGHT)
    camera = PinholeCamera(intrinsics=intrinsics, extrinsics=CameraExtrinsics(position=Vec3.zero(), rotation=Quat.identity()))
    depth = _flat_depth(value=3.0)  # a fronto-parallel plane at z=3 in camera space

    points = depth_map_to_points(depth, camera)
    result = ReconstructionResult(points=points, camera_poses=[], registration_status="success")

    detection = detect_planes(result, seed=42, min_inliers=10)
    assert len(detection.planes) >= 1
    plane = detection.planes[0]
    # All points are at z=3 (camera space == world space here) -> the
    # detected plane's normal should be along Z.
    assert abs(abs(plane.normal[2]) - 1.0) < 1e-6
