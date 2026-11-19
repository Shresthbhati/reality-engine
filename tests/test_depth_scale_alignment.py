"""Unit and Contract Tests for Depth Scale Alignment and Validation.

Validates:
1. Strict rejection of None, invalid type, non-2D, non-positive, or collapsed/constant depth maps.
2. Verified affine disparity and linear depth scale alignment against reference 3D points.
3. Provenance and metadata contract preservation (scale, shift, R^2, RMS error, inlier count).
"""

from __future__ import annotations

import numpy as np

from engine.math import Quat, Vec3
from engine.multiengine.depth_scale_alignment import (
    align_relative_depth_to_metric,
    validate_depth_map,
)
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera


def test_validate_depth_map_rejects_none_and_invalid_shapes() -> None:
    ok, err = validate_depth_map(None)
    assert not ok
    assert err is not None and "DEPTH_MAP_MISSING" in err

    ok, err = validate_depth_map("not-an-array")  # type: ignore[arg-type]
    assert not ok
    assert err is not None and "DEPTH_MAP_INVALID_TYPE" in err

    ok, err = validate_depth_map(np.zeros((10, 10, 3)))
    assert not ok
    assert err is not None and "DEPTH_MAP_INVALID_SHAPE" in err

    ok, err = validate_depth_map(np.zeros((0, 0)))
    assert not ok
    assert err is not None and "DEPTH_MAP_EMPTY" in err


def test_validate_depth_map_rejects_negative_and_all_nan() -> None:
    # All NaN
    nan_arr = np.full((10, 10), np.nan, dtype=np.float32)
    ok, err = validate_depth_map(nan_arr, allow_nan=True)
    assert not ok
    assert err is not None and "DEPTH_MAP_ALL_NAN" in err

    # Negative values
    neg_arr = np.array([[-1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ok, err = validate_depth_map(neg_arr)
    assert not ok
    assert err is not None and "DEPTH_MAP_NEGATIVE" in err


def test_validate_depth_map_rejects_constant_or_collapsed_field() -> None:
    # Exactly constant array (e.g. constant fallback array)
    const_arr = np.full((100, 100), 9.75, dtype=np.float32)
    ok, err = validate_depth_map(const_arr, min_std=1e-4)
    assert not ok
    assert err is not None and "DEPTH_MAP_COLLAPSED" in err

    # Tiny jitter below 1e-4 threshold
    jitter_arr = 9.75 + np.random.uniform(-1e-6, 1e-6, (100, 100)).astype(np.float32)
    ok, err = validate_depth_map(jitter_arr, min_std=1e-4)
    assert not ok
    assert err is not None and "DEPTH_MAP_COLLAPSED" in err


def test_validate_depth_map_accepts_valid_depth_field() -> None:
    valid_arr = np.linspace(1.0, 10.0, 10000, dtype=np.float32).reshape(100, 100)
    ok, err = validate_depth_map(valid_arr, min_std=1e-4)
    assert ok
    assert err is None


def test_align_relative_depth_to_metric_affine_regression() -> None:
    """Creates a synthetic camera, known 3D points, and synthetic relative disparity:

    1/z = s * d_rel + t. Verifies that scale and shift are accurately recovered.
    """
    camera = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity()),
    )

    true_scale = 0.5
    true_shift = 0.1

    pts_3d: list[tuple[float, float, float]] = []
    pts_2d: list[tuple[float, float]] = []
    rel_depth_map = np.linspace(0.1, 2.0, 480 * 640, dtype=np.float32).reshape(480, 640)

    # Generate 50 points with depth between 2.0m and 8.0m at unique pixels
    for i in range(50):
        z_true = 2.0 + (i / 50.0) * 6.0
        u = 50.0 + i * 10.0
        v = 50.0 + i * 7.0
        pts_2d.append((u, v))
        pt_w = camera.unproject(u, v, z_true)
        pts_3d.append((pt_w.x, pt_w.y, pt_w.z))

        # True disparity: 1 / z = true_scale * d_rel + true_shift
        # => d_rel = ((1 / z) - true_shift) / true_scale
        d_rel = float(((1.0 / z_true) - true_shift) / true_scale)
        rel_depth_map[int(round(v)), int(round(u))] = d_rel

    result = align_relative_depth_to_metric(
        depth_map=rel_depth_map,
        camera=camera,
        reference_points_3d=pts_3d,
        pixel_coords_2d=pts_2d,
        scale_source="synthetic_test",
        min_inliers=15,
    )

    assert result.is_aligned
    assert result.scale_state == "METRIC_SCALE_ALIGNED"
    assert result.calibration_method == "affine_disparity_regression"
    assert result.scale is not None and abs(result.scale - true_scale) < 1e-3
    assert result.shift is not None and abs(result.shift - true_shift) < 1e-3
    assert result.r_squared is not None and result.r_squared > 0.99
    assert result.residual_rms_error_m is not None and result.residual_rms_error_m < 0.05
    assert result.inlier_count >= 40


def test_align_relative_depth_to_metric_rejects_insufficient_points() -> None:
    camera = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity()),
    )
    rel_depth_map = np.linspace(1.0, 5.0, 480 * 640, dtype=np.float32).reshape(480, 640)
    # Only 5 points when min_inliers=15
    pts_3d = [(0.0, 0.0, float(i + 2)) for i in range(5)]
    pts_2d = [(float(50 + i * 10), float(50 + i * 10)) for i in range(5)]

    result = align_relative_depth_to_metric(
        depth_map=rel_depth_map,
        camera=camera,
        reference_points_3d=pts_3d,
        pixel_coords_2d=pts_2d,
        min_inliers=15,
    )

    assert not result.is_aligned
    assert result.scale_state == "RELATIVE_UNSCALED"
    assert result.rejection_reason is not None and "INSUFFICIENT_CORRESPONDENCES" in result.rejection_reason
