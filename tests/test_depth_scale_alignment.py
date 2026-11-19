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
    MetricAnchor,
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


def test_align_relative_depth_without_metric_anchor_yields_sfm_unmetric() -> None:
    """When no metric anchor is provided (e.g. standard COLMAP sparse points),

    the output scale state is SFM_ALIGNED_UNMETRIC and units are COLMAP_SFM_UNITS.
    It strictly refuses to claim is_physically_metric=True.
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

    for i in range(50):
        z_true = 2.0 + (i / 50.0) * 6.0
        u = 50.0 + i * 10.0
        v = 50.0 + i * 7.0
        pts_2d.append((u, v))
        pt_w = camera.unproject(u, v, z_true)
        pts_3d.append((pt_w.x, pt_w.y, pt_w.z))

        d_rel = float(((1.0 / z_true) - true_shift) / true_scale)
        rel_depth_map[int(round(v)), int(round(u))] = d_rel

    result = align_relative_depth_to_metric(
        depth_map=rel_depth_map,
        camera=camera,
        reference_points_3d=pts_3d,
        pixel_coords_2d=pts_2d,
        scale_source="colmap_sparse_points",
        metric_anchor=None,  # No independent metric anchor
        min_inliers=15,
    )

    assert result.is_aligned
    assert result.scale_state == "SFM_ALIGNED_UNMETRIC"
    assert result.units == "COLMAP_SFM_UNITS"
    assert not result.is_physically_metric
    assert result.calibration_method == "affine_disparity_regression"
    assert result.scale is not None and abs(result.scale - true_scale) < 1e-3
    assert result.shift is not None and abs(result.shift - true_shift) < 1e-3
    assert result.r_squared is not None and result.r_squared > 0.99
    assert result.residual_rms_error_m is not None and result.residual_rms_error_m < 0.05
    assert result.held_out_rms_error_m is not None and result.held_out_rms_error_m < 0.05
    assert result.spatial_residual_std_m is not None
    assert result.held_out_count > 0
    assert result.scale_uncertainty is not None
    assert result.shift_uncertainty is not None


def test_align_relative_depth_with_verified_metric_anchor() -> None:
    """When a verified metric anchor is provided, scale is converted to physical METRES

    and scale_state is METRIC_SCALE_ALIGNED with is_physically_metric=True.
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

    for i in range(50):
        z_true = 2.0 + (i / 50.0) * 6.0
        u = 50.0 + i * 10.0
        v = 50.0 + i * 7.0
        pts_2d.append((u, v))
        pt_w = camera.unproject(u, v, z_true)
        pts_3d.append((pt_w.x, pt_w.y, pt_w.z))

        d_rel = float(((1.0 / z_true) - true_shift) / true_scale)
        rel_depth_map[int(round(v)), int(round(u))] = d_rel

    # Anchor between point 0 and point 49: known physical laser distance = 6.0 meters
    p0 = pts_3d[0]
    p49 = pts_3d[49]
    dist_sfm = float(np.linalg.norm(np.array(p0) - np.array(p49)))
    metric_anchor = MetricAnchor(
        anchor_type="laser_distance_meter",
        measured_distance_m=dist_sfm * 2.0,  # Physical distance is 2x SfM units
        uncertainty_m=0.005,
        reference_point_a=p0,
        reference_point_b=p49,
        notes="Physical Leica Disto D2 measurement",
    )

    result = align_relative_depth_to_metric(
        depth_map=rel_depth_map,
        camera=camera,
        reference_points_3d=pts_3d,
        pixel_coords_2d=pts_2d,
        scale_source="colmap_with_laser_anchor",
        metric_anchor=metric_anchor,
        min_inliers=15,
    )

    assert result.is_aligned
    assert result.scale_state == "METRIC_SCALE_ALIGNED"
    assert result.units == "METRES"
    assert result.is_physically_metric
    assert result.metric_anchor is not None
    assert result.metric_anchor.measured_distance_m == dist_sfm * 2.0
    # Aligned depth map values should be scaled by 2.0
    assert result.aligned_depth_map is not None
    # Check that scale factor of 2.0 was applied to scale and shift
    assert abs(result.scale - true_scale * 2.0) < 1e-2


def test_align_relative_depth_rejects_invalid_metric_anchor() -> None:
    camera = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity()),
    )
    rel_depth_map = np.linspace(1.0, 5.0, 480 * 640, dtype=np.float32).reshape(480, 640)
    pts_3d = [(0.0, 0.0, float(i + 2)) for i in range(20)]
    pts_2d = [(float(50 + i * 10), float(50 + i * 10)) for i in range(20)]

    # Negative measured distance
    invalid_anchor = MetricAnchor(
        anchor_type="laser",
        measured_distance_m=-5.0,
        reference_point_a=(0.0, 0.0, 1.0),
        reference_point_b=(0.0, 0.0, 2.0),
    )
    result = align_relative_depth_to_metric(
        depth_map=rel_depth_map,
        camera=camera,
        reference_points_3d=pts_3d,
        pixel_coords_2d=pts_2d,
        metric_anchor=invalid_anchor,
    )
    assert not result.is_aligned
    assert result.scale_state == "INVALID"
    assert result.rejection_reason is not None and "measured_distance_m must be strictly positive" in result.rejection_reason

    # Coincident reference points (distance == 0)
    zero_dist_anchor = MetricAnchor(
        anchor_type="laser",
        measured_distance_m=10.0,
        reference_point_a=(1.0, 2.0, 3.0),
        reference_point_b=(1.0, 2.0, 3.0),
    )
    result2 = align_relative_depth_to_metric(
        depth_map=rel_depth_map,
        camera=camera,
        reference_points_3d=pts_3d,
        pixel_coords_2d=pts_2d,
        metric_anchor=zero_dist_anchor,
    )
    assert not result2.is_aligned
    assert result2.scale_state == "INVALID"
    assert result2.rejection_reason is not None and "Distance between reference points is zero" in result2.rejection_reason


def test_adversarial_metric_anchor_missing_reference_points() -> None:
    """A positive measured distance must be rejected if reference points are missing."""
    camera = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity()),
    )
    rel_depth_map = np.linspace(1.0, 5.0, 480 * 640, dtype=np.float32).reshape(480, 640)
    pts_3d = [(0.0, 0.0, float(i + 2)) for i in range(20)]
    pts_2d = [(float(50 + i * 10), float(50 + i * 10)) for i in range(20)]

    anchor = MetricAnchor(
        anchor_type="laser_survey",
        measured_distance_m=5.0,
        reference_point_a=None,
        reference_point_b=None,
    )
    res = align_relative_depth_to_metric(
        depth_map=rel_depth_map,
        camera=camera,
        reference_points_3d=pts_3d,
        pixel_coords_2d=pts_2d,
        metric_anchor=anchor,
    )
    assert not res.is_aligned
    assert res.scale_state == "INVALID"
    assert res.rejection_reason is not None and "Missing reference points" in res.rejection_reason


def test_adversarial_metric_anchor_non_finite_values() -> None:
    """Non-finite values (NaN, Inf) in coordinates, distance or uncertainty must be rejected."""
    camera = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity()),
    )
    rel_depth_map = np.linspace(1.0, 5.0, 480 * 640, dtype=np.float32).reshape(480, 640)
    pts_3d = [(0.0, 0.0, float(i + 2)) for i in range(20)]
    pts_2d = [(float(50 + i * 10), float(50 + i * 10)) for i in range(20)]

    # NaN in measured distance
    anchor_nan_dist = MetricAnchor(
        anchor_type="laser",
        measured_distance_m=float("nan"),
        reference_point_a=(0.0, 0.0, 0.0),
        reference_point_b=(1.0, 1.0, 1.0),
    )
    res = align_relative_depth_to_metric(rel_depth_map, camera, pts_3d, pts_2d, metric_anchor=anchor_nan_dist)
    assert not res.is_aligned
    assert res.scale_state == "INVALID"
    assert "measured_distance_m must be a finite number" in res.rejection_reason

    # Inf in reference point
    anchor_inf_pt = MetricAnchor(
        anchor_type="laser",
        measured_distance_m=5.0,
        reference_point_a=(0.0, 0.0, float("inf")),
        reference_point_b=(1.0, 1.0, 1.0),
    )
    res = align_relative_depth_to_metric(rel_depth_map, camera, pts_3d, pts_2d, metric_anchor=anchor_inf_pt)
    assert not res.is_aligned
    assert res.scale_state == "INVALID"
    assert "contains non-finite or non-numeric values" in res.rejection_reason

    # Negative uncertainty
    anchor_neg_unc = MetricAnchor(
        anchor_type="laser",
        measured_distance_m=5.0,
        uncertainty_m=-0.1,
        reference_point_a=(0.0, 0.0, 0.0),
        reference_point_b=(1.0, 1.0, 1.0),
    )
    res = align_relative_depth_to_metric(rel_depth_map, camera, pts_3d, pts_2d, metric_anchor=anchor_neg_unc)
    assert not res.is_aligned
    assert res.scale_state == "INVALID"
    assert "uncertainty_m cannot be negative" in res.rejection_reason


def test_adversarial_synthetic_anchor_rejected_by_default() -> None:
    """Synthetic ground truth anchors are rejected for production physical accuracy."""
    camera = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity()),
    )
    rel_depth_map = np.linspace(1.0, 5.0, 480 * 640, dtype=np.float32).reshape(480, 640)
    pts_3d = [(0.0, 0.0, float(i + 2)) for i in range(20)]
    pts_2d = [(float(50 + i * 10), float(50 + i * 10)) for i in range(20)]

    synth_anchor = MetricAnchor(
        anchor_type="synthetic_ground_truth",
        measured_distance_m=5.0,
        reference_point_a=(0.0, 0.0, 1.0),
        reference_point_b=(0.0, 0.0, 6.0),
    )
    # Default: allow_synthetic=False -> rejected!
    res = align_relative_depth_to_metric(rel_depth_map, camera, pts_3d, pts_2d, metric_anchor=synth_anchor)
    assert not res.is_aligned
    assert res.scale_state == "INVALID"
    assert "Synthetic ground truth cannot establish production physical accuracy" in res.rejection_reason

    # When allow_synthetic=True: is_physically_metric MUST remain False!
    true_scale = 0.5
    true_shift = 0.1
    for i in range(20):
        z_true = 2.0 + (i / 20.0) * 6.0
        d_rel = float(((1.0 / z_true) - true_shift) / true_scale)
        u, v = pts_2d[i]
        rel_depth_map[int(round(v)), int(round(u))] = d_rel
        pt_w = camera.unproject(u, v, z_true)
        pts_3d[i] = (pt_w.x, pt_w.y, pt_w.z)

    synth_anchor_valid = MetricAnchor(
        anchor_type="synthetic_ground_truth",
        measured_distance_m=5.0,
        reference_point_a=pts_3d[0],
        reference_point_b=pts_3d[10],
    )
    res_allowed = align_relative_depth_to_metric(
        rel_depth_map, camera, pts_3d, pts_2d, metric_anchor=synth_anchor_valid, allow_synthetic=True
    )
    assert res_allowed.is_aligned
    assert not res_allowed.is_physically_metric
    assert res_allowed.scale_state == "SYNTHETIC_SCALE_ALIGNED"


def test_adversarial_high_training_r2_with_poor_held_out_error() -> None:
    """Overfitted or corrupted fits with poor held-out generalization must be rejected."""
    camera = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity()),
    )
    rel_depth_map = np.linspace(1.0, 5.0, 480 * 640, dtype=np.float32).reshape(480, 640)

    # 30 points where training set fits linear relation perfectly, but held-out test points diverge
    rng_test = np.random.RandomState(42)
    shuffled = rng_test.permutation(30)
    test_indices = set(shuffled[21:])

    pts_3d: list[tuple[float, float, float]] = []
    pts_2d: list[tuple[float, float]] = []

    for i in range(30):
        u = float(20 + i * 15)
        v = float(20 + i * 10)
        pts_2d.append((u, v))
        z = float(2.0 + i * 0.2)
        pt_w = camera.unproject(u, v, z)
        pts_3d.append((pt_w.x, pt_w.y, pt_w.z))

        # Make training points perfectly aligned, but corrupt held-out points
        if i not in test_indices:
            rel_depth_map[int(v), int(u)] = z * 2.0  # linear relation
        else:
            rel_depth_map[int(v), int(u)] = 100.0  # severe divergence on held-out test set

    res = align_relative_depth_to_metric(
        depth_map=rel_depth_map,
        camera=camera,
        reference_points_3d=pts_3d,
        pixel_coords_2d=pts_2d,
        min_inliers=15,
        train_ratio=0.7,
    )
    assert not res.is_aligned
    assert res.scale_state == "RELATIVE_UNSCALED"
    assert "POOR_HELD_OUT_EVALUATION" in res.rejection_reason


def test_align_relative_depth_rejects_poor_fit_low_r2() -> None:
    camera = PinholeCamera(
        intrinsics=CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0, width=640, height=480),
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity()),
    )
    # Random unaligned depth map that has no correlation with true depths
    rng = np.random.RandomState(123)
    random_depth = rng.uniform(0.5, 10.0, (480, 640)).astype(np.float32)
    pts_3d = [(0.0, 0.0, float(i + 2)) for i in range(20)]
    pts_2d = [(float(50 + i * 10), float(50 + i * 10)) for i in range(20)]

    result = align_relative_depth_to_metric(
        depth_map=random_depth,
        camera=camera,
        reference_points_3d=pts_3d,
        pixel_coords_2d=pts_2d,
        min_inliers=15,
    )
    assert not result.is_aligned
    assert result.scale_state == "RELATIVE_UNSCALED"
    assert result.rejection_reason is not None and "POOR_SCALE_ALIGNMENT_FIT" in result.rejection_reason


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
