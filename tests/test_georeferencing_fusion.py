"""Comprehensive unit tests for Georeferencing and Global Coordinate Fusion Engine (Block 19 & Block 37)."""

import math
import numpy as np
import pytest

from reconstruction.scale.georeferencing import (
    GeodeticPoint,
    align_georeference_ransac,
    ecef_to_lla,
    enu_to_lla,
    georeference_world_from_gnss,
    lla_to_ecef,
    lla_to_enu,
    solve_umeyama_sim3,
)
from world_ir.coordinates import Frame
from world_ir.world_v1 import WorldIR


def test_lla_to_ecef_and_back_roundtrip():
    """Verify WGS84 Geodetic to ECEF and inverse Bowring round-trip within millimeter precision."""
    test_cases = [
        (37.7749, -122.4194, 30.5),     # San Francisco
        (51.5074, -0.1278, 15.0),       # London
        (35.6762, 139.6503, 40.0),      # Tokyo
        (-33.8688, 151.2093, 25.0),     # Sydney
        (0.0, 0.0, 0.0),                # Prime meridian / equator
    ]

    for lat, lon, alt in test_cases:
        x, y, z = lla_to_ecef(lat, lon, alt)
        lat_back, lon_back, alt_back = ecef_to_lla(x, y, z)

        assert abs(lat_back - lat) < 1e-7, f"Latitude discrepancy: {lat_back} vs {lat}"
        assert abs(lon_back - lon) < 1e-7, f"Longitude discrepancy: {lon_back} vs {lon}"
        assert abs(alt_back - alt) < 1e-3, f"Altitude discrepancy: {alt_back} vs {alt}"


def test_lla_to_enu_and_back_roundtrip():
    """Verify local East-North-Up tangent plane conversions roundtrip cleanly."""
    ref_lat, ref_lon, ref_alt = 37.7749, -122.4194, 10.0

    # Test points offset by several hundred meters in various directions
    test_offsets = [
        (37.7758, -122.4180, 15.0),
        (37.7740, -122.4205, 5.0),
        (37.7760, -122.4210, 22.0),
    ]

    for lat, lon, alt in test_offsets:
        enu = lla_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)
        lat_b, lon_b, alt_b = enu_to_lla(enu[0], enu[1], enu[2], ref_lat, ref_lon, ref_alt)

        assert abs(lat_b - lat) < 1e-6
        assert abs(lon_b - lon) < 1e-6
        assert abs(alt_b - alt) < 1e-2


def test_umeyama_sim3_exact_recovery():
    """Verify Umeyama closed-form Sim(3) solver recovers exact known rotation, scale, and translation."""
    # Synthetic source points
    src = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0],
        [0.0, 15.0, 0.0],
        [5.0, 5.0, 8.0],
        [12.0, 8.0, -3.0],
    ], dtype=np.float64)

    # Known ground-truth transformation parameters
    gt_scale = 2.5
    theta = math.radians(35.0)
    gt_rot = np.array([
        [math.cos(theta), -math.sin(theta), 0.0],
        [math.sin(theta), math.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    gt_trans = np.array([120.0, -45.0, 15.0], dtype=np.float64)

    # Generate exact destination points
    dst = gt_scale * (src @ gt_rot.T) + gt_trans

    scale, rot, trans = solve_umeyama_sim3(src, dst, with_scale=True)

    assert abs(scale - gt_scale) < 1e-5
    np.testing.assert_allclose(rot, gt_rot, atol=1e-5)
    np.testing.assert_allclose(trans, gt_trans, atol=1e-5)


def test_align_georeference_ransac_filters_multipath_outliers():
    """Verify RANSAC successfully filters bad GNSS multipath spikes and converges to high inlier accuracy."""
    ref_origin = GeodeticPoint(37.7749, -122.4194, 10.0)

    # 10 local camera station positions along a capture trajectory
    src_pts = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 1.0, 0.2],
        [10.0, 2.0, 0.3],
        [15.0, 3.0, 0.5],
        [20.0, 4.0, 0.6],
        [25.0, 5.0, 0.8],
        [30.0, 6.0, 1.0],
        [35.0, 7.0, 1.1],
        [40.0, 8.0, 1.2],
        [45.0, 9.0, 1.3],
    ], dtype=np.float64)

    gt_scale = 1.05
    gt_rot = np.eye(3)
    gt_trans = np.array([50.0, 25.0, 2.0])

    dst_enu = gt_scale * (src_pts @ gt_rot.T) + gt_trans

    # Inject 2 severe GNSS multipath outliers (> 20 meters error)
    dst_enu[3] += np.array([25.0, -30.0, 15.0])
    dst_enu[7] += np.array([-40.0, 35.0, -20.0])

    result = align_georeference_ransac(
        src_local_pts=src_pts,
        dst_enu_pts=dst_enu,
        ref_origin=ref_origin,
        max_inlier_residual_m=2.0,
    )

    assert result.success is True
    assert 3 not in result.inlier_indices, "Outlier index 3 should be rejected by RANSAC"
    assert 7 not in result.inlier_indices, "Outlier index 7 should be rejected by RANSAC"
    assert len(result.inlier_indices) == 8
    assert result.diagnostics.rmse_horizontal_m < 0.1
    assert result.diagnostics.rmse_vertical_m < 0.1


def test_georeference_world_from_gnss_promotes_world_ir():
    """Verify georeference_world_from_gnss seamlessly updates WorldIR to Frame.ENU and records transforms."""
    world = WorldIR()
    assert world.coordinate_frame == Frame.WORLD

    camera_poses = [
        ("cam_01", np.array([0.0, 0.0, 0.0])),
        ("cam_02", np.array([10.0, 0.0, 0.0])),
        ("cam_03", np.array([20.0, 0.0, 0.0])),
        ("cam_04", np.array([20.0, 15.0, 0.0])),
        ("cam_05", np.array([0.0, 15.0, 0.0])),
    ]

    ref_lat, ref_lon, ref_alt = 28.6139, 77.2090, 215.0

    # Generate synthetic GNSS coordinates at expected ENU offsets
    gnss_records = []
    for ev_id, local_xyz in camera_poses:
        # Scale 1.0, translation E=100, N=200, U=0
        enu_e = float(local_xyz[0]) + 100.0
        enu_n = float(local_xyz[1]) + 200.0
        enu_u = float(local_xyz[2])
        lat, lon, alt = enu_to_lla(enu_e, enu_n, enu_u, ref_lat, ref_lon, ref_alt)
        gnss_records.append((ev_id, lat, lon, alt))

    result = georeference_world_from_gnss(
        world=world,
        camera_poses=camera_poses,
        gnss_records=gnss_records,
    )

    assert result.success is True
    assert world.coordinate_frame == Frame.ENU
    assert "world_to_enu" in world.transforms
    assert world.transforms["world_to_enu"].source_frame == Frame.WORLD
    assert world.transforms["world_to_enu"].target_frame == Frame.ENU
    assert "georeference" in world.metadata
    assert world.metadata["georeference"]["status"] == "georeferenced"
    assert result.diagnostics.rmse_total_3d_m < 0.1
