"""Tests for Epic E03-1: Multi-View Dense Cloud Fusion.

Validates:
1. Multi-view ray unprojection and co-visibility depth consistency voting.
2. Rejection of monocular single-view depth estimation hallucinations.
3. Metric density verification (> 500 pts/m^3).
4. Chamfer distance (<= 15mm) and F-Score against ground truth geometry.
5. Multi-view radiance color blending across co-visible camera views.
6. Consistent surface normal orientation towards observing cameras.
7. Zero-configuration automatic stride and view-threshold tuning.
"""

import math
from pathlib import Path
import numpy as np
import pytest

from engine.math.math3 import Quat, Vec3
from perception.depth.interface import DepthMap
from reconstruction.calibration.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera
from reconstruction.mvs.dense_cloud import DenseCloudResult, MultiViewDenseCloudBuilder
from reconstruction.mvs.point_cloud_ops import (
    chamfer_distance,
    compute_point_cloud_density,
    estimate_point_normals,
    export_point_cloud_ply,
    f_score_tau,
    statistical_outlier_removal,
    voxel_downsample,
)


def test_chamfer_distance_and_f_score_exact_and_shifted():
    # 1. Exact identical sets
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ], dtype=np.float64)

    cd, a_to_b, b_to_a = chamfer_distance(pts, pts, return_components=True)
    assert cd == pytest.approx(0.0, abs=1e-6)
    assert a_to_b == pytest.approx(0.0, abs=1e-6)
    assert b_to_a == pytest.approx(0.0, abs=1e-6)

    prec, rec, f1 = f_score_tau(pts, pts, tau=0.01)
    assert prec == pytest.approx(1.0)
    assert rec == pytest.approx(1.0)
    assert f1 == pytest.approx(1.0)

    # 2. Rigid shift by 10mm (0.01m) along X axis
    shift = np.array([0.01, 0.0, 0.0])
    pts_shifted = pts + shift

    cd_shifted, d_a_b, d_b_a = chamfer_distance(pts, pts_shifted, return_components=True)
    assert d_a_b == pytest.approx(0.01, abs=1e-5)
    assert d_b_a == pytest.approx(0.01, abs=1e-5)
    assert cd_shifted == pytest.approx(0.02, abs=1e-5)

    # At tau = 0.015m, all points are within tolerance
    p15, r15, f15 = f_score_tau(pts, pts_shifted, tau=0.015)
    assert p15 == pytest.approx(1.0)
    assert r15 == pytest.approx(1.0)
    assert f15 == pytest.approx(1.0)

    # At tau = 0.005m, no points are within tolerance
    p05, r05, f05 = f_score_tau(pts, pts_shifted, tau=0.005)
    assert p05 == pytest.approx(0.0)
    assert r05 == pytest.approx(0.0)
    assert f05 == pytest.approx(0.0)


def test_point_cloud_ply_export_and_inspection(tmp_path: Path):
    ply_path = tmp_path / "test_cloud.ply"
    pts = np.array([
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
    ], dtype=np.float32)
    cols = np.array([
        [255, 0, 0],
        [0, 255, 0],
    ], dtype=np.uint8)
    norms = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)

    # Test binary PLY
    export_point_cloud_ply(ply_path, pts, colors=cols, normals=norms, binary=True)
    assert ply_path.exists()
    assert ply_path.stat().st_size > 50

    with open(ply_path, "rb") as f:
        header = f.read(250)
        assert b"format binary_little_endian" in header
        assert b"element vertex 2" in header

    # Test ASCII PLY
    ascii_path = tmp_path / "test_cloud_ascii.ply"
    export_point_cloud_ply(ascii_path, pts, colors=cols, normals=norms, binary=False)
    assert ascii_path.exists()
    content = ascii_path.read_text(encoding="ascii")
    assert "format ascii 1.0" in content
    assert "element vertex 2" in content
    assert "0.000000 1.000000 2.000000" in content


def test_multiview_depth_hallucination_rejection():
    """Verify co-visibility consistency rejects single-view depth estimation hallucinations."""
    intrinsics = CameraIntrinsics(fx=400.0, fy=400.0, cx=160.0, cy=120.0, width=320, height=240)

    # Camera 1 at origin
    cam1 = PinholeCamera(
        intrinsics=intrinsics,
        extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
    )
    # Camera 2 displaced horizontally by 0.3m
    cam2 = PinholeCamera(
        intrinsics=intrinsics,
        extrinsics=CameraExtrinsics(position=Vec3(0.3, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
    )

    # True physical wall is at Z = 2.5m in world coordinates
    # Both cameras see the wall at depth ~ 2.5m
    depth1_arr = np.full((240, 320), 2.5, dtype=np.float64)
    depth2_arr = np.full((240, 320), 2.5, dtype=np.float64)

    # Inject an artificial monocular depth hallucination in Camera 1:
    # A 40x40 pixel block falsely estimated at Z = 1.0m (floating in mid-air)
    depth1_arr[100:140, 140:180] = 1.0

    d1 = DepthMap(evidence_id="cam1", width=320, height=240, values=depth1_arr.tolist(), unit="meters")
    d2 = DepthMap(evidence_id="cam2", width=320, height=240, values=depth2_arr.tolist(), unit="meters")

    # 1. With consistency check and min_consistent_views=2, hallucination must be filtered
    builder_strict = MultiViewDenseCloudBuilder(
        stride=4,
        consistency_check=True,
        depth_tolerance=0.08,
        min_consistent_views=2,
        voxel_size_m=0.05,
    )
    result_strict = builder_strict.fuse(
        cameras={"cam1": cam1, "cam2": cam2},
        depth_maps={"cam1": d1, "cam2": d2},
    )

    assert result_strict.point_count > 0
    # Every accepted point must lie near the real wall at Z = 2.5m; none at Z = 1.0m
    assert np.all(result_strict.points[:, 2] >= 2.3)
    assert not np.any(result_strict.points[:, 2] < 1.5)

    # 2. When consistency check is disabled, hallucination points leak through
    builder_naive = MultiViewDenseCloudBuilder(
        stride=4,
        consistency_check=False,
        voxel_size_m=0.05,
    )
    result_naive = builder_naive.fuse(
        cameras={"cam1": cam1, "cam2": cam2},
        depth_maps={"cam1": d1, "cam2": d2},
    )
    assert np.any(result_naive.points[:, 2] < 1.5)


def test_dense_fusion_synthetic_room_tls_ground_truth():
    """Verify density (> 500 pts/m^3) and Chamfer distance (<= 15mm) against TLS ground truth."""
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=200.0, cy=150.0, width=400, height=300)

    # 3 cameras looking towards a target flat wall at z = 3.0m spanning [-1.5, 1.5] x [-1.0, 1.0]
    cameras = {
        "cam0": PinholeCamera(
            intrinsics=intrinsics,
            extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
        ),
        "cam1": PinholeCamera(
            intrinsics=intrinsics,
            extrinsics=CameraExtrinsics(position=Vec3(0.25, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
        ),
        "cam2": PinholeCamera(
            intrinsics=intrinsics,
            extrinsics=CameraExtrinsics(position=Vec3(-0.25, 0.05, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)),
        ),
    }

    # Ground truth wall geometry at Z = 3.0m
    wall_z = 3.0
    depth_maps = {}
    for cid, cam in cameras.items():
        # Wall is at z = 3.0m in world space, cam has z = 0.0, so depth is 3.0m
        vals = np.full((300, 400), wall_z, dtype=np.float64)
        depth_maps[cid] = DepthMap(evidence_id=cid, width=400, height=300, values=vals.tolist(), unit="meters")

    builder = MultiViewDenseCloudBuilder(
        stride=2,
        consistency_check=True,
        depth_tolerance=0.05,
        min_consistent_views=2,
        voxel_size_m=0.02,  # 2cm resolution
        sor_std_ratio=2.0,
    )

    result = builder.fuse(cameras=cameras, depth_maps=depth_maps)

    # Generate dense TLS ground truth point cloud on the planar wall covering observed bounds
    bmin, bmax = result.bounds_min, result.bounds_max
    gx = np.linspace(bmin[0], bmax[0], int((bmax[0] - bmin[0]) / 0.005))
    gy = np.linspace(bmin[1], bmax[1], int((bmax[1] - bmin[1]) / 0.005))
    gxx, gyy = np.meshgrid(gx, gy)
    tls_ground_truth = np.stack([gxx.ravel(), gyy.ravel(), np.full(gxx.size, wall_z)], axis=1)

    assert result.point_count > 1000
    assert result.view_count == 3
    assert result.normals is not None

    # Verification Criterion 1: Point cloud density > 500 pts/m^3
    assert result.density_points_per_m3 > 500.0, f"Density was {result.density_points_per_m3} pts/m^3"

    # Verification Criterion 2: Chamfer distance <= 15mm (0.015m)
    cd = chamfer_distance(result.points, tls_ground_truth)
    assert cd <= 0.015, f"Chamfer distance was {cd * 1000.0:.2f} mm (exceeded 15mm limit)"

    # Verification Criterion 3: F-Score@15mm >= 95%
    precision, recall, f1 = f_score_tau(result.points, tls_ground_truth, tau=0.015)
    assert precision >= 0.95, f"Precision was {precision * 100:.1f}%"
    assert recall >= 0.95, f"Recall was {recall * 100:.1f}%"
    assert f1 >= 0.95, f"F1 score was {f1 * 100:.1f}%"

    # Verification Criterion 4: Surface normals oriented toward cameras (+Z points away, normal must point toward -Z / cam)
    # Since camera is at Z=0 and wall is at Z=3, normal oriented towards camera must have negative Z component
    mean_normal = np.mean(result.normals, axis=0)
    assert mean_normal[2] < -0.85


def test_radiance_color_blending():
    """Verify multi-view radiance blending averages color smoothly across views."""
    intrinsics = CameraIntrinsics(fx=400.0, fy=400.0, cx=100.0, cy=100.0, width=200, height=200)

    cam1 = PinholeCamera(intrinsics=intrinsics, extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)))
    cam2 = PinholeCamera(intrinsics=intrinsics, extrinsics=CameraExtrinsics(position=Vec3(0.1, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)))

    depth_arr = np.full((200, 200), 2.0, dtype=np.float64)
    d1 = DepthMap(evidence_id="c1", width=200, height=200, values=depth_arr.tolist(), unit="meters")
    d2 = DepthMap(evidence_id="c2", width=200, height=200, values=depth_arr.tolist(), unit="meters")

    # Cam1 observes surface with RGB [100, 100, 100]
    # Cam2 observes surface with RGB [200, 200, 200]
    img1 = np.full((200, 200, 3), 100, dtype=np.uint8)
    img2 = np.full((200, 200, 3), 200, dtype=np.uint8)

    builder = MultiViewDenseCloudBuilder(
        stride=4,
        consistency_check=True,
        min_consistent_views=2,
        radiance_blending=True,
        voxel_size_m=0.04,
    )

    result = builder.fuse(
        cameras={"c1": cam1, "c2": cam2},
        depth_maps={"c1": d1, "c2": d2},
        images_rgb={"c1": img1, "c2": img2},
    )

    assert result.colors is not None
    assert result.point_count > 0
    # Mean color across blended views should be near 150
    mean_col = np.mean(result.colors, axis=0)
    assert abs(mean_col[0] - 150.0) < 5.0
    assert abs(mean_col[1] - 150.0) < 5.0
    assert abs(mean_col[2] - 150.0) < 5.0


def test_reconstructed_point_conversion_and_diagnostics():
    """Verify conversion of DenseCloudResult to ReconstructedPoint instances."""
    intrinsics = CameraIntrinsics(fx=300.0, fy=300.0, cx=80.0, cy=60.0, width=160, height=120)
    cam = PinholeCamera(intrinsics=intrinsics, extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)))

    depth = DepthMap(evidence_id="v1", width=160, height=120, values=[[2.0] * 160 for _ in range(120)], unit="meters")
    img = np.full((120, 160, 3), 180, dtype=np.uint8)

    builder = MultiViewDenseCloudBuilder(stride=8, voxel_size_m=0.05)
    result = builder.fuse(cameras={"v1": cam}, depth_maps={"v1": depth}, images_rgb={"v1": img})

    assert result.point_count > 0
    rec_points = result.to_reconstructed_points(default_track_prefix="dense_mvs", source_evidence_ids=["v1"])

    assert len(rec_points) == result.point_count
    pt0 = rec_points[0]
    assert pt0.track_id.startswith("dense_mvs:")
    assert pt0.source_evidence_ids == ["v1"]
    assert pt0.color == (180, 180, 180)
    assert 0.0 <= pt0.uncertainty.confidence <= 1.0


def test_results_over_technologies_auto_tuning():
    """Verify automatic stride and parameter resolution with zero user configuration."""
    intrinsics = CameraIntrinsics(fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480)
    cam1 = PinholeCamera(intrinsics=intrinsics, extrinsics=CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)))
    cam2 = PinholeCamera(intrinsics=intrinsics, extrinsics=CameraExtrinsics(position=Vec3(0.2, 0.0, 0.0), rotation=Quat(1.0, 0.0, 0.0, 0.0)))

    d1 = DepthMap(evidence_id="cam1", width=640, height=480, values=np.full((480, 640), 2.5).tolist(), unit="meters")
    d2 = DepthMap(evidence_id="cam2", width=640, height=480, values=np.full((480, 640), 2.5).tolist(), unit="meters")

    # Instantiate with stride=None and min_consistent_views=None
    builder = MultiViewDenseCloudBuilder(stride=None, min_consistent_views=None)
    result = builder.fuse(cameras={"cam1": cam1, "cam2": cam2}, depth_maps={"cam1": d1, "cam2": d2})

    assert result.point_count > 0
    assert result.diagnostics["status"] == "success"
    # Auto stride for 640x480 is 2
    assert result.diagnostics["stride"] == 2
    # Auto min_consistent_views for 2 views is 2
    assert result.diagnostics["min_consistent_views"] == 2
