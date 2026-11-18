"""Deterministic tests for robust multi-sensor evidence fusion (P6-02 / Phase 9).

Validates:
  1. Statistical consensus: agreeing sensors (LiDAR, stereo, RGB-D) yield tight inverse-variance estimate.
  2. Outlier rejection: anomalous readings (e.g. 10-sigma multipath reflection / mono failure)
     are downweighted and catalogued in `outliers` rather than corrupting the consensus.
  3. Preserved audit trail: `Provenance.CONFLICT` honestly marked on outlier presence; zero raw data lost.
  4. Multi-sensor 3D point fusion across multiple streams.
"""

from __future__ import annotations

from engine.core.units import Unit
from provenance import Provenance
from reconstruction.fusion.fusion import FusableObservation
from reconstruction.fusion.multi_source import DepthSourceObservation
from reconstruction.fusion.robust_fusion import (
    fuse_multisensor_points_3d,
    robust_fuse_depth_observations,
    robust_fuse_quantity,
)

TRUE_DEPTH_M = 12.0


def test_robust_fusion_consensus():
    """When all sensors report values consistent with their claimed precision,

    the robust estimator matches standard inverse-variance weighted mean and flags zero outliers.
    """
    lidar = DepthSourceObservation(
        source="lidar",
        depth_m=TRUE_DEPTH_M + 0.01,
        precision_m=0.02,
        provenance=Provenance.OBSERVED,
        confidence=0.98,
    )
    rgbd = DepthSourceObservation(
        source="rgbd",
        depth_m=TRUE_DEPTH_M - 0.02,
        precision_m=0.03,
        provenance=Provenance.OBSERVED,
        confidence=0.95,
    )
    stereo = DepthSourceObservation(
        source="stereo",
        depth_m=TRUE_DEPTH_M + 0.05,
        precision_m=0.08,
        provenance=Provenance.OBSERVED,
        confidence=0.85,
    )

    result = robust_fuse_depth_observations([lidar, rgbd, stereo], point_id="p-consensus")

    assert not result.has_outliers()
    assert len(result.inliers) == 3
    assert abs(result.value - TRUE_DEPTH_M) < 0.02
    assert result.precision < 0.02  # fused precision strictly tighter than best sensor alone
    assert result.unit is Unit.METER
    assert result.provenance is Provenance.ESTIMATED


def test_robust_fusion_outlier_downweighting():
    """When a rogue sensor reports a 10-sigma anomaly (e.g. monocular failure at 22m),

    robust fusion prevents it from corrupting the spatial consensus around 12m,
    catalogues it in `outliers`, and sets Provenance to CONFLICT.
    """
    lidar = DepthSourceObservation(
        source="lidar",
        depth_m=TRUE_DEPTH_M + 0.01,
        precision_m=0.02,
        provenance=Provenance.OBSERVED,
        confidence=0.98,
    )
    rgbd = DepthSourceObservation(
        source="rgbd",
        depth_m=TRUE_DEPTH_M - 0.01,
        precision_m=0.03,
        provenance=Provenance.OBSERVED,
        confidence=0.95,
    )
    # Rogue outlier: reports 22.0 meters (10 meters off true depth)
    rogue_mono = DepthSourceObservation(
        source="monocular_depth",
        depth_m=TRUE_DEPTH_M + 10.0,
        precision_m=0.15,
        provenance=Provenance.ESTIMATED,
        confidence=0.70,
    )

    result = robust_fuse_depth_observations(
        [lidar, rgbd, rogue_mono], point_id="p-outlier", loss="huber", tuning_k=2.5
    )

    # 1. Consensus preserved: fused estimate remains very close to LiDAR + RGB-D
    assert abs(result.value - TRUE_DEPTH_M) < 0.03
    assert abs(result.value - 12.0) < 0.05

    # 2. Outlier preserved in audit trail (not destroyed or hidden)
    assert result.has_outliers()
    assert any(o.source == "monocular_depth" for o in result.outliers)
    assert len(result.inliers) == 2

    # 3. Provenance honestly flagged as CONFLICT
    assert result.provenance is Provenance.CONFLICT

    # 4. Outlier's effective weight was severely decayed
    w_lidar = result.weights["lidar"]
    w_mono = result.weights["monocular_depth"]
    assert w_lidar > 100.0 * w_mono


def test_robust_fusion_tukey_trimming():
    """Tukey biweight completely zeroes weights for extreme gross outliers."""
    o1 = FusableObservation(
        value=5.0,
        unit=Unit.METER,
        source="sensor_a",
        provenance=Provenance.OBSERVED,
        confidence=0.9,
        precision=0.05,
    )
    o2 = FusableObservation(
        value=5.02,
        unit=Unit.METER,
        source="sensor_b",
        provenance=Provenance.OBSERVED,
        confidence=0.9,
        precision=0.05,
    )
    o_bad = FusableObservation(
        value=50.0,
        unit=Unit.METER,
        source="sensor_broken",
        provenance=Provenance.OBSERVED,
        confidence=0.5,
        precision=0.5,
    )

    res = robust_fuse_quantity([o1, o2, o_bad], quantity="dist", loss="tukey")
    assert abs(res.value - 5.01) < 0.02
    assert res.weights["sensor_broken"] == 0.0
    assert any(o.source == "sensor_broken" for o in res.outliers)


def test_robust_multisensor_points_3d():
    """Fuses 3D point cloud streams across multiple sensors with noise."""
    sensor_lidar = [
        (1.0, 2.0, 3.0),
        (2.0, 3.0, 4.0),
        (3.0, 4.0, 5.0),
    ]
    sensor_depth = [
        (1.02, 1.98, 3.01),
        (1.99, 3.02, 3.99),
        (3.01, 3.99, 5.02),
    ]
    # Sensor 3 has a gross glitch on point 2
    sensor_stereo = [
        (1.01, 2.01, 2.99),
        (20.0, 30.0, 40.0),  # gross glitch!
        (2.99, 4.01, 4.98),
    ]

    points_by_source = {
        "lidar": sensor_lidar,
        "depth": sensor_depth,
        "stereo": sensor_stereo,
    }
    precisions = {
        "lidar": 0.02,
        "depth": 0.04,
        "stereo": 0.05,
    }

    fused_pts = fuse_multisensor_points_3d(points_by_source, precisions)
    assert len(fused_pts) == 3

    # Point 0: tight consensus
    assert abs(fused_pts[0][0] - 1.0) < 0.03
    assert abs(fused_pts[0][0] - 1.0) < 0.03
    assert abs(fused_pts[0][1] - 2.0) < 0.03
    assert abs(fused_pts[0][2] - 3.0) < 0.03

    # Point 1: glitch on stereo ignored, stays near (2.0, 3.0, 4.0)
    assert abs(fused_pts[1][0] - 2.0) < 0.05
    assert abs(fused_pts[1][1] - 3.0) < 0.05
    assert abs(fused_pts[1][2] - 4.0) < 0.05

    # Point 2: tight consensus
    assert abs(fused_pts[2][0] - 3.0) < 0.03
    assert abs(fused_pts[2][1] - 4.0) < 0.03
    assert abs(fused_pts[2][2] - 5.0) < 0.03
