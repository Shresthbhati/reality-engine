"""Advanced test suite for robust multi-sensor fusion.

Validates:
- Small sample size handling (n=1, n=2, n=3)
- Two-sensor disagreement and conflict detection
- Bimodal distribution identification and competing mode tracking
- Inter-sensor correlation variance scaling
- Strict rejection of missing or non-positive precision (no fabricated defaults)
- Coordinate frame and physical unit consistency checks
- Ratio-test ambiguity rejection in registered point cloud fusion
- Duplicate match prevention (mutual 1:1 correspondence)
- Full unmatched point retention
"""

from __future__ import annotations

import pytest

from engine.core.units import Unit
from provenance import Provenance
from reconstruction.fusion.fusion import FusableObservation
from reconstruction.fusion.robust_fusion import (
    FusionError,
    fuse_multisensor_points_3d,
    fuse_registered_point_clouds,
    robust_fuse_quantity,
)


def test_small_sample_fusion_single_observation() -> None:
    obs = FusableObservation(
        value=3.1415,
        unit=Unit.METER,
        source="lidar_primary",
        provenance=Provenance.OBSERVED,
        confidence=0.99,
        precision=0.005,
    )
    result = robust_fuse_quantity([obs], quantity="wall_distance")
    assert result.method == "passthrough"
    assert result.value == pytest.approx(3.1415, abs=1e-5)
    assert result.precision == pytest.approx(0.005, abs=1e-5)
    assert not result.has_conflicts()
    assert not result.is_bimodal


def test_small_sample_two_sensors_agreeing() -> None:
    obs1 = FusableObservation(
        value=2.00,
        unit=Unit.METER,
        source="lidar",
        provenance=Provenance.OBSERVED,
        confidence=0.95,
        precision=0.01,
    )
    obs2 = FusableObservation(
        value=2.01,
        unit=Unit.METER,
        source="depth_cam",
        provenance=Provenance.OBSERVED,
        confidence=0.90,
        precision=0.02,
    )
    result = robust_fuse_quantity([obs1, obs2], quantity="corridor_width")
    assert 2.00 <= result.value <= 2.01
    assert result.precision < 0.01  # Precision improves when two sensors agree
    assert not result.is_bimodal
    assert not result.has_outliers()


def test_two_sensor_disagreement_detection() -> None:
    # Sensor 1 says 1.50m (0.01m precision), Sensor 2 says 1.80m (0.01m precision) -> 30 sigma disagreement
    obs1 = FusableObservation(
        value=1.50,
        unit=Unit.METER,
        source="lidar",
        provenance=Provenance.OBSERVED,
        confidence=0.95,
        precision=0.01,
    )
    obs2 = FusableObservation(
        value=1.80,
        unit=Unit.METER,
        source="ultrasonic",
        provenance=Provenance.OBSERVED,
        confidence=0.90,
        precision=0.01,
    )
    result = robust_fuse_quantity([obs1, obs2], quantity="clearance_height", tuning_k=2.0)
    assert result.has_conflicts() or result.scale_estimate > 0.05
    assert len(result.conflicts) > 0 or result.is_bimodal


def test_bimodal_distribution_tracking() -> None:
    # Bimodal cluster: two measurements near 1.0m, two measurements near 2.5m
    observations = [
        FusableObservation(value=1.00, unit=Unit.METER, source="cam_view1", provenance=Provenance.OBSERVED, confidence=0.9, precision=0.02),
        FusableObservation(value=1.02, unit=Unit.METER, source="cam_view2", provenance=Provenance.OBSERVED, confidence=0.9, precision=0.02),
        FusableObservation(value=2.48, unit=Unit.METER, source="sonar_1", provenance=Provenance.OBSERVED, confidence=0.85, precision=0.03),
        FusableObservation(value=2.52, unit=Unit.METER, source="sonar_2", provenance=Provenance.OBSERVED, confidence=0.85, precision=0.03),
    ]
    result = robust_fuse_quantity(observations, quantity="doorway_depth")
    assert result.has_conflicts()
    assert result.is_bimodal
    assert len(result.competing_modes) >= 2


def test_correlated_observations_inflate_uncertainty() -> None:
    # Repeated measurements from correlated stereo pipelines
    obs_uncorrelated = [
        FusableObservation(value=4.0, unit=Unit.METER, source=f"cam_{i}", provenance=Provenance.OBSERVED, confidence=0.9, precision=0.04)
        for i in range(4)
    ]
    obs_correlated = [
        FusableObservation(value=4.0, unit=Unit.METER, source=f"cam_{i}", provenance=Provenance.OBSERVED, confidence=0.9, precision=0.04)
        for i in range(4)
    ]

    res_uncorrelated = robust_fuse_quantity(obs_uncorrelated, quantity="ceiling_height", correlation=0.0)
    res_correlated = robust_fuse_quantity(obs_correlated, quantity="ceiling_height", correlation=0.6)

    assert res_correlated.value == pytest.approx(4.0, abs=1e-4)
    assert res_uncorrelated.value == pytest.approx(4.0, abs=1e-4)
    # Correlated observations must have higher fused standard error (greater uncertainty)
    assert res_correlated.precision > res_uncorrelated.precision


def test_missing_or_non_positive_precision_rejected() -> None:
    with pytest.raises((ValueError, FusionError)):
        FusableObservation(
            value=2.0,
            unit=Unit.METER,
            source="lidar",
            provenance=Provenance.OBSERVED,
            confidence=0.9,
            precision=0.0,  # Zero precision prohibited!
        )

    points = {"s1": [(0.0, 0.0, 0.0)], "s2": [(0.01, 0.0, 0.0)]}
    with pytest.raises(ValueError, match="Missing or invalid precision|missing or non-positive precision"):
        fuse_registered_point_clouds(
            points_by_source=points,
            precisions_by_source={"s1": 0.01},  # "s2" is missing precision
        )


def test_registered_point_clouds_coordinate_frame_validation() -> None:
    points = {
        "lidar": [(0.0, 0.0, 0.0)],
        "stereo": [(0.0, 0.0, 0.0)],
    }
    precisions = {"lidar": 0.01, "stereo": 0.02}

    # Different coordinate frames must raise FusionError
    with pytest.raises(FusionError, match="coordinate frame mismatch"):
        fuse_registered_point_clouds(
            points_by_source=points,
            precisions_by_source=precisions,
            coordinate_frames_by_source={"lidar": "world_enu", "stereo": "camera_optical"},
        )


def test_registered_point_clouds_unit_validation() -> None:
    points = {
        "lidar": [(0.0, 0.0, 0.0)],
        "stereo": [(0.0, 0.0, 0.0)],
    }
    precisions = {"lidar": 0.01, "stereo": 0.02}

    # Incompatible unit must raise FusionError
    with pytest.raises(FusionError, match="unit mismatch"):
        fuse_registered_point_clouds(
            points_by_source=points,
            precisions_by_source=precisions,
            units_by_source={"lidar": Unit.METER, "stereo": Unit.SECOND},
        )


def test_registered_point_clouds_ambiguity_ratio_test() -> None:
    # Reference point at (1.0, 0.0, 0.0)
    # Secondary points at (1.01, 0.0, 0.0) and (1.012, 0.0, 0.0)
    # Both are within 0.05m radius, distance ratio = 0.01 / 0.012 = 0.833, or very close
    # Setting threshold 0.80 rejects this ambiguous match
    ref_points = [(1.0, 0.0, 0.0)]
    sec_points = [(1.010, 0.0, 0.0), (1.011, 0.0, 0.0)]

    res = fuse_registered_point_clouds(
        points_by_source={"ref": ref_points, "sec": sec_points},
        precisions_by_source={"ref": 0.01, "sec": 0.02},
        correspondence_radius_m=0.05,
        ambiguity_ratio_threshold=0.80,
    )

    assert res.ambiguous_matches_count > 0
    # Ambiguous point is not fused, both remain unmatched
    assert res.total_fused == 0
    assert len(res.unmatched_by_source["ref"]) == 1
    assert len(res.unmatched_by_source["sec"]) == 2


def test_registered_point_clouds_duplicate_match_prevention() -> None:
    # Two reference points: (1.0, 0.0, 0.0) and (1.01, 0.0, 0.0)
    # Only one secondary point: (1.005, 0.0, 0.0)
    # Only the first reference point can match it; the second reference point cannot steal or duplicate it
    ref_points = [(1.0, 0.0, 0.0), (1.01, 0.0, 0.0)]
    sec_points = [(1.005, 0.0, 0.0)]

    res = fuse_registered_point_clouds(
        points_by_source={"ref": ref_points, "sec": sec_points},
        precisions_by_source={"ref": 0.01, "sec": 0.02},
        correspondence_radius_m=0.05,
    )

    assert res.duplicate_matches_prevented == 1
    assert res.total_fused == 1
    assert len(res.unmatched_by_source["ref"]) == 1
    assert len(res.unmatched_by_source["sec"]) == 0


def test_fuse_multisensor_points_3d_forwarding() -> None:
    points = {
        "lidar": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        "stereo": [(0.01, 0.0, 0.0), (0.99, 0.01, 0.0)],
    }
    precisions = {"lidar": 0.01, "stereo": 0.02}

    fused = fuse_multisensor_points_3d(
        points_by_source=points,
        precisions_by_source=precisions,
        spatial_registration=True,
        correspondence_radius_m=0.05,
        coordinate_frames_by_source={"lidar": "world_enu", "stereo": "world_enu"},
    )

    assert len(fused) == 2
    assert abs(fused[0][0]) < 0.01
    assert abs(fused[1][0] - 1.0) < 0.02
