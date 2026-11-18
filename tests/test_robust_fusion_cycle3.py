"""Cycle 3 verification tests for robust multi-sensor evidence fusion.

Validates:
  1. Small sample N=2 consensus when sensors agree.
  2. Two-sensor bimodal disagreement detection: competing modes preserved, is_bimodal=True, no false averaging.
  3. Correlated observations (rho > 0) properly inflate uncertainty rather than falsely claiming 1/sqrt(N) scaling.
  4. Registered point cloud fusion: spatial nearest-neighbor correspondence, zero silent truncation.
"""

from __future__ import annotations

import math

import pytest

from engine.core.units import Unit
from provenance import Provenance
from reconstruction.fusion.fusion import FusableObservation
from reconstruction.fusion.robust_fusion import (
    fuse_registered_point_clouds,
    robust_fuse_quantity,
)


def test_two_sensor_agreement_consensus():
    """Two agreeing sensors produce exact Bayesian inverse-variance consensus."""
    o1 = FusableObservation(
        value=10.0,
        unit=Unit.METER,
        source="lidar",
        provenance=Provenance.OBSERVED,
        confidence=0.98,
        precision=0.02,
    )
    o2 = FusableObservation(
        value=10.02,
        unit=Unit.METER,
        source="stereo",
        provenance=Provenance.OBSERVED,
        confidence=0.90,
        precision=0.04,
    )

    res = robust_fuse_quantity([o1, o2], quantity="test_agree")
    assert not res.is_bimodal
    assert res.provenance is Provenance.ESTIMATED
    # Fused mean should be between 10.0 and 10.02, closer to LiDAR
    assert 10.0 < res.value < 10.02
    # Fused precision tighter than either sensor alone
    assert res.precision < 0.02


def test_two_sensor_bimodal_disagreement():
    """Two disagreeing sensors (e.g. LiDAR 2.0m vs stereo 5.0m) detect bimodal conflict.

    The engine must NOT average them to 3.5m, must preserve both modes, and set is_bimodal=True.
    """
    o_lidar = FusableObservation(
        value=2.0,
        unit=Unit.METER,
        source="lidar",
        provenance=Provenance.OBSERVED,
        confidence=0.98,
        precision=0.05,
    )
    o_stereo = FusableObservation(
        value=5.0,
        unit=Unit.METER,
        source="stereo",
        provenance=Provenance.OBSERVED,
        confidence=0.85,
        precision=0.10,
    )

    res = robust_fuse_quantity([o_lidar, o_stereo], quantity="test_conflict")
    assert res.is_bimodal is True
    assert res.provenance is Provenance.CONFLICT
    assert len(res.competing_modes) == 2
    # Modes must correspond to the two actual sensor claims
    mode_vals = sorted([m[0] for m in res.competing_modes])
    assert pytest.approx(mode_vals[0]) == 2.0
    assert pytest.approx(mode_vals[1]) == 5.0

    # Primary estimate should match the higher precision sensor, NEVER an invalid average
    assert abs(res.value - 2.0) < 0.01
    assert abs(res.value - 3.5) > 1.0


def test_correlated_observations_variance_adjustment():
    """Correlated observations (rho > 0) must not falsely inflate precision."""
    obs_independent = [
        FusableObservation(value=4.0, unit=Unit.METER, source=f"s_{i}", provenance=Provenance.OBSERVED, confidence=0.9, precision=0.10)
        for i in range(4)
    ]
    res_indep = robust_fuse_quantity(obs_independent, quantity="q", correlation=0.0)

    obs_correlated = [
        FusableObservation(value=4.0, unit=Unit.METER, source=f"s_{i}", provenance=Provenance.OBSERVED, confidence=0.9, precision=0.10)
        for i in range(4)
    ]
    res_corr = robust_fuse_quantity(obs_correlated, quantity="q", correlation=0.6)

    # Independent precision: sigma / sqrt(4) = 0.05
    assert pytest.approx(res_indep.precision, rel=1e-2) == 0.05
    # Correlated precision must be strictly wider than independent
    assert res_corr.precision > res_indep.precision
    expected_corr_prec = math.sqrt((0.10**2 / 4.0) * (1.0 + 3.0 * 0.6))
    assert pytest.approx(res_corr.precision, rel=1e-2) == expected_corr_prec


def test_registered_point_cloud_fusion_no_silent_truncation():
    """Verify registered point cloud fusion preserves unmatched points without truncation."""
    # LiDAR has 3 points
    cloud_lidar = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
    ]
    # Stereo has 2 points near point 0 and 1, plus 1 distant point (5.0, 5.0, 5.0)
    cloud_stereo = [
        (0.01, 0.0, 0.0),
        (0.99, 0.01, 0.0),
        (5.0, 5.0, 5.0),
    ]

    points_by_source = {
        "lidar": cloud_lidar,
        "stereo": cloud_stereo,
    }
    precisions = {
        "lidar": 0.02,
        "stereo": 0.05,
    }

    res = fuse_registered_point_clouds(
        points_by_source=points_by_source,
        precisions_by_source=precisions,
        correspondence_radius_m=0.05,
    )

    # 2 points matched within 5cm: (0,0,0) and (1,0,0)
    assert res.total_fused == 2
    assert len(res.fused_points) == 2
    assert abs(res.fused_points[0][0]) < 0.01
    assert abs(res.fused_points[1][0] - 1.0) < 0.02

    # Unmatched points must NOT be lost: LiDAR (2,0,0) and Stereo (5,5,5)
    assert res.total_unmatched == 2
    assert len(res.unmatched_by_source["lidar"]) == 1
    assert res.unmatched_by_source["lidar"][0] == (2.0, 0.0, 0.0)
    assert len(res.unmatched_by_source["stereo"]) == 1
    assert res.unmatched_by_source["stereo"][0] == (5.0, 5.0, 5.0)
