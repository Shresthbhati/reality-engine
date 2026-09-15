"""Deterministic tests for multi-source depth fusion (P6-02).

Ground truth is known by construction: each source observes the SAME
true depth plus a fixed, known offset within its claimed precision.
These are the classic inverse-variance-weighting sanity checks --
fused value closer to truth than either source alone, fused precision
tighter than either alone -- plus the conflict-rejection path from
fusion.py (sources disagreeing beyond combined uncertainty must be
flagged, never silently averaged into nonsense).
"""

from __future__ import annotations

import math

import pytest

from engine.core.units import Unit
from provenance import Provenance
from reconstruction.fusion.fusion import FusionError
from reconstruction.fusion.multi_source import (
    DepthSourceObservation,
    fuse_depth_observations,
)

TRUE_DEPTH_M = 10.0


def test_fused_estimate_beats_either_source_alone():
    # LiDAR: tight, slightly high. Monocular: loose, slightly low.
    lidar = DepthSourceObservation(
        source="lidar",
        depth_m=TRUE_DEPTH_M + 0.02,
        precision_m=0.03,
        provenance=Provenance.OBSERVED,
        confidence=0.95,
    )
    mono = DepthSourceObservation(
        source="monocular_midas",
        depth_m=TRUE_DEPTH_M - 0.15,
        precision_m=0.20,
        provenance=Provenance.ESTIMATED,
        confidence=0.6,
    )

    fused = fuse_depth_observations([lidar, mono], point_id="p0")

    lidar_error = abs(lidar.depth_m - TRUE_DEPTH_M)
    mono_error = abs(mono.depth_m - TRUE_DEPTH_M)
    fused_error = abs(fused.value - TRUE_DEPTH_M)

    # Fused value is closer to truth than the weaker source, and no
    # worse than the stronger source by more than its own precision.
    assert fused_error < mono_error
    assert fused_error <= lidar_error + lidar.precision_m

    # Fused precision (the classic inverse-variance-weighting result)
    # is strictly tighter than either contributor's on its own.
    assert fused.precision < lidar.precision_m
    assert fused.precision < mono.precision_m

    # A tighter source pulls the fused value further toward itself:
    # lidar's precision (0.03) dwarfs monocular's (0.20), so the fused
    # value should sit much closer to lidar's claim than the midpoint.
    midpoint = (lidar.depth_m + mono.depth_m) / 2.0
    assert abs(fused.value - lidar.depth_m) < abs(fused.value - midpoint)

    assert fused.unit is Unit.METER
    assert fused.provenance is Provenance.ESTIMATED
    assert not fused.has_conflicts()
    assert fused.method == "inverse_variance_weighted_mean"


def test_uncertainty_weights_influence_the_estimate():
    """A tighter source must move the fused value more than a looser
    one -- direct evidence the weighting is real, not an unweighted
    average."""
    tight = DepthSourceObservation(
        source="rgbd",
        depth_m=TRUE_DEPTH_M,
        precision_m=0.01,
        provenance=Provenance.OBSERVED,
        confidence=0.98,
    )
    loose = DepthSourceObservation(
        source="stereo",
        depth_m=TRUE_DEPTH_M + 1.0,
        precision_m=1.0,
        provenance=Provenance.OBSERVED,
        confidence=0.7,
    )

    fused = fuse_depth_observations([tight, loose], point_id="p1")

    # An unweighted average would land at +0.5; inverse-variance
    # weighting must pull it far closer to the tight source.
    unweighted_average = (tight.depth_m + loose.depth_m) / 2.0
    assert abs(fused.value - tight.depth_m) < abs(fused.value - unweighted_average)
    assert fused.value < unweighted_average


def test_disagreeing_sources_flagged_as_conflict_not_blended():
    # Two tight sources 5m apart -- far beyond any plausible combined
    # uncertainty. This must surface as CONFLICT, not a quiet average.
    a = DepthSourceObservation(
        source="lidar",
        depth_m=TRUE_DEPTH_M,
        precision_m=0.02,
        provenance=Provenance.OBSERVED,
        confidence=0.95,
    )
    b = DepthSourceObservation(
        source="mvs",
        depth_m=TRUE_DEPTH_M + 5.0,
        precision_m=0.02,
        provenance=Provenance.OBSERVED,
        confidence=0.9,
    )

    fused = fuse_depth_observations([a, b], point_id="p2")

    assert fused.provenance is Provenance.CONFLICT
    assert fused.has_conflicts()
    assert len(fused.conflicts) == 1
    conflict = fused.conflicts[0]
    assert {conflict.source_a, conflict.source_b} == {"lidar", "mvs"}
    assert conflict.sigma_multiple > 5.0

    # Raw evidence is never destroyed even under conflict.
    assert {o.source for o in fused.contributing} == {"lidar", "mvs"}
    # Confidence drops to the weakest contributor's, never inflated.
    assert fused.confidence == pytest.approx(min(a.confidence, b.confidence))


def test_single_source_is_exact_passthrough():
    only = DepthSourceObservation(
        source="neural_depth",
        depth_m=TRUE_DEPTH_M,
        precision_m=0.05,
        provenance=Provenance.ESTIMATED,
        confidence=0.7,
    )
    fused = fuse_depth_observations([only], point_id="p3")

    assert fused.value == only.depth_m
    assert fused.precision == only.precision_m
    assert fused.provenance is only.provenance
    assert fused.method == "passthrough"
    assert fused.agreement_rms is None


def test_empty_observations_rejected_honestly():
    with pytest.raises(FusionError):
        fuse_depth_observations([], point_id="p4")


def test_three_plus_sources_generic_weighting():
    """Sanity check the module is not hardcoded to exactly two
    sources -- MVS + RGB-D + LiDAR simultaneously."""
    lidar = DepthSourceObservation("lidar", TRUE_DEPTH_M + 0.01, 0.02, Provenance.OBSERVED, 0.95)
    rgbd = DepthSourceObservation("rgbd", TRUE_DEPTH_M - 0.03, 0.05, Provenance.OBSERVED, 0.9)
    mvs = DepthSourceObservation("mvs", TRUE_DEPTH_M + 0.08, 0.10, Provenance.ESTIMATED, 0.8)

    fused = fuse_depth_observations([lidar, rgbd, mvs], point_id="p5")

    assert not fused.has_conflicts()
    assert fused.precision < min(lidar.precision_m, rgbd.precision_m, mvs.precision_m)
    assert math.isfinite(fused.value)
    assert len(fused.contributing) == 3
