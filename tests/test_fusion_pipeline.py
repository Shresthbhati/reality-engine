"""Tests for reconstruction/fusion/consumer.py (P6-02 remainder): the
pipeline consumer that ingests PLURAL point sources (sparse SfM +
depth-unprojected RGB-D), associates co-observed world points, runs
fuse_depth_observations, and emits fused points ready for WorldIR
geometry write-back.

Honesty rules under test:
- unassociated points are NOT fabricated into fusions -- they pass
  through as single-source observations (passthrough provenance);
- conflicts are recorded per point, never silently resolved by picking
  a winner;
- the stage report records per-source counts and association facts so
  the skip/failure reasons are auditable.
"""

import pytest

from engine.physics.math3 import Vec3
from provenance import Provenance
from reconstruction.backend.interface import ReconstructedPoint
from reconstruction.fusion.consumer import fuse_pipeline_points
from reconstruction.fusion.multi_source import DepthSourceObservation


def _sparse_point(x, y, z):
    return ReconstructedPoint(
        position=(x, y, z), track_id=f"sparse-{x}-{y}",
        source_evidence_ids=["img1", "img2"],
    )


def _depth_point(x, y, z):
    return ReconstructedPoint(
        position=(x, y, z), track_id=f"depth-cam0-00010-00020",
        source_evidence_ids=["depth-cam0"],
    )


class TestFusePipelinePoints:
    def test_co_observed_point_fuses_two_sources(self):
        # A sparse triangulated point and a depth unprojection of the
        # SAME physical corner (1 cm apart): fusion must run and
        # produce an ESTIMATED fused estimate.
        sparse = [_sparse_point(1.0, 0.0, 2.0)]
        depth = [_depth_point(1.005, 0.0, 2.0)]
        result = fuse_pipeline_points(
            sparse_points=sparse, depth_points=depth,
            association_tolerance_m=0.05,
        )
        assert result["n_associated"] == 1
        fused = result["fused"][0]
        assert fused.provenance is Provenance.ESTIMATED
        # Fusion runs on each source's RANGE estimate; both points sit
        # at range sqrt(1^2 + 0 + 2^2) ~= 2.236, so the fused range
        # must land between the two claims' ranges (which differ by
        # <1 mm here).
        r_true_lo = (1.0 ** 2 + 0.0 + 2.0 ** 2) ** 0.5
        r_true_hi = (1.005 ** 2 + 0.0 + 2.0 ** 2) ** 0.5
        assert r_true_lo <= fused.value <= r_true_hi
        assert fused.agreement_rms is not None and fused.agreement_rms < 0.01

    def test_unassociated_points_pass_through(self):
        # A depth point with no sparse neighbor within tolerance is a
        # single-source observation: passthrough, never dropped, never
        # fused with an imagined partner.
        sparse = [_sparse_point(10.0, 0.0, 2.0)]  # far away
        depth = [_depth_point(1.0, 0.0, 2.0)]
        result = fuse_pipeline_points(
            sparse_points=sparse, depth_points=depth,
            association_tolerance_m=0.05,
        )
        assert result["n_associated"] == 0
        assert result["n_passthrough_sparse"] == 1
        assert result["n_passthrough_depth"] == 1
        assert len(result["fused"]) == 2

    def test_conflict_recorded_not_resolved_by_winner(self):
        # Sources disagree by 1 m with tight precisions: a CONFLICT
        # must be recorded on that point's estimate.
        sparse = [_sparse_point(1.0, 0.0, 2.0)]
        depth = [_depth_point(2.0, 0.0, 2.0)]
        result = fuse_pipeline_points(
            sparse_points=sparse, depth_points=depth,
            association_tolerance_m=1.5,
        )
        assert result["n_associated"] == 1
        assert result["n_conflicts"] == 1

    def test_empty_inputs_report_honestly(self):
        result = fuse_pipeline_points(
            sparse_points=[], depth_points=[],
            association_tolerance_m=0.05,
        )
        assert result["n_associated"] == 0
        assert result["fused"] == []
        assert "status" in result

    def test_facts_are_auditable(self):
        sparse = [_sparse_point(1.0, 0.0, 2.0)]
        depth = [_depth_point(1.001, 0.0, 2.0)]
        result = fuse_pipeline_points(
            sparse_points=sparse, depth_points=depth,
            association_tolerance_m=0.05,
        )
        assert result["n_sparse"] == 1
        assert result["n_depth"] == 1
        assert result["association_tolerance_m"] == 0.05
