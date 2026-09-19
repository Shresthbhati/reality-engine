"""Tests for universal evidence-quality assessment (P7-04 foundation:
"universal evidence-grounded reconstruction" directive items 2 (quality
analysis) + 32 (capture feedback) + 11 (detail budget)).

The contract under test -- every rule is the repository's honesty
convention applied to evidence quality:

  - GSD is MEASURED from real geometry: the actual camera-to-surface
    distance for each point (|p - camera_center|, not z-depth), the
    actual focal length, and the actual in-image location of the
    projection. GSD = distance * pixel / focal_length. No average
    distance is assumed; the median of real per-point distances is
    the reported value.
  - A point is observed_by N cameras iff it projects INSIDE the image
    bounds of N cameras. A point in zero cameras is reported in
    `unprojectable_point_ids` -- never dropped silently, never guessed
    with an occlusion model we do not have.
  - The detail tier is DERIVED from measured facts (GSD + view count)
    by documented thresholds. Deriving is not fabricating: the tier
    says "the evidence supports this much detail", nothing more.
  - Domain-agnostic: no architectural classes, no scene-type language.
    A column, a pipe, a car fender, and a rock are just points.
  - Capture recommendations derive from the same measured facts,
    phrased for the Capture application (closed-loop §32/§33).
"""

from __future__ import annotations

import math
import statistics

import pytest

from engine.math import Quat, Vec3
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.calibration.camera import (
    CameraExtrinsics,
    CameraIntrinsics,
    camera_from_pose,
)

from perception.quality.assessment import (
    DETAIL_TIER_THRESHOLDS,
    assess_evidence_quality,
    recommend_capture,
)


def _intrinsics(width=640, height=480, fx=640.0):
    """fx = width -> horizontal half-FOV atan(0.5) ~= 26.6 deg, so a
    camera sees |x| <= z/2 at depth z; GSD = z_m / fx m/px (1 mm/px at
    0.64 m, 1 px/m at 1 m for fx=1000)."""
    return CameraIntrinsics(fx=fx, fy=fx, cx=width / 2, cy=height / 2,
                            width=width, height=height)


def _camera(eid, pos, intrinsics=None):
    intrinsics = intrinsics or _intrinsics()
    pose = ReconstructedCameraPose(
        evidence_id=eid,
        position=(pos[0], pos[1], pos[2]),
        rotation=(1.0, 0.0, 0.0, 0.0),  # identity: camera looks +Z
    )
    return camera_from_pose(intrinsics, pose)


def _point(pid, pos, evidence_ids):
    return ReconstructedPoint(position=pos, track_id=pid,
                              source_evidence_ids=list(evidence_ids))


def _result(points, cameras, status="success"):
    """Builds the ReconstructionResult whose camera_poses correspond
    1:1 (same order) to the PinholeCameras passed to the assessor.
    Evidence ids follow the tests' convention: c0, c1, ... in order."""
    poses = [ReconstructedCameraPose(
        evidence_id=f"c{i}",
        position=(c.extrinsics.position.x, c.extrinsics.position.y,
                  c.extrinsics.position.z),
        rotation=(1.0, 0.0, 0.0, 0.0))
        for i, c in enumerate(cameras)]
    return ReconstructionResult(points=points, camera_poses=poses,
                                registration_status=status)


class TestGsdMeasurement:
    def test_gsd_is_median_of_real_distances(self):
        # Cameras at z=0 looking +Z; points on a z=2 wall. Expected GSD
        # computed EXACTLY from the fixture: median of real per-point
        # euclidean distances, each over that camera's focal length.
        cams = [_camera("c0", (0.0, 0.0, 0.0)), _camera("c1", (0.4, 0.0, 0.0))]
        pts = [_point(f"p{i}", (0.1 * i - 0.15, 0.05, 2.0), ["c0", "c1"])
               for i in range(5)]
        all_gsds = []
        for p in pts:
            for cam in cams:
                d = math.dist(p.position, (cam.extrinsics.position.x,
                                           cam.extrinsics.position.y,
                                           cam.extrinsics.position.z))
                all_gsds.append(d * 1000.0 / cam.intrinsics.fx)
        expected = statistics.median(all_gsds)
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert report.gsd_mm_per_px == pytest.approx(expected, rel=1e-9)

    def test_gsd_uses_slant_distance_not_depth(self):
        # An off-axis point IN bounds (x/z = 0.4 < tan(45 deg)) at the
        # same z has a LARGER euclidean camera distance; the measured
        # GSD must reflect the slant, not z-depth.
        cams = [_camera("c0", (0.0, 0.0, 0.0))]
        # depth 2 m, lateral 0.8 m -> distance sqrt(4.64).
        pts = [_point("p0", (0.8, 0.0, 2.0), ["c0"])]
        report = assess_evidence_quality(_result(pts, cams), cams)
        expected = math.sqrt(4.64) * 1000.0 / 640.0
        assert report.gsd_mm_per_px == pytest.approx(expected, rel=1e-9)

    def test_gsd_requires_intrinsics_not_fabricated(self):
        cams = [_camera("c0", (0.0, 0.0, 0.0))]
        pts = [_point("p0", (0.0, 0.0, 2.0), ["c0"])]
        result = _result(pts, cams)
        # Strip intrinsics: no focal length exists -> refuse, never guess.
        with pytest.raises(ValueError, match="intrinsics"):
            assess_evidence_quality(result, [])

    def test_gsd_is_none_when_no_point_is_observed(self):
        cams = [_camera("c0", (0.0, 0.0, 0.0))]
        pts = [_point("p0", (0.0, 0.0, -2.0), ["c0"])]  # behind camera
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert report.gsd_mm_per_px is None


class TestViewCounts:
    def test_view_count_counts_in_bounds_projections(self):
        cams = [_camera("c0", (0.0, 0.0, 0.0)), _camera("c1", (0.4, 0.0, 0.0)),
                _camera("c2", (10.0, 0.0, 0.0))]  # c2 looks away from wall
        pts = [_point("p0", (0.1, 0.05, 2.0), ["c0"]),
               _point("p1", (0.0, 0.0, 2.0), ["c0"])]
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert report.view_counts == {"p0": 2, "p1": 2}

    def test_unprojectable_points_are_reported_not_dropped(self):
        cams = [_camera("c0", (0.0, 0.0, 0.0))]
        pts = [_point("p0", (0.0, 0.0, -2.0), ["c0"]),   # behind
               _point("p1", (50.0, 0.0, 2.0), ["c0"])]   # out of bounds
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert set(report.unprojectable_point_ids) == {"p0", "p1"}
        assert report.view_counts == {}

    def test_source_evidence_ids_are_not_treated_as_views(self):
        # The point CLAIMS two sources but only c0 exists; the camera
        # projection is the truth, not the metadata claim.
        cams = [_camera("c0", (0.0, 0.0, 0.0))]
        pts = [_point("p0", (0.1, 0.0, 2.0), ["c0", "c1"])]
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert report.view_counts["p0"] == 1


class TestDetailTiers:
    def test_tiers_from_measured_gsd_and_views(self):
        assert DETAIL_TIER_THRESHOLDS["fine_gsd_mm"] < DETAIL_TIER_THRESHOLDS["medium_gsd_mm"]
        # Two overlapping cameras, close enough that the near point
        # stays inside BOTH frusta; tier must reflect GSD.
        cams = [_camera("c0", (0.0, 0.0, 0.0)), _camera("c1", (0.05, 0.0, 0.0))]
        near = [_point("p0", (0.0, 0.0, 0.2), ["c0", "c1"])]  # 0.31 mm/px
        far = [_point("p1", (0.0, 0.0, 20.0), ["c0", "c1"])]  # ~31 mm/px
        r_near = assess_evidence_quality(_result(near, cams), cams)
        r_far = assess_evidence_quality(_result(far, cams), cams)
        assert r_near.detail_tier == "fine"
        assert r_far.detail_tier == "coarse"

    def test_low_view_count_caps_tier(self):
        # Excellent GSD but observed by exactly one camera: fine detail
        # needs multi-view corroboration; single-view caps the tier.
        cams = [_camera("c0", (0.0, 0.0, 0.0))]
        pts = [_point("p0", (0.0, 0.0, 0.2), ["c0"])]
        report = assess_evidence_quality(_result(pts, cams), cams,
                                         min_views_for_fine=2)
        assert report.detail_tier == "medium"

    def test_tier_is_unsupported_without_observed_points(self):
        cams = [_camera("c0", (0.0, 0.0, 0.0))]
        pts = [_point("p0", (0.0, 0.0, -2.0), ["c0"])]
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert report.detail_tier == "unsupported"


class TestCoverage:
    def test_observed_fraction_is_measured(self):
        cams = [_camera("c0", (0.0, 0.0, 0.0))]
        pts = [_point("p0", (0.0, 0.0, 2.0), ["c0"]),
               _point("p1", (0.0, 0.0, -2.0), ["c0"]),
               _point("p2", (0.0, 0.0, 2.5), ["c0"])]
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert report.observed_fraction == pytest.approx(2.0 / 3.0)


class TestCaptureRecommendations:
    def test_no_recommendation_when_evidence_good(self):
        # 10 points inside BOTH cameras' frusta overlap: c0 sees
        # |x| <= 0.25 at z=0.5; c1 (at x=0.3) sees x >= 0.05. Overlap:
        # [0.05, 0.25] -> every point gets 2 views, GSD fine.
        cams = [_camera("c0", (0.0, 0.0, 0.0)), _camera("c1", (0.3, 0.0, 0.0))]
        pts = [_point(f"p{i}", (0.05 + 0.02 * i, 0.0, 0.5), ["c0", "c1"])
               for i in range(10)]
        report = assess_evidence_quality(_result(pts, cams), cams)
        recs = recommend_capture(report)
        # Prose view only; the machine-readable scene_budget is always
        # present but is not a prose recommendation.
        assert [v for k, v in recs.items() if k != "scene_budget"] == []

    def test_coarse_gsd_recommends_close_range(self):
        cams = [_camera("c0", (0.0, 0.0, 0.0))]
        pts = [_point("p0", (0.0, 0.0, 20.0), ["c0"])]
        report = assess_evidence_quality(_result(pts, cams), cams)
        recs = recommend_capture(report)
        prose = [v for k, v in recs.items() if k != "scene_budget"]
        assert any("close-range" in r for r in prose)

    def test_unprojectable_points_recommends_coverage(self):
        cams = [_camera("c0", (0.0, 0.0, 0.0))]
        pts = [_point("p0", (0.0, 0.0, -2.0), ["c0"]),
               _point("p1", (0.0, 0.0, -2.5), ["c0"])]
        report = assess_evidence_quality(_result(pts, cams), cams)
        recs = recommend_capture(report)
        prose = [v for k, v in recs.items() if k != "scene_budget"]
        assert any("coverage" in r.lower() for r in prose)

    def test_low_view_counts_recommends_more_views(self):
        # Frustum fact: |x| <= z/2 (see _intrinsics). At z=2, c0 (x=0)
        # sees x in [-1, 1); c1 (x=0.3) sees x in [-0.7, 1.3). Points
        # in [-0.98, -0.755] are seen by c0 ONLY: exactly 1 view each,
        # so the multi-view recommendation fires.
        cams = [_camera("c0", (0.0, 0.0, 0.0)), _camera("c1", (0.3, 0.0, 0.0))]
        pts = [_point(f"p{i}", (-0.98 + 0.025 * i, 0.0, 2.0), ["c0"])
               for i in range(10)]
        report = assess_evidence_quality(_result(pts, cams), cams)
        recs = recommend_capture(report)
        prose = [v for k, v in recs.items() if k != "scene_budget"]
        assert any("view" in r.lower() for r in prose)


class TestDomainAgnosticismAndDeterminism:
    def test_report_serializes_roundtrip(self):
        cams = [_camera("c0", (0.0, 0.0, 0.0))]
        pts = [_point("p0", (0.0, 0.0, 2.0), ["c0"])]
        report = assess_evidence_quality(_result(pts, cams), cams)
        d = report.to_dict()
        assert d["gsd_mm_per_px"] == report.gsd_mm_per_px
        assert d["detail_tier"] == report.detail_tier

    def test_deterministic(self):
        cams = [_camera("c0", (0.0, 0.0, 0.0)), _camera("c1", (0.4, 0.0, 0.0))]
        pts = [_point(f"p{i}", (0.1 * i, 0.02 * i, 2.0), ["c0", "c1"])
               for i in range(8)]
        r1 = assess_evidence_quality(_result(pts, cams), cams)
        r2 = assess_evidence_quality(_result(pts, cams), cams)
        assert r1.to_dict() == r2.to_dict()

    def test_empty_scene_is_honest(self):
        report = assess_evidence_quality(_result([], []), [])
        assert report.gsd_mm_per_px is None
        assert report.detail_tier == "unsupported"
        assert report.observed_fraction == 0.0


class TestViewAngleDiversity:
    """P7-04 expansion (directive section 11): view-angle diversity is
    MEASURED per point -- the angular spread of the observing cameras'
    ray-to-normal angles. A degenerate local neighborhood (collinear /
    coincident kNN) has no defensible normal; those points contribute
    nothing rather than a fabricated normal."""

    def _div_cam(self, eid, pos, pitch_deg):
        """Camera looking +Z pitched by pitch_deg about +X (w,x,y,z)."""
        import math as _m
        half = _m.radians(pitch_deg) / 2.0
        pose = ReconstructedCameraPose(
            evidence_id=eid,
            position=pos,
            rotation=(_m.cos(half), _m.sin(half), 0.0, 0.0),
        )
        return camera_from_pose(_intrinsics(), pose)

    def _aimed_cam(self, eid, pos):
        """Camera at `pos` with its optical axis aimed exactly at the
        ring fixture's cluster center (0,0,2): yaw about +Y then pitch
        about the local +X, both as half-angle quaternions."""
        import math as _m
        dx, dy, dz = -pos[0], -pos[1], 2.0 - pos[2]
        yaw = _m.atan2(dx, dz)          # about +Y
        pitch = -_m.atan2(dy, _m.hypot(dx, dz))  # about +X
        qy = (_m.cos(yaw / 2), 0.0, _m.sin(yaw / 2), 0.0)
        qx = (_m.cos(pitch / 2), _m.sin(pitch / 2), 0.0, 0.0)
        from engine.physics.math3 import Quat
        q = Quat(*qy).multiply(Quat(*qx))
        pose = ReconstructedCameraPose(
            evidence_id=eid, position=pos,
            rotation=(q.w, q.x, q.y, q.z))
        return camera_from_pose(_intrinsics(), pose)

    def test_flat_wall_single_view_direction_low_diversity(self):
        # All cameras in a row looking at a planar z=2 wall: every ray
        # hits the normal head-on -> angular spread ~ 0.
        cams = [self._div_cam(f"c{i}", (0.4 * i - 0.4, 0.0, 0.0), 0.0)
                for i in range(4)]
        # Planar z=2 patch spread in BOTH x and y (collinear points
        # would be a degenerate neighborhood, not a wall).
        pts = [_point(f"p{i}", (0.2 * (i % 3) - 0.3, 0.1 * (i // 3) - 0.05,
                                2.0), ["c0", "c1", "c2"])
               for i in range(6)]
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert report.view_angle_diversity_deg is not None
        assert report.view_angle_diversity_deg < 30.0
        assert report.view_angle_diversity_n == 6

    def test_converging_ring_high_diversity(self):
        # Cameras ring the point cluster at wide angles: rays approach
        # the surface normal from very different directions -> large
        # angular spread.
        # Elevation CONTRAST is what spreads ray-to-normal angles on a
        # horizontal patch: c0 nearly overhead (~0 deg), c1/c2 far and
        # low (~59 deg). A same-height ring grazes at a constant angle
        # and measures ~0 by construction -- not a diverse capture.
        cams = [self._aimed_cam("c0", (0.0, 0.0, 0.5)),
                self._aimed_cam("c1", (2.5, 0.0, 0.5)),
                self._aimed_cam("c2", (-2.5, 0.0, 0.5))]
        pts = [_point(f"p{i}", (0.05 * i - 0.1, 0.03 * ((i % 2) * 2 - 1),
                                2.0), ["c0", "c1", "c2"])
               for i in range(4)]
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert report.view_angle_diversity_deg is not None
        assert report.view_angle_diversity_deg > 45.0

    def test_degenerate_neighborhood_excluded(self):
        # Collinear points: kNN covariance is rank-1 -> no defensible
        # normal -> the point contributes nothing (n = 0), not a
        # fabricated angle.
        cams = [self._div_cam("c0", (0.0, 0.0, 0.0), 0.0)]
        pts = [_point(f"p{i}", (0.1 * i, 0.0, 2.0), ["c0"]) for i in range(5)]
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert report.view_angle_diversity_deg is None
        assert report.view_angle_diversity_n == 0

    def test_single_observer_contributes_nothing(self):
        # One camera = no angular spread to measure; the point is
        # excluded from the median even though its normal is fine.
        cams = [self._div_cam("c0", (0.0, 0.0, 0.0), 0.0),
                self._div_cam("c1", (5.0, 0.0, 0.0), 0.0)]  # c1 looks away
        pts = [_point("p0", (0.0, 0.0, 2.0), ["c0"]),
               _point("p1", (0.0, 0.0, 2.5), ["c0"])]
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert report.view_angle_diversity_deg is None
        assert report.view_angle_diversity_n == 0

    def test_unknown_when_nothing_observed(self):
        cams = [self._div_cam("c0", (0.0, 0.0, 0.0), 0.0)]
        pts = [_point("p0", (0.0, 0.0, -2.0), ["c0"])]  # behind camera
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert report.view_angle_diversity_deg is None
        assert report.view_angle_diversity_n == 0


class TestViewAngleDiversitySamplingRegression:
    """Regression (2026-09-18): when eligible points exceed max_points,
    the stride sample reindexes pts -- the per-point loop must index by
    the SAMPLED row, not the original point id (IndexError before the
    fix; caught by the CLI vertical-slice on a 25-point fixture)."""

    def test_stride_sampling_indexes_sampled_rows(self):
        cams = [TestViewAngleDiversity()._div_cam(
            f"c{i}", (0.4 * i - 0.6, 0.0, 0.0), 0.0) for i in range(4)]
        # 600 eligible (>=2 observers) planar points -> the default
        # max_points=512 stride sample kicks in and reindexes pts.
        pts = [_point(f"p{i}",
                      (0.2 * (i % 20) - 2.0, 0.1 * (i // 20) - 1.0, 2.0),
                      ["c0", "c1", "c2"])
               for i in range(600)]
        report = assess_evidence_quality(_result(pts, cams), cams)
        assert report.view_angle_diversity_deg is not None
        assert report.view_angle_diversity_n > 0
        assert report.view_angle_diversity_n <= 512
