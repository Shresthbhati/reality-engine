"""Tests for the spatial detail budget (P7-05).

Proves the spatial budget localizes the SAME measured evidence-quality
facts the scene-level budget already uses -- it does not invent new
numbers. A high-detail region (lots of views, fine GSD) gets a finer
budget than a sparse region, and the spatial budget must never exceed
the scene-level budget (it is a partition, not a promotion).
"""

from __future__ import annotations

from perception.quality.assessment import (
    assess_evidence_quality,
    recommend_capture,
)
from perception.quality.detail_budget import DetailBudget
from perception.quality.spatial_budget import (
    SpatialRegion,
    spatial_detail_budget,
)
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.calibration.camera import (
    CameraIntrinsics,
    camera_from_pose,
)


def _intrinsics(width=640, height=480, fx=640.0):
    # fx = width -> half-FOV atan(0.5) ~= 26.6 deg; GSD = dist_m / fx m/px.
    return CameraIntrinsics(fx=fx, fy=fx, cx=width / 2, cy=height / 2,
                            width=width, height=height)


def _camera(eid, pos, intrinsics=None):
    intrinsics = intrinsics or _intrinsics()
    pose = ReconstructedCameraPose(
        evidence_id=eid,
        position=(pos[0], pos[1], pos[2]),
        rotation=(1.0, 0.0, 0.0, 0.0),
    )
    return camera_from_pose(intrinsics, pose)


def _point(pid, pos, evidence_ids):
    return ReconstructedPoint(position=pos, track_id=pid,
                              source_evidence_ids=list(evidence_ids))


def _result(points, cameras, status="success"):
    poses = [ReconstructedCameraPose(
        evidence_id=f"c{i}",
        position=(c.extrinsics.position.x, c.extrinsics.position.y,
                  c.extrinsics.position.z),
        rotation=(1.0, 0.0, 0.0, 0.0))
        for i, c in enumerate(cameras)]
    return ReconstructionResult(points=points, camera_poses=poses,
                                registration_status=status)


def _two_close_cams(z=2.0, spacing=0.3):
    """Two cameras 0.3 m apart on X, both looking +Z.

    Camera models: fx=640, sensor 640x480, cx=320, cy=240.
    A point at depth z is visible to both cameras when:

        max(-0.5*z, 0.3-0.5*z) <= x < min(0.5*z, 0.3+0.5*z)
        -0.375*z <= y < 0.375*z

    At z=2.0 that is x in [-0.7, 1.0), y in [-0.75, 0.75).
    """


class TestSpatialBudgetBasics:
    def test_empty_scene_is_empty_not_error(self):
        budget = spatial_detail_budget(_result([], []), [])
        assert budget.total_points == 0
        assert budget.regions == []
        assert budget.scene_budget.justified_level == "L0"

    def test_partition_covers_every_point_once(self):
        cams, z = _two_close_cams()
        pts = [_point(f"p{i}", (0.2 * (i % 10 - 5), 0.2 * (i // 10 - 5), z),
               ["c0", "c1"])
               for i in range(1000)]
        result = _result(pts, cams)
        budget = spatial_detail_budget(result, cams, max_depth=2,
                                       min_points_per_leaf=50)
        covered = sum(r.point_count for r in budget.regions)
        assert covered == len(pts)
        all_ids = [pid for r in budget.regions for pid in r.point_ids]
        assert len(all_ids) == len(set(all_ids))

    def test_finer_gsd_region_gets_higher_level_than_coarser(self):
        """Near (fine-GSD) cluster vs far (coarse-GSD) cluster, equal
        views: the near cluster must get a strictly finer level."""
        cams, _ = _two_close_cams(z=2.0)
        near = [_point(f"n{i}", (0.02 * i, 0.0, 1.0), ["c0", "c1"])
                for i in range(80)]
        far = [_point(f"f{i}", (0.02 * i, 0.0, 6.0), ["c0", "c1"])
               for i in range(80)]
        result = _result(near + far, cams)
        budget = spatial_detail_budget(result, cams, max_depth=1,
                                       min_points_per_leaf=5)
        near_level = None
        far_level = None
        for r in budget.regions:
            if r.point_ids and r.point_ids[0].startswith("n"):
                near_level = r.budget.justified_level
            if r.point_ids and r.point_ids[0].startswith("f"):
                far_level = r.budget.justified_level
        assert near_level == "L4"
        assert far_level == "L2"


    def test_spatial_partition_never_exceeds_scene_level(self):
        """The spatial budget is a localization: no region may claim a
        finer justified level than the scene aggregate."""
        cams, z = _two_close_cams()
        pts = [_point(f"p{i}", (0.2 * (i % 10 - 5), 0.2 * (i // 10 - 5), z),
               ["c0", "c1"]) for i in range(400)]
        result = _result(pts, cams)
        budget = spatial_detail_budget(result, cams, max_depth=2,
                                       min_points_per_leaf=50)
        order = ["L0", "L1", "L2", "L3", "L4"]
        scene_rank = order.index(budget.scene_budget.justified_level)
        for r in budget.regions:
            assert order.index(r.budget.justified_level) <= scene_rank

    def test_coverage_cap_drops_single_view_region(self):
        """Fine-GSD region seen by ONE camera is capped at L2, not L4."""
        cams = [_camera("c0", (0.0, 0.0, 0.0))]
        pts = [_point(f"p{i}", (0.02 * i, 0.0, 1.0), ["c0"]) for i in range(80)]
        result = _result(pts, cams)
        budget = spatial_detail_budget(result, cams, max_depth=1,
                                       min_points_per_leaf=5)
        for r in budget.regions:
            assert r.budget.justified_level == "L2"
            assert r.budget.coverage_capped is True

    def test_region_inherits_real_camera_models(self):
        """Spatial regions must NOT invent focal lengths."""
        cams, z = _two_close_cams()
        pts = [_point(f"p{i}", (i * 0.05, 0.0, z), ["c0", "c1"])
               for i in range(60)]
        result = _result(pts, cams)
        budget = spatial_detail_budget(result, cams, max_depth=1,
                                       min_points_per_leaf=10)
        assert all(r.budget.max_gsd_mm_per_px is not None
                   for r in budget.regions)

    def test_serializes_and_roundtrips(self):
        cams, z = _two_close_cams()
        pts = [_point(f"p{i}", (0.2 * (i % 10 - 5), 0.2 * (i // 10 - 5), z),
               ["c0", "c1"]) for i in range(200)]
        result = _result(pts, cams)
        budget = spatial_detail_budget(result, cams, max_depth=1,
                                       min_points_per_leaf=50)
        d = budget.to_dict()
        assert d["total_points"] == 200
        assert d["partitioning"]["n_regions"] == len(budget.regions)
        assert d["regions"][0]["budget"]["justified_level"]
        for r in budget.regions:
            assert r.report.gsd_mm_per_px is not None

    def test_recommendation_scene_budget_matches(self):
        cams, z = _two_close_cams()
        pts = [_point(f"p{i}", (0.2 * (i % 10 - 5), 0.2 * (i // 10 - 5), z),
               ["c0", "c1"]) for i in range(200)]
        result = _result(pts, cams)
        report = assess_evidence_quality(result, cams)
        recs = recommend_capture(report)
        assert "scene_budget" in recs
        assert isinstance(recs["scene_budget"], DetailBudget)
        budget = spatial_detail_budget(result, cams, max_depth=1,
                                       min_points_per_leaf=50)
        assert (budget.scene_budget.justified_level
                == recs["scene_budget"].justified_level)

    def test_deterministic(self):
        cams, z = _two_close_cams()
        pts = [_point(f"p{i}", (0.2 * (i % 10 - 5), 0.2 * (i // 10 - 5), z),
               ["c0", "c1"]) for i in range(200)]
        result = _result(pts, cams)
        b1 = spatial_detail_budget(result, cams, max_depth=2,
                                   min_points_per_leaf=50).to_dict()
        b2 = spatial_detail_budget(result, cams, max_depth=2,
                                   min_points_per_leaf=50).to_dict()
        assert b1 == b2
