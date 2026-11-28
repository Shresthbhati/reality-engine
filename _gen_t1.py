#!/usr/bin/env python3
# Generator: rewrite tests/test_spatial_budget.py with frustum-safe fixtures.
import io

PART1 = r'''"""Tests for the spatial detail budget (P7-05).

Proves the spatial budget localizes the SAME measured evidence-quality
facts the scene-level budget already uses -- it does not invent new
numbers. A high-detail region (lots of views, fine GSD) gets a finer
budget than a sparse region, and the spatial budget must never exceed
the scene-level budget (it is a partition, not a promotion).

Fixture note: a camera at origin with fx=width=640 (half-FOV atan(.5))
sees, at depth z, |x| < 0.5*z and |y| < 0.75*z. All test point clouds are
kept strictly inside this frustum for the configured cameras so a
region's measured GSD is real, not a fixture artifact.
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
    return [_camera("c0", (0.0, 0.0, 0.0)), _camera("c1", (spacing, 0.0, 0.0))], z


def _wall(n, z, xmax, prefix="p"):
    """n points on a horizontal line at depth z, |x| <= xmax (kept well
    inside both cameras' shared frustum)."""
    span = xmax * 2.0
    return [_point(f"{prefix}{i}", ((i / max(1, n - 1)) * span - xmax, 0.0, z),
                   ["c0", "c1"]) for i in range(n)]


class TestSpatialBudgetBasics:
    def test_empty_scene_is_empty_not_error(self):
        budget = spatial_detail_budget(_result([], []), [])
        assert budget.total_points == 0
        assert budget.regions == []
        assert budget.scene_budget.justified_level == "L0"

    def test_partition_covers_every_point_once(self):
        cams, z = _two_close_cams()
        pts = _wall(1000, z, xmax=0.9)
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
        near = _wall(80, z=1.0, xmax=0.4, prefix="n")
        far = _wall(80, z=6.0, xmax=2.0, prefix="f")
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
'''

with io.open("tests/test_spatial_budget.py", "w", encoding="utf-8") as f:
    f.write(PART1)
print("part1 written", len(PART1))
