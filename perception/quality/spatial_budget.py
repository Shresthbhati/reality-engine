"""Spatial detail budget (P7-05, directive sections 9-11).

Scene-level ``DetailBudget`` (perception/quality/detail_budget.py) answers
"how much detail can the whole capture support?". The universal pipeline
needs the spatial answer: where in the scene is the evidence fine enough
to spend reconstruction compute, and where is it too sparse to warrant
more than a survey?

This module partitions the reconstructed point cloud into spatial leaves
(an octree) and derives a measured ``DetailBudget`` per leaf region by
reusing the existing honest assessor (``assess_evidence_quality``) and
budget law (``detail_budget_for``). No new GSD/coverage math is invented
here -- every number is still a measured or documented fact. The spatial
step only localizes the same facts so the detail-discovery/ROI stage can
spend compute where the evidence actually supports it.

Honesty inherited from the rest of the quality stack:
- a region observed by no camera yields an honest "unsupported" /
  zero-allocation budget -- never an error and never an inflated tier
  borrowed from a neighbor;
- cameras are the scene's real camera models; regions do not get their
  own focal-length guesses;
- empty scenes yield an empty region list (the assessor already refuses
  intrinsics-less input at the scene level).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from perception.quality.assessment import (
    EvidenceQualityReport,
    assess_evidence_quality,
)
from perception.quality.detail_budget import DetailBudget, detail_budget_for
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.calibration.camera import PinholeCamera

_Bounds = Tuple[float, float, float, float, float, float]


@dataclass
class SpatialRegion:
    """One spatial leaf and its measured detail budget."""

    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    point_count: int
    point_ids: List[str]
    report: EvidenceQualityReport
    budget: DetailBudget
    level: int = 0

    def to_dict(self) -> dict:
        return {
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
            "point_count": self.point_count,
            "point_ids": list(self.point_ids),
            "report": self.report.to_dict(),
            "budget": self.budget.to_dict(),
            "level": self.level,
        }


@dataclass
class SpatialDetailBudget:
    """Scene-wide spatial detail budget."""

    regions: List[SpatialRegion]
    world_bounds_min: Tuple[float, float, float]
    world_bounds_max: Tuple[float, float, float]
    total_points: int
    scene_budget: DetailBudget
    max_depth: int
    min_points_per_leaf: int
    n_leaves_target: int

    def to_dict(self) -> dict:
        return {
            "world_bounds_min": list(self.world_bounds_min),
            "world_bounds_max": list(self.world_bounds_max),
            "total_points": self.total_points,
            "scene_budget": self.scene_budget.to_dict(),
            "partitioning": {
                "max_depth": self.max_depth,
                "min_points_per_leaf": self.min_points_per_leaf,
                "n_leaves_target": self.n_leaves_target,
                "n_regions": len(self.regions),
            },
            "regions": [r.to_dict() for r in self.regions],
        }



def _point_aabb(points):
    if not points:
        return None
    px = [float(p.position[0]) for p in points]
    py = [float(p.position[1]) for p in points]
    pz = [float(p.position[2]) for p in points]
    return (min(px), min(py), min(pz), max(px), max(py), max(pz))


def _split_axis(bounds):
    extents = (bounds[3] - bounds[0], bounds[4] - bounds[1], bounds[5] - bounds[2])
    return max(range(3), key=lambda i: extents[i])


def _point_in_leaf(p, bmin, bmax):
    x, y, z = (float(c) for c in p.position)
    return (bmin[0] <= x <= bmax[0] and bmin[1] <= y <= bmax[1]
            and bmin[2] <= z <= bmax[2])


def _octree_leaves(points, bounds, depth, max_depth, min_points_per_leaf):
    """Recursively subdivide ``bounds`` until depth limits or the leaf is
    small enough. Returns ``(bounds, points, level)`` for each leaf.

    Degenerate (collapsed) bounds are not subdivided: a region where all
    retained points share a coordinate still reports its (tight) budget
    honestly rather than looping forever.
    """
    pts = [p for p in points if _point_in_leaf(p, bounds[0:3], bounds[3:6])]
    if not pts:
        return []
    tight = _point_aabb(pts)
    if tight is None:
        return []
    bounds = tight
    if depth >= max_depth or len(pts) <= min_points_per_leaf:
        return [(bounds, pts, depth)]
    axis = _split_axis(bounds)
    mid = (bounds[axis] + bounds[axis + 3]) / 2.0
    if mid != mid or abs(mid) == float("inf"):
        return [(bounds, pts, depth)]
    lo = list(bounds)
    hi = list(bounds)
    lo[axis + 3] = mid
    hi[axis] = mid
    leaves = []
    leaves.extend(_octree_leaves(
        pts, (lo[0], lo[1], lo[2], lo[3], lo[4], lo[5]),
        depth + 1, max_depth, min_points_per_leaf))
    leaves.extend(_octree_leaves(
        pts, (hi[0], hi[1], hi[2], hi[3], hi[4], hi[5]),
        depth + 1, max_depth, min_points_per_leaf))
    # If subdivision produced no children (points on the split plane),
    # stop here to avoid infinite recursion.
    return leaves or [(bounds, pts, depth)]


def _region_budget(region_points, poses, cameras, thresholds=None,
                   min_views_for_fine=None):
    """Per-region assessment reusing the scene-level honest assessor.

    A region is a sub-set of points observed by the scene's cameras; the
    GSD / view-count / unprojectable logic is identical, so we build a
    regional ReconstructionResult and run the real assessor rather than
    duplicating any of its math.
    """
    regional = ReconstructionResult(
        points=list(region_points),
        camera_poses=list(poses),
        registration_status="success",
    )
    report = assess_evidence_quality(regional, cameras, thresholds,
                                     min_views_for_fine)
    budget = detail_budget_for(report)
    return report, budget


def spatial_detail_budget(result, cameras, *, max_depth=3,
                          min_points_per_leaf=200, thresholds=None,
                          min_views_for_fine=None):
    """Partition ``result.points`` into octree leaves and derive a
    measured ``DetailBudget`` per leaf.

    The scene-level ``scene_budget`` is produced by the SAME
    ``detail_budget`` law on the scene-level report, so aggregate and
    per-region budgets are always consistent.
    """
    poses = result.camera_poses
    points = list(result.points) if result.points else []
    world_bounds = _point_aabb(points)
    if world_bounds is None:
        empty_report = assess_evidence_quality(
            ReconstructionResult(points=[], camera_poses=list(poses),
                                 registration_status="success"),
            cameras, thresholds, min_views_for_fine)
        empty_budget = detail_budget_for(empty_report)
        return SpatialDetailBudget(
            regions=[], world_bounds_min=(0.0, 0.0, 0.0),
            world_bounds_max=(0.0, 0.0, 0.0), total_points=0,
            scene_budget=empty_budget, max_depth=max_depth,
            min_points_per_leaf=min_points_per_leaf, n_leaves_target=1)

    leaves = _octree_leaves(points, world_bounds, 0, max_depth,
                            min_points_per_leaf)
    regions = []
    for bnds, leaf_pts, level in leaves:
        report, budget = _region_budget(leaf_pts, poses, cameras,
                                        thresholds, min_views_for_fine)
        regions.append(SpatialRegion(
            bounds_min=(bnds[0], bnds[1], bnds[2]),
            bounds_max=(bnds[3], bnds[4], bnds[5]),
            point_count=len(leaf_pts),
            point_ids=[p.track_id for p in leaf_pts],
            report=report, budget=budget, level=level))

    scene_report = assess_evidence_quality(result, cameras, thresholds,
                                           min_views_for_fine)
    scene_budget = detail_budget_for(scene_report)
    return SpatialDetailBudget(
        regions=regions,
        world_bounds_min=(world_bounds[0], world_bounds[1], world_bounds[2]),
        world_bounds_max=(world_bounds[3], world_bounds[4], world_bounds[5]),
        total_points=len(points),
        scene_budget=scene_budget, max_depth=max_depth,
        min_points_per_leaf=min_points_per_leaf,
        n_leaves_target=max(1, len(regions)))
