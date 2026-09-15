"""Trajectory quality diagnostics (P3-03): ATE, RPE, drift, and
tracking state -- standard TUM RGB-D benchmark metrics (Sturm et al.
2012), not invented math. Pure functions over two `Trajectory`
objects already in the same world frame (this repo's Trajectory
contract); no SE(3) alignment step is performed here -- if estimated
and ground-truth trajectories are not already co-registered, run them
through P4-01 registration first. That is a deliberate scope cut, not
an oversight.
"""

from __future__ import annotations

import math
import statistics
from enum import Enum
from typing import List, NamedTuple

from trajectories.trajectory import Trajectory, TrajectoryError


class TrackingState(str, Enum):
    """Backend-reported tracking quality. This module does not compute
    it (a real VIO backend knows its own state); it exists so
    diagnostics results can carry one alongside the numeric metrics."""

    TRACKING = "tracking"
    LOST = "lost"
    RELOCALIZING = "relocalizing"
    LOCALIZED = "localized"


class ErrorStats(NamedTuple):
    rmse: float
    mean: float
    median: float
    std: float
    count: int


def _stats(errors: List[float]) -> ErrorStats:
    if not errors:
        raise TrajectoryError("no matching timestamps between trajectories")
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    return ErrorStats(
        rmse=rmse,
        mean=statistics.mean(errors),
        median=statistics.median(errors),
        std=statistics.pstdev(errors) if len(errors) > 1 else 0.0,
        count=len(errors),
    )


def _common_timestamps(est: Trajectory, gt: Trajectory) -> List[int]:
    est_ts = {f.timestamp_ns for f in est.frames}
    gt_ts = {f.timestamp_ns for f in gt.frames}
    return sorted(est_ts & gt_ts)


def absolute_trajectory_error(est: Trajectory, gt: Trajectory) -> ErrorStats:
    """ATE: per-timestamp translation error between matching poses,
    at exact-matching timestamps only (no interpolation, same
    no-guessing rule as Trajectory.at)."""
    common = _common_timestamps(est, gt)
    errors = [
        (est.at(ts).pose.translation - gt.at(ts).pose.translation).length()
        for ts in common
    ]
    return _stats(errors)


def relative_pose_error(est: Trajectory, gt: Trajectory, delta_ns: int) -> ErrorStats:
    """RPE: translation-error magnitude of the relative motion over
    `delta_ns`, compared between est and gt, at matching timestamp
    pairs (t, t+delta_ns) where both exist in both trajectories."""
    if delta_ns <= 0:
        raise TrajectoryError(f"delta_ns must be > 0, got {delta_ns}")
    common = set(_common_timestamps(est, gt))
    pairs = sorted(t for t in common if (t + delta_ns) in common)
    errors = []
    for t0 in pairs:
        t1 = t0 + delta_ns
        est_rel = est.at(t0).pose.inverse().compose(est.at(t1).pose)
        gt_rel = gt.at(t0).pose.inverse().compose(gt.at(t1).pose)
        drift = gt_rel.inverse().compose(est_rel)
        errors.append(drift.translation.length())
    return _stats(errors)


def endpoint_drift(traj: Trajectory) -> float:
    """Loop-closure proxy: distance between the first and last pose,
    in meters. Near-zero on a path that returns to its start; not a
    substitute for real loop-closure detection (ceiling noted).
    ponytail: straight-line endpoint gap, add real loop-closure
    detection (place recognition) if this proxy proves insufficient.
    """
    return (traj.frames[-1].pose.translation - traj.frames[0].pose.translation).length()
