"""Canonical trajectory representation, interpolation, and diagnostics."""

from trajectories.interpolation import (
    interpolate_pose,
    resample_trajectory,
    slerp,
    smooth_trajectory,
)
from trajectories.trajectory import (
    DriftEstimate,
    FrameSource,
    Trajectory,
    TrajectoryError,
    TrajectoryFrame,
)

__all__ = [
    "DriftEstimate",
    "FrameSource",
    "Trajectory",
    "TrajectoryError",
    "TrajectoryFrame",
    "interpolate_pose",
    "resample_trajectory",
    "slerp",
    "smooth_trajectory",
]
