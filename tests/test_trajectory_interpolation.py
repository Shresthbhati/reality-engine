"""Tests for continuous trajectory interpolation, SLERP, and smoothing."""

from __future__ import annotations

import math

import pytest

from engine.math import Quat, Vec3
from reconstruction.calibration.transforms import RigidTransform
from trajectories.interpolation import (
    interpolate_pose,
    resample_trajectory,
    slerp,
    smooth_trajectory,
)
from trajectories.trajectory import (
    FrameSource,
    Trajectory,
    TrajectoryError,
    TrajectoryFrame,
)


def test_slerp_interpolates_rotation_geodesic():
    # q1 = identity (0 deg rotation)
    q1 = Quat.identity()
    # q2 = 90 deg around Z axis
    half_angle = math.radians(45.0)
    q2 = Quat(math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)).normalized()

    # t = 0.0 -> q1
    q_start = slerp(q1, q2, 0.0)
    assert math.isclose(q_start.w, 1.0, rel_tol=1e-5)

    # t = 1.0 -> q2
    q_end = slerp(q1, q2, 1.0)
    assert math.isclose(q_end.w, q2.w, rel_tol=1e-5)
    assert math.isclose(q_end.z, q2.z, rel_tol=1e-5)

    # t = 0.5 -> 45 deg around Z axis
    q_mid = slerp(q1, q2, 0.5)
    expected_half = math.radians(22.5)
    assert math.isclose(q_mid.w, math.cos(expected_half), rel_tol=1e-4)
    assert math.isclose(q_mid.z, math.sin(expected_half), rel_tol=1e-4)


def test_interpolate_pose_between_trajectory_frames():
    p1 = RigidTransform("body", "world", Quat.identity(), Vec3(0.0, 0.0, 0.0))
    p2 = RigidTransform("body", "world", Quat.identity(), Vec3(10.0, 20.0, 30.0))

    f1 = TrajectoryFrame(timestamp_ns=1_000_000_000, pose=p1)
    f2 = TrajectoryFrame(timestamp_ns=2_000_000_000, pose=p2)

    traj = Trajectory(frames=(f1, f2))

    # Exact boundary queries
    pose_start = interpolate_pose(traj, 1_000_000_000)
    assert pose_start.translation == Vec3(0.0, 0.0, 0.0)

    pose_end = interpolate_pose(traj, 2_000_000_000)
    assert pose_end.translation == Vec3(10.0, 20.0, 30.0)

    # Midpoint query at t = 1.5s
    pose_mid = interpolate_pose(traj, 1_500_000_000)
    assert math.isclose(pose_mid.translation.x, 5.0, abs_tol=1e-5)
    assert math.isclose(pose_mid.translation.y, 10.0, abs_tol=1e-5)
    assert math.isclose(pose_mid.translation.z, 15.0, abs_tol=1e-5)

    # Query out of range raises TrajectoryError
    with pytest.raises(TrajectoryError):
        interpolate_pose(traj, 500_000_000)
    with pytest.raises(TrajectoryError):
        interpolate_pose(traj, 2_500_000_000)


def test_resample_trajectory_uniform_rate():
    # 2 seconds duration: t = 1.0s to 3.0s
    p1 = RigidTransform("body", "world", Quat.identity(), Vec3(0.0, 0.0, 0.0))
    p2 = RigidTransform("body", "world", Quat.identity(), Vec3(2.0, 4.0, 6.0))
    traj = Trajectory(
        frames=(
            TrajectoryFrame(1_000_000_000, p1),
            TrajectoryFrame(3_000_000_000, p2),
        ),
        frame_source=FrameSource.SFM,
    )

    # Resample at 2 Hz -> dt = 0.5s -> points at 1.0, 1.5, 2.0, 2.5, 3.0 (5 frames)
    resampled = resample_trajectory(traj, rate_hz=2.0)
    assert len(resampled.frames) == 5
    assert resampled.start_ns == 1_000_000_000
    assert resampled.end_ns == 3_000_000_000
    assert math.isclose(resampled.frames[2].pose.translation.x, 1.0, abs_tol=1e-5)


def test_smooth_trajectory_reduces_jitter():
    # Trajectory with position jitter on Y axis
    frames = []
    base_t = 1_000_000_000
    step = 100_000_000  # 100ms
    y_vals = [0.0, 0.1, 0.8, 0.3, 0.9, 0.5, 1.0]  # spike at index 2 and 4

    for i, y in enumerate(y_vals):
        pose = RigidTransform("body", "world", Quat.identity(), Vec3(float(i), y, 0.0))
        frames.append(TrajectoryFrame(base_t + i * step, pose))

    traj = Trajectory(frames=tuple(frames))
    smoothed = smooth_trajectory(traj, window_size=3)

    assert len(smoothed) == len(traj)
    # Check endpoints preserved
    assert smoothed.frames[0].pose.translation.y == 0.0
    assert smoothed.frames[-1].pose.translation.y == 1.0
    # Check spike at index 2 (0.8) is attenuated by window average (0.1 + 0.8 + 0.3)/3 = 0.4
    assert math.isclose(smoothed.frames[2].pose.translation.y, 0.4, rel_tol=1e-3)
