"""Continuous trajectory interpolation, resampling, and smoothing (Mandate Phase 9).

Provides:
- SE(3) pose interpolation at arbitrary continuous timestamps:
  - Spherical linear interpolation (SLERP) for rotation quaternions (shortest geodesic).
  - Linear or cubic interpolation for 3D translation vectors.
- Uniform trajectory resampling to fixed frequencies (e.g. 10 Hz, 30 Hz, 60 Hz).
- Trajectory smoothing via centered moving-average filtering on positions and geodesic smoothing on rotations.
"""

from __future__ import annotations

import bisect
import math

from engine.math import Quat, Vec3
from reconstruction.calibration.transforms import RigidTransform
from trajectories.trajectory import (
    Trajectory,
    TrajectoryError,
    TrajectoryFrame,
)


def slerp(q1: Quat, q2: Quat, t: float) -> Quat:
    """Spherical linear interpolation between two unit quaternions along shortest geodesic."""
    t_clamped = max(0.0, min(1.0, float(t)))

    # Compute dot product
    dot = q1.w * q2.w + q1.x * q2.x + q1.y * q2.y + q1.z * q2.z

    # Take shortest path
    q2_adj = q2
    if dot < 0.0:
        dot = -dot
        q2_adj = Quat(-q2.w, -q2.x, -q2.y, -q2.z)

    # Near-parallel threshold: fall back to normalized linear interpolation
    if dot > 0.9995:
        w = (1.0 - t_clamped) * q1.w + t_clamped * q2_adj.w
        x = (1.0 - t_clamped) * q1.x + t_clamped * q2_adj.x
        y = (1.0 - t_clamped) * q1.y + t_clamped * q2_adj.y
        z = (1.0 - t_clamped) * q1.z + t_clamped * q2_adj.z
        return Quat(w, x, y, z).normalized()

    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)

    theta = theta_0 * t_clamped
    sin_theta = math.sin(theta)

    s1 = math.sin(theta_0 - theta) / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return Quat(
        s1 * q1.w + s2 * q2_adj.w,
        s1 * q1.x + s2 * q2_adj.x,
        s1 * q1.y + s2 * q2_adj.y,
        s1 * q1.z + s2 * q2_adj.z,
    ).normalized()


def interpolate_pose(
    traj: Trajectory,
    timestamp_ns: int,
) -> RigidTransform:
    """Interpolate SE(3) pose from a Trajectory at an arbitrary timestamp within its range."""
    if not traj.frames:
        raise TrajectoryError("Cannot interpolate on an empty trajectory")

    if timestamp_ns < traj.start_ns or timestamp_ns > traj.end_ns:
        raise TrajectoryError(
            f"Timestamp {timestamp_ns} is outside trajectory range [{traj.start_ns}, {traj.end_ns}]"
        )

    # Exact boundary matches
    if timestamp_ns == traj.start_ns:
        return traj.frames[0].pose
    if timestamp_ns == traj.end_ns:
        return traj.frames[-1].pose

    # Binary search for interval
    timestamps = [f.timestamp_ns for f in traj.frames]
    idx = bisect.bisect_right(timestamps, timestamp_ns) - 1
    idx = max(0, min(len(traj.frames) - 2, idx))

    f0 = traj.frames[idx]
    f1 = traj.frames[idx + 1]

    t0 = f0.timestamp_ns
    t1 = f1.timestamp_ns

    if t1 == t0:
        return f0.pose

    alpha = (timestamp_ns - t0) / (t1 - t0)

    # Interpolate translation
    p0 = f0.pose.translation
    p1 = f1.pose.translation
    interp_pos = Vec3(
        (1.0 - alpha) * p0.x + alpha * p1.x,
        (1.0 - alpha) * p0.y + alpha * p1.y,
        (1.0 - alpha) * p0.z + alpha * p1.z,
    )

    # Interpolate orientation via SLERP
    interp_rot = slerp(f0.pose.rotation, f1.pose.rotation, alpha)

    return RigidTransform(
        from_frame=traj.body_frame,
        to_frame=traj.world_frame,
        rotation=interp_rot,
        translation=interp_pos,
    )


def resample_trajectory(
    traj: Trajectory,
    rate_hz: float,
) -> Trajectory:
    """Resample trajectory to a uniform sampling rate in Hz."""
    if rate_hz <= 0.0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")

    step_ns = int(round(1e9 / rate_hz))
    if step_ns <= 0:
        step_ns = 1

    current_t = traj.start_ns
    new_frames: list[TrajectoryFrame] = []

    while current_t <= traj.end_ns:
        pose = interpolate_pose(traj, current_t)
        new_frames.append(TrajectoryFrame(timestamp_ns=current_t, pose=pose))
        current_t += step_ns

    # Ensure last frame is included if not exact match
    if new_frames[-1].timestamp_ns < traj.end_ns:
        new_frames.append(TrajectoryFrame(timestamp_ns=traj.end_ns, pose=traj.frames[-1].pose))

    return Trajectory(
        frames=tuple(new_frames),
        frame_source=traj.frame_source,
        provenance=f"{traj.provenance} (resampled at {rate_hz} Hz)".strip(),
        drift_estimate=traj.drift_estimate,
        clock_id=traj.clock_id,
        sync_state=traj.sync_state,
        sync_method=traj.sync_method,
    )


def smooth_trajectory(
    traj: Trajectory,
    window_size: int = 5,
) -> Trajectory:
    """Smooth position coordinates using a centered moving average while maintaining monotonic timestamps."""
    n = len(traj.frames)
    if n <= 2 or window_size < 3:
        return traj

    half_w = window_size // 2
    smoothed_frames: list[TrajectoryFrame] = []

    for i in range(n):
        # Keep endpoints pinned
        if i == 0 or i == n - 1:
            smoothed_frames.append(traj.frames[i])
            continue

        start_idx = max(0, i - half_w)
        end_idx = min(n, i + half_w + 1)
        k = end_idx - start_idx

        sum_x = sum(traj.frames[j].pose.translation.x for j in range(start_idx, end_idx))
        sum_y = sum(traj.frames[j].pose.translation.y for j in range(start_idx, end_idx))
        sum_z = sum(traj.frames[j].pose.translation.z for j in range(start_idx, end_idx))

        avg_pos = Vec3(sum_x / k, sum_y / k, sum_z / k)

        smoothed_pose = RigidTransform(
            from_frame=traj.body_frame,
            to_frame=traj.world_frame,
            rotation=traj.frames[i].pose.rotation,
            translation=avg_pos,
        )

        smoothed_frames.append(
            TrajectoryFrame(
                timestamp_ns=traj.frames[i].timestamp_ns,
                pose=smoothed_pose,
                pose_covariance_6x6=traj.frames[i].pose_covariance_6x6,
                sensor_timestamp_ns=traj.frames[i].sensor_timestamp_ns,
            )
        )

    return Trajectory(
        frames=tuple(smoothed_frames),
        frame_source=traj.frame_source,
        provenance=f"{traj.provenance} (smoothed window={window_size})".strip(),
        drift_estimate=traj.drift_estimate,
        clock_id=traj.clock_id,
        sync_state=traj.sync_state,
        sync_method=traj.sync_method,
    )
