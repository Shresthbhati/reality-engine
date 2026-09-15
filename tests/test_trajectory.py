"""Tests for trajectories/trajectory.py (P3-01 canonical trajectory
model): construction validation (monotonic timestamps, consistent
frame identity) and serialization roundtrip, per
docs/future/vio/VIO_TRAJECTORY.md acceptance criteria.
"""

import pytest

from engine.physics.math3 import Quat, Vec3
from reconstruction.calibration.transforms import RigidTransform
from trajectories.trajectory import (
    DriftEstimate,
    FrameSource,
    Trajectory,
    TrajectoryError,
    TrajectoryFrame,
)


def _pose(x: float, from_frame="body", to_frame="world") -> RigidTransform:
    return RigidTransform(
        from_frame=from_frame,
        to_frame=to_frame,
        rotation=Quat.identity(),
        translation=Vec3(x, 0.0, 0.0),
    )


def _frames(n: int, step_ns: int = 1_000_000) -> tuple:
    return tuple(
        TrajectoryFrame(timestamp_ns=i * step_ns, pose=_pose(float(i))) for i in range(n)
    )


class TestTrajectoryConstruction:
    def test_single_frame_ok(self):
        traj = Trajectory(frames=_frames(1))
        assert len(traj) == 1
        assert traj.start_ns == traj.end_ns == 0

    def test_monotonic_frames_ok(self):
        traj = Trajectory(frames=_frames(5))
        assert len(traj) == 5
        assert traj.start_ns == 0
        assert traj.end_ns == 4_000_000
        assert traj.duration_ns == 4_000_000

    def test_empty_frames_rejected(self):
        with pytest.raises(TrajectoryError, match="at least one frame"):
            Trajectory(frames=())

    def test_non_increasing_timestamps_rejected(self):
        frames = (
            TrajectoryFrame(timestamp_ns=1000, pose=_pose(0.0)),
            TrajectoryFrame(timestamp_ns=1000, pose=_pose(1.0)),
        )
        with pytest.raises(TrajectoryError, match="strictly increasing"):
            Trajectory(frames=frames)

    def test_decreasing_timestamps_rejected(self):
        frames = (
            TrajectoryFrame(timestamp_ns=2000, pose=_pose(0.0)),
            TrajectoryFrame(timestamp_ns=1000, pose=_pose(1.0)),
        )
        with pytest.raises(TrajectoryError, match="strictly increasing"):
            Trajectory(frames=frames)

    def test_inconsistent_from_frame_rejected(self):
        frames = (
            TrajectoryFrame(timestamp_ns=1000, pose=_pose(0.0, from_frame="body")),
            TrajectoryFrame(timestamp_ns=2000, pose=_pose(1.0, from_frame="imu")),
        )
        with pytest.raises(TrajectoryError, match="inconsistent frame identity"):
            Trajectory(frames=frames)

    def test_inconsistent_to_frame_rejected(self):
        frames = (
            TrajectoryFrame(timestamp_ns=1000, pose=_pose(0.0, to_frame="world")),
            TrajectoryFrame(timestamp_ns=2000, pose=_pose(1.0, to_frame="map")),
        )
        with pytest.raises(TrajectoryError, match="inconsistent frame identity"):
            Trajectory(frames=frames)

    def test_body_and_world_frame_properties(self):
        traj = Trajectory(frames=_frames(3))
        assert traj.body_frame == "body"
        assert traj.world_frame == "world"


class TestTrajectoryFrameCovariance:
    def test_valid_6x6_covariance_accepted(self):
        cov = tuple(float(i) for i in range(36))
        frame = TrajectoryFrame(timestamp_ns=0, pose=_pose(0.0), pose_covariance_6x6=cov)
        assert frame.pose_covariance_6x6 == cov

    def test_wrong_length_covariance_rejected(self):
        with pytest.raises(TrajectoryError, match="36 entries"):
            TrajectoryFrame(timestamp_ns=0, pose=_pose(0.0), pose_covariance_6x6=(1.0, 2.0))


class TestDriftEstimate:
    def test_unknown_is_default(self):
        traj = Trajectory(frames=_frames(2))
        assert traj.drift_estimate.known is False
        assert traj.drift_estimate.scalar_m is None

    def test_unknown_with_values_rejected(self):
        with pytest.raises(TrajectoryError, match="UNKNOWN drift estimate"):
            DriftEstimate(known=False, scalar_m=1.0)

    def test_known_scalar_ok(self):
        drift = DriftEstimate(known=True, scalar_m=0.05)
        assert drift.scalar_m == 0.05

    def test_known_negative_scalar_rejected(self):
        with pytest.raises(TrajectoryError, match=">= 0"):
            DriftEstimate(known=True, scalar_m=-0.1)

    def test_known_without_any_value_rejected(self):
        with pytest.raises(TrajectoryError, match="scalar_m and/or covariance_6x6"):
            DriftEstimate(known=True)

    def test_known_wrong_covariance_length_rejected(self):
        with pytest.raises(TrajectoryError, match="36 entries"):
            DriftEstimate(known=True, covariance_6x6=(1.0, 2.0, 3.0))


class TestTrajectoryAt:
    def test_at_exact_timestamp(self):
        traj = Trajectory(frames=_frames(3, step_ns=500))
        frame = traj.at(500)
        assert frame.timestamp_ns == 500

    def test_at_missing_timestamp_raises(self):
        traj = Trajectory(frames=_frames(3, step_ns=500))
        with pytest.raises(TrajectoryError, match="does not interpolate"):
            traj.at(250)


class TestSerializationRoundtrip:
    def test_trajectory_roundtrip(self):
        traj = Trajectory(
            frames=_frames(4),
            frame_source=FrameSource.VIO,
            provenance="orb_slam3 v1.0, real-time monocular-inertial",
            drift_estimate=DriftEstimate(known=True, scalar_m=0.02),
        )
        restored = Trajectory.from_dict(traj.to_dict())
        assert restored == traj

    def test_trajectory_roundtrip_unknown_drift(self):
        traj = Trajectory(frames=_frames(2), frame_source=FrameSource.SFM)
        restored = Trajectory.from_dict(traj.to_dict())
        assert restored.drift_estimate.known is False
        assert restored == traj

    def test_frame_roundtrip_with_covariance(self):
        cov = tuple(float(i) * 0.1 for i in range(36))
        frame = TrajectoryFrame(timestamp_ns=42, pose=_pose(1.0), pose_covariance_6x6=cov)
        restored = TrajectoryFrame.from_dict(frame.to_dict())
        assert restored == frame

    def test_drift_estimate_roundtrip_covariance(self):
        cov = tuple(float(i) for i in range(36))
        drift = DriftEstimate(known=True, covariance_6x6=cov)
        restored = DriftEstimate.from_dict(drift.to_dict())
        assert restored == drift

    def test_frame_source_all_values_roundtrip(self):
        for source in FrameSource:
            traj = Trajectory(frames=_frames(1), frame_source=source)
            restored = Trajectory.from_dict(traj.to_dict())
            assert restored.frame_source == source
