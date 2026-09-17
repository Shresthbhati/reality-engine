"""Tests for trajectories/diagnostics.py (P3-03): ATE/RPE against
known analytic values, endpoint drift, and error handling.
"""

import pytest

from engine.math import Quat, Vec3
from reconstruction.calibration.transforms import RigidTransform
from trajectories.diagnostics import (
    absolute_trajectory_error,
    endpoint_drift,
    relative_pose_error,
)
from trajectories.trajectory import Trajectory, TrajectoryError, TrajectoryFrame


def _traj(offsets, step_ns=1000):
    return Trajectory(
        frames=tuple(
            TrajectoryFrame(
                timestamp_ns=i * step_ns,
                pose=RigidTransform(
                    from_frame="body", to_frame="world",
                    rotation=Quat.identity(), translation=Vec3(x, 0.0, 0.0),
                ),
            )
            for i, x in enumerate(offsets)
        )
    )


class TestATE:
    def test_zero_error_when_identical(self):
        gt = _traj([0.0, 1.0, 2.0])
        stats = absolute_trajectory_error(gt, gt)
        assert stats.rmse == 0.0
        assert stats.count == 3

    def test_constant_offset_analytic(self):
        gt = _traj([0.0, 1.0, 2.0, 3.0])
        est = _traj([0.3, 1.3, 2.3, 3.3])
        stats = absolute_trajectory_error(est, gt)
        assert stats.rmse == pytest.approx(0.3)
        assert stats.mean == pytest.approx(0.3)
        assert stats.std == pytest.approx(0.0, abs=1e-9)

    def test_no_overlap_raises(self):
        a = _traj([0.0, 1.0], step_ns=1000)
        b = _traj([0.0, 1.0], step_ns=2000)
        # only t=0 overlaps -> still one match, not an error
        stats = absolute_trajectory_error(a, b)
        assert stats.count == 1

        c = Trajectory(frames=(TrajectoryFrame(
            999, RigidTransform("body", "world", Quat.identity(), Vec3(0, 0, 0))
        ),))
        with pytest.raises(TrajectoryError, match="no matching timestamps"):
            absolute_trajectory_error(a, c)


class TestRPE:
    def test_zero_when_identical(self):
        gt = _traj([0.0, 1.0, 2.0, 3.0])
        stats = relative_pose_error(gt, gt, delta_ns=1000)
        assert stats.rmse == pytest.approx(0.0, abs=1e-9)

    def test_constant_offset_cancels(self):
        # a constant translation offset produces zero RELATIVE error
        gt = _traj([0.0, 1.0, 2.0, 3.0])
        est = _traj([0.5, 1.5, 2.5, 3.5])
        stats = relative_pose_error(est, gt, delta_ns=1000)
        assert stats.rmse == pytest.approx(0.0, abs=1e-9)

    def test_drifting_error_grows(self):
        gt = _traj([0.0, 1.0, 2.0, 3.0])
        est = _traj([0.0, 1.1, 2.3, 3.6])  # accelerating drift
        stats = relative_pose_error(est, gt, delta_ns=1000)
        assert stats.rmse > 0.0

    def test_non_positive_delta_raises(self):
        gt = _traj([0.0, 1.0])
        with pytest.raises(TrajectoryError, match="delta_ns must be > 0"):
            relative_pose_error(gt, gt, delta_ns=0)


class TestEndpointDrift:
    def test_loop_closure_zero(self):
        gt = _traj([0.0, 1.0, 2.0, 0.0])
        assert endpoint_drift(gt) == pytest.approx(0.0)

    def test_open_path_nonzero(self):
        gt = _traj([0.0, 1.0, 2.0, 5.0])
        assert endpoint_drift(gt) == pytest.approx(5.0)
