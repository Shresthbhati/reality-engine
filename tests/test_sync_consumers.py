"""Tests for synchronized-time consumers (directive E-B): the clock
model (P2-01) becoming load-bearing downstream.

Contract (docs/future/synchronization/TIME_SYNCHRONIZATION.md +
TrajectoryFrame's own docstring, which has promised "post
ClockModel.apply" timestamps since P3-01 without anyone applying it):

  - `apply_clock_model` maps a backend's sensor-stamped trajectory
    onto the global timeline (t_global = a*t_sensor + b, ns integer
    rounding), sets clock_id/sync_state/sync_method on the result,
    and PRESERVES every original sensor stamp alongside its global
    stamp -- originals are never overwritten.
  - No model supplied -> pass-through: stamps unchanged, trajectory
    stays declared "<undeclared>" (never silently claims a clock it
    was not synchronized to).
  - Known answers to ns precision for offset-only and drift models.
  - Honest failure: an UNSYNCHRONIZED sentinel model (the container
    synchronize_stream returns when every backend declines) leaves
    the trajectory UNSYNCHRONIZED -- stamps unchanged, no exception,
    no fabricated global timeline.
  - `estimate_trajectory` integrates the model via its declared
    clock_id seam, and records the synchronization into provenance
    so downstream consumers can tell a synchronized trajectory from
    an undeclared one.
  - Synchronization never reorders (a > 0 preserves monotonicity).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.math import Quat, Vec3
from evidence.clocks import ClockModel, SynchronizationError, SyncMethod, SyncUncertainty
from reconstruction.calibration.transforms import RigidTransform
from trajectories.backend import selection as traj_selection
from trajectories.backend.interface import TrajectoryEstimationRequest
from trajectories.backend.sync import apply_clock_model
from trajectories.trajectory import (
    DriftEstimate,
    FrameSource,
    Trajectory,
    TrajectoryFrame,
)


def _pose(x: float) -> RigidTransform:
    return RigidTransform(
        from_frame="body",
        to_frame="world",
        rotation=Quat.identity(),
        translation=Vec3(x, 0.0, 0.0),
    )


def _trajectory(stamps_ns) -> Trajectory:
    frames = tuple(
        TrajectoryFrame(timestamp_ns=t, pose=_pose(t * 1e-9)) for t in stamps_ns
    )
    return Trajectory(
        frames=frames,
        frame_source=FrameSource.VIO,
        provenance="test-backend",
        drift_estimate=DriftEstimate.unknown(),
    )


def _request(clock_id: str) -> TrajectoryEstimationRequest:
    return TrajectoryEstimationRequest(images_path=Path("."), clock_id=clock_id)


def _unsync_model(clock_id: str = "cam0") -> ClockModel:
    """The exact sentinel synchronize_stream returns when every
    backend declines: identity parameters, 'no synchronization
    performed' basis."""
    return ClockModel(
        clock_id=clock_id,
        a=1.0,
        b=0.0,
        method=SyncMethod.KNOWN_OFFSET,
        uncertainty=SyncUncertainty(basis="no synchronization performed"),
    )


# ------------------------------------------------------------------
# 1. Pass-through without a model
# ------------------------------------------------------------------


class TestPassThrough:
    def test_no_model_stamps_unchanged_and_declared_undeclared(self):
        traj = _trajectory((1_000, 2_000, 3_000))
        out = apply_clock_model(traj, None)
        assert [f.timestamp_ns for f in out.frames] == [1_000, 2_000, 3_000]
        assert out.clock_id == "<undeclared>"
        assert out.sync_state == "<undeclared>"
        assert out.sync_method is None

    def test_no_model_does_not_claim_synchronization(self):
        out = apply_clock_model(_trajectory((1, 2)), None)
        assert "synchroniz" not in out.provenance.lower()


# ------------------------------------------------------------------
# 2. Model application (known answers)
# ------------------------------------------------------------------


class TestModelApplication:
    def test_offset_only_model_shifts_every_frame(self):
        model = ClockModel(clock_id="cam0", a=1.0, b=2_500_000.0,
                           method=SyncMethod.KNOWN_OFFSET)
        out = apply_clock_model(_trajectory((1_000_000, 2_000_000, 3_000_000)), model)
        assert [f.timestamp_ns for f in out.frames] == [3_500_000, 4_500_000, 5_500_000]
        assert out.clock_id == "cam0"
        assert out.sync_state == "SYNCHRONIZED"
        assert out.sync_method == "known_offset"

    def test_drift_model_known_answer_to_ns(self):
        # a=1.01, b=-5e6: sensor 1e9 ns -> 1.005e9 ns exactly.
        model = ClockModel(clock_id="cam0", a=1.01, b=-5_000_000.0,
                           method=SyncMethod.SIGNAL_CORRELATION)
        out = apply_clock_model(_trajectory((1_000_000_000, 2_000_000_000)), model)
        assert [f.timestamp_ns for f in out.frames] == [1_005_000_000, 2_015_000_000]
        assert out.sync_method == "signal_correlation"

    def test_original_sensor_stamps_survive(self):
        model = ClockModel(clock_id="cam0", a=1.0, b=100.0)
        out = apply_clock_model(_trajectory((10, 20)), model)
        assert out.frames[0].timestamp_ns == 110
        # Originals preserved alongside, never overwritten (spec).
        assert out.frames[0].sensor_timestamp_ns == 10
        assert out.frames[1].sensor_timestamp_ns == 20

    def test_input_trajectory_not_mutated(self):
        model = ClockModel(clock_id="cam0", a=1.0, b=100.0)
        src = _trajectory((10, 20))
        apply_clock_model(src, model)
        assert src.frames[0].timestamp_ns == 10
        assert src.frames[0].sensor_timestamp_ns is None

    def test_sync_never_reorders(self):
        model = ClockModel(clock_id="cam0", a=1.5, b=7.0)
        stamps = (100, 200, 300)
        out = apply_clock_model(_trajectory(stamps), model)
        assert [f.timestamp_ns for f in out.frames] == [157, 307, 457]

    def test_roundtrip_keeps_sync_metadata(self):
        model = ClockModel(clock_id="cam0", a=1.0, b=50.0, method=SyncMethod.TRIGGER)
        out = apply_clock_model(_trajectory((1, 2)), model)
        restored = Trajectory.from_dict(out.to_dict())
        assert restored.clock_id == "cam0"
        assert restored.sync_state == "SYNCHRONIZED"
        assert restored.sync_method == "trigger"
        assert restored.frames[0].sensor_timestamp_ns == 1


# ------------------------------------------------------------------
# 3. Honest failure: UNSYNCHRONIZED sentinel model
# ------------------------------------------------------------------


class TestUnsynchronizableDeclines:
    def test_sentinel_model_leaves_trajectory_unsynchronized(self):
        out = apply_clock_model(_trajectory((1_000, 2_000)), _unsync_model())
        assert [f.timestamp_ns for f in out.frames] == [1_000, 2_000]
        assert out.clock_id == "cam0"
        assert out.sync_state == "UNSYNCHRONIZED"
        assert out.sync_method is None

    def test_time_collapse_is_an_error_not_silence(self):
        # A model with sub-ns effective resolution would map distinct
        # sensor stamps to equal global stamps. Silently collapsing
        # time would corrupt every downstream consumer -- this is a
        # real fault and must raise.
        model = ClockModel(clock_id="cam0", a=1e-12, b=0.0)
        with pytest.raises(SynchronizationError):
            apply_clock_model(_trajectory((1_000, 2_000)), model)

    def test_sentinel_model_originals_still_recorded(self):
        out = apply_clock_model(_trajectory((10, 20)), _unsync_model())
        assert out.frames[0].sensor_timestamp_ns == 10


# ------------------------------------------------------------------
# 4. Integration: estimate_trajectory consumes the clock model
# ------------------------------------------------------------------


class _ScriptedBackend:
    """Minimal stand-in returning a fixed sensor-stamped trajectory."""

    name = "scripted"

    def __init__(self, stamps_ns):
        self._stamps = tuple(stamps_ns)

    def is_available(self) -> bool:
        return True

    def estimate(self, request: TrajectoryEstimationRequest) -> Trajectory:
        return _trajectory(self._stamps)


class TestEstimateTrajectorySync:
    def test_model_applied_through_selection(self):
        model = ClockModel(clock_id="cam0", a=1.0, b=2_500_000.0,
                           method=SyncMethod.KNOWN_OFFSET)
        traj, _attempts = traj_selection.estimate_trajectory(
            _request("cam0"), backends=(_ScriptedBackend((1_000_000,)),),
            clock_model=model,
        )
        assert traj.frames[0].timestamp_ns == 3_500_000
        assert traj.clock_id == "cam0"
        assert traj.sync_state == "SYNCHRONIZED"
        assert traj.sync_method == "known_offset"

    def test_no_model_selection_is_pass_through(self):
        traj, _ = traj_selection.estimate_trajectory(
            _request("<undeclared>"), backends=(_ScriptedBackend((42,)),)
        )
        assert traj.frames[0].timestamp_ns == 42
        assert traj.clock_id == "<undeclared>"
        assert traj.sync_state == "<undeclared>"

    def test_unsynchronizable_selection_degrades_not_raises(self):
        traj, _ = traj_selection.estimate_trajectory(
            _request("cam0"), backends=(_ScriptedBackend((7, 8)),),
            clock_model=_unsync_model(),
        )
        assert traj.sync_state == "UNSYNCHRONIZED"
        assert [f.timestamp_ns for f in traj.frames] == [7, 8]

    def test_provenance_names_the_sync(self):
        model = ClockModel(clock_id="cam0", a=1.0, b=1.0, method=SyncMethod.TRIGGER)
        traj, _ = traj_selection.estimate_trajectory(
            _request("cam0"), backends=(_ScriptedBackend((1,)),), clock_model=model
        )
        assert "clock_id=cam0" in traj.provenance
        assert "trigger" in traj.provenance


class TestEndToEndWithRealSync:
    """The full chain: SensorStream -> synchronize_stream (real
    backend, declared anchors) -> ClockModel -> estimate_trajectory ->
    synchronized canonical Trajectory."""

    def test_stream_to_synchronized_trajectory(self):
        from evidence.clocks import synchronize_stream
        from evidence.sensors import SensorStream

        # Ground truth: t_global = 1.001 * t_sensor + 100.0 (seconds).
        def truth(t):
            return 1.001 * t + 100.0

        anchors = [(t, truth(t)) for t in (0.0, 10.0, 20.0, 30.0)]
        stream = SensorStream(
            kind="imu", source_path="test/imu.jsonl",
            samples=tuple(type("S", (), {"t": t})() for t, _ in anchors),
        )
        alignment = synchronize_stream(
            stream, "cam0", {"gnss_anchor_pairs": anchors}
        )
        assert alignment.diagnostics.method == "gnss_pps"

        # Backend stamps are sensor-clock ns at 10 Hz starting at t=1 s.
        sensor_stamps_s = (1.0, 2.0, 3.0)
        backend_stamps = tuple(round(t * 1e9) for t in sensor_stamps_s)
        traj, _ = traj_selection.estimate_trajectory(
            _request("cam0"),
            backends=(_ScriptedBackend(backend_stamps),),
            clock_model=alignment.model,
            model_unit="s",  # stream-fitted model: sidecar units are seconds
        )
        assert traj.sync_state == "SYNCHRONIZED"
        assert traj.clock_id == "cam0"
        # Known answers: t_global = 1.001 * t_sensor + 100.
        for frame, t_s in zip(traj.frames, sensor_stamps_s):
            expected_ns = round((1.001 * t_s + 100.0) * 1e9)
            assert frame.timestamp_ns == expected_ns
            assert frame.sensor_timestamp_ns == round(t_s * 1e9)
