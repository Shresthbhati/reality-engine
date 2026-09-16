"""Synchronized-time consumers (P2-01 becoming load-bearing, per
docs/future/synchronization/TIME_SYNCHRONIZATION.md): applying a
ClockModel to a backend's sensor-stamped Trajectory.

Rules carried over from the clock model itself:

  - t_global = a * t_sensor + b, with a > 0 (monotonicity is
    preserved, never reordered).
  - Original sensor timestamps are NEVER overwritten. A synchronized
    TrajectoryFrame carries its global stamp in ``timestamp_ns`` and
    keeps the sensor stamp alongside in ``sensor_timestamp_ns`` --
    absent (None) means "never synchronized", not zero.
  - An UNSYNCHRONIZED outcome (the sentinel model synchronize_stream
    returns when every backend declines, ``uncertainty.basis ==
    "no synchronization performed"``) is honest: stamps pass through
    unchanged, the result is marked UNSYNCHRONIZED, nothing raises,
    nothing is fabricated.
  - No model (None) is pass-through: stamps unchanged, the result
    keeps the undeclared marker and never claims a clock.

Order preservation is checked, not assumed: a>0 clock models are
monotone, but if a degenerate/resolution-collapse model ever maps two
strictly-increasing stamps to equal global stamps, that is a real
fault and raises -- silently collapsing time would corrupt every
downstream consumer.

UNITS (explicit, not hidden): trajectory stamps are canonical
nanoseconds, but a ClockModel is fit in the units of whatever produced
it -- sensor sidecar streams are in SECONDS. ``a`` is a dimensionless
drift ratio (unit-independent), but ``b`` carries the model's unit.
Applying a seconds-space b to ns stamps would corrupt time by 10^9
(the exact bug the end-to-end test caught), so `apply_clock_model`
takes an explicit `model_unit` ("ns" default; "s" converts b to ns)
and `estimate_trajectory` forwards it. There is no implicit unit.

The input Trajectory is never mutated.
"""

from __future__ import annotations

from typing import Optional

from evidence.clocks import SynchronizationError, SyncMethod
from trajectories.trajectory import Trajectory, TrajectoryFrame


#: The uncertainty basis string synchronize_stream uses for its
#: "every backend declined" container model. Keeping it in one place
#: (and importing it in clocks.py is deliberately avoided -- no
#: reverse dependency) means the sentinel detection cannot drift.
UNSYNC_BASIS = "no synchronization performed"

#: Marker for "no clock declared on this trajectory".
UNDECLARED_CLOCK = "<undeclared>"


def is_unsynchronized_model(model) -> bool:
    """True iff `model` is the sentinel synchronize_stream returns
    when every backend declined -- identity parameters carrying no
    synchronization information. Detection is by the sentinel basis
    string, not by a==1 and b==0 (a genuine shared-clock model is
    also identity-parameterized and MUST NOT be degraded)."""
    return (
        model is not None
        and model.uncertainty is not None
        and model.uncertainty.basis == UNSYNC_BASIS
    )


NS_PER_S = 1_000_000_000


def apply_clock_model(trajectory: Trajectory, model, model_unit: str = "ns") -> Trajectory:
    """Return a new Trajectory with `model` applied to every frame.

    See module docstring for the contract. `model` is a ClockModel
    (evidence.clocks) or None for pass-through. `model_unit` is the
    time unit the model was fit in -- "ns" (canonical) or "s" (sensor
    streams fit from second-denominated sidecars); trajectory stamps
    are always ns.
    """
    if model_unit not in ("ns", "s"):
        raise SynchronizationError(
            f"model_unit must be 'ns' or 's', got {model_unit!r}"
        )
    if model is None:
        return _pass_through(trajectory)

    if is_unsynchronized_model(model):
        return _decline(trajectory, model)

    # b carries the model's unit; a is dimensionless.
    b_ns = model.b * NS_PER_S if model_unit == "s" else model.b

    frames = []
    prev_global: Optional[int] = None
    for frame in trajectory.frames:
        global_ns = model.a * float(frame.timestamp_ns) + b_ns
        global_ns_i = int(round(global_ns))
        if prev_global is not None and global_ns_i <= prev_global:
            raise SynchronizationError(
                f"clock model collapses time: sensor stamp "
                f"{frame.timestamp_ns}ns maps to {global_ns_i}ns, not after "
                f"the previous frame's {prev_global}ns -- a>0 models are "
                f"monotone, so this indicates a degenerate model"
            )
        prev_global = global_ns_i
        frames.append(
            TrajectoryFrame(
                timestamp_ns=global_ns_i,
                pose=frame.pose,
                pose_covariance_6x6=frame.pose_covariance_6x6,
                sensor_timestamp_ns=frame.timestamp_ns,
            )
        )
    return Trajectory(
        frames=tuple(frames),
        frame_source=trajectory.frame_source,
        provenance=(
            f"{trajectory.provenance} "
            f"[synchronized onto clock_id={model.clock_id} "
            f"by {model.method.value}]"
        ),
        drift_estimate=trajectory.drift_estimate,
        clock_id=model.clock_id,
        sync_state="SYNCHRONIZED",
        sync_method=model.method.value,
    )


def _pass_through(trajectory: Trajectory) -> Trajectory:
    """No model supplied: stamps unchanged, no clock claimed."""
    return Trajectory(
        frames=trajectory.frames,
        frame_source=trajectory.frame_source,
        provenance=trajectory.provenance,
        drift_estimate=trajectory.drift_estimate,
        clock_id=UNDECLARED_CLOCK,
        sync_state=UNDECLARED_CLOCK,
        sync_method=None,
    )


def _decline(trajectory: Trajectory, model) -> Trajectory:
    """Every backend declined: honest UNSYNCHRONIZED degradation --
    stamps unchanged (duplicates included), originals recorded, no
    fabricated timeline, nothing raised."""
    frames = tuple(
        TrajectoryFrame(
            timestamp_ns=f.timestamp_ns,
            pose=f.pose,
            pose_covariance_6x6=f.pose_covariance_6x6,
            sensor_timestamp_ns=f.timestamp_ns,
        )
        for f in trajectory.frames
    )
    return Trajectory(
        frames=frames,
        frame_source=trajectory.frame_source,
        provenance=(
            f"{trajectory.provenance} [UNSYNCHRONIZED: no backend could "
            f"synchronize clock_id={model.clock_id}; original stamps retained]"
        ),
        drift_estimate=trajectory.drift_estimate,
        clock_id=model.clock_id,
        sync_state="UNSYNCHRONIZED",
        sync_method=None,
    )
