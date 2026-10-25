"""Attach a ``Trajectory`` to a ``WorldIR`` as an observation record.

Until this module a trajectory (VIO / SfM / EKF output, P3) never reached a world: ``WorldIR`` carries per-entity
transforms and sensor ``Observation`` records, and ``build_frame_graph_from_world`` documented that a trajectory
"is not stored in WorldIR directly". This is the minimal honest bridge: one ``Observation`` per trajectory, holding
the poses verbatim (``t_ns, tx, ty, tz, qw, qx, qy, qz`` rows, camera/body -> world), the clock and synchronisation
state, the drift estimate and where the poses came from.

Frame honesty: the trajectory's frames are mapped to the graph's ``Frame`` enum (``world_ir.frame_graph``). If the
trajectory's target frame is not the world's own ``coordinate_frame`` (a VIO frame is a gauge-fixed session frame
until registration anchors it), that is RECORDED (``registered_into_world: False``) -- no transform is invented to
make them agree.

Idempotent: the observation id is derived from the pose content, so attaching the same trajectory twice replaces
rather than duplicates.
"""

from __future__ import annotations

import hashlib
import json
from typing import Dict, Optional

from provenance import Uncertainty
from world_ir.coordinates import Frame
from world_ir.frame_graph import _trajectory_frame
from world_ir.schema_v1 import Observation


def _rows(trajectory) -> list:
    out = []
    for f in trajectory.frames:
        r, t = f.pose.rotation, f.pose.translation
        out.append([int(f.timestamp_ns), t.x, t.y, t.z, r.w, r.x, r.y, r.z])
    return out


def attach_trajectory(world, trajectory, *, frame_map: Optional[Dict[str, Frame]] = None) -> str:
    """Record ``trajectory`` in ``world.observations``; returns the observation id."""
    rows = _rows(trajectory)
    payload = json.dumps(rows, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    pose0 = trajectory.frames[0].pose
    body = _trajectory_frame(pose0.from_frame, frame_map)
    target = _trajectory_frame(pose0.to_frame, frame_map)
    registered = target == world.coordinate_frame
    drift = trajectory.drift_estimate
    if drift.known:
        drift_text = f"drift {drift.scalar_m:.3f} m" if drift.scalar_m is not None else "drift covariance given"
    else:
        drift_text = "drift NOT estimated (unknown, not zero)"
    note = (f"{trajectory.frame_source.value} trajectory, {len(rows)} poses, sync {trajectory.sync_state}; "
            f"{drift_text}"
            + ("" if registered else f"; expressed in {target.value}, not the world frame "
               f"({world.coordinate_frame.value}) -- registration has not anchored it"))
    obs = Observation(
        id=f"obs-trajectory-{digest[:12]}",
        sensor_type=trajectory.frame_source.value,
        timestamp=trajectory.start_ns / 1e9,
        frame_id=target.value,
        data_hash=digest,
        metadata={
            "kind": "trajectory",
            "frame_source": trajectory.frame_source.value,
            "body_frame": body.value, "world_frame": target.value,
            "registered_into_world": registered,
            "clock_id": trajectory.clock_id, "sync_state": trajectory.sync_state, "sync_method": trajectory.sync_method,
            "n_poses": len(rows), "start_ns": trajectory.start_ns, "end_ns": trajectory.end_ns,
            "drift": drift.to_dict(), "provenance": trajectory.provenance,
            "pose_columns": ["t_ns", "tx", "ty", "tz", "qw", "qx", "qy", "qz"], "poses": rows,
        },
        confidence=0.5,   # not measured: a trajectory carries no calibrated confidence of its own
        uncertainty=Uncertainty(confidence=0.5, note=note),
    )
    world.observations[obs.id] = obs
    return obs.id
