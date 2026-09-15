"""OpenVINS adapter (P3-02).

License: GPL-3.0 with linking exceptions per upstream -- verify at
binding time (recorded in .agent/LICENSES.yaml). OpenVINS is normally
run as a ROS node; its `ov_eval` companion tool exports the estimated
trajectory in TUM format for benchmarking, which this adapter treats
as the handoff format. The exact launch invocation (ROS launch file or
`run_subscribe_msckf` binary + config) is supplied by the caller via
`request.options["command"]` -- see subprocess_backend.py.
"""

from __future__ import annotations

from trajectories.backend.subprocess_backend import SubprocessTrajectoryBackend
from trajectories.trajectory import FrameSource


class OpenVINSBackend(SubprocessTrajectoryBackend):
    name = "openvins"
    binary_names = ("run_subscribe_msckf", "openvins")
    default_output_filename = "stamped_traj_estimate.txt"
    frame_source = FrameSource.VIO
