"""Basalt adapter (P3-02).

License: AGPL-3.0 per upstream -- verify at binding time (recorded in
.agent/LICENSES.yaml; AGPL is viral for distributed derivatives, the
strictest of the three candidates -- subprocess boundary is not
optional here). Basalt's `basalt_vio` binary exports its estimated
trajectory in TUM format via `--save-trajectory tum`, treated as this
adapter's handoff format. Exact invocation supplied by the caller via
`request.options["command"]` -- see subprocess_backend.py.
"""

from __future__ import annotations

from trajectories.backend.subprocess_backend import SubprocessTrajectoryBackend
from trajectories.trajectory import FrameSource


class BasaltBackend(SubprocessTrajectoryBackend):
    name = "basalt"
    binary_names = ("basalt_vio", "basalt")
    default_output_filename = "trajectory.txt"
    frame_source = FrameSource.VIO
