"""ORB-SLAM3 adapter (P3-02).

License: GPL-3.0 -- subprocess-only integration, source never
vendored/linked (recorded in .agent/LICENSES.yaml federation
candidates). ORB-SLAM3 ships per-dataset example binaries (mono_tum,
mono_euroc, mono_inertial_euroc, ...) rather than one stable CLI, so
the exact binary + argv are supplied by the caller via
`request.options["command"]` -- see subprocess_backend.py's docstring
for why this is not guessed.

Output: `SaveKeyFrameTrajectoryTUM` / `SaveTrajectoryTUM`, ORB-SLAM3's
own TUM-format export, is the documented handoff into this repo's
`trajectories.backend.tum_format.parse_tum_trajectory`.
"""

from __future__ import annotations

from trajectories.backend.subprocess_backend import SubprocessTrajectoryBackend
from trajectories.trajectory import FrameSource


class ORBSLAM3Backend(SubprocessTrajectoryBackend):
    name = "orb_slam3"
    #: No single canonical binary name across builds; common example
    #: names are checked, in the order ORB-SLAM3's own repo lists them.
    binary_names = ("orb_slam3", "mono_inertial_euroc", "mono_euroc", "mono_tum")
    default_output_filename = "KeyFrameTrajectory.txt"
    frame_source = FrameSource.VIO
