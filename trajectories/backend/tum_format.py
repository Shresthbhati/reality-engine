"""Parser for the TUM trajectory text format.

TUM format (from the TUM RGB-D benchmark tools, adopted as the common
evaluation/output format by ORB-SLAM3's ``SaveKeyFrameTrajectoryTUM``,
OpenVINS's ``ov_eval`` trajectory recorder, and Basalt's VIO result
export) is one pose per line:

    timestamp tx ty tz qx qy qz qw

whitespace-separated, ``#``-prefixed and blank lines are comments.
Position in meters, quaternion in the (x, y, z, w) convention (the
file format's, not this repo's -- reordered on parse). This module is
a pure function over the file format so it is testable without any of
the three binaries installed; the adapters (orb_slam3_backend.py etc.)
call it after running their subprocess.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from engine.math import Quat, Vec3
from reconstruction.calibration.transforms import RigidTransform
from trajectories.backend.interface import TrajectoryBackendRunError
from trajectories.trajectory import FrameSource, Trajectory, TrajectoryFrame


def parse_tum_trajectory(
    path: Path,
    *,
    from_frame: str,
    to_frame: str,
    frame_source: FrameSource,
    provenance: str,
) -> Trajectory:
    """Parse a TUM-format trajectory file into a canonical `Trajectory`.

    Timestamps in the file are seconds (float, TUM convention);
    converted to integer nanoseconds for `TrajectoryFrame` (P3-01's
    monotonic-timestamp contract is integer nanoseconds). Raises
    `TrajectoryBackendRunError` on a missing file, malformed line, or
    a non-finite/non-unit quaternion -- a backend that produced
    garbage output is a run failure, not something to silently drop
    rows from.
    """
    if not path.is_file():
        raise TrajectoryBackendRunError(f"TUM trajectory file not found: {path}")

    frames: List[TrajectoryFrame] = []
    text = path.read_text(encoding="utf-8")
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise TrajectoryBackendRunError(
                f"{path}:{lineno}: expected 8 TUM fields "
                f"(t tx ty tz qx qy qz qw), got {len(parts)}: {line!r}"
            )
        try:
            t, tx, ty, tz, qx, qy, qz, qw = (float(p) for p in parts)
        except ValueError as exc:
            raise TrajectoryBackendRunError(f"{path}:{lineno}: non-numeric field: {line!r}") from exc

        timestamp_ns = round(t * 1_000_000_000)
        rotation = Quat(qw, qx, qy, qz)
        try:
            pose = RigidTransform(
                from_frame=from_frame,
                to_frame=to_frame,
                rotation=rotation,
                translation=Vec3(tx, ty, tz),
            )
        except Exception as exc:  # RigidTransform validates finiteness/unit-norm
            raise TrajectoryBackendRunError(
                f"{path}:{lineno}: invalid pose ({exc})"
            ) from exc
        frames.append(TrajectoryFrame(timestamp_ns=timestamp_ns, pose=pose))

    if not frames:
        raise TrajectoryBackendRunError(f"{path}: no trajectory rows parsed")

    try:
        return Trajectory(
            frames=tuple(frames),
            frame_source=frame_source,
            provenance=provenance,
        )
    except Exception as exc:  # Trajectory validates monotonicity
        raise TrajectoryBackendRunError(f"{path}: {exc}") from exc
