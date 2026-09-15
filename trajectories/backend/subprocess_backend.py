"""Shared subprocess-adapter base for VIO backends (P3-02).

ORB-SLAM3, OpenVINS, and Basalt are research systems built per-user
from source with no single stable CLI across builds/datasets (unlike
`colmap`, which ships one documented binary/subcommand surface). This
base class is honest about that: it never guesses a tool's exact
invocation flags. The caller MUST supply the full argv via
`request.options["command"]` (a list of strings, with ``{images}``,
``{imu}``, ``{workdir}`` placeholders substituted from the request).
Without it, `estimate` raises `TrajectoryBackendRunError` naming
exactly what's missing -- never a fabricated command line.

What IS real here: binary-presence detection (`shutil.which`),
subprocess execution with the real diagnostic on failure (matching
`reconstruction.backend.colmap_backend.ReconstructionStepError`'s
discipline), and parsing the tool's real TUM-format trajectory output.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from trajectories.backend.interface import (
    ITrajectoryBackend,
    TrajectoryBackendRunError,
    TrajectoryBackendUnavailableError,
    TrajectoryEstimationRequest,
)
from trajectories.backend.tum_format import parse_tum_trajectory
from trajectories.trajectory import FrameSource, Trajectory


class SubprocessTrajectoryBackend(ITrajectoryBackend):
    """Common shape: find a binary on PATH, run a caller-supplied
    command, parse its TUM-format trajectory output. Subclasses set
    `binary_names`, `default_output_filename`, and `frame_source`."""

    #: Candidate executable names to check with shutil.which (build
    #: naming varies -- e.g. ORB-SLAM3 ships per-example binaries).
    binary_names: Tuple[str, ...] = ()
    #: Filename the tool conventionally writes its TUM trajectory to,
    #: used when the caller doesn't override via options["output_file"].
    default_output_filename: str = "trajectory.txt"
    frame_source: FrameSource = FrameSource.UNKNOWN

    def _resolve_binary(self) -> Optional[str]:
        for candidate in self.binary_names:
            found = shutil.which(candidate)
            if found:
                return found
        return None

    def is_available(self) -> bool:
        return self._resolve_binary() is not None

    def estimate(self, request: TrajectoryEstimationRequest) -> Trajectory:
        binary_path = self._resolve_binary()
        if binary_path is None:
            raise TrajectoryBackendUnavailableError(
                f"{self.name}: none of {self.binary_names!r} found on PATH"
            )

        command_template = request.options.get("command")
        if not command_template:
            raise TrajectoryBackendRunError(
                f"{self.name}: no invocation supplied -- pass "
                f"options['command'] (argv list; supports {{images}}, "
                f"{{imu}}, {{workdir}} placeholders). {self.name}'s exact "
                f"flags are build/dataset-specific and are not guessed."
            )
        command = self._render_command(command_template, request)

        request.workdir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            command,
            cwd=str(request.workdir),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stderr or "").strip().splitlines()[-15:])
            raise TrajectoryBackendRunError(
                f"{self.name} failed (exit {proc.returncode})\n"
                f"command: {' '.join(command)}\n"
                f"stderr tail:\n{tail}"
            )

        output_option = request.options.get("output_file")
        output_path = (
            Path(output_option) if output_option else request.workdir / self.default_output_filename
        )
        return parse_tum_trajectory(
            output_path,
            from_frame=str(request.options.get("body_frame", "body")),
            to_frame=str(request.options.get("world_frame", "world")),
            frame_source=self.frame_source,
            provenance=f"{self.name} ({binary_path}); command: {' '.join(command)}",
        )

    @staticmethod
    def _render_command(
        template: Sequence[str], request: TrajectoryEstimationRequest
    ) -> List[str]:
        substitutions = {
            "images": str(request.images_path),
            "imu": str(request.imu_path) if request.imu_path else "",
            "workdir": str(request.workdir),
        }
        return [str(arg).format(**substitutions) for arg in template]
