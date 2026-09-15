"""Backend selection for VIO trajectory estimation (P3-02).

Mirrors `evidence.clocks.synchronize_stream`'s try-in-order, honest-
failure shape: skip backends whose binary isn't installed (cheap,
`is_available()`), attempt the rest in DEFAULT_BACKENDS order, and if
every backend is unavailable or fails, raise
`TrajectoryBackendUnavailableError` naming every attempt -- never
return a fabricated or partially-estimated trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from trajectories.backend.basalt_backend import BasaltBackend
from trajectories.backend.interface import (
    ITrajectoryBackend,
    TrajectoryBackendRunError,
    TrajectoryBackendUnavailableError,
    TrajectoryEstimationRequest,
)
from trajectories.backend.openvins_backend import OpenVINSBackend
from trajectories.backend.orb_slam3_backend import ORBSLAM3Backend
from trajectories.trajectory import Trajectory

#: Federation order: ORB-SLAM3 (most mature/widely deployed for
#: mono-inertial), OpenVINS (MSCKF, lighter-weight), Basalt (strictest
#: license -- tried last). Each entry is a fresh instance; backends are
#: stateless.
DEFAULT_BACKENDS: Tuple[ITrajectoryBackend, ...] = (
    ORBSLAM3Backend(),
    OpenVINSBackend(),
    BasaltBackend(),
)


@dataclass(frozen=True)
class BackendAttempt:
    """One backend's outcome, recorded whether it succeeded or not --
    the diagnostics a caller needs to understand why a particular
    backend was or wasn't used."""

    backend_name: str
    available: bool
    error: str = ""


def estimate_trajectory(
    request: TrajectoryEstimationRequest,
    backends: Sequence[ITrajectoryBackend] = DEFAULT_BACKENDS,
) -> Tuple[Trajectory, Tuple[BackendAttempt, ...]]:
    """Try `backends` in order; the first that produces a Trajectory
    wins. Returns the trajectory plus the full attempt log (including
    the backends that were skipped or failed) so callers can record
    provenance honestly rather than silently picking a backend.

    Raises `TrajectoryBackendUnavailableError` if every backend is
    unavailable, and `TrajectoryBackendRunError` if at least one was
    available but all attempts failed -- the distinction matters: "no
    VIO binary installed" and "VIO ran and failed" are different
    failure modes an operator needs to tell apart.
    """
    if not backends:
        raise TrajectoryBackendUnavailableError("backends sequence must not be empty")

    attempts: List[BackendAttempt] = []
    any_available = False
    run_errors: List[str] = []

    for backend in backends:
        if not backend.is_available():
            attempts.append(BackendAttempt(backend.name, available=False))
            continue
        any_available = True
        try:
            trajectory = backend.estimate(request)
            attempts.append(BackendAttempt(backend.name, available=True))
            return trajectory, tuple(attempts)
        except TrajectoryBackendUnavailableError as exc:
            attempts.append(BackendAttempt(backend.name, available=False, error=str(exc)))
        except TrajectoryBackendRunError as exc:
            attempts.append(BackendAttempt(backend.name, available=True, error=str(exc)))
            run_errors.append(f"{backend.name}: {exc}")

    if not any_available:
        names = ", ".join(b.name for b in backends)
        raise TrajectoryBackendUnavailableError(
            f"no VIO backend available (checked: {names}) -- install "
            f"ORB-SLAM3, OpenVINS, or Basalt, or gate this call"
        )
    raise TrajectoryBackendRunError(
        "every available VIO backend failed:\n" + "\n".join(run_errors)
    )
