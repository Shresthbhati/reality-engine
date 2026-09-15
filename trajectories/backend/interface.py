"""VIO/localization backend abstraction (P3-02).

Mirrors the two existing backend-seam patterns in this repo --
`reconstruction.backend.interface.IReconstructionBackend` (subprocess
CV backend) and `evidence.clocks.ISynchronizationBackend` (try-in-order
selection with honest failure) -- rather than inventing a third shape.

Per the ledger's standing rule (P3-02: "Do NOT build a mediocre custom
VIO first -- federate"), this module federates to mature external
binaries (ORB-SLAM3, OpenVINS, Basalt) behind one interface. No
backend estimates VIO itself; every adapter shells out to a real
binary and parses its real output, or raises
`TrajectoryBackendUnavailableError` when that binary is not installed
-- never a fabricated or interpolated trajectory.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from trajectories.trajectory import Trajectory


class TrajectoryBackendUnavailableError(RuntimeError):
    """The backend's binary (or a required input) is not available on
    this machine. Never caught silently -- selection.py records the
    reason and tries the next backend; if every backend declines, the
    caller gets an honest failure, not a fabricated trajectory."""


class TrajectoryBackendRunError(RuntimeError):
    """The backend binary ran but exited non-zero, or its output could
    not be parsed. Carries the real diagnostic (stderr tail / parse
    error), matching ReconstructionStepError's discipline of never
    collapsing a real failure into an opaque message."""


@dataclass(frozen=True)
class TrajectoryEstimationRequest:
    """What a VIO/VO backend needs to estimate a trajectory. Backends
    that don't use IMU (visual-only fallback) simply ignore
    ``imu_path``; a backend that REQUIRES IMU and doesn't get one must
    raise, not silently run visual-only and call it VIO."""

    #: Directory of ordered frame images, or a video file -- backend-specific.
    images_path: Path
    #: IMU CSV/text log, if present (format is backend-specific).
    imu_path: Optional[Path] = None
    #: Working directory for the backend's intermediate/output files.
    workdir: Path = field(default_factory=lambda: Path("."))
    #: Clock id the resulting timestamps are declared to be on (fed
    #: into ClockModel elsewhere -- this module does not synchronize).
    clock_id: str = "<undeclared>"
    #: Extra backend-specific options (config file path, camera model,
    #: vocabulary path, ...). Deliberately untyped -- each adapter
    #: documents the keys it reads.
    options: Dict[str, object] = field(default_factory=dict)


class ITrajectoryBackend(ABC):
    """One VIO/VO backend behind a common contract.

    `estimate` must raise `TrajectoryBackendUnavailableError` (binary
    missing, required input missing) or `TrajectoryBackendRunError`
    (binary ran and failed, or output unparseable) rather than return
    a degraded/guessed Trajectory. `is_available` is a cheap,
    side-effect-free presence check (binary on PATH) used by selection
    to skip backends without running them.
    """

    #: Registry name, also the default provenance label.
    name: str = "abstract"

    @abstractmethod
    def is_available(self) -> bool:
        """True iff this backend's binary is installed. Must not raise."""
        ...

    @abstractmethod
    def estimate(self, request: TrajectoryEstimationRequest) -> Trajectory:
        ...
