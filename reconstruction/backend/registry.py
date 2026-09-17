"""Reconstruction backend registry + selection (P5-01/P5-02).

Mirrors `trajectories.backend.selection`'s pattern (binary-presence
discovery via shutil.which, honest unavailability, no fabricated
capability data) applied to the five SfM/MVS tools named in the spec:
COLMAP (the only one with a real adapter --
`reconstruction.backend.colmap_backend.ColmapReconstructionBackend`),
OpenMVS, AliceVision/Meshroom, OpenSfM, OpenDroneMap. The latter four
have no adapters in this repo (nobody has built them, per CAPABILITIES
honesty) -- discovery reports their binary presence/absence only;
selection never ranks a backend that has no adapter to actually run.

Ranking is explicit deterministic policy (constitution: "selection is
explicit policy, not a language model"), not learned or guessed: a
fixed preference order, filtered down to backends that are both
available AND runnable (has an adapter).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from reconstruction.backend.colmap_backend import ColmapReconstructionBackend
from reconstruction.backend.interface import IReconstructionBackend


@dataclass(frozen=True)
class BackendDescriptor:
    """One candidate backend's static identity: what to check for on
    PATH, its declared license, and (if this repo has one) a factory
    for the real adapter. `adapter_factory=None` means this backend is
    known and checked for, but nothing in this repo can run it yet --
    an honest MISSING, not a silently-skipped entry."""

    name: str
    binary_names: Tuple[str, ...]
    license: str
    adapter_factory: Optional[Callable[[], IReconstructionBackend]] = None


@dataclass(frozen=True)
class BackendAvailability:
    """Discovery result for one descriptor."""

    name: str
    available: bool
    binary_path: str = ""
    license: str = ""
    has_adapter: bool = False


#: Declared candidates in the spec's order. Only COLMAP has a real
#: adapter; the rest are discovery-only until one is built.
DEFAULT_REGISTRY: Tuple[BackendDescriptor, ...] = (
    BackendDescriptor("colmap", ("colmap",), "BSD-3-Clause", ColmapReconstructionBackend),
    BackendDescriptor("openmvs", ("OpenMVS", "DensifyPointCloud"), "AGPL-3.0"),
    BackendDescriptor("alicevision_meshroom", ("meshroom_batch", "aliceVision_cameraInit"), "MPL-2.0"),
    BackendDescriptor("opensfm", ("opensfm",), "BSD-2-Clause"),
    BackendDescriptor("opendronemap", ("odm",), "AGPL-3.0"),
)


def _resolve(descriptor: BackendDescriptor) -> str:
    """First matching binary's resolved path, or "" if none found."""
    for candidate in descriptor.binary_names:
        found = shutil.which(candidate)
        if found:
            return found
    return ""


def discover_backends(
    registry: Sequence[BackendDescriptor] = DEFAULT_REGISTRY,
) -> Tuple[BackendAvailability, ...]:
    """Check PATH for every declared candidate. Never raises -- a
    missing binary is a normal, expected result, reported honestly."""
    results = []
    for descriptor in registry:
        binary_path = _resolve(descriptor)
        results.append(
            BackendAvailability(
                name=descriptor.name,
                available=bool(binary_path),
                binary_path=binary_path,
                license=descriptor.license,
                has_adapter=descriptor.adapter_factory is not None,
            )
        )
    return tuple(results)


def select_backends(
    registry: Sequence[BackendDescriptor] = DEFAULT_REGISTRY,
) -> Tuple[str, ...]:
    """Ranked, RUNNABLE candidate names: available on this machine AND
    this repo has an adapter for it, in the registry's declared
    preference order. Installed-but-no-adapter and adapter-but-not-
    installed are both correctly excluded -- neither is runnable."""
    return tuple(
        d.name for d in registry
        if d.adapter_factory is not None and _resolve(d)
    )
