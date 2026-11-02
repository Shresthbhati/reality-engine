"""Dense/MVS backend registry (plan F3).

The sparse registry (`reconstruction.backend.registry`) ranks metrology tools that need a binary on PATH. Dense depth has a
different availability rule: the CPU plane sweep needs only OpenCV, so it is always a candidate, while COLMAP's MVS
needs COLMAP. Selection returns every RUNNABLE dense backend in preference order; the pipeline can then try them in turn
and record which one actually contributed (never silently claim a backend that did not run).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from reconstruction.backend import plane_sweep


@dataclass(frozen=True)
class DenseBackendDescriptor:
    name: str
    license: str
    #: callable(**kwargs) -> list[(world xyz, rgb)], or None when this repo cannot run it
    runner: Optional[Callable[..., list]] = None
    #: why it is unavailable when runner is None
    unavailable_reason: str = ""


DEFAULT_DENSE_REGISTRY: Tuple[DenseBackendDescriptor, ...] = (
    DenseBackendDescriptor("colmap_mvs", "BSD-3-Clause", runner=None,
                           unavailable_reason="COLMAP dense MVS is driven by the sparse stage, not by this callable registry"),
    DenseBackendDescriptor("plane_sweep_cpu", "Apache-2.0 (OpenCV)", runner=plane_sweep.dense_plane_sweep),
)


def discover_dense_backends(
    registry: Sequence[DenseBackendDescriptor] = DEFAULT_DENSE_REGISTRY,
) -> Tuple[DenseBackendDescriptor, ...]:
    return tuple(d for d in registry if d.runner is not None)


def select_dense_backends(
    registry: Sequence[DenseBackendDescriptor] = DEFAULT_DENSE_REGISTRY,
) -> Tuple[str, ...]:
    return tuple(d.name for d in registry if d.runner is not None)
