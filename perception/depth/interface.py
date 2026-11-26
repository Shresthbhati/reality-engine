"""Depth estimation backend abstraction (spec sec 8 PERCEPTION ENGINE,
sec 11 STANDARDIZED ADAPTERS, sec 15 CAMERA UNDERSTANDING).

Mirrors reconstruction/backend/interface.py's pattern: any WorldIR-facing
code depends on IDepthBackend, never on a specific depth model (Depth
Anything, MiDaS, ZoeDepth, or otherwise). Concrete backends live beside it
(``depth_anything_backend``, ``midAS_backend``); each states its own
calibration status rather than letting a consumer infer it from the values.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from evidence.session import EvidenceItem
from provenance import Uncertainty


@dataclass(frozen=True)
class DepthMap:
    """Per-pixel depth for one evidence item (photo or video frame).

    `values` is row-major and may be represented as nested lists or a 2-D
    float32 array. The array representation avoids the memory overhead of
    nested Python floats; consumers read either representation the same way.
    """
    evidence_id: str
    width: int
    height: int
    #: values[row][col]. A list of lists OR a 2-D float32 numpy array: producers pass arrays (about 8x smaller than nested
    #: Python floats, RE-PERF-000010) and consumers read both the same way (len, indexing, ``np.asarray``). float32, not
    #: float16: a float16 scalar times a Python float stays float16 and would silently coarsen un-projection.
    values: Any
    unit: str = "relative"  # "relative" (unscaled, e.g. monocular) or "meters" (metric/scaled)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)
    #: What the numbers ARE, because "relative" alone does not say which way is far. "depth":
    #: larger = farther. "inverse_depth": larger = nearer (disparity-like, the Depth Anything V2
    #: relative output). A consumer that inverts the wrong one produces an inverted room, so this
    #: is explicit and `reconstruction.depth_to_points.metricize_relative_depth` refuses the wrong
    #: one instead of guessing from the value range.
    quantity: str = "depth"
    #: Per-pixel measurement mask (row-major, same shape as ``values``; True = measured). Pixels the
    #: producer could not measure are NaN in ``values`` and False here: the producer never invents a
    #: depth for a hole, and this keeps the hole visible to a consumer that averages or unprojects.
    valid_mask: Any = None

    def __post_init__(self):
        if self.quantity not in ("depth", "inverse_depth"):
            raise ValueError(f"quantity must be 'depth' or 'inverse_depth', got {self.quantity!r}")
        flags = getattr(self.values, "flags", None)     # an array is shared between stages: freeze it (no numpy import here)
        if flags is not None:
            flags.writeable = False
        if len(self.values) != self.height:
            raise ValueError(f"values has {len(self.values)} rows, expected height={self.height}")
        if len(self.values) > 0 and len(self.values[0]) != self.width:
            raise ValueError(f"values row length {len(self.values[0])}, expected width={self.width}")
        if self.valid_mask is not None:
            mask_flags = getattr(self.valid_mask, "flags", None)
            if mask_flags is not None:
                mask_flags.writeable = False
            if len(self.valid_mask) != self.height or (self.height and len(self.valid_mask[0]) != self.width):
                raise ValueError("valid_mask must have the same shape as values")

    @property
    def invalid_pixels(self) -> int | None:
        """How many pixels are holes, or None when the producer supplied no mask."""
        if self.valid_mask is None:
            return None
        return int(sum(1 for row in self.valid_mask for cell in row if not cell))


class IDepthBackend(ABC):
    @abstractmethod
    def estimate_depth(self, evidence: list[EvidenceItem]) -> list[DepthMap]:
        """Produce one DepthMap per PHOTO/VIDEO-frame evidence item the
        backend could actually process.

        Must skip an item it cannot produce real depth for -- returning
        fewer maps than inputs is the honest failure mode; a dummy
        all-zero or all-same-value map would be exactly the fabricated
        output the project conventions forbid.
        """
        ...
