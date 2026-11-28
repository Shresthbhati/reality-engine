"""Depth estimation backend abstraction (spec sec 8 PERCEPTION ENGINE,
sec 11 STANDARDIZED ADAPTERS, sec 15 CAMERA UNDERSTANDING).

Mirrors reconstruction/backend/interface.py's pattern: any WorldIR-facing
code depends on IDepthBackend, never on a specific depth model (Depth
Anything, MiDaS, ZoeDepth, or otherwise). No concrete backend implements
this yet -- per docs/REALITY_ENGINE_AUDIT.md, `perception/` was README-only
before this pass, and this is the interface only, not an integration.
Adding a concrete adapter is separate, real work: research the candidate,
check its license and model-checkpoint license, benchmark it, then wire it
in behind this interface -- never the other way around.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from evidence.session import EvidenceItem
from provenance import Uncertainty


@dataclass(frozen=True)
class DepthMap:
    """Per-pixel depth for one evidence item (photo or video frame).

    `values` is a plain row-major list of rows, not a numpy array --
    this interface has zero dependencies beyond the stdlib, matching
    this repo's dependency discipline (pyproject.toml declares none).
    A concrete adapter wrapping a real depth model will likely convert
    to/from numpy internally; that conversion is the adapter's job, not
    this interface's.
    """
    evidence_id: str
    width: int
    height: int
    values: List[List[float]]  # values[row][col]
    unit: str = "relative"  # "relative" (unscaled, e.g. monocular) or "meters" (metric/scaled)
    uncertainty: Uncertainty = field(default_factory=Uncertainty)

    def __post_init__(self):
        if len(self.values) != self.height:
            raise ValueError(f"values has {len(self.values)} rows, expected height={self.height}")
        if self.values and len(self.values[0]) != self.width:
            raise ValueError(f"values row length {len(self.values[0])}, expected width={self.width}")


class IDepthBackend(ABC):
    @abstractmethod
    def estimate_depth(self, evidence: List[EvidenceItem]) -> List[DepthMap]:
        """Produce one DepthMap per PHOTO/VIDEO-frame evidence item the
        backend could actually process.

        Must skip an item it cannot produce real depth for -- returning
        fewer maps than inputs is the honest failure mode; a dummy
        all-zero or all-same-value map would be exactly the fabricated
        output the project conventions forbid.
        """
        ...
