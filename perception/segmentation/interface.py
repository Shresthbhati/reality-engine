"""Segmentation backend abstraction (spec sec 8 PERCEPTION ENGINE,
sec 18 SEGMENTATION, sec 11 STANDARDIZED ADAPTERS).

Same pattern as IDepthBackend/IReconstructionBackend: no concrete
backend (SAM2 or otherwise) implements this yet. `model_name`/
`model_version` on SegmentationResult exist because spec sec 48
(PROVENANCE) requires tracing a semantic claim back to what produced
it -- "why does the engine believe this is a wall?" must be answerable
down to the model, not just "segmentation ran".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from evidence.session import EvidenceItem
from provenance import Uncertainty


@dataclass(frozen=True)
class SegmentedRegion:
    """One detected region (instance or semantic class) within one
    evidence item's image plane.

    `label` is whatever class name the backend produces -- open-vocabulary
    or fixed-ontology, the backend's choice. This interface does not
    invent or constrain labels; mapping a backend's raw label onto the
    Reality Engine ontology (spec sec 21) is a separate step downstream
    of this one, not done here.
    """
    region_id: str
    evidence_id: str
    label: str
    mask: List[List[bool]]  # mask[row][col], same resolution as the source image
    confidence: float
    uncertainty: Uncertainty = field(default_factory=Uncertainty)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True)
class SegmentationResult:
    evidence_id: str
    regions: List[SegmentedRegion]
    model_name: str
    model_version: str = ""


class ISegmentationBackend(ABC):
    @abstractmethod
    def segment(self, evidence: List[EvidenceItem]) -> List[SegmentationResult]:
        """Produce one SegmentationResult per evidence item the backend
        could actually process. Must skip, not fabricate, an item it
        cannot segment (e.g. unsupported format, model failure)."""
        ...
