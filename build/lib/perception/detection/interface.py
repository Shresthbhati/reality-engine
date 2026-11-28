"""Object detection backend abstraction (mapping campaign P0.7).

Before this module `perception/detection/` held only a README: nothing in
the engine produced a real object detection, so the lifting -> resolution
-> WorldIR promotion chain had no real upstream. This interface follows
the same pattern as `ISegmentationBackend`/`IDepthBackend`: the abstract
contract lives here; concrete backends own their model acquisition and
declare unavailability honestly.

`Detection` is deliberately minimal: class, box, score, source evidence,
model. Mapping onto masks (segmentation) is a separate backend's job --
a detector that also emits masks (e.g. Mask R-CNN) may implement
`ISegmentationBackend` too (see `perception/detection/maskrcnn_backend.py`,
which does exactly that).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from evidence.session import EvidenceItem
from provenance import Uncertainty


@dataclass(frozen=True)
class Detection:
    """One detected object in one evidence item's image plane.

    `box` is (x_min, y_min, x_max, y_max) in pixel coordinates of the
    source image. `label` is the backend's raw class name; ontology
    mapping to the Reality Engine taxonomy is downstream (the promotion
    stage keeps the raw label in `semantic_labels`)."""

    detection_id: str
    evidence_id: str
    label: str
    box: tuple  # (x_min, y_min, x_max, y_max) pixels
    score: float
    model_name: str
    model_version: str = ""
    uncertainty: Uncertainty = field(default_factory=Uncertainty)

    def __post_init__(self):
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score}")
        x_min, y_min, x_max, y_max = self.box
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"degenerate box {self.box!r}")

    def to_dict(self) -> dict:
        return {
            "detection_id": self.detection_id,
            "evidence_id": self.evidence_id,
            "label": self.label,
            "box": list(self.box),
            "score": self.score,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


class IDetectorBackend(ABC):
    @abstractmethod
    def detect(self, evidence: list) -> list:
        """Produce one DetectionResult per evidence item the backend could
        actually process; skip (never fabricate) items it cannot."""
        ...


@dataclass(frozen=True)
class DetectionResult:
    evidence_id: str
    detections: list  # List[Detection]
    model_name: str
    model_version: str = ""
