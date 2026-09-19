"""2D->3D instance lifting abstraction (spec sec 8 PERCEPTION ENGINE,
sec 19 2D->3D INSTANCE LIFTING).

Same pattern as IDepthBackend/ISegmentationBackend/IReconstructionBackend:
no concrete backend (appearance-embedding re-identification, geometric
consistency via camera poses, or otherwise) implements this yet. Per
docs/REALITY_ENGINE_AUDIT.md, this addresses a critical missing
intelligence layer -- today a "chair" detected in frame 1, frame 2, and
frame 3 becomes three unrelated SegmentedRegions, not one CHAIR entity.
Nothing here performs that association: no tracking algorithm, no
appearance-embedding model, no geometric/epipolar consistency check --
just the interface boundary a real tracker will sit behind. Adding a
concrete backend is separate, real work: research the candidate approach
(embedding re-id, SfM-based reprojection, or a hybrid), benchmark it,
then wire it in behind this interface -- never the other way around.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from perception.segmentation.interface import SegmentationResult, SegmentedRegion
from provenance import Uncertainty


@dataclass(frozen=True)
class InstanceTrack:
    """A group of SegmentedRegions, across possibly-different evidence
    items/frames, that a backend believes are observations of the same
    real-world object.

    `label` is backend-owned, not invented by this interface -- same
    philosophy as SegmentedRegion.label: this interface does not decide
    what a tracked object is called, only that a set of regions were
    associated together under some label. Mapping onto the Reality
    Engine ontology (spec sec 21) is a separate step downstream of this
    one, not done here.
    """
    track_id: str
    regions: List[SegmentedRegion]
    label: str
    confidence: float
    uncertainty: Uncertainty = field(default_factory=Uncertainty)

    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


class ITrackBackend(ABC):
    @abstractmethod
    def link_instances(self, results: List[SegmentationResult]) -> List[InstanceTrack]:
        """Associate SegmentedRegions across evidence items/frames into
        InstanceTracks representing the same real-world object.

        Must skip, not fabricate, an association it cannot support --
        an ungrouped region left out of every track is the honest
        failure mode; forcing every region into some track would be
        exactly the fabricated identity this project's conventions
        forbid.
        """
        ...
