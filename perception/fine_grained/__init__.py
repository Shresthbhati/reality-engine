"""Fine-Grained Perception and Multi-View Component Recognition (Universal Directive Rule 5 & 6)."""

from perception.fine_grained.multiview_associator import (
    AssociatedEntityCluster,
    MultiViewIdentityAssociator,
    VisualObservation3D,
)
from perception.fine_grained.occlusion import (
    OcclusionEngine,
    VisibilityCheckResult,
)
from perception.fine_grained.pipeline import (
    FineGrainedPerceptionEngine,
    FineGrainedPerceptionOptions,
    PerceptionStageSummary,
)

__all__ = [
    "AssociatedEntityCluster",
    "FineGrainedPerceptionEngine",
    "FineGrainedPerceptionOptions",
    "MultiViewIdentityAssociator",
    "OcclusionEngine",
    "PerceptionStageSummary",
    "VisibilityCheckResult",
    "VisualObservation3D",
]
