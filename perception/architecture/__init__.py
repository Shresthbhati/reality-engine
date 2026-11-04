"""Architectural element classification, structural perception, and multi-room topology."""

from perception.architecture.adapters import (
    ContractAdaptationResult,
    SurfaceSpan,
    WallContinuityResult,
    adapt_architecture_to_contract,
    classify_glazing_opening,
    evaluate_wall_continuity_under_occlusion,
)
from perception.architecture.shared_walls import (
    SharedWallConsolidation,
    SharedWallDeduplicationReport,
    deduplicate_shared_walls,
)
from perception.architecture.structural_network import (
    StructuralAnalysisReport,
    StructuralBay,
    StructuralRole,
    analyze_structural_system,
)

__all__ = [
    "ContractAdaptationResult",
    "SharedWallConsolidation",
    "SharedWallDeduplicationReport",
    "StructuralAnalysisReport",
    "StructuralBay",
    "StructuralRole",
    "SurfaceSpan",
    "WallContinuityResult",
    "adapt_architecture_to_contract",
    "analyze_structural_system",
    "classify_glazing_opening",
    "deduplicate_shared_walls",
    "evaluate_wall_continuity_under_occlusion",
]

