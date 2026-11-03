"""Architectural element classification, structural perception, and multi-room topology."""

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
    "SharedWallConsolidation",
    "SharedWallDeduplicationReport",
    "StructuralAnalysisReport",
    "StructuralBay",
    "StructuralRole",
    "analyze_structural_system",
    "deduplicate_shared_walls",
]
