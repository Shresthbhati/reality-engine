"""Multi-Engine Comparative Evaluation Package.

Exports the standardized comparison harness and metric structures for:
- Camera trajectory alignment (ATE / RPE via Umeyama Sim(3))
- Depth estimation comparison (Affine scale/shift alignment, MAE, edge correlation)
- Object detection consensus & conflict evaluation (Greedy bipartite IoU matching)
- Segmentation mask overlap and hole topology evaluation
- Mesh and point cloud geometry comparison (Chamfer distance)
"""

from __future__ import annotations

from .multi_engine_comparator import (
    ComparativeEvaluationHarness,
    DepthComparisonResult,
    DetectionComparisonResult,
    MaskComparisonResult,
    MeshComparisonResult,
    PoseComparisonResult,
)

__all__ = [
    "ComparativeEvaluationHarness",
    "DepthComparisonResult",
    "DetectionComparisonResult",
    "MaskComparisonResult",
    "MeshComparisonResult",
    "PoseComparisonResult",
]
