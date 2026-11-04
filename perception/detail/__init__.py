"""Universal detail engine (universal-perception directive PHASE 2-3 & Wave 2 Block 16).

See discovery.py for scope: evidence-backed discovery of WHERE detail
is recoverable, domain-agnostic -- points in, measured curvature +
per-cell budgets out. The ROI/refinement layers consume these
candidates.

See adaptive_inspection.py for adaptive zoom viewpoint scoring, micro-geometry
fastener inspection, and fine-grained fixture attribute enrichment.
"""

from .adaptive_inspection import (
    AdaptiveInspectionReport,
    FastenerInspection,
    InspectionRefused,
    InspectionViewSelection,
    apply_adaptive_inspection_to_world,
    perform_adaptive_inspection,
    select_inspection_views,
)
from .crop_pass import (
    CropPassResult,
    HighResCropPass,
    ImageCrop,
    RejectedDetailDetection,
)
from .discovery import DetailCandidate, discover_detail
from .pipeline import DetailPipelineReport, run_detail_pipeline
from .roi import RegionOfInterest, generate_rois

__all__ = [
    "AdaptiveInspectionReport",
    "CropPassResult",
    "DetailCandidate",
    "DetailPipelineReport",
    "FastenerInspection",
    "HighResCropPass",
    "ImageCrop",
    "InspectionRefused",
    "InspectionViewSelection",
    "RegionOfInterest",
    "RejectedDetailDetection",
    "apply_adaptive_inspection_to_world",
    "discover_detail",
    "generate_rois",
    "perform_adaptive_inspection",
    "run_detail_pipeline",
    "select_inspection_views",
]
