"""Perception segmentation package.

Exports:
- ISegmentationBackend: Abstract interface for segmentation backends
- SegmentedRegion: One detected region within an evidence item
- SegmentationResult: Full segmentation output for one evidence item
- SAMSegmentationBackend: Real SAM (Meta AI) implementation (requires .[perception])
- SAM2SegmentationBackend: Real SAM 2 (Meta AI) implementation (requires sam2)
- SegmentationBackendUnavailableError: Raised when deps/checkpoint missing
"""

from .interface import ISegmentationBackend, SegmentedRegion, SegmentationResult
from .sam_backend import SAMSegmentationBackend, SegmentationBackendUnavailableError
from .sam2_backend import SAM2SegmentationBackend

__all__ = [
    "ISegmentationBackend",
    "SegmentedRegion",
    "SegmentationResult",
    "SAMSegmentationBackend",
    "SAM2SegmentationBackend",
    "SegmentationBackendUnavailableError",
]
