"""Perception segmentation package.

Exports:
- ISegmentationBackend: Abstract interface for segmentation backends
- SegmentedRegion: One detected region within an evidence item
- SegmentationResult: Full segmentation output for one evidence item
- SAMSegmentationBackend: Real SAM (Meta AI) implementation (requires .[perception])
- SegmentationBackendUnavailableError: Raised when deps/checkpoint missing
"""

from .interface import ISegmentationBackend, SegmentedRegion, SegmentationResult
from .sam_backend import SAMSegmentationBackend, SegmentationBackendUnavailableError

__all__ = [
    "ISegmentationBackend",
    "SegmentedRegion",
    "SegmentationResult",
    "SAMSegmentationBackend",
    "SegmentationBackendUnavailableError",
]