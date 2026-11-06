"""Perception depth estimation package.

Exports:
- IDepthBackend: Abstract interface for depth estimation backends
- DepthMap: Per-pixel depth output
- HeuristicDepthPriorBackend: Heuristic scene-statistics depth prior (unit='meters', scale_state='estimated', Apache 2.0)
- DepthAnythingV2Backend: Depth Anything V2 Small implementation (Apache 2.0)
- MiDaSDepthBackend: MiDaS (Intel ISL) implementation
- DepthBackendUnavailableError: Raised when deps/checkpoint missing
"""

from .depth_anything_backend import DepthAnythingV2Backend
from .interface import DepthMap, IDepthBackend
from .metric_depth_backend import HeuristicDepthPriorBackend
from .midAS_backend import DepthBackendUnavailableError, MiDaSDepthBackend

__all__ = [
    "DepthAnythingV2Backend",
    "DepthBackendUnavailableError",
    "DepthMap",
    "HeuristicDepthPriorBackend",
    "IDepthBackend",
    "MiDaSDepthBackend",
]
