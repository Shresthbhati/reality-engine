"""Perception depth estimation package.

Exports:
- IDepthBackend: Abstract interface for depth estimation backends
- DepthMap: Per-pixel depth output
- MiDaSDepthBackend: Real MiDaS (Intel ISL) implementation (requires .[perception])
- DepthBackendUnavailableError: Raised when deps/checkpoint missing
"""

from .interface import IDepthBackend, DepthMap
from .midAS_backend import MiDaSDepthBackend, DepthBackendUnavailableError

__all__ = [
    "IDepthBackend",
    "DepthMap",
    "MiDaSDepthBackend",
    "DepthBackendUnavailableError",
]