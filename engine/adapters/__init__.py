"""Standardized Multi-Engine Adapters Package.

Exports independent, production-grade adapters for:
- Structure from Motion (pycolmap)
- Monocular Depth Estimation (Depth Anything V2, MiDaS DPT)
- Open-Vocabulary Object Detection (OWL-ViT, Grounding DINO, Florence-2)
- Instance Segmentation (SAM 1 Box-Prompted)
- Feature Matching (LightGlue ALIKED / DISK)
- Volumetric Reconstruction (Open3D ScalableTSDFVolume)
"""

from __future__ import annotations

from .base import (
    BaseEngineAdapter,
    DependencyMissingError,
    EnginePredictionResult,
    EngineUnavailableError,
    InputValidationError,
    LicenceBlockedError,
    ProducerReport,
)
from .depth_anything_v2 import DepthAnythingV2Adapter
from .depth_midas_dpt import MidasDPTAdapter
from .detector_florence2 import Florence2DetectorAdapter
from .detector_grounding_dino import GroundingDinoDetectorAdapter
from .detector_owlvit import OwlVitDetectorAdapter
from .matcher_lightglue import LightGlueMatcherAdapter
from .segmenter_sam import SamPromptedSegmenterAdapter
from .sfm_pycolmap import PycolmapSfMAdapter
from .volumetric_open3d import Open3DTSDFAdapter

__all__ = [
    "BaseEngineAdapter",
    "DependencyMissingError",
    "DepthAnythingV2Adapter",
    "EnginePredictionResult",
    "EngineUnavailableError",
    "Florence2DetectorAdapter",
    "GroundingDinoDetectorAdapter",
    "InputValidationError",
    "LicenceBlockedError",
    "LightGlueMatcherAdapter",
    "MidasDPTAdapter",
    "Open3DTSDFAdapter",
    "OwlVitDetectorAdapter",
    "ProducerReport",
    "PycolmapSfMAdapter",
    "SamPromptedSegmenterAdapter",
]
