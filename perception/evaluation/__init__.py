"""Perception Evaluation, Benchmarking, and Failure Diagnostics Package.

Exports:
- TiledInferenceEngine for high-resolution small-object detection
- SmallObjectBenchmarkHarness
- SmallObjectBenchmarkReport
- SmallObjectDetection
- EngineFailureDetector
- FailureDiagnosis
- FailureKind
"""

from __future__ import annotations

from .failure_detector import (
    EngineFailureDetector,
    FailureDiagnosis,
    FailureKind,
)
from .small_object_benchmark import (
    SmallObjectBenchmarkHarness,
    SmallObjectBenchmarkReport,
    SmallObjectDetection,
    TiledInferenceEngine,
    TileSpecification,
)

__all__ = [
    "EngineFailureDetector",
    "FailureDiagnosis",
    "FailureKind",
    "SmallObjectBenchmarkHarness",
    "SmallObjectBenchmarkReport",
    "SmallObjectDetection",
    "TileSpecification",
    "TiledInferenceEngine",
]
