"""Object detection backends and open-vocabulary detection pipeline."""

from perception.detection.interface import (
    Detection,
    DetectionResult,
    Detector,
    DetectorUnavailable,
    IDetectorBackend,
    ImageDetection,
    get_detector,
    register_detector,
)
from perception.detection.open_vocab import (
    DetectionEvaluation,
    OpenVocabDetector,
    box_iou,
    evaluate_detection_metrics,
)

__all__ = [
    "Detection",
    "DetectionEvaluation",
    "DetectionResult",
    "Detector",
    "DetectorUnavailable",
    "IDetectorBackend",
    "ImageDetection",
    "OpenVocabDetector",
    "box_iou",
    "evaluate_detection_metrics",
    "get_detector",
    "register_detector",
]
