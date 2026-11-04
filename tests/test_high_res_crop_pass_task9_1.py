"""Unit tests for Blueprint Task 9.1: High-Res Crop Pass (Sockets, Switches) with min_pixels gating.

Verifies:
- High-res crop extraction around candidate wall/fixture ROIs with context margin.
- Multi-scale detection of sockets and switches on crops.
- Resolution-aware min_pixels gating: detections with pixel_size >= min_pixels accepted;
  detections below min_pixels rejected and never claimed.
- Rule Zero compliance: outputs world_ir.observations.Observation instances only.
"""

import numpy as np
import pytest

from perception.detail.crop_pass import (
    CropPassResult,
    HighResCropPass,
    ImageCrop,
    RejectedDetailDetection,
)
from perception.detection.interface import Detector, ImageDetection
from perception.detection.open_vocab import OpenVocabDetector
from perception.ontology import get_default_ontology
from world_ir.observations import Observation


class MockCropDetector(Detector):
    """Deterministic mock detector returning controllable detections on crops."""

    def __init__(self, detections: list[ImageDetection]):
        self._detections = detections

    def is_available(self) -> bool:
        return True

    def detect(self, image: np.ndarray, prompts: list[str], threshold: float = 0.25) -> list[ImageDetection]:
        return [d for d in self._detections if d.score >= threshold]


def test_extract_crops():
    crop_pass = HighResCropPass(crop_padding_ratio=0.10)
    full_img = np.zeros((1000, 1000, 3), dtype=np.uint8)

    regions = [
        (100.0, 100.0, 300.0, 300.0),  # 200x200 box
        (500.0, 500.0, 600.0, 600.0),  # 100x100 box
    ]
    crops = crop_pass.extract_crops(full_img, regions)
    assert len(crops) == 2

    c0 = crops[0]
    # 200 + 2 * (200 * 0.10) = 240
    assert c0.width == 240
    assert c0.height == 240
    assert c0.origin_x == 80
    assert c0.origin_y == 80


def test_crop_pass_resolution_gating_accepts_resolvable_sockets():
    """Verify sockets with pixel area >= min_pixels (100 px) are accepted as Observations."""
    # A 15x15 socket in the crop -> 225 pixels >= 100 px threshold
    mock_det = MockCropDetector([
        ImageDetection(label="electrical socket", score=0.88, box_xyxy=(20.0, 20.0, 35.0, 35.0)),
    ])
    detector = OpenVocabDetector(backend=mock_det)
    crop_pass = HighResCropPass(detector=detector, target_classes=("fixtures/electrical/socket",))

    full_img = np.zeros((800, 800, 3), dtype=np.uint8)
    regions = [(200.0, 200.0, 400.0, 400.0)]

    result = crop_pass.run_pass(full_img, regions, frame_id="cam_01")

    assert len(result.accepted_observations) == 1
    assert len(result.rejected_detections) == 0

    obs = result.accepted_observations[0]
    assert isinstance(obs, Observation)
    assert obs.label == "fixtures/electrical/socket"
    assert obs.bbox_px is not None
    area = (obs.bbox_px[2] - obs.bbox_px[0]) * (obs.bbox_px[3] - obs.bbox_px[1])
    assert area == 225.0
    assert obs.confidence == 0.88
    assert obs.image_id == "cam_01"


def test_crop_pass_resolution_gating_rejects_sub_resolution_switches():
    """Verify switches with pixel area < min_pixels (100 px) are REJECTED and never claimed."""
    # A 7x8 switch in the crop -> 56 pixels < 100 px threshold
    mock_det = MockCropDetector([
        ImageDetection(label="light switch", score=0.92, box_xyxy=(50.0, 50.0, 57.0, 58.0)),
    ])
    detector = OpenVocabDetector(backend=mock_det)
    crop_pass = HighResCropPass(detector=detector, target_classes=("fixtures/electrical/switch",))

    full_img = np.zeros((800, 800, 3), dtype=np.uint8)
    regions = [(100.0, 100.0, 300.0, 300.0)]

    result = crop_pass.run_pass(full_img, regions, frame_id="cam_02")

    # None claimed as observation
    assert len(result.accepted_observations) == 0
    # Recorded as rejected detection
    assert len(result.rejected_detections) == 1
    rej = result.rejected_detections[0]
    assert rej.class_path == "fixtures/electrical/switch"
    assert rej.pixel_size == 56.0
    assert rej.min_pixels == 100
    assert "below_min_pixels" in rej.reason or "< min_pixels" in rej.reason


def test_crop_pass_mixed_scene():
    """Verify in a mixed scene with both resolvable and unresolvable objects, only valid ones pass."""
    mock_det = MockCropDetector([
        # Resolvable socket: 16x16 = 256 px >= 100
        ImageDetection(label="electrical socket", score=0.85, box_xyxy=(10.0, 10.0, 26.0, 26.0)),
        # Unresolvable switch: 6x6 = 36 px < 100
        ImageDetection(label="light switch", score=0.79, box_xyxy=(40.0, 40.0, 46.0, 46.0)),
    ])
    detector = OpenVocabDetector(backend=mock_det)
    crop_pass = HighResCropPass(detector=detector, target_classes=("fixtures/electrical/socket", "fixtures/electrical/switch"))

    full_img = np.zeros((600, 600, 3), dtype=np.uint8)
    regions = [(50.0, 50.0, 250.0, 250.0)]

    result = crop_pass.run_pass(full_img, regions)

    assert len(result.accepted_observations) == 1
    assert result.accepted_observations[0].label == "fixtures/electrical/socket"
    assert len(result.rejected_detections) == 1
    assert result.rejected_detections[0].class_path == "fixtures/electrical/switch"
    assert result.summary["accepted_total"] == 1
    assert result.summary["rejected_total"] == 1
