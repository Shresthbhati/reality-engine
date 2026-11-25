"""Contract tests for the single normalized detection boundary (perception/detection/normalized.py)."""

from __future__ import annotations

import numpy as np
import pytest

from perception.detection.backends import FakeDetector
from perception.detection.interface import Detection, ImageDetection
from perception.detection.normalized import (
    CoordinateConvention,
    DetectionStatus,
    InvalidDetectionError,
    NormalizedDetection,
    NormalizedDetectionBatch,
    convert_and_validate_box,
    normalize_backend_output,
    normalize_detection,
)


def test_pixel_xyxy_and_normalized_xyxy_conversion():
    px, norm = convert_and_validate_box(
        (100, 50, 300, 250),
        convention=CoordinateConvention.PIXEL_XYXY,
        image_width=1000,
        image_height=500,
    )
    assert px == (100.0, 50.0, 300.0, 250.0)
    assert norm == (0.1, 0.1, 0.3, 0.5)

    px2, norm2 = convert_and_validate_box(
        (0.1, 0.2, 0.5, 0.8),
        convention=CoordinateConvention.NORMALIZED_XYXY,
        image_width=800,
        image_height=600,
    )
    assert px2 == (80.0, 120.0, 400.0, 480.0)
    assert norm2 == (0.1, 0.2, 0.5, 0.8)


@pytest.mark.parametrize(
    "bad_box",
    [
        (-10, 10, 50, 50),         # negative x0
        (10, -5, 50, 50),          # negative y0
        (10, 10, 650, 50),         # x1 > width
        (10, 10, 50, 490),         # y1 > height
        (100, 10, 100, 50),        # zero width
        (10, 100, 50, 50),         # inverted y
        (float("nan"), 10, 50, 50),
        (10, 10, float("inf"), 50),
    ],
)
def test_invalid_negative_and_out_of_image_boxes_rejected(bad_box):
    with pytest.raises(InvalidDetectionError):
        convert_and_validate_box(
            bad_box,
            convention=CoordinateConvention.PIXEL_XYXY,
            image_width=640,
            image_height=480,
        )


def test_normalizes_all_existing_backend_types_and_preserves_missing_confidence_and_unknown_labels():
    # 1. MaskRCNN Detection (.box, .score)
    det_mrcnn = Detection(
        detection_id="d1",
        evidence_id="ev1",
        label="chair",
        box=(10.0, 20.0, 110.0, 220.0),
        score=0.91,
        model_name="maskrcnn_resnet50_fpn",
        model_version="coco_v1",
    )
    n1 = normalize_detection(det_mrcnn, image_width=640, image_height=480, device="cuda")
    assert n1.bbox_px == (10.0, 20.0, 110.0, 220.0)
    assert n1.confidence == pytest.approx(0.91)
    assert n1.has_mask is False
    assert "no_mask" in n1.quality_flags

    # 2. ImageDetection (.box_xyxy, .score, .mask) from OwlVit / Torchvision / FakeDetector
    mask = np.zeros((480, 640), dtype=bool)
    mask[20:100, 30:130] = True
    det_owl = ImageDetection(
        label="wooden archway",
        score=0.64,
        box_xyxy=(30.0, 20.0, 130.0, 100.0),
        mask=mask,
        view_id="view_2",
        source="owlvit",
    )
    n2 = normalize_detection(det_owl, image_width=640, image_height=480)
    assert n2.label == "wooden archway"
    assert n2.canonical_label == "wooden archway"  # unknown category preserved, not forced
    assert n2.has_mask is True
    assert n2.mask is not None and n2.mask.shape == (480, 640)

    # 3. Florence-2 dict (no confidence -> remains None, never fabricated)
    det_florence = {"label": "wainscoting_panel", "box_xyxy": (50.0, 60.0, 250.0, 300.0)}
    n3 = normalize_detection(det_florence, image_width=1024, image_height=768, model_name="florence2")
    assert n3.confidence is None
    assert "missing_confidence" in n3.quality_flags
    assert n3.canonical_label == "wainscoting_panel"
    assert n3.image_width == 1024 and n3.image_height == 768


def test_empty_detections_vs_model_failure_and_batch_rejection():
    empty_batch = normalize_backend_output(
        [],
        image_width=640,
        image_height=480,
        model_name="owlvit",
        image_id="img_0",
    )
    assert empty_batch.status is DetectionStatus.EMPTY
    assert empty_batch.detections == ()
    assert empty_batch.error is None

    failed_batch = normalize_backend_output(
        None,
        image_width=640,
        image_height=480,
        model_name="grounding_dino",
        image_id="img_0",
        error="CUDA out of memory",
    )
    assert failed_batch.status is DetectionStatus.FAILED
    assert failed_batch.error == "CUDA out of memory"
    assert failed_batch.detections == ()

    # Mixed valid + invalid boxes: valid kept, invalid counted honestly
    fake = FakeDetector({"door": [((10, 10, 100, 300), 0.8), ((-5, 10, 50, 50), 0.7)]})
    raw = fake.detect(np.zeros((480, 640, 3), dtype=np.uint8), ["door"])
    mixed_batch = normalize_backend_output(raw, image_width=640, image_height=480, model_name="fake")
    assert mixed_batch.status is DetectionStatus.OK
    assert len(mixed_batch.detections) == 1
    assert mixed_batch.rejected_count == 1
    assert "negative box coordinates" in mixed_batch.rejection_reasons[0]


def test_different_image_resolutions_and_mask_shape_mismatch():
    bad_mask = np.zeros((100, 100), dtype=bool)
    with pytest.raises(InvalidDetectionError):
        normalize_detection(
            {"label": "wall", "box_xyxy": (10, 10, 200, 200), "mask": bad_mask},
            image_width=1024,
            image_height=768,
        )

    with pytest.raises(InvalidDetectionError):
        NormalizedDetectionBatch(
            image_id="x",
            image_width=640,
            image_height=480,
            status=DetectionStatus.FAILED,
            error=None,  # forbidden
        )
