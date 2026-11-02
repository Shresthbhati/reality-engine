from pathlib import Path

import numpy as np
import pytest

from perception.detection import interface
from perception.detection.backends import FakeDetector, OwlVitDetector, TorchvisionCocoDetector
from perception.detection.interface import DetectorUnavailable, NullDetector, get_detector
from perception.detection.masks import apply_mask, transient_mask
from perception.detection.policy import STRUCTURE_QUERIES, TRANSIENT_QUERIES, filter_detections


def test_null_detector_raises_never_empty():
    d = NullDetector()
    ok, reason = d.available()
    assert not ok and reason.startswith("no detector installed:")
    with pytest.raises(DetectorUnavailable):
        d.detect(np.zeros((4, 4, 3), np.uint8), ["person"])


def test_registry_returns_null_when_nothing_available(monkeypatch):
    monkeypatch.setattr(OwlVitDetector, "available", lambda self: (False, "owl missing"))
    monkeypatch.setattr(TorchvisionCocoDetector, "available", lambda self: (False, "tv missing"))
    det = get_detector()
    assert isinstance(det, NullDetector)
    reason = det.available()[1]
    assert reason.startswith("no detector installed:") and "owl missing" in reason and "tv missing" in reason
    with pytest.raises(DetectorUnavailable):
        det.detect(np.zeros((4, 4, 3), np.uint8), [])


def test_registry_picks_first_available(monkeypatch):
    monkeypatch.setattr(OwlVitDetector, "available", lambda self: (False, "x"))
    monkeypatch.setattr(TorchvisionCocoDetector, "available", lambda self: (True, "ok"))
    assert get_detector().name == "torchvision-coco"
    assert isinstance(get_detector("nope"), NullDetector)
    assert isinstance(get_detector("owlvit"), NullDetector)


def test_backends_unavailable_without_weights(monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **k: None)
    ok, reason = OwlVitDetector().available()
    assert not ok and "huggingface-cli download google/owlvit-base-patch32" in reason
    monkeypatch.setattr(TorchvisionCocoDetector, "_ckpt", lambda self: Path("/nonexistent/w.pth"))
    ok, reason = TorchvisionCocoDetector().available()
    assert not ok and "fasterrcnn_mobilenet_v3_large_320_fpn" in reason
    for cls in (OwlVitDetector, TorchvisionCocoDetector):
        with pytest.raises(DetectorUnavailable):
            cls().detect(np.zeros((8, 8, 3), np.uint8), ["person"])


def test_fake_end_to_end_person_and_barricade():
    img = np.full((60, 80, 3), 128, np.uint8)
    fake = FakeDetector({"person": [((10, 10, 40, 50), 0.95)],
                         "police barricade": [((30, 10, 60, 50), 0.9)],
                         "unicorn": [((0, 0, 3, 3), 0.5)]})
    assert fake.available()[0]
    dets = fake.detect(img, STRUCTURE_QUERIES + TRANSIENT_QUERIES)
    kept, dropped, unknown = filter_detections(dets)
    assert [d.label for d in kept] == ["police barricade"]
    assert [d.label for d in dropped] == ["person"]
    assert [d.label for d in unknown] == ["unicorn"]
    mask = transient_mask(img.shape, dets, margin=0)
    assert mask[30, 20] and not mask[30, 45] and not mask[1, 1]
    out, valid = apply_mask(img, mask, "inpaint")
    assert out.shape == img.shape and valid[30, 45] and not valid[30, 20]


def test_fake_callable():
    d = FakeDetector(lambda image, queries: [interface.ImageDetection("bench", 1.0, (0, 0, 1, 1))])
    assert d.detect(np.zeros((2, 2, 3), np.uint8), [])[0].label == "bench"
