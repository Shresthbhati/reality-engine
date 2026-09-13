"""Real-model integration test for the Mask R-CNN detector backend.

Dependency-gated exactly like tests/test_midas_backend.py's real-model
test: skipped (with reason) when torch/torchvision or the COCO weights
are unavailable -- never silently, and never replaced with fakes.

Verified behavior on real weights (observed, this environment): a
genuine photograph yields labeled detections (e.g. 'book', 'person')
with scores >= threshold and instance masks resampled to the exact
source-image dimensions that lift_region_to_3d requires. The synthetic
room fixture legitimately produces ZERO detections (white-noise
textures are out of COCO's distribution) -- that honesty is asserted
too, so a future regression that fabricates detections cannot hide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evidence.session import EvidenceItem, EvidenceKind
from perception.detection.interface import IDetectorBackend
from perception.detection.maskrcnn_backend import (
    DEFAULT_SCORE_THRESHOLD,
    MaskRCNNDetector,
    DetectionBackendUnavailableError,
)


def _torchvision_available() -> bool:
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except ImportError:
        return False


def _coco_weights_cached() -> bool:
    from perception.model_registry import default_cache_dir
    return any(default_cache_dir().glob("maskrcnn_resnet50_fpn_coco*"))


pytestmark = [
    pytest.mark.skipif(
        not _torchvision_available(),
        reason="torch/torchvision not installed -- real model test",
    ),
    pytest.mark.skipif(
        _torchvision_available() and not _coco_weights_cached(),
        reason="Mask R-CNN COCO weights not cached -- run once online to seed; "
        "never downloading inside tests",
    ),
]


# A real photograph (not committed): any decodable local JPEG works.
_REAL_PHOTO = Path.home() / "Downloads" / "1000007228.jpg"


@pytest.fixture
def real_photo_item():
    if not _REAL_PHOTO.exists():
        pytest.skip(f"local real photo {_REAL_PHOTO} not present")
    return EvidenceItem(
        id="real_photo", kind=EvidenceKind.PHOTO, source_uri=str(_REAL_PHOTO)
    )


class TestMaskRCNNInterface:
    def test_implements_both_interfaces(self):
        assert issubclass(MaskRCNNDetector, IDetectorBackend)
        from perception.segmentation.interface import ISegmentationBackend
        assert issubclass(MaskRCNNDetector, ISegmentationBackend)

    def test_unavailable_raises_not_falls_back(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def _no_torch(name, *a, **k):
            if name.startswith(("torch", "numpy")):
                raise ImportError(f"blocked for test: {name}")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", _no_torch)
        with pytest.raises(DetectionBackendUnavailableError, match="torch/torchvision"):
            MaskRCNNDetector()


@pytest.mark.slow
class TestRealModel:
    def test_real_photo_detection_and_masks(self, real_photo_item):
        det = MaskRCNNDetector(score_threshold=DEFAULT_SCORE_THRESHOLD)
        detections = det.detect([real_photo_item])
        segments = det.segment([real_photo_item])

        assert len(detections) == 1
        assert len(segments) == 1
        assert detections[0].detections, "real photo must produce detections"
        for d in detections[0].detections:
            assert 0.5 <= d.score <= 1.0
            assert d.model_name == "maskrcnn_resnet50_fpn"
            x0, y0, x1, y1 = d.box
            assert x1 > x0 and y1 > y0

        from PIL import Image
        width, height = Image.open(_REAL_PHOTO).size
        for region in segments[0].regions:
            assert region.confidence >= DEFAULT_SCORE_THRESHOLD
            # lift_region_to_3d's hard contract: mask dims == image dims.
            assert len(region.mask) == height
            assert len(region.mask[0]) == width

    def test_synthetic_fixture_is_out_of_distribution(self):
        """Honest boundary: the synthetic room render yields no COCO
        detections. Guards against any future 'always return something'
        regression that would fabricate perception."""
        repo_root = Path(__file__).resolve().parents[1]
        fixture = repo_root.parent.parent / "datasets" / "real_room_capture" / "IMG_0000.jpg"
        if not fixture.exists():
            pytest.skip("synthetic fixture not present in this checkout")
        det = MaskRCNNDetector(score_threshold=0.5)
        item = EvidenceItem(id="IMG_0000", kind=EvidenceKind.PHOTO, source_uri=str(fixture))
        results = det.detect([item])
        assert results and results[0].detections == []
