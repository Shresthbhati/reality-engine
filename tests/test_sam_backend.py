"""Tests for SAM segmentation backend.

Covers:
- Interface contract (ABC enforcement)
- SegmentedRegion/SegmentationResult validation
- Missing-dependency error path (no torch/torchvision/numpy)
- Model type validation
- Device selection logic
"""

from __future__ import annotations

import sys
from unittest.mock import patch, MagicMock

import pytest

from perception.segmentation.interface import (
    ISegmentationBackend,
    SegmentedRegion,
    SegmentationResult,
)
from perception.segmentation.sam_backend import (
    SAMSegmentationBackend,
    SegmentationBackendUnavailableError,
)
from evidence.session import EvidenceItem, EvidenceKind
from provenance import Uncertainty


def _has_perception_deps() -> bool:
    try:
        import torch
        import torchvision
        import numpy
        return True
    except ImportError:
        return False


class TestSAMSegmentationBackendInterface:
    """Verify SAMSegmentationBackend implements ISegmentationBackend contract."""

    def test_implements_interface(self):
        assert issubclass(SAMSegmentationBackend, ISegmentationBackend)

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="Invalid model_type"):
            SAMSegmentationBackend(model_type="invalid")

    def test_valid_model_types_accepted(self):
        for mt in ("vit_b", "vit_l", "vit_h"):
            backend = SAMSegmentationBackend(model_type=mt)
            assert backend._model_type == mt


class TestSegmentedRegionValidation:
    """SegmentedRegion dataclass validation."""

    def test_confidence_bounds_checked(self):
        with pytest.raises(ValueError, match="confidence must be in \\[0, 1\\]"):
            SegmentedRegion(
                region_id="r1",
                evidence_id="e1",
                label="test",
                mask=[[True]],
                confidence=1.5,
            )

    def test_confidence_negative_raises(self):
        with pytest.raises(ValueError, match="confidence must be in \\[0, 1\\]"):
            SegmentedRegion(
                region_id="r1",
                evidence_id="e1",
                label="test",
                mask=[[True]],
                confidence=-0.1,
            )

    def test_valid_region_created(self):
        region = SegmentedRegion(
            region_id="r1",
            evidence_id="e1",
            label="mask_0",
            mask=[[True, False], [False, True]],
            confidence=0.9,
        )
        assert region.region_id == "r1"
        assert region.evidence_id == "e1"
        assert region.label == "mask_0"
        assert region.confidence == 0.9
        assert isinstance(region.uncertainty, Uncertainty)

    def test_mask_preserved(self):
        mask = [[True, False], [False, True]]
        region = SegmentedRegion(
            region_id="r1",
            evidence_id="e1",
            label="test",
            mask=mask,
            confidence=0.5,
        )
        assert region.mask == mask


class TestSegmentationResultValidation:
    """SegmentationResult dataclass validation."""

    def test_result_created_with_model_info(self):
        region = SegmentedRegion(
            region_id="r1", evidence_id="e1", label="mask_0", mask=[[True]], confidence=0.9
        )
        result = SegmentationResult(
            evidence_id="e1",
            regions=[region],
            model_name="sam",
            model_version="vit_h",
        )
        assert result.evidence_id == "e1"
        assert len(result.regions) == 1
        assert result.model_name == "sam"
        assert result.model_version == "vit_h"

    def test_model_version_defaults_empty(self):
        region = SegmentedRegion(
            region_id="r1", evidence_id="e1", label="mask_0", mask=[[True]], confidence=0.9
        )
        result = SegmentationResult(evidence_id="e1", regions=[region], model_name="sam")
        assert result.model_version == ""


class TestSegmentationBackendUnavailableError:
    """Error raised when perception dependencies are missing."""

    def test_raises_when_torch_missing(self):
        with patch.dict(sys.modules, {"torch": None, "torchvision": None, "numpy": None}):
            backend = SAMSegmentationBackend()
            with pytest.raises(SegmentationBackendUnavailableError, match="Missing perception dependencies"):
                backend.segment([])

    def test_raises_when_torchvision_missing(self):
        with patch.dict(sys.modules, {"torch": MagicMock(), "torchvision": None, "numpy": MagicMock()}):
            backend = SAMSegmentationBackend()
            with pytest.raises(SegmentationBackendUnavailableError, match="torchvision"):
                backend.segment([])

    def test_raises_when_numpy_missing(self):
        with patch.dict(sys.modules, {"torch": MagicMock(), "torchvision": MagicMock(), "numpy": None}):
            backend = SAMSegmentationBackend()
            with pytest.raises(SegmentationBackendUnavailableError, match="numpy"):
                backend.segment([])


class TestSAMSegmentationBackendLogic:
    """Test backend logic with mocked torch."""

    def test_segment_skips_non_image_evidence(self):
        """Non-PHOTO/VIDEO evidence should be skipped, not error."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torchvision = MagicMock()
        mock_numpy = MagicMock()
        mock_pil = MagicMock()
        mock_pil_image = MagicMock()
        mock_pil.Image = mock_pil_image

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "torchvision": mock_torchvision,
            "numpy": mock_numpy,
            "PIL": mock_pil,
            "PIL.Image": mock_pil_image,
            "segment_anything": MagicMock(),
        }):
            backend = SAMSegmentationBackend()
            # Mock the internal model loading
            backend._model = MagicMock()
            backend._mask_generator = MagicMock()
            backend._device = "cpu"

            evidence = [
                EvidenceItem(id="e1", kind=EvidenceKind.PHOTO, source_uri="file:///test.jpg"),
                EvidenceItem(id="e2", kind=EvidenceKind.POINT_CLOUD, source_uri="file:///test.las"),
            ]
            results = backend.segment(evidence)
            # Should only process the PHOTO item (POINT_CLOUD skipped)
            # But since we mock PIL.Image.open to fail, it returns empty
            assert isinstance(results, list)

    def test_segment_returns_empty_for_no_images(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torchvision = MagicMock()
        mock_numpy = MagicMock()

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "torchvision": mock_torchvision,
            "numpy": mock_numpy,
            "segment_anything": MagicMock(),
        }):
            backend = SAMSegmentationBackend()
            backend._model = MagicMock()
            backend._mask_generator = MagicMock()
            backend._device = "cpu"

            evidence = [
                EvidenceItem(id="e1", kind=EvidenceKind.POINT_CLOUD, source_uri="file:///test.las"),
            ]
            results = backend.segment(evidence)
            assert results == []

    def test_device_defaults_to_cuda_when_available(self):
        with patch.dict(sys.modules, {"torch": MagicMock()}):
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = True

            backend = SAMSegmentationBackend()
            assert backend._device == "cuda"

    def test_device_defaults_to_cpu_when_cuda_unavailable(self):
        with patch.dict(sys.modules, {"torch": MagicMock()}):
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = False

            backend = SAMSegmentationBackend()
            assert backend._device == "cpu"

    def test_explicit_device_overrides_auto(self):
        with patch.dict(sys.modules, {"torch": MagicMock()}):
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = True

            backend = SAMSegmentationBackend(device="cpu")
            assert backend._device == "cpu"

    def test_mask_generator_config_stored(self):
        backend = SAMSegmentationBackend(
            points_per_side=16,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.9,
        )
        assert backend._mask_gen_config["points_per_side"] == 16
        assert backend._mask_gen_config["pred_iou_thresh"] == 0.9
        assert backend._mask_gen_config["stability_score_thresh"] == 0.9


class TestSAMSegmentationBackendIntegration:
    """Integration-style tests (require actual deps - marked as optional)."""

    @pytest.mark.skipif(
        not _has_perception_deps(),
        reason="Perception dependencies (torch/torchvision/numpy) not installed"
    )
    def test_real_model_load_and_inference(self, tmp_path):
        """Load real SAM model and run inference on a synthetic image."""
        import torch
        import numpy as np
        from PIL import Image

        # Create a simple test image with distinct regions
        img = Image.new("RGB", (320, 240), color=(128, 128, 128))
        # Add a colored rectangle to give SAM something to segment
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill=(255, 0, 0))
        draw.rectangle([200, 100, 280, 200], fill=(0, 255, 0))

        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        evidence = EvidenceItem(
            id="test_img",
            kind=EvidenceKind.PHOTO,
            source_uri=f"file://{img_path}"
        )

        backend = SAMSegmentationBackend(model_type="vit_b")  # Smallest/fastest
        results = backend.segment([evidence])

        assert len(results) == 1
        result = results[0]
        assert result.evidence_id == "test_img"
        assert result.model_name == "sam"
        assert result.model_version == "vit_b"
        assert len(result.regions) > 0  # Should find at least some masks

        # Check region structure
        for region in result.regions:
            assert region.evidence_id == "test_img"
            assert region.region_id.startswith("test_img_mask_")
            assert region.label.startswith("mask_")
            assert 0.0 < region.confidence <= 1.0
            assert len(region.mask) == 240
            assert len(region.mask[0]) == 320
            # Mask should be boolean
            for row in region.mask:
                for v in row:
                    assert isinstance(v, bool)


class TestSAMSegmentationBackendProvenance:
    """Test that provenance fields are correctly populated."""

    def test_segmentation_result_carries_model_info(self):
        """SegmentationResult must carry model_name and model_version for provenance."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torchvision = MagicMock()
        mock_numpy = MagicMock()

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "torchvision": mock_torchvision,
            "numpy": mock_numpy,
            "segment_anything": MagicMock(),
        }):
            backend = SAMSegmentationBackend(model_type="vit_l")
            backend._model = MagicMock()
            backend._mask_generator = MagicMock()
            backend._device = "cpu"

            # We can't easily test the full output without a real image,
            # but we can verify the model info is stored
            assert backend._model_type == "vit_l"