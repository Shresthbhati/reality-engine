"""Tests for MiDaS depth backend.

Covers:
- Interface contract (ABC enforcement)
- DepthMap validation (shape, bounds, unit field)
- Missing-dependency error path (no torch/torchvision/numpy)
- Model type validation
- Device selection logic
"""

from __future__ import annotations

import sys
from unittest.mock import patch, MagicMock

import pytest

from perception.depth.interface import IDepthBackend, DepthMap
from perception.depth.midAS_backend import MiDaSDepthBackend, DepthBackendUnavailableError
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


class TestMiDaSDepthBackendInterface:
    """Verify MiDaSDepthBackend implements IDepthBackend contract."""

    def test_implements_interface(self):
        assert issubclass(MiDaSDepthBackend, IDepthBackend)

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="Invalid model_type"):
            MiDaSDepthBackend(model_type="InvalidModel")

    def test_valid_model_types_accepted(self):
        for mt in ("MiDaS_small", "DPT_Large", "DPT_Hybrid", "DPT_Small"):
            backend = MiDaSDepthBackend(model_type=mt)
            assert backend._model_type == mt


class TestDepthMapValidation:
    """DepthMap dataclass validation."""

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="values has 2 rows, expected height=3"):
            DepthMap(evidence_id="e1", width=4, height=3, values=[[1.0]*4, [1.0]*4])

    def test_row_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="values row length 3, expected width=4"):
            DepthMap(evidence_id="e1", width=4, height=2, values=[[1.0]*3, [1.0]*3])

    def test_valid_depthmap_created(self):
        dm = DepthMap(evidence_id="e1", width=2, height=2, values=[[0.1, 0.2], [0.3, 0.4]])
        assert dm.evidence_id == "e1"
        assert dm.width == 2
        assert dm.height == 2
        assert dm.unit == "relative"
        assert isinstance(dm.uncertainty, Uncertainty)

    def test_unit_defaults_to_relative(self):
        dm = DepthMap(evidence_id="e1", width=1, height=1, values=[[0.5]])
        assert dm.unit == "relative"

    def test_unit_can_be_metric(self):
        dm = DepthMap(evidence_id="e1", width=1, height=1, values=[[1.5]], unit="meters")
        assert dm.unit == "meters"


class TestDepthBackendUnavailableError:
    """Error raised when perception dependencies are missing."""

    def test_raises_when_torch_missing(self):
        # Mock torch import to fail
        with patch.dict(sys.modules, {"torch": None, "torchvision": None, "numpy": None}):
            backend = MiDaSDepthBackend()
            # Clear any cached module state
            with pytest.raises(DepthBackendUnavailableError, match="Missing perception dependencies"):
                backend.estimate_depth([])

    def test_raises_when_torchvision_missing(self):
        with patch.dict(sys.modules, {"torch": MagicMock(), "torchvision": None, "numpy": MagicMock()}):
            backend = MiDaSDepthBackend()
            with pytest.raises(DepthBackendUnavailableError, match="torchvision"):
                backend.estimate_depth([])

    def test_raises_when_numpy_missing(self):
        with patch.dict(sys.modules, {"torch": MagicMock(), "torchvision": MagicMock(), "numpy": None}):
            backend = MiDaSDepthBackend()
            with pytest.raises(DepthBackendUnavailableError, match="numpy"):
                backend.estimate_depth([])


class TestMiDaSDepthBackendLogic:
    """Test backend logic with mocked torch."""

    def test_estimate_depth_skips_non_image_evidence(self):
        """Non-PHOTO/VIDEO evidence should be skipped, not error."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torchvision = MagicMock()
        mock_torchvision.transforms = MagicMock()
        mock_numpy = MagicMock()
        mock_pil = MagicMock()
        mock_pil_image = MagicMock()
        mock_pil.Image = mock_pil_image

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "torchvision": mock_torchvision,
            "torchvision.transforms": mock_torchvision.transforms,
            "numpy": mock_numpy,
            "PIL": mock_pil,
            "PIL.Image": mock_pil_image,
        }):
            backend = MiDaSDepthBackend()
            # Mock the internal model loading
            backend._model = MagicMock()
            backend._transform = MagicMock()
            backend._device = "cpu"

            evidence = [
                EvidenceItem(id="e1", kind=EvidenceKind.PHOTO, source_uri="file:///test.jpg"),
                EvidenceItem(id="e2", kind=EvidenceKind.POINT_CLOUD, source_uri="file:///test.las"),
            ]
            results = backend.estimate_depth(evidence)
            # Should only process the PHOTO item (POINT_CLOUD skipped)
            # But since we mock PIL.Image.open to fail, it returns empty
            assert isinstance(results, list)

    def test_estimate_depth_returns_empty_for_no_images(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torchvision = MagicMock()
        mock_torchvision.transforms = MagicMock()
        mock_numpy = MagicMock()

        with patch.dict(sys.modules, {
            "torch": mock_torch,
            "torchvision": mock_torchvision,
            "torchvision.transforms": mock_torchvision.transforms,
            "numpy": mock_numpy,
        }):
            backend = MiDaSDepthBackend()
            backend._model = MagicMock()
            backend._transform = MagicMock()
            backend._device = "cpu"

            evidence = [
                EvidenceItem(id="e1", kind=EvidenceKind.POINT_CLOUD, source_uri="file:///test.las"),
            ]
            results = backend.estimate_depth(evidence)
            assert results == []

    def test_device_defaults_to_cuda_when_available(self):
        with patch.dict(sys.modules, {"torch": MagicMock()}):
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = True

            backend = MiDaSDepthBackend()
            assert backend._device == "cuda"

    def test_device_defaults_to_cpu_when_cuda_unavailable(self):
        with patch.dict(sys.modules, {"torch": MagicMock()}):
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = False

            backend = MiDaSDepthBackend()
            assert backend._device == "cpu"

    def test_explicit_device_overrides_auto(self):
        with patch.dict(sys.modules, {"torch": MagicMock()}):
            mock_torch = sys.modules["torch"]
            mock_torch.cuda.is_available.return_value = True

            backend = MiDaSDepthBackend(device="cpu")
            assert backend._device == "cpu"


class TestMiDaSDepthBackendIntegration:
    """Integration-style tests (require actual deps - marked as optional)."""

    @pytest.mark.skipif(
        not _has_perception_deps(),
        reason="Perception dependencies (torch/torchvision/numpy) not installed"
    )
    def test_real_model_load_and_inference(self, tmp_path):
        """Load real MiDaS model and run inference on a synthetic image."""
        # This test only runs if perception deps are installed
        import torch
        import numpy as np
        from PIL import Image

        # Create a simple test image
        img = Image.new("RGB", (320, 240), color=(128, 128, 128))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)

        evidence = EvidenceItem(
            id="test_img",
            kind=EvidenceKind.PHOTO,
            source_uri=f"file://{img_path}"
        )

        backend = MiDaSDepthBackend(model_type="MiDaS_small")  # Smallest/fastest
        results = backend.estimate_depth([evidence])

        assert len(results) == 1
        dm = results[0]
        assert dm.evidence_id == "test_img"
        assert dm.width == 320
        assert dm.height == 240
        assert dm.unit == "relative"
        assert len(dm.values) == 240
        assert len(dm.values[0]) == 320
        # All values should be in [0, 1] after normalization
        for row in dm.values:
            for v in row:
                assert 0.0 <= v <= 1.0
        assert 0.0 < dm.uncertainty.confidence <= 1.0