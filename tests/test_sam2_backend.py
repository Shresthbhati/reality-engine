"""Unit and benchmark comparison tests for SAM 2 segmentation backend (Task 13.6).

Verifies:
1. Interface conformance (ISegmentationBackend)
2. Model configurations and aliases (Hiera-T/S/B+/L)
3. Unavailability error handling and diagnostics
4. Provenance tracking (model_name="sam2")
5. VRAM release via unload()
6. Comparative benchmark metrics vs SAM 1 (parameters, VRAM footprint, speedup)
"""

from __future__ import annotations

import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from evidence.session import EvidenceItem, EvidenceKind
from perception.segmentation import (
    ISegmentationBackend,
    SAM2SegmentationBackend,
    SegmentationBackendUnavailableError,
    SegmentationResult,
)


class TestSAM2SegmentationBackendInterface:
    """Verify SAM2SegmentationBackend satisfies ISegmentationBackend contract."""

    def test_implements_interface(self):
        assert issubclass(SAM2SegmentationBackend, ISegmentationBackend)

    def test_rejects_invalid_model_type(self):
        with pytest.raises(ValueError, match="Invalid model_type"):
            SAM2SegmentationBackend(model_type="nonexistent_model")

    @pytest.mark.parametrize(
        "model_type,expected_canonical",
        [
            ("sam2_hiera_tiny", "sam2_hiera_tiny"),
            ("hiera_t", "sam2_hiera_tiny"),
            ("tiny", "sam2_hiera_tiny"),
            ("sam2_hiera_small", "sam2_hiera_small"),
            ("hiera_s", "sam2_hiera_small"),
            ("sam2_hiera_base_plus", "sam2_hiera_base_plus"),
            ("hiera_b+", "sam2_hiera_base_plus"),
            ("sam2_hiera_large", "sam2_hiera_large"),
            ("hiera_l", "sam2_hiera_large"),
        ],
    )
    def test_accepts_model_types_and_aliases(self, model_type, expected_canonical):
        backend = SAM2SegmentationBackend(model_type=model_type, device="cpu")
        assert backend.model_type == expected_canonical

    def test_device_selection(self):
        backend = SAM2SegmentationBackend(device="cpu")
        assert backend.device == "cpu"

    def test_unload_clears_models(self):
        backend = SAM2SegmentationBackend(device="cpu")
        backend._model = object()
        backend._mask_generator = object()
        backend.unload()
        assert backend._model is None
        assert backend._mask_generator is None


class TestSAM2SegmentationBackendLogic:
    """Test error handling, diagnostics, and item filtering."""

    def test_empty_evidence_returns_empty(self):
        backend = SAM2SegmentationBackend(device="cpu")
        assert backend.segment([]) == []
        assert backend.diagnostics == []

    def test_non_image_evidence_ignored(self):
        backend = SAM2SegmentationBackend(device="cpu")
        non_image = [
            EvidenceItem(id="e1", kind=EvidenceKind.POINT_CLOUD, source_uri="file:///test.las"),
            EvidenceItem(id="e2", kind=EvidenceKind.DEPTH, source_uri="file:///test.exr"),
        ]
        assert backend.segment(non_image) == []
        assert backend.diagnostics == []

    def test_missing_evidence_file_skips_with_diagnostic(self, tmp_path):
        backend = SAM2SegmentationBackend(device="cpu")
        missing_item = EvidenceItem(
            id="photo_miss",
            kind=EvidenceKind.PHOTO,
            source_uri=f"file://{tmp_path / 'missing.jpg'}",
        )
        with patch("perception.segmentation.sam2_backend._ensure_sam2_deps"), patch.object(backend, "_load_model"):
            res = backend.segment([missing_item])
            assert res == []
            assert len(backend.diagnostics) == 1
            assert backend.diagnostics[0]["status"] == "skipped"
            assert "source file not found" in backend.diagnostics[0]["reason"]

    def test_missing_checkpoint_raises_unavailable(self, tmp_path):
        backend = SAM2SegmentationBackend(
            model_type="sam2_hiera_tiny",
            checkpoint_path=str(tmp_path / "nonexistent.pt"),
            device="cpu",
        )
        with pytest.raises(SegmentationBackendUnavailableError, match="checkpoint not found"):
            backend._resolve_checkpoint()

    def test_cached_checkpoint_resolution(self, tmp_path, monkeypatch):
        cached_file = tmp_path / "sam2_hiera_tiny.pt"
        cached_file.write_bytes(b"mock_weights")

        import torch
        monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path.parent))
        checkpoints_dir = tmp_path.parent / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        target_file = checkpoints_dir / "sam2_hiera_tiny.pt"
        target_file.write_bytes(b"mock_weights")

        backend = SAM2SegmentationBackend(model_type="sam2_hiera_tiny", device="cpu")
        assert backend._resolve_checkpoint() == str(target_file)


class TestSAM2SegmentationBackendExecution:
    """Test segmentation execution and result formatting with mocked model."""

    def test_segment_produces_valid_segmentation_result(self, tmp_path):
        from PIL import Image

        img_path = tmp_path / "room.jpg"
        img = Image.new("RGB", (64, 64), color="blue")
        img.save(img_path)

        item = EvidenceItem(
            id="photo_1",
            kind=EvidenceKind.PHOTO,
            source_uri=f"file://{img_path}",
        )

        backend = SAM2SegmentationBackend(model_type="sam2_hiera_tiny", device="cpu")

        mock_mask = [[True if (i < 32 and j < 32) else False for j in range(64)] for i in range(64)]
        mock_generator = MagicMock()
        mock_generator.generate.return_value = [
            {
                "segmentation": mock_mask,
                "area": 1024,
                "bbox": [0, 0, 32, 32],
                "predicted_iou": 0.94,
                "stability_score": 0.96,
            }
        ]

        with patch("perception.segmentation.sam2_backend._ensure_sam2_deps"), patch.object(backend, "_load_model"):
            backend._mask_generator = mock_generator
            results = backend.segment([item])

            assert len(results) == 1
            res = results[0]
            assert isinstance(res, SegmentationResult)
            assert res.evidence_id == "photo_1"
            assert res.model_name == "sam2"
            assert res.model_version == "sam2_hiera_tiny"
            assert len(res.regions) == 1

            region = res.regions[0]
            assert region.region_id == "sam2_r_0000"
            assert region.confidence == pytest.approx(0.95, abs=0.01)
            assert region.mask[0][0] is True
            assert region.mask[40][40] is False
            assert "sam2 sam2_hiera_tiny mask" in region.uncertainty.note

            assert len(backend.diagnostics) == 1
            assert backend.diagnostics[0]["status"] == "ok"
            assert backend.diagnostics[0]["regions_count"] == 1


class TestSAM2VsSAM1BenchmarkComparison:
    """Benchmark evaluation comparing SAM 2 architectures against SAM 1."""

    @pytest.mark.parametrize(
        "sam2_variant,min_speedup,max_vram",
        [
            ("sam2_hiera_tiny", 3.0, 500.0),
            ("sam2_hiera_small", 2.5, 700.0),
            ("sam2_hiera_base_plus", 2.0, 1000.0),
            ("sam2_hiera_large", 1.5, 1800.0),
        ],
    )
    def test_sam2_architectural_improvements_over_sam1(self, sam2_variant, min_speedup, max_vram):
        report = SAM2SegmentationBackend.compare_against_sam1(sam2_variant)

        assert report["estimated_speedup"] >= min_speedup
        assert report["sam2_vram_mb"] <= max_vram
        assert report["sam2_parameters_m"] < 636.0
        assert report["sam2_vram_mb"] < 2500.0

        if "tiny" in sam2_variant or "small" in sam2_variant:
            assert report["sam2_parameters_m"] < 91.0
            assert report["vram_reduction_percent"] > 30.0

        assert report["multiscale_hierarchical_features"] is True
