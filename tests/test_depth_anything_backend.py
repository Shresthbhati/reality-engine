"""Tests for Depth Anything V2 depth estimation backend.

Covers:
- Interface contract (IDepthBackend compliance)
- Output DepthMap validation (dimensions, values, range, unit)
- Non-image evidence skipping
- Device selection logic (CUDA/CPU)
- Real model inference on test fixture
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from evidence.session import EvidenceItem, EvidenceKind
from perception.depth.interface import DepthMap, IDepthBackend
from perception.depth.depth_anything_backend import DepthAnythingV2Backend


class TestDepthAnythingV2BackendInterface:
    def test_implements_interface(self):
        assert issubclass(DepthAnythingV2Backend, IDepthBackend)

    def test_custom_device_passed(self):
        backend = DepthAnythingV2Backend(device="cpu")
        assert backend._custom_device == "cpu"


class TestDepthAnythingV2BackendLogic:
    def test_skips_non_image_evidence(self):
        backend = DepthAnythingV2Backend()
        non_image = [
            EvidenceItem(
                id="pc1",
                kind=EvidenceKind.POINT_CLOUD,
                source_uri="file:///dummy.ply",
            )
        ]
        assert backend.estimate_depth(non_image) == []

    def test_skips_missing_files_honestly(self, tmp_path):
        backend = DepthAnythingV2Backend()
        missing = [
            EvidenceItem(
                id="missing_img",
                kind=EvidenceKind.PHOTO,
                source_uri=str(tmp_path / "does_not_exist.jpg"),
            )
        ]
        assert backend.estimate_depth(missing) == []


class TestDepthAnythingV2BackendIntegration:
    @pytest.fixture()
    def sample_photo(self, tmp_path):
        img_path = tmp_path / "test_scene.png"
        rng = np.random.default_rng(42)
        arr = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
        Image.fromarray(arr).save(img_path)
        return EvidenceItem(
            id="ev_001",
            kind=EvidenceKind.PHOTO,
            source_uri=str(img_path),
        )

    def test_real_model_depth_estimation(self, sample_photo):
        backend = DepthAnythingV2Backend()
        results = backend.estimate_depth([sample_photo])

        assert len(results) == 1
        dm = results[0]
        assert isinstance(dm, DepthMap)
        assert dm.evidence_id == "ev_001"
        assert dm.width == 160
        assert dm.height == 120
        assert len(dm.values) == 120
        assert len(dm.values[0]) == 160
        assert dm.unit == "relative"

        # Check normalization bounds
        val_arr = np.array(dm.values)
        assert val_arr.min() >= 0.0
        assert val_arr.max() <= 1.0 + 1e-5
        assert dm.uncertainty.note == "depth_anything_v2_small"
        assert dm.uncertainty.confidence > 0.8

    def test_pipeline_depth_stage_dispatches_depth_anything(self, sample_photo):
        from engine.pipeline.vertical_slice import VerticalSliceOptions, _depth_stage
        from reconstruction.backend.interface import ReconstructionResult, ReconstructedCameraPose

        options = VerticalSliceOptions(
            depth_model="depth_anything_v2_small",
            intrinsics=(100.0, 100.0, 80.0, 60.0),
            image_size=(160, 120),
        )
        fake_recon = ReconstructionResult(
            camera_poses=[
                ReconstructedCameraPose(
                    evidence_id="ev_001",
                    position=(0.0, 0.0, 0.0),
                    rotation=(1.0, 0.0, 0.0, 0.0),
                )
            ],
            points=[],
            registration_status="complete",
        )
        facts, metric_maps = _depth_stage(fake_recon, [sample_photo], options, scale_state="metric")
        assert facts["status"] == "ran"
        assert facts["model"] == "depth_anything_v2_small"
        assert facts["maps"] == 1

