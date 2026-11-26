"""Tests for Depth Anything V2 depth estimation backend.

Covers:
- Interface contract (IDepthBackend compliance)
- Output DepthMap validation (dimensions, values, range, unit)
- Non-image evidence skipping
- Device selection logic (CUDA/CPU)
- Real model inference on test fixture
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from PIL import Image
import numpy as np
import pytest

from evidence.session import EvidenceItem, EvidenceKind
from perception.depth.interface import DepthMap, IDepthBackend
from perception.depth.depth_anything_backend import DepthAnythingV2Backend


_HAVE_ML = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
requires_ml = pytest.mark.skipif(not _HAVE_ML, reason="torch and transformers are not installed (optional extras)")


class TestDepthAnythingV2BackendInterface:
    def test_implements_interface(self):
        assert issubclass(DepthAnythingV2Backend, IDepthBackend)

    def test_custom_device_passed(self):
        backend = DepthAnythingV2Backend(device="cpu")
        assert backend._custom_device == "cpu"


@requires_ml
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
        assert backend.skipped == []          # a different evidence kind is not a failure

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
        assert len(backend.skipped) == 1
        assert backend.skipped[0]["evidence_id"] == "missing_img"
        assert "does not exist" in backend.skipped[0]["reason"]

    def test_an_undecodable_file_is_recorded_not_swallowed(self, tmp_path):
        backend = DepthAnythingV2Backend()
        bad = tmp_path / "not_an_image.jpg"
        bad.write_bytes(b"not a jpeg")
        item = EvidenceItem(id="bad_img", kind=EvidenceKind.PHOTO, source_uri=str(bad))
        assert backend.estimate_depth([item]) == []
        assert len(backend.skipped) == 1
        assert "bad_img" == backend.skipped[0]["evidence_id"]
        assert backend.skipped[0]["reason"]


@requires_ml
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
        assert dm.quantity == "inverse_depth"    # larger = nearer, stated explicitly

        # No [0, 1] normalisation: the model's own inverse-depth scale is kept, which is what
        # makes two views comparable at all.
        val_arr = np.array(dm.values, dtype=np.float64)
        finite = np.isfinite(val_arr)
        assert finite.mean() > 0.99
        assert np.all(val_arr[finite] > 0.0)
        assert dm.valid_mask is not None
        assert np.array_equal(np.asarray(dm.valid_mask), finite)
        assert dm.invalid_pixels == int((~finite).sum())
        assert "inverse depth" in (dm.uncertainty.note or "")
        assert dm.uncertainty.confidence > 0.8

    def test_the_raw_model_scale_is_not_rescaled_per_image(self, sample_photo, monkeypatch):
        """The regression for the min-max normalisation: a known raw prediction must come
        back unchanged (a per-image [0, 1] normalisation would turn 2/4/8 into 0/0.33/1)."""
        backend = DepthAnythingV2Backend()
        raw = np.full((120, 160), 1.0, dtype=np.float32)
        raw[0, 0], raw[0, 1], raw[1, 0], raw[1, 1] = 2.0, 4.0, 8.0, -1.0
        monkeypatch.setattr(backend, "_ensure_model", lambda: None)
        monkeypatch.setattr(backend, "_predict", lambda image, w, h: raw)
        results = backend.estimate_depth([sample_photo])

        assert len(results) == 1
        dm = results[0]
        vals = np.asarray(dm.values, dtype=np.float64)
        assert (dm.width, dm.height) == (160, 120)
        assert vals[0, 0] == pytest.approx(2.0)
        assert vals[0, 1] == pytest.approx(4.0)
        assert vals[1, 0] == pytest.approx(8.0)     # > 1: proves no normalisation
        assert np.isnan(vals[1, 1])                 # a non-positive prediction is a hole
        assert not dm.valid_mask[1, 1] and dm.invalid_pixels == 1
        assert dm.quantity == "inverse_depth"

    def test_masked_arrays_marks_non_positive_and_non_finite_pixels(self):
        from perception.depth.depth_anything_backend import _masked_arrays

        values, valid = _masked_arrays(np.array([[0.5, 0.0], [np.inf, np.nan]], dtype=np.float32))
        assert valid.tolist() == [[True, False], [False, False]]
        assert np.isfinite(values[0, 0]) and np.isnan(values[0, 1])
        assert np.isnan(values[1, 0]) and np.isnan(values[1, 1])

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
