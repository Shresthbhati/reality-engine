"""A5 Tests (1, 2, 3, 12): SAM execution, mask coordinate correctness, empty-mask diagnosis, and missing output safety."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    CoordinateSystemSpec,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    ObservationLifecycle,
    ScoreSemantics,
)
from perception.intelligence.mask_geometry import (
    compute_tight_box_from_mask,
    decode_rle_to_mask,
    extract_mask_polygon_px,
    refine_boxes_from_verified_masks,
)
from perception.intelligence.segmentation_v2 import (
    attach_masks_to_observations,
    build_segmentation_observation_from_mask,
    probe_available_segmenters,
)
from perception.segmentation_adapter import (
    _encode_rle_binary,
    _resolve_image_bboxes,
    _sam2_run,
    _select_best_2d_mask,
)


def _obs(
    obs_id: str,
    image_id: str,
    label: str,
    bbox: tuple[float, float, float, float],
    *,
    category: str = "furniture",
    width: int = 320,
    height: int = 240,
) -> NormalizedPerceptionObservation:
    return NormalizedPerceptionObservation(
        observation_id=obs_id,
        schema_version=SCHEMA_VERSION,
        raw_label=label,
        canonical_label=label,
        taxonomy_id=f"re:{label}",
        category=category,
        bbox_px=bbox,
        coords=CoordinateSystemSpec("image_px", "pixels", width, height),
        scores=ScoreSemantics(raw_score=0.85, score_emitted=True),
        provenance=DetectionProvenance(
            source_image_id=image_id,
            execution=ModelExecutionProvenance("grounding_dino", "grounding_dino", "test_ckpt"),
        ),
    )


def test_1_probe_and_live_sam2_execution_on_real_photo() -> None:
    probes = probe_available_segmenters()
    assert "sam2" in probes
    assert "sam3" in probes
    assert "sam_hq" in probes

    img_path = Path("datasets/south_building/images/P1180141.JPG")
    if not (probes["sam2"]["available"] and img_path.exists()):
        pytest.skip("SAM2 or south_building image unavailable")

    out = _sam2_run(
        [str(img_path)],
        bboxes={img_path.name: [(120.0, 120.0, 420.0, 420.0)]},
        run_id="test_sam2_live",
    )
    assert out.status == "succeeded"
    assert len(out.observations) == 1
    rec = out.observations[0]
    assert rec.metadata["image_width"] == 1024
    assert rec.metadata["image_height"] == 768
    assert rec.metadata["mask_area_px"] > 0
    assert rec.metadata["is_empty"] is False
    assert rec.metadata["rle_mask"]["size"] == [768, 1024]


def test_2_mask_coordinate_correctness_rle_roundtrip_and_dimension_guard() -> None:
    mask = np.zeros((240, 320), dtype=bool)
    mask[50:150, 80:220] = True
    rle = _encode_rle_binary(mask)
    decoded = decode_rle_to_mask(rle)
    assert decoded is not None
    assert decoded.shape == (240, 320)
    assert np.array_equal(decoded, mask)

    # Dimension mismatch must be rejected, never silently resized or fabricated
    obs = _obs("o_dim", "img_1", "chair", (80, 50, 220, 150), width=320, height=240)
    wrong_mask = np.ones((120, 160), dtype=bool)
    seg, err = build_segmentation_observation_from_mask(obs, wrong_mask, model_id="sam2")
    assert seg is None
    assert err is not None and "dimension_mismatch" in err

    # Tight box & polygon extraction
    loose_obs = _obs("o_loose", "img_1", "chair", (40, 20, 280, 210), width=320, height=240)
    updated, rep = attach_masks_to_observations(
        [loose_obs],
        precomputed_masks={"o_loose": mask},
    )
    assert rep.status == "succeeded"
    assert rep.masks_produced == 1
    refined, summaries = refine_boxes_from_verified_masks(updated)
    assert len(summaries) == 1
    assert summaries[0].is_valid is True
    assert summaries[0].polygon_vertex_count >= 4
    tight = compute_tight_box_from_mask(mask)
    assert tight == (80.0, 50.0, 220.0, 150.0)
    assert refined[0].bbox_px == tight
    assert refined[0].revision.lifecycle is ObservationLifecycle.REVISED


def test_3_empty_mask_detection_never_fabricates_mask() -> None:
    obs = _obs("o_empty", "img_1", "chair", (50, 50, 150, 180), width=320, height=240)
    empty_mask = np.zeros((240, 320), dtype=bool)
    updated, rep = attach_masks_to_observations(
        [obs],
        precomputed_masks={"o_empty": empty_mask},
    )
    assert rep.empty_masks == 1
    assert rep.masks_produced == 0
    assert updated[0].segmentation is None
    assert any("empty_mask" in d for d in rep.diagnostics)


def test_12_missing_engine_output_and_multimask_channel_selection() -> None:
    # Missing image source leaves segmentation=None (never fabricates)
    obs = _obs("o_miss", "nonexistent_img.jpg", "table", (20, 40, 200, 180))
    updated, rep = attach_masks_to_observations(
        [obs],
        images_rgb={},
        image_paths={},
        preferred_backend="sam2",
    )
    assert updated[0].segmentation is None
    assert rep.masks_produced == 0
    assert rep.status in ("failed", "unavailable")

    # Multimask channel selection picks highest IoU score channel
    cand = np.zeros((3, 10, 10), dtype=bool)
    cand[0, 0:2, 0:2] = True
    cand[1, 2:8, 2:8] = True
    cand[2, 0:1, 0:1] = True
    iou_scores = np.array([[0.45, 0.92, 0.30]], dtype=np.float32)
    best = _select_best_2d_mask(cand, iou_scores=iou_scores, box_idx=0)
    assert int(best.sum()) == 36

    # Basename resolution works when full path key is absent
    resolved = _resolve_image_bboxes({"P1180141.JPG": [(1, 2, 3, 4)]}, "datasets/south_building/images/P1180141.JPG")
    assert resolved == [(1.0, 2.0, 3.0, 4.0)]

    # Empty polygon check
    assert extract_mask_polygon_px(np.zeros((20, 20), dtype=bool)) == ()


def test_1b_mask_path_runs_with_torch_and_weights_stubbed(monkeypatch, tmp_path) -> None:
    """The SAM2 mask path is provable where torch and the checkpoint are absent.

    ``test_1`` needs the real optional extra plus cached weights, so it skips on a
    CI image. This variant stubs torch and transformers and therefore never skips;
    it also pins that this module stays collectable without torch, which the
    adapter's former module-level ``import torch`` broke.
    """
    import sys
    import types

    from PIL import Image as PILImage

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    class _Cuda:
        def is_available(self):
            return False

        def reset_peak_memory_stats(self, index=0):
            pass

        def max_memory_allocated(self, index=0):
            return 0

        def empty_cache(self):
            pass

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = _Cuda()
    fake_torch.no_grad = lambda: _NoGrad()

    class _Tensor:
        def __init__(self, arr):
            self._arr = np.asarray(arr)

        def detach(self):
            return self

        def to(self, device):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    class _Inputs(dict):
        def to(self, device):
            return self

    class _Processor:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return cls()

        def __call__(self, images, input_boxes, return_tensors):
            w, h = images.size
            return _Inputs({"original_sizes": _Tensor([[h, w]])})

        def post_process_masks(self, pred_masks, original_sizes):
            return [[_Tensor(np.ones((240, 320), dtype=bool))]]

    class _Model:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            return cls()

        def __call__(self, **inputs):
            return types.SimpleNamespace(pred_masks=None, iou_scores=None)

        def to(self, device):
            return self

        def eval(self):
            return self

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Sam2Model = _Model
    fake_transformers.Sam2Processor = _Processor
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    img = tmp_path / "stub.png"
    PILImage.fromarray(np.zeros((240, 320, 3), dtype=np.uint8)).save(img)

    out = _sam2_run(
        [str(img)],
        bboxes={img.name: [(10.0, 10.0, 100.0, 100.0)]},  # basename prompt resolution
        run_id="test_sam2_stubbed",
    )

    assert out.status == "succeeded", out.error_message
    assert len(out.observations) == 1
    rec = out.observations[0]
    assert rec.mask is not None
    assert rec.mask.shape == (240, 320)
    assert rec.mask.dtype == bool and rec.mask.all()
    assert rec.metadata["rle_mask"]["size"] == [240, 320]
    assert rec.metadata["mask_area_px"] == 240 * 320
    assert rec.model_id.architecture_id == "sam2"
    assert rec.model_id.hardware_class == "cpu"
    assert rec.coordinate_frame.scale_state == "image"
