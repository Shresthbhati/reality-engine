"""Segmentation adapter test: SAM 2 masks must flow into ObservationRecord.mask,
not merely mask_area_px metadata. The transformers module is replaced with a fake
(via sys.modules, since transformers 5.16.1 blocks attribute monkeypatching) so no
SAM 2 weights are downloaded or required.

This test targets the current adapter contract: post_process_masks returns a
list of per-image mask stacks (H, W) or (N, H, W), and iou_scores selects the
best channel.
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from perception.segmentation_adapter import _sam2_run, _select_best_2d_mask


class _FakeTensor:
    """Stand-in for a torch tensor: detach().cpu().numpy() -> ndarray."""

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


class _FakeOutputs:
    def __init__(self, pred_masks):
        self.pred_masks = pred_masks
        self.iou_scores = None


class _FakeModel:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls()

    def __call__(self, **inputs):
        return _FakeOutputs(None)

    def to(self, device):
        return self

    def eval(self):
        return self


class _FakeProcessor:
    def __init__(self, masks):
        self._masks = masks  # list of bool masks for the single image

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls([])

    def __call__(self, images, input_boxes, return_tensors):
        class _Inputs(dict):
            def to(self, device):
                return self

        w, h = images.size
        return _Inputs({"original_sizes": _FakeTensor([[h, w]])})

    def post_process_masks(self, pred_masks, original_sizes):
        # One image -> one list of mask tensors. Each is a 2-D (H, W) bool mask.
        return [[_FakeTensor(m) for m in self._masks]]


class _FakeTransformersModule:
    Sam2Model = _FakeModel
    Sam2Processor = _FakeProcessor


def _install_fake_transformers(monkeypatch, masks):
    fake = _FakeTransformersModule()
    monkeypatch.setitem(sys.modules, "transformers", fake)
    fake.Sam2Processor.from_pretrained = classmethod(
        lambda cls, name, **kwargs: _FakeProcessor(masks))
    return fake


def _make_png(tmp_path):
    from PIL import Image as PILImage

    img = tmp_path / "img.png"
    PILImage.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img)
    return str(img)


def test_select_best_2d_mask_selects_channel_by_iou():
    masks = np.stack([np.zeros((4, 4)), np.ones((4, 4))]).astype(np.float32)
    iou = _FakeTensor([[0.2, 0.9]])
    out = _select_best_2d_mask(_FakeTensor(masks), iou_scores=iou, box_idx=0)
    assert out.shape == (4, 4) and out.dtype == bool
    assert out.all()  # channel 1 had the higher IoU


def test_sam2_adapter_puts_mask_into_observation_record(tmp_path, monkeypatch):
    path = _make_png(tmp_path)
    masks = [
        np.zeros((100, 100), bool),
        np.ones((100, 100), bool),
    ]
    _install_fake_transformers(monkeypatch, masks)

    out = _sam2_run([path], run_id="run1", bboxes={path: [(0, 0, 4, 4), (10, 10, 20, 20)]})

    assert out.status == "succeeded", out.error_message
    assert len(out.observations) == 2
    for obs in out.observations:
        assert obs.category == "segmentation"
        assert obs.model_id.architecture_id == "sam2"
        assert obs.mask is not None, "mask must be present on the observation record"
        assert obs.mask.shape == (100, 100)
        assert obs.mask.dtype == bool
        assert obs.metadata["mask_area_px"] == int(obs.mask.sum())
    assert not out.observations[0].mask.any()
    assert out.observations[1].mask.all()


def test_sam2_adapter_skips_image_without_boxes(tmp_path, monkeypatch):
    path = _make_png(tmp_path)
    _install_fake_transformers(monkeypatch, [])

    out = _sam2_run([path], run_id="run1")

    # No bboxes -> SAM2 needs prompts -> skips every image honestly.
    assert len(out.observations) == 0
    assert out.summary["n_masks"] == 0


def test_sam2_adapter_model_load_failure_reported_honestly(tmp_path, monkeypatch):
    path = _make_png(tmp_path)

    class _BadModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            raise RuntimeError("weights not found")

    fake = _FakeTransformersModule()
    fake.Sam2Model = _BadModel
    monkeypatch.setitem(sys.modules, "transformers", fake)

    out = _sam2_run([path], run_id="run1", bboxes={path: [(0, 0, 4, 4)]})

    assert out.status == "failed"
    assert "weights not found" in out.error_message
    assert len(out.observations) == 0
