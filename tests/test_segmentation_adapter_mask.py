"""Segmentation adapter tests: SAM 2 masks must flow into ObservationRecord.mask,
not merely mask_area_px metadata.

``torch`` and ``transformers`` are optional extras (the CI image has neither), so
both are stubbed through ``sys.modules`` — transformers 5.16.1 blocks attribute
monkeypatching — and these tests run rather than skip. The adapter imports torch
inside the functions that need it, which is what makes this possible;
``test_adapter_imports_without_torch`` pins that contract in a real subprocess and
is the regression test for the module-level ``import torch`` that used to make
this module, and every test importing it, uncollectable in CI.

The adapter contract under test: ``post_process_masks`` returns a list of
per-image mask stacks (H, W) or (N, H, W), and iou_scores selects the best
channel.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

from perception.mask_persistence import load_mask_store, persist_masks, save_mask_store
from perception.segmentation_adapter import (
    _SAM2_MODEL,
    _sam2_available,
    _sam2_run,
    _select_best_2d_mask,
)
from world_ir.mask_store import MaskStore

_REPO_ROOT = Path(__file__).resolve().parents[1]


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


class _FakeCuda:
    """CUDA facade that records peak-statistics management."""

    def __init__(self, available: bool, run_peak_bytes: int):
        self._available = available
        self._run_peak_bytes = run_peak_bytes
        self.reset_calls = 0
        self.peak_bytes = 0

    def is_available(self) -> bool:
        return self._available

    def reset_peak_memory_stats(self, index: int = 0) -> None:
        self.reset_calls += 1
        self.peak_bytes = self._run_peak_bytes

    def max_memory_allocated(self, index: int = 0) -> int:
        return self.peak_bytes

    def empty_cache(self) -> None:
        pass


class _FakeTorch(types.ModuleType):
    """Minimal torch module exposing exactly what the adapter calls."""

    def __init__(self, *, cuda: bool, run_peak_bytes: int = 0):
        super().__init__("torch")
        self.cuda = _FakeCuda(cuda, run_peak_bytes)

    def no_grad(self):
        class _NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        return _NoGrad()


def _install_fake_torch(
    monkeypatch, *, cuda: bool = False, run_peak_bytes: int = 0
) -> _FakeTorch:
    fake = _FakeTorch(cuda=cuda, run_peak_bytes=run_peak_bytes)
    monkeypatch.setitem(sys.modules, "torch", fake)
    return fake


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
        lambda cls, name, **kwargs: _FakeProcessor(masks)
    )
    return fake


def _make_png(tmp_path, name: str = "img.png"):
    from PIL import Image as PILImage

    img = tmp_path / name
    PILImage.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img)
    return str(img)


def test_adapter_imports_without_torch() -> None:
    """Importing the adapter must not require the optional torch extra.

    Runs in a subprocess so a partially imported module cannot mask the failure,
    and pins the precise-dependency report at the same time.
    """
    script = textwrap.dedent(
        """
        import sys
        sys.modules["torch"] = None          # any `import torch` now raises ImportError
        sys.modules["transformers"] = None
        import perception.segmentation_adapter as adapter
        print("imported:", adapter.__name__)
        print("reason:", adapter._sam2_available())
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "imported: perception.segmentation_adapter" in proc.stdout
    assert "torch" in proc.stdout  # the reason names the actual missing package


def test_sam2_available_names_the_missing_dependency(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    reason = _sam2_available()
    assert reason is not None
    assert "torch" in reason


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
    _install_fake_torch(monkeypatch)  # no CUDA -> CPU execution path
    _install_fake_transformers(monkeypatch, masks)

    out = _sam2_run([path], run_id="run1", bboxes={path: [(0, 0, 4, 4), (10, 10, 20, 20)]})

    assert out.status == "succeeded", out.error_message
    assert len(out.observations) == 2
    for obs in out.observations:
        assert obs.category == "segmentation"
        assert obs.source_image_id == path
        assert obs.model_id.architecture_id == "sam2"
        assert obs.model_id.checkpoint_id == _SAM2_MODEL
        assert obs.model_id.hardware_class == "cpu"  # honest device provenance
        assert obs.coordinate_frame.scale_state == "image"
        assert obs.coordinate_frame.units == "pixels"
        assert obs.mask is not None, "mask must be present on the observation record"
        assert obs.mask.shape == (100, 100)
        assert obs.mask.dtype == bool
        assert obs.metadata["mask_area_px"] == int(obs.mask.sum())
        assert obs.mask_ref is None  # stamped later by persist_masks
    assert not out.observations[0].mask.any()
    assert out.observations[1].mask.all()
    assert out.peak_vram_mb is None  # CPU run reports no VRAM figure


def test_sam2_adapter_skips_image_without_boxes(tmp_path, monkeypatch):
    path = _make_png(tmp_path)
    _install_fake_torch(monkeypatch)
    _install_fake_transformers(monkeypatch, [])

    out = _sam2_run([path], run_id="run1")

    # No bboxes -> SAM2 needs prompts -> skips every image honestly.
    assert len(out.observations) == 0
    assert out.summary["n_masks"] == 0


def test_missing_checkpoint_is_unavailable_and_never_downloads(tmp_path, monkeypatch):
    """A cold cache must be reported, never patched over with a network download."""
    path = _make_png(tmp_path)
    attempted: list[dict] = []

    class _UncachedModel:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            attempted.append(dict(kwargs))
            raise FileNotFoundError(f"{name} is not in the local cache")

    fake = _FakeTransformersModule()
    fake.Sam2Model = _UncachedModel
    monkeypatch.setitem(sys.modules, "transformers", fake)
    _install_fake_torch(monkeypatch)

    out = _sam2_run([path], run_id="run1", bboxes={path: [(0, 0, 4, 4)]})

    assert out.status == "unavailable", out.error_message
    assert out.observations == ()
    assert _SAM2_MODEL in out.error_message
    assert "huggingface-cli download" in out.error_message  # explicit download path
    # Every load attempt was local-only: no second, network-enabled retry.
    assert attempted, "the adapter must still try the local cache"
    assert all(call.get("local_files_only") is True for call in attempted), attempted


def test_model_loaded_once_for_many_images(tmp_path, monkeypatch):
    paths = [_make_png(tmp_path, f"{name}.png") for name in ("a", "b", "c")]
    calls = {"model": 0, "processor": 0}
    masks = [np.ones((100, 100), bool)]

    class _CountingModel(_FakeModel):
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["model"] += 1
            return cls()

    class _CountingProcessor(_FakeProcessor):
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            calls["processor"] += 1
            return cls(masks)

    fake = _FakeTransformersModule()
    fake.Sam2Model = _CountingModel
    fake.Sam2Processor = _CountingProcessor
    monkeypatch.setitem(sys.modules, "transformers", fake)
    _install_fake_torch(monkeypatch)

    out = _sam2_run(paths, run_id="many", bboxes={p: [(0, 0, 4, 4)] for p in paths})

    assert out.status == "succeeded", out.error_message
    assert calls == {"model": 1, "processor": 1}  # loaded once, not once per image
    assert [o.source_image_id for o in out.observations] == paths  # image identity


def test_peak_vram_is_reset_per_run(tmp_path, monkeypatch):
    """peak_vram_mb must describe this run, not the process high-water mark."""
    path = _make_png(tmp_path)
    _install_fake_transformers(monkeypatch, [np.ones((100, 100), bool)])
    fake_torch = _install_fake_torch(
        monkeypatch, cuda=True, run_peak_bytes=300 * 1024 * 1024
    )

    out = _sam2_run([path], run_id="vram", bboxes={path: [(0, 0, 4, 4)]})

    assert fake_torch.cuda.reset_calls == 1  # cleared before the model is loaded
    assert out.peak_vram_mb == pytest.approx(300.0)
    assert out.observations[0].model_id.hardware_class == "cuda"

    # A CPU run on a CUDA-capable host must not report a VRAM figure at all.
    cpu_run = _sam2_run(
        [path], run_id="vram-cpu", bboxes={path: [(0, 0, 4, 4)]}, device="cpu"
    )
    assert cpu_run.status == "succeeded", cpu_run.error_message
    assert cpu_run.peak_vram_mb is None
    assert cpu_run.observations[0].model_id.hardware_class == "cpu"
    assert fake_torch.cuda.reset_calls == 1  # no reset for a non-CUDA run


def test_wrong_sized_mask_is_reported_not_fabricated(tmp_path, monkeypatch):
    path = _make_png(tmp_path)
    _install_fake_torch(monkeypatch)
    _install_fake_transformers(monkeypatch, [np.ones((50, 50), bool)])

    out = _sam2_run([path], run_id="bad", bboxes={path: [(0, 0, 4, 4)]})

    assert out.status == "failed"
    assert out.observations == ()  # never a resized or invented mask
    assert "shape" in out.error_message
    assert out.summary["errors"] == 1


def test_adapter_masks_persist_to_mask_store_and_reload(tmp_path, monkeypatch):
    """SAM2 masks must survive the production persistence leg, not just sit in memory."""
    path = _make_png(tmp_path)
    masks = [np.zeros((100, 100), bool), np.ones((100, 100), bool)]
    _install_fake_torch(monkeypatch)
    _install_fake_transformers(monkeypatch, masks)

    out = _sam2_run(
        [path], run_id="persist", bboxes={path: [(0, 0, 4, 4), (10, 10, 20, 20)]}
    )
    assert out.status == "succeeded", out.error_message

    store = MaskStore()
    stamped = persist_masks(list(out.observations), store)
    assert [rec.mask_ref for rec in stamped] == [0, 1]

    npz = tmp_path / "masks.npz"
    save_mask_store(store, npz)
    reloaded = load_mask_store(npz)
    for rec in stamped:
        assert np.array_equal(reloaded.get(rec.mask_ref), rec.mask)
