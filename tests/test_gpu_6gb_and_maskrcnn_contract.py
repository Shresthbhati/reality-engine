"""Tests for 6 GB GPU VRAM accounting, LightGlue rescaling, offline depth loading,
and Mask R-CNN ObservationRecord/MaskStore contract (PR-A).
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from perception.engine_adapters import _lightglue_run
from perception.engine_evidence import (
    CoordinateFrame,
    ModelIdentity,
    ObservationRecord,
    UnifiedEngineOutput,
)
from perception.engine_orchestrator import EngineSpec, MultiEngineOrchestrator, merge_engine_outputs
from perception.mask_persistence import persist_masks
from world_ir.mask_store import MaskStore


def _make_obs(engine_id: str, label: str = "chair") -> ObservationRecord:
    return ObservationRecord(
        evidence_id=f"{engine_id}:img0:0",
        source_image_id="img0.jpg",
        coordinate_frame=CoordinateFrame(frame_id="image", scale_state="image", units="pixels"),
        model_id=ModelIdentity(
            engine_id=engine_id,
            architecture_id=engine_id,
            checkpoint_id="test",
            license="MIT",
            hardware_class="cpu",
        ),
        label=label,
        category="object",
        confidence=0.9,
        calibration_quality="unverified",
    )


def test_merge_engine_outputs_vram_peak_takes_max_across_waves() -> None:
    """MultiEngineReport.vram_peak_mb must be max across all engines, not last wave."""
    out_heavy = UnifiedEngineOutput(
        engine_id="sam2",
        run_id="r1",
        observations=(_make_obs("segmentation"),),
        status="succeeded",
        wall_time_s=1.0,
        peak_vram_mb=5432.5,
    )
    out_light = UnifiedEngineOutput(
        engine_id="depth_anything",
        run_id="r1",
        observations=(_make_obs("depth"),),
        status="succeeded",
        wall_time_s=0.5,
        peak_vram_mb=89.625,
    )
    report = merge_engine_outputs([out_heavy, out_light])
    assert report.vram_peak_mb == 5432.5


def test_orchestrator_run_all_reports_max_vram_across_engines(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    arr = np.linspace(0, 255, 64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    Image.fromarray(arr).save(img)

    orch = MultiEngineOrchestrator()
    fake_specs = {
        "heavy": EngineSpec("heavy", "heavy", "v1", dependencies=()),
        "light": EngineSpec("light", "light", "v1", dependencies=("heavy",)),
    }
    fake_fns = {
        "heavy": lambda paths, **ctx: UnifiedEngineOutput(
            engine_id="heavy", run_id="r", observations=(_make_obs("heavy"),),
            status="succeeded", wall_time_s=0.1, peak_vram_mb=4200.0,
        ),
        "light": lambda paths, **ctx: UnifiedEngineOutput(
            engine_id="light", run_id="r", observations=(_make_obs("light"),),
            status="succeeded", wall_time_s=0.1, peak_vram_mb=95.0,
        ),
    }
    vram_seq = iter([4200.0, 4200.0, 95.0, 95.0, 95.0])
    with (
        patch.dict("perception.engine_orchestrator._ENGINES", fake_specs, clear=True),
        patch.dict("perception.engine_orchestrator._ENGINE_FNS", fake_fns, clear=True),
        patch("perception.engine_orchestrator._get_vram_peak_mb", side_effect=lambda: next(vram_seq, 95.0)),
    ):
        report = orch.run_all([str(img)], context={"run_id": "r_max"})
    assert report.vram_peak_mb == 4200.0


def test_lightglue_downscales_and_rescales_keypoints(tmp_path: Path) -> None:
    """Large images (> max_long_edge_px) are downscaled for LightGlue and keypoints rescaled."""
    p1 = tmp_path / "img1.jpg"
    p2 = tmp_path / "img2.jpg"
    Image.new("RGB", (4000, 3000), color=(200, 100, 50)).save(p1)
    Image.new("RGB", (4000, 3000), color=(200, 100, 50)).save(p2)

    captured_shapes: list[tuple[int, int]] = []

    class FakeTensor:
        def __init__(self, arr: np.ndarray) -> None:
            self.arr = arr

        @property
        def shape(self):
            return self.arr.shape

        @property
        def ndim(self):
            return self.arr.ndim

        def to(self, _dev):
            return self

        def permute(self, *axes):
            return FakeTensor(np.transpose(self.arr, axes))

        def float(self):
            return self

        def long(self):
            return FakeTensor(self.arr.astype(np.int64))

        def div(self, x):
            return FakeTensor(self.arr / x)

        def __truediv__(self, x):
            return FakeTensor(self.arr / x)

        def contiguous(self):
            return self

        def unsqueeze(self, _dim):
            return FakeTensor(np.expand_dims(self.arr, axis=0))

        def squeeze(self, _dim=0):
            return FakeTensor(np.squeeze(self.arr, axis=0))

        def cpu(self):
            return self

        def detach(self):
            return self

        def reshape(self, *shape):
            return FakeTensor(self.arr.reshape(*shape))

        def numpy(self):
            return self.arr

        def __getitem__(self, key):
            return FakeTensor(self.arr[key])

    class FakeExtractor:
        def eval(self):
            return self

        def to(self, _dev):
            return self

        def permute(self, *axes):
            return FakeTensor(np.transpose(self.arr, axes))

        def float(self):
            return self

        def long(self):
            return FakeTensor(self.arr.astype(np.int64))

        def div(self, x):
            return FakeTensor(self.arr / x)

        def __truediv__(self, x):
            return FakeTensor(self.arr / x)

        def contiguous(self):
            return self

        def __call__(self, tensor):
            # tensor shape is (1, C, H, W)
            captured_shapes.append((tensor.shape[-2], tensor.shape[-1]))
            pts = np.array([
                [100.0, 50.0], [200.0, 100.0], [400.0, 150.0], [600.0, 300.0],
                [800.0, 500.0], [1000.0, 200.0], [1200.0, 800.0], [1400.0, 900.0],
            ], dtype=np.float32)
            kpts = FakeTensor(pts)
            desc = FakeTensor(np.ones((8, 128), dtype=np.float32))
            kscores = FakeTensor(np.ones(8, dtype=np.float32) * 0.9)
            return [types.SimpleNamespace(keypoints=kpts, descriptors=desc, keypoint_scores=kscores)]

    class FakeMatcher:
        def eval(self):
            return self

        def to(self, _dev):
            return self

        def permute(self, *axes):
            return FakeTensor(np.transpose(self.arr, axes))

        def float(self):
            return self

        def long(self):
            return FakeTensor(self.arr.astype(np.int64))

        def div(self, x):
            return FakeTensor(self.arr / x)

        def __truediv__(self, x):
            return FakeTensor(self.arr / x)

        def contiguous(self):
            return self

        def match_pair(self, _ext, _t0, _t1):
            pass

        def __call__(self, _inp):
            return {
                "matches0": FakeTensor(np.arange(8, dtype=np.int64).reshape(1, 8)),
                "matches": [FakeTensor(np.column_stack([np.arange(8), np.arange(8)]).astype(np.int64))],
                "scores": [FakeTensor(np.ones(8, dtype=np.float32) * 0.9)],
            }

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            reset_peak_memory_stats=lambda *_a: None,
            max_memory_allocated=lambda *_a: 0,
            empty_cache=lambda: None,
        ),
        no_grad=lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None),
        from_numpy=lambda arr: FakeTensor(arr),
    )
    fake_kf = types.SimpleNamespace(
        ALIKED=lambda **_kw: FakeExtractor(),
        LightGlue=lambda **_kw: FakeMatcher(),
    )
    fake_kornia = types.SimpleNamespace(feature=fake_kf)

    with patch.dict(sys.modules, {
        "torch": fake_torch,
        "kornia": fake_kornia,
        "kornia.feature": fake_kf,
    }):
        out = _lightglue_run([str(p1), str(p2)], max_long_edge_px=1600, run_id="lg_scale")

    assert out.status == "succeeded", f"error={out.error_message!r} summary={out.summary!r}"
    assert len(captured_shapes) == 2
    assert captured_shapes[0] == (1200, 1600)
    assert len(out.observations) == 1
    obs = out.observations[0]
    assert obs.model_id.hardware_class == "cpu"
    assert obs.metadata["keypoints_a"] == 8
    assert obs.metadata["matches"] == 8


def test_maskrcnn_module_has_no_top_level_torch_import() -> None:
    src = Path("perception/maskrcnn_adapter.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    top_imports = [
        node.names[0].name
        for node in tree.body
        if isinstance(node, ast.Import)
    ] + [
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    assert "torch" not in top_imports


def test_maskrcnn_emits_mask_bbox_ontology_and_persists_to_maskstore(tmp_path: Path) -> None:
    p1 = tmp_path / "room.jpg"
    arr = np.linspace(0, 255, 48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
    Image.fromarray(arr).save(p1)

    fake_mask = np.zeros((1, 48, 64), dtype=np.float32)
    fake_mask[0, 10:30, 12:36] = 0.95

    class FakeArr:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        @property
        def shape(self):
            return self._arr.shape

        def tolist(self):
            return self._arr.tolist()

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

        def __getitem__(self, item):
            val = self._arr[item]
            return FakeArr(val) if isinstance(val, np.ndarray) else val

        def __len__(self):
            return len(self._arr)

    fake_pred = {
        "scores": FakeArr(np.array([0.92], dtype=np.float32)),
        "labels": FakeArr(np.array([1], dtype=np.int64)),
        "boxes": FakeArr(np.array([[12.0, 10.0, 36.0, 30.0]], dtype=np.float32)),
        "masks": FakeArr(np.expand_dims(fake_mask, axis=0)),
    }

    class FakeModel:
        def to(self, _d):
            return self

        def eval(self):
            return self

        def __call__(self, _imgs):
            return [fake_pred]

    class FakeTensor:
        def __init__(self, arr: np.ndarray) -> None:
            self.arr = arr

        def permute(self, *_a):
            return self

        def to(self, _d):
            return self

        def float(self):
            return self

        def long(self):
            return FakeTensor(self.arr.astype(np.int64))

        def div(self, x):
            return FakeTensor(self.arr / x)

        def __truediv__(self, x):
            return FakeTensor(self.arr / x)

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            reset_peak_memory_stats=lambda *_a: None,
            max_memory_allocated=lambda *_a: 0,
            empty_cache=lambda: None,
        ),
        no_grad=lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None),
        from_numpy=lambda arr: FakeTensor(arr),
    )
    _w = types.SimpleNamespace(meta={"categories": ["__background__", "chair"]}, url="local")
    fake_weights = types.SimpleNamespace(DEFAULT=_w, COCO_V1=_w)
    fake_tv_det = types.SimpleNamespace(
        maskrcnn_resnet50_fpn=lambda weights=None: FakeModel(),
        MaskRCNN_ResNet50_FPN_Weights=fake_weights,
    )
    fake_tv = types.SimpleNamespace(
        __version__="0.0-test",
        models=types.SimpleNamespace(detection=fake_tv_det),
        transforms=types.SimpleNamespace(functional=types.SimpleNamespace()),
    )

    with patch.dict(sys.modules, {
        "torch": fake_torch,
        "torchvision": fake_tv,
        "torchvision.models": fake_tv.models,
        "torchvision.models.detection": fake_tv_det,
        "torchvision.transforms": fake_tv.transforms,
        "torchvision.transforms.functional": fake_tv.transforms.functional,
    }):
        from perception.maskrcnn_adapter import _maskrcnn_run

        out = _maskrcnn_run([str(p1)], run_id="mrcnn_test", score_threshold=0.5, min_mask_pixels=10)

    assert out.status == "succeeded", f"error={out.error_message!r} summary={out.summary!r}"
    seg_obs = [o for o in out.observations if o.category == "instance_segmentation"]
    assert len(seg_obs) == 1
    assert seg_obs[0].mask is not None
    assert seg_obs[0].mask.shape == (48, 64)
    assert seg_obs[0].bbox_px == (12.0, 10.0, 36.0, 30.0)
    assert "ontology_path" in seg_obs[0].metadata
    assert seg_obs[0].metadata["raw_label"] == "chair"

    store = MaskStore()
    stamped = persist_masks(list(out.observations), store)
    assert len(store) == 1
    assert any(o.mask_ref == 0 for o in stamped)


def test_depth_adapter_uses_local_files_only_and_grades_partial_failure(tmp_path: Path) -> None:
    good = tmp_path / "good.jpg"
    bad = tmp_path / "missing.jpg"
    arr = np.linspace(0, 255, 64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    Image.fromarray(arr).save(good)

    calls: list[dict] = []

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, _name, **kwargs):
            calls.append(kwargs)
            return cls()

        def __call__(self, images=None, return_tensors="pt"):
            return {"pixel_values": MagicMock(to=lambda _d: "pv")}

    class FakeDepthOut:
        def __init__(self) -> None:
            self.predicted_depth = MagicMock(
                detach=lambda: MagicMock(
                    cpu=lambda: MagicMock(
                        numpy=lambda: np.ones((1, 64, 64), dtype=np.float32) * 2.5
                    )
                )
            )

    class FakeModel:
        @classmethod
        def from_pretrained(cls, _name, **kwargs):
            calls.append(kwargs)
            return cls()

        def to(self, _d):
            return self

        def eval(self):
            return self

        def __call__(self, **_kw):
            return FakeDepthOut()

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            reset_peak_memory_stats=lambda *_a: None,
            max_memory_allocated=lambda *_a: 0,
            empty_cache=lambda: None,
        ),
        no_grad=lambda: MagicMock(__enter__=lambda s: None, __exit__=lambda s, *a: None),
    )
    fake_tf = types.SimpleNamespace(
        AutoImageProcessor=FakeProcessor,
        AutoModelForDepthEstimation=FakeModel,
    )

    with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_tf}):
        from perception.depth_adapter import _depth_run

        out = _depth_run([str(good), str(bad)], run_id="depth_partial")

    assert all(c.get("local_files_only") is True for c in calls)
    assert out.status == "partial"
    assert len(out.observations) == 1
    assert out.observations[0].model_id.hardware_class == "cpu"
    assert out.summary["failed_images"] == [str(bad)]
