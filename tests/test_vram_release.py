"""Each model stage must give its VRAM back on success AND on exception (repair-roadmap P0 item 4).

CUDA is mocked (a fake ``torch`` whose cuda reports available and counts ``empty_cache``), so this runs on
any machine and proves the release path, not a measurement.
"""

from __future__ import annotations

import sys
import types

import pytest

from engine.core.device import drop_models, unloading


@pytest.fixture
def fake_cuda(monkeypatch):
    calls = {"empty_cache": 0}
    cuda = types.SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    return calls


class _Backend:
    def __init__(self):
        self._model = object()
        self._processor = object()

    def unload(self):
        drop_models(self, "_model", "_processor")


def test_drop_models_nulls_attrs_and_empties_cache(fake_cuda):
    b = _Backend()
    b.unload()
    assert b._model is None and b._processor is None
    assert fake_cuda["empty_cache"] == 1


def test_unloading_releases_on_success(fake_cuda):
    b = _Backend()
    with unloading(b):
        assert b._model is not None  # still resident inside the scope
    assert b._model is None
    assert fake_cuda["empty_cache"] == 1


def test_unloading_releases_on_exception(fake_cuda):
    b = _Backend()
    with pytest.raises(RuntimeError, match="boom"):
        with unloading(b):
            raise RuntimeError("boom")
    assert b._model is None
    assert fake_cuda["empty_cache"] == 1


@pytest.mark.parametrize("path,cls,attrs", [
    ("perception.depth.midAS_backend", "MiDaSDepthBackend", ("_model", "_transform")),
    ("perception.depth.depth_anything_backend", "DepthAnythingV2Backend", ("_model", "_processor")),
    ("perception.segmentation.sam_backend", "SAMSegmentationBackend", ("_model", "_mask_generator")),
])
def test_real_backends_unload_their_models(fake_cuda, path, cls, attrs):
    import importlib

    backend = getattr(importlib.import_module(path), cls)(device="cpu")
    for a in attrs:
        setattr(backend, a, object())
    with unloading(backend):
        pass
    assert all(getattr(backend, a) is None for a in attrs)
    assert fake_cuda["empty_cache"] == 1


def test_furniture_model_cache_unload(fake_cuda, monkeypatch):
    from engine.pipeline import single_image_furniture as sif

    b = _Backend()
    monkeypatch.setattr(sif, "_MODEL", b)
    sif.unload_furniture_model()
    assert sif._MODEL is None and b._model is None
    assert fake_cuda["empty_cache"] == 1
    sif.unload_furniture_model()  # idempotent with nothing cached
