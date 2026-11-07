"""A shared 6 GB GPU can be full when the layout models load: that must fall back to CPU, not fail the whole job."""
import numpy as np
import pytest

from perception.layout import learned_models as lm


def test_cuda_out_of_memory_retries_once_on_cpu(monkeypatch):
    m = lm.LayoutModels()
    seen = []

    def once(image):
        seen.append(m._device)
        if m._device != "cpu":
            raise RuntimeError("CUDA error: out of memory")
        return "ok"

    monkeypatch.setattr(m, "_analyze_once", once)
    monkeypatch.setattr(m, "unload", lambda: None)
    assert m.analyze(object()) == "ok"
    assert seen == [None, "cpu"]


def test_other_errors_are_not_swallowed(monkeypatch):
    m = lm.LayoutModels()
    monkeypatch.setattr(m, "_analyze_once", lambda image: (_ for _ in ()).throw(ValueError("bad image")))
    with pytest.raises(ValueError):
        m.analyze(object())
