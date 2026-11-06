"""OneFormer must raise the standard unavailable error when its model cannot load -- never fabricated masks."""

from __future__ import annotations

import pytest

from perception.segmentation.oneformer_backend import OneFormerSegmentationBackend
from perception.segmentation.sam_backend import SegmentationBackendUnavailableError


def test_unloadable_model_raises_instead_of_returning_masks(monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__

    def deny_transformers(name, *a, **k):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("transformers blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", deny_transformers)

    class _Ev:
        id = "e1"
        path = tmp_path / "x.png"

    with pytest.raises(SegmentationBackendUnavailableError):
        OneFormerSegmentationBackend().segment(_Ev())
