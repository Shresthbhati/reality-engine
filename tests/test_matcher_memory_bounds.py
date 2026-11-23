"""The learned matcher must bound its memory: capped keypoints per image, and every model/feature released by close()."""
from __future__ import annotations

import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from reconstruction.matching.lightglue import LightGlueMatcher  # noqa: E402


class _StubDisk:
    """Stands in for kornia's DISK: records the cap it was asked for and returns MORE keypoints than that if the cap is ignored."""

    def __init__(self):
        self.asked_n = None

    def __call__(self, img, n=None):
        self.asked_n = n
        k = 5000 if n is None else n
        return [types.SimpleNamespace(keypoints=torch.rand(k, 2) * 64, descriptors=torch.rand(k, 128))]


def _image(tmp_path):
    p = tmp_path / "a.png"
    cv2.imwrite(str(p), (np.random.rand(64, 64, 3) * 255).astype(np.uint8))
    return p


def test_the_keypoint_cap_reaches_the_detector(tmp_path):
    m = LightGlueMatcher({"a": _image(tmp_path)}, max_keypoints=300, device="cpu")
    m._detector, m._matcher, m._device = _StubDisk(), object(), torch.device("cpu")
    kpts_np, kpts, descs, _ = m._features("a")
    assert m._detector.asked_n == 300 and len(kpts) == 300 == len(descs) == len(kpts_np)


def test_the_default_cap_is_bounded_and_overridable(tmp_path, monkeypatch):
    monkeypatch.delenv("RE_MATCH_MAX_KEYPOINTS", raising=False)
    assert LightGlueMatcher({}).max_keypoints == 2048
    monkeypatch.setenv("RE_MATCH_MAX_KEYPOINTS", "512")
    assert LightGlueMatcher({}).max_keypoints == 512


def test_close_releases_models_and_cached_features(tmp_path):
    m = LightGlueMatcher({"a": _image(tmp_path)}, max_keypoints=50, device="cpu")
    m._detector, m._matcher, m._device = _StubDisk(), object(), torch.device("cpu")
    m._features("a")
    assert m._feat
    m.close()
    assert not m._feat and m._detector is None and m._matcher is None
