"""perception.device: one device-selection rule for every model backend.

Passes on both CPU-only and CUDA machines: it asserts the selection is
consistent with what torch reports, and that a tensor op actually runs on the
chosen device.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from perception.device import DeviceInfo, release_vram, select_device


def test_env_override_forces_cpu(monkeypatch):
    monkeypatch.setenv("REALITY_DEVICE", "cpu")
    info = select_device()
    assert info == DeviceInfo("cpu", False, "cpu", None, None, "REALITY_DEVICE=cpu")
    assert not info.is_gpu


def test_selection_matches_torch_and_runs_a_real_op(monkeypatch):
    monkeypatch.delenv("REALITY_DEVICE", raising=False)
    info = select_device()
    if torch.cuda.is_available():
        assert info.device == "cuda" and info.fp16 and info.free_vram_mb and info.total_vram_mb
        dtype = torch.float16
    else:
        assert info.device == "cpu" and not info.fp16
        dtype = torch.float32
    x = torch.ones(64, 64, device=info.device, dtype=dtype)
    assert float((x @ x).float().sum()) == 64 * 64 * 64
    release_vram()  # must be safe on either device


def test_too_little_free_vram_falls_back_to_cpu_with_the_reason(monkeypatch):
    monkeypatch.delenv("REALITY_DEVICE", raising=False)
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    info = select_device(min_free_vram_mb=10**9)  # impossible bound
    assert info.device == "cpu" and "VRAM free" in info.reason
