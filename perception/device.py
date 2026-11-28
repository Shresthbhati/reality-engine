"""Device configuration and CUDA/CPU fallback helper for perception backends.

Provides a single point of truth for device, FP16, and free-VRAM guard.
Every model backend should import and use this helper rather than calling
torch.device() directly.
"""

from __future__ import annotations

import os
from typing import Literal

import torch

# ---------------------------------------------------------------------------
# Global device configuration (set once at process start)
# ---------------------------------------------------------------------------

_device: torch.device | None = None
_fp16_enabled: bool = False
_FREE_VRAM_GUARD_GB = int(os.getenv("REALITY_FREE_VRAM_GB", "4"))

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def init(device_preference: Literal["cuda", "cpu"] | None = None) -> tuple[torch.device, bool]:
    global _device, _fp16_enabled
    if device_preference is not None:
        _device = torch.device(device_preference)
    else:
        _device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _fp16_enabled = _check_fp16_eligibility()
    return _device, _fp16_enabled


def device() -> torch.device:
    if _device is None:
        init()
    return _device


def is_cuda() -> bool:
    return device().type == "cuda"


def is_cpu() -> bool:
    return device().type == "cpu"


def is_fp16() -> bool:
    return _fp16_enabled


def free_vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.mem_get_info()[0] / 1024 ** 3
    return 0.0


def _check_fp16_eligibility() -> bool:
    if not torch.cuda.is_available():
        return False
    return free_vram_gb() > _FREE_VRAM_GUARD_GB


def enable_fp16(force: bool = False) -> None:
    global _fp16_enabled
    if force or _check_fp16_eligibility():
        _fp16_enabled = True


def disable_fp16() -> None:
    global _fp16_enabled
    _fp16_enabled = False


class device_context:
    def __init__(self, new_device: str | torch.device, fp16: bool | None = None):
        self._old_device = device()
        self._old_fp16 = is_fp16()
        self._new_device = torch.device(new_device)
        self._fp16_new = fp16 if fp16 is not None else self._old_fp16

    def __enter__(self):
        init(device_preference=self._new_device.type)
        if self._fp16_new and not _check_fp16_eligibility():
            disable_fp16()
        return self

    def __exit__(self, *exc_info):
        init(device_preference=self._old_device.type)
        if self._old_fp16:
            enable_fp16()
        return False


def mark_resident(enable: bool = True) -> None:
    global _fp16_enabled
    if enable:
        _fp16_enabled = False
    else:
        _fp16_enabled = _check_fp16_eligibility()


# Module-level initialisation
if _device is None:
    init()