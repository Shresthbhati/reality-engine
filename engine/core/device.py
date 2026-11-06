"""Compute-device selection shared by every model backend.

Decision D1 of the ML reconstruction plan: a 6 GB laptop GPU runs ONE heavy
model at a time, in fp16, and every backend still works on CPU (reported as
lower confidence by the caller, never a hard requirement). Backends call
`select_device()` instead of probing torch themselves, and call
`release_vram()` after unloading a model so the next one has room.

`REALITY_DEVICE=cpu` forces the CPU path (tests, GPU-less CI, debugging).
"""

from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

#: Below this much free VRAM a heavy model is not started on the GPU: it would
#: either OOM or evict the display. Callers can pass their own bound.
DEFAULT_MIN_FREE_VRAM_MB = 1024


@dataclass(frozen=True)
class DeviceInfo:
    device: str  # "cuda" | "cpu"
    fp16: bool  # run the model in half precision (GPU only)
    name: str
    free_vram_mb: Optional[int]
    total_vram_mb: Optional[int]
    reason: str  # why this device was chosen, shown in diagnostics

    @property
    def is_gpu(self) -> bool:
        return self.device == "cuda"


def _cpu(reason: str) -> DeviceInfo:
    return DeviceInfo("cpu", False, "cpu", None, None, reason)


def select_device(min_free_vram_mb: int = DEFAULT_MIN_FREE_VRAM_MB) -> DeviceInfo:
    if os.environ.get("REALITY_DEVICE", "").strip().lower() == "cpu":
        return _cpu("REALITY_DEVICE=cpu")
    try:
        import torch
    except ImportError:
        return _cpu("torch not installed")
    if not torch.cuda.is_available():
        return _cpu("CUDA unavailable (CPU torch build or no NVIDIA GPU)")

    name = str(torch.cuda.get_device_name(0))
    try:
        free, total = torch.cuda.mem_get_info()
        free_mb, total_mb = int(free) // (1024 * 1024), int(total) // (1024 * 1024)
    except (RuntimeError, TypeError, ValueError):
        # some drivers can't report memory; a working CUDA device is still usable
        return DeviceInfo("cuda", True, name, None, None, "CUDA available (VRAM query failed)")
    if free_mb < min_free_vram_mb:
        return DeviceInfo(
            "cpu", False, name, free_mb, total_mb,
            f"only {free_mb} MB VRAM free, need {min_free_vram_mb} MB",
        )
    return DeviceInfo("cuda", True, name, free_mb, total_mb, "CUDA available with enough free VRAM")


@contextmanager
def unloading(backend):
    """Yield ``backend`` and, on success or exception, call its ``unload()`` so the
    model's VRAM is returned before the next stage loads its own."""
    try:
        yield backend
    finally:
        unload = getattr(backend, "unload", None)  # fixture/fake backends hold no model
        if unload is not None:
            unload()


def drop_models(obj, *attrs: str) -> None:
    """Null the named model attributes on ``obj`` and flush the CUDA allocator."""
    for name in attrs:
        setattr(obj, name, None)
    release_vram()


def release_vram() -> None:
    """Call after dropping the last reference to a model: frees the cached
    allocator blocks so the next model fits in the 6 GB budget."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
