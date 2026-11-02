"""Compute-device selection shared by model backends.
Forwarded to engine.core.device for layer decoupling.
"""

from engine.core.device import (
    DEFAULT_MIN_FREE_VRAM_MB,
    DeviceInfo,
    release_vram,
    select_device,
)

__all__ = [
    "DEFAULT_MIN_FREE_VRAM_MB",
    "DeviceInfo",
    "release_vram",
    "select_device",
]
