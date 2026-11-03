"""Compute-device selection shared by model backends (Blueprint Task 4.2).
Forwarded to engine.core.device for layer decoupling, plus single-model VRAM residency scheduling.
"""

from engine.core.device import (
    DEFAULT_MIN_FREE_VRAM_MB,
    DeviceInfo,
    release_vram,
    select_device,
)
from perception.stage_runner import (
    RTX4050_MAX_USABLE_VRAM_MB,
    ModelResidentScheduler,
    PerceptionStageReport,
    PerceptionStageRunner,
    TaskExecutionRecord,
)

__all__ = [
    "DEFAULT_MIN_FREE_VRAM_MB",
    "DeviceInfo",
    "ModelResidentScheduler",
    "PerceptionStageReport",
    "PerceptionStageRunner",
    "RTX4050_MAX_USABLE_VRAM_MB",
    "TaskExecutionRecord",
    "release_vram",
    "select_device",
]
