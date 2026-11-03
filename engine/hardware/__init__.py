"""Hardware-Aware Execution Subsystem (Directive Rule 10 - RTX 4050 6GB)."""

from engine.hardware.device_manager import (
    ExecutionDevice,
    HardwareExecutionOrchestrator,
    HardwareMemoryBudget,
    ModelMemoryRecord,
    POINT_CLOUD_TILING_THRESHOLD,
    RTX4050_MAX_USABLE_VRAM_MB,
    RTX4050_OS_RESERVE_MB,
    RTX4050_TOTAL_VRAM_MB,
    SpatialTile,
    TileBatchingScheduler,
)

__all__ = [
    "RTX4050_TOTAL_VRAM_MB",
    "RTX4050_OS_RESERVE_MB",
    "RTX4050_MAX_USABLE_VRAM_MB",
    "POINT_CLOUD_TILING_THRESHOLD",
    "ExecutionDevice",
    "ModelMemoryRecord",
    "SpatialTile",
    "HardwareMemoryBudget",
    "TileBatchingScheduler",
    "HardwareExecutionOrchestrator",
]
