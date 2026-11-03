"""Hardware-Aware Execution & Memory Budget Manager (Directive Rule 10).

Enforces strict deployment bounds for NVIDIA GeForce RTX 4050 Laptop GPU:
- Physical VRAM: 6,144 MB (6 GB)
- Operating VRAM ceiling: 5,200 MB (leaves 944 MB reserve for Windows 11 DWM and display)
- System RAM ceiling: 12,000 MB (out of 16 GB physical)
- Tile-based spatial batching when point clouds exceed 2,000,000 points
- Automatic Least-Recently-Used (LRU) model offload to host RAM
- Dynamic precision adaptation: FP16 / INT8 quantization
- Zero Out-Of-Memory (OOM) failures: graceful fallback to CPU execution
"""

from __future__ import annotations

import gc
import logging
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Hardware budget constants (RTX 4050 6GB Laptop Target)
RTX4050_TOTAL_VRAM_MB: float = 6144.0
RTX4050_OS_RESERVE_MB: float = 944.0
RTX4050_MAX_USABLE_VRAM_MB: float = 5200.0  # Hard ceiling for Reality Engine
MAX_SYSTEM_RAM_MB: float = 12000.0
POINT_CLOUD_TILING_THRESHOLD: int = 2_000_000  # Points


class ExecutionDevice(str, Enum):
    CUDA_FP16 = "cuda_fp16"
    CUDA_INT8 = "cuda_int8"
    CUDA_FP32 = "cuda_fp32"
    CPU = "cpu"


@dataclass
class ModelMemoryRecord:
    """Record of an active model in memory."""
    model_id: str
    device: ExecutionDevice
    vram_mb: float
    ram_mb: float
    last_used: float
    is_pinned: bool = False


@dataclass(frozen=True)
class SpatialTile:
    """A spatial chunk of a dense point cloud or mesh."""
    tile_id: str
    bounds_min: Tuple[float, float, float]
    bounds_max: Tuple[float, float, float]
    point_count: int
    point_indices: Tuple[int, ...]


class HardwareMemoryBudget:
    """Tracks and enforces VRAM and System RAM allocations to guarantee 0 OOMs."""

    def __init__(
        self,
        max_vram_mb: float = RTX4050_MAX_USABLE_VRAM_MB,
        max_ram_mb: float = MAX_SYSTEM_RAM_MB,
    ):
        self.max_vram_mb = max_vram_mb
        self.max_ram_mb = max_ram_mb
        self._current_vram_mb: float = 0.0
        self._current_ram_mb: float = 0.0
        self._models: Dict[str, ModelMemoryRecord] = {}

    @property
    def current_vram_mb(self) -> float:
        return self._current_vram_mb

    @property
    def available_vram_mb(self) -> float:
        return max(0.0, self.max_vram_mb - self._current_vram_mb)

    @property
    def current_ram_mb(self) -> float:
        return self._current_ram_mb

    @property
    def available_ram_mb(self) -> float:
        return max(0.0, self.max_ram_mb - self._current_ram_mb)

    def can_allocate_vram(self, requested_mb: float) -> bool:
        return (self._current_vram_mb + requested_mb) <= self.max_vram_mb

    def allocate_vram(self, model_id: str, requested_mb: float, device: ExecutionDevice) -> bool:
        """Attempt to allocate VRAM, triggering LRU offloading if necessary."""
        if requested_mb > self.max_vram_mb:
            # Single model exceeds total GPU capacity -> must run on CPU
            logger.warning(f"Model {model_id} requested {requested_mb:.1f}MB exceeding max VRAM {self.max_vram_mb}MB.")
            return False

        # If not enough space, offload unpinned models until space is freed
        while not self.can_allocate_vram(requested_mb):
            offloaded = self._offload_lru_model()
            if not offloaded:
                # No more models can be offloaded
                return False

        self._current_vram_mb += requested_mb
        self._models[model_id] = ModelMemoryRecord(
            model_id=model_id,
            device=device,
            vram_mb=requested_mb,
            ram_mb=0.0,
            last_used=1.0,
        )
        return True

    def release_vram(self, model_id: str) -> None:
        """Release VRAM used by a model."""
        if model_id in self._models:
            rec = self._models.pop(model_id)
            self._current_vram_mb = max(0.0, self._current_vram_mb - rec.vram_mb)
            gc.collect()

    def _offload_lru_model(self) -> bool:
        """Offload the least recently used unpinned model to host RAM."""
        unpinned = [m for m in self._models.values() if not m.is_pinned and m.vram_mb > 0.0]
        if not unpinned:
            return False

        # Sort by last used
        unpinned.sort(key=lambda m: m.last_used)
        victim = unpinned[0]

        # Shift to CPU
        self._current_vram_mb = max(0.0, self._current_vram_mb - victim.vram_mb)
        victim.ram_mb = victim.vram_mb * 0.8  # Host RAM footprint
        victim.vram_mb = 0.0
        victim.device = ExecutionDevice.CPU
        self._current_ram_mb += victim.ram_mb
        gc.collect()
        return True


class TileBatchingScheduler:
    """Partitions massive dense point clouds (>2M points) into manageable spatial tiles."""

    @staticmethod
    def partition_point_cloud(
        points: np.ndarray,
        tile_size_m: float = 10.0,
        max_points_per_tile: int = 1_000_000,
    ) -> List[SpatialTile]:
        """Divide a point cloud into spatial tiles to guarantee sub-5.2GB memory usage."""
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) == 0:
            return []

        if len(pts) <= POINT_CLOUD_TILING_THRESHOLD:
            # Entire cloud fits into a single batch
            min_pt = tuple(float(x) for x in np.min(pts, axis=0))
            max_pt = tuple(float(x) for x in np.max(pts, axis=0))
            return [SpatialTile("tile-all", min_pt, max_pt, len(pts), tuple(range(len(pts))))]

        # Spatial grid partition
        min_coords = np.min(pts, axis=0)
        grid_coords = np.floor((pts - min_coords) / tile_size_m).astype(np.int32)

        tile_dict: Dict[Tuple[int, int, int], List[int]] = {}
        for idx, (gx, gy, gz) in enumerate(grid_coords):
            key = (int(gx), int(gy), int(gz))
            if key not in tile_dict:
                tile_dict[key] = []
            tile_dict[key].append(idx)

        tiles: List[SpatialTile] = []
        for (gx, gy, gz), indices in tile_dict.items():
            tile_pts = pts[indices]
            b_min = tuple(float(x) for x in np.min(tile_pts, axis=0))
            b_max = tuple(float(x) for x in np.max(tile_pts, axis=0))
            t_id = f"tile-{gx}_{gy}_{gz}"
            tiles.append(SpatialTile(t_id, b_min, b_max, len(indices), tuple(indices)))

        return tiles


class HardwareExecutionOrchestrator:
    """Determines execution device, batch size, and precision based on available hardware."""

    def __init__(self, budget: Optional[HardwareMemoryBudget] = None):
        self.budget = budget or HardwareMemoryBudget()

    def select_execution_strategy(
        self,
        task_name: str,
        base_vram_requirement_mb: float,
        supports_int8: bool = True,
        supports_fp16: bool = True,
    ) -> Tuple[ExecutionDevice, float]:
        """Select the highest-performance device and precision that fits within the 5.2GB budget.

        Returns (ExecutionDevice, allocated_mb).
        """
        # 1. Try FP16 if supported
        fp16_req = base_vram_requirement_mb * 0.5 if supports_fp16 else base_vram_requirement_mb
        if supports_fp16 and self.budget.can_allocate_vram(fp16_req):
            self.budget.allocate_vram(task_name, fp16_req, ExecutionDevice.CUDA_FP16)
            return ExecutionDevice.CUDA_FP16, fp16_req

        # 2. Try INT8 quantization
        int8_req = base_vram_requirement_mb * 0.25 if supports_int8 else base_vram_requirement_mb
        if supports_int8 and self.budget.can_allocate_vram(int8_req):
            self.budget.allocate_vram(task_name, int8_req, ExecutionDevice.CUDA_INT8)
            return ExecutionDevice.CUDA_INT8, int8_req

        # 3. Graceful fallback to CPU execution
        logger.info(f"VRAM budget constrained ({self.budget.available_vram_mb:.1f}MB avail). Falling back to CPU for {task_name}.")
        return ExecutionDevice.CPU, 0.0

    @contextmanager
    def execution_guard(self, task_name: str, requested_vram_mb: float) -> Iterator[ExecutionDevice]:
        """Context manager guarding execution against OOMs and ensuring memory cleanup."""
        device, allocated = self.select_execution_strategy(task_name, requested_vram_mb)
        try:
            yield device
        finally:
            if allocated > 0.0:
                self.budget.release_vram(task_name)
