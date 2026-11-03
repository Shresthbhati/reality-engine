"""Tests for Hardware-Aware Execution & RTX 4050 6GB Budgeting (Directive Rule 10).

Verifies:
- Hard 5.2GB usable VRAM ceiling on RTX 4050 6GB (leaving 944MB OS reserve)
- Least-Recently-Used (LRU) model offloading to host RAM under memory pressure
- Graceful degradation from CUDA FP16 -> INT8 -> CPU fallback
- Spatial point-cloud tiling when dense clouds exceed 2M points
- Context manager memory guard ensuring zero VRAM leaks
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.hardware.device_manager import (
    ExecutionDevice,
    HardwareExecutionOrchestrator,
    HardwareMemoryBudget,
    RTX4050_MAX_USABLE_VRAM_MB,
    RTX4050_OS_RESERVE_MB,
    RTX4050_TOTAL_VRAM_MB,
    TileBatchingScheduler,
)


class TestHardwareAwareExecutionRule10:
    def test_vram_ceiling_and_os_reserve(self):
        """Verify RTX 4050 Laptop physical and operating bounds."""
        budget = HardwareMemoryBudget()

        assert budget.max_vram_mb == 5200.0
        assert RTX4050_TOTAL_VRAM_MB == 6144.0
        assert RTX4050_OS_RESERVE_MB == 944.0
        assert RTX4050_TOTAL_VRAM_MB - RTX4050_OS_RESERVE_MB == budget.max_vram_mb
        assert budget.available_vram_mb == 5200.0

    def test_lru_offloading_under_pressure(self):
        """Verify automatic offloading of idle models when approaching 5.2GB ceiling."""
        budget = HardwareMemoryBudget(max_vram_mb=5000.0)

        # Allocate Model A (2000 MB) and Model B (2000 MB) -> Total 4000 MB
        ok_a = budget.allocate_vram("model_a", 2000.0, ExecutionDevice.CUDA_FP16)
        ok_b = budget.allocate_vram("model_b", 2000.0, ExecutionDevice.CUDA_FP16)
        assert ok_a and ok_b
        assert budget.current_vram_mb == 4000.0

        # Mark model_a as used at t=1.0 and model_b at t=2.0
        budget._models["model_a"].last_used = 1.0
        budget._models["model_b"].last_used = 2.0

        # Now request Model C (2500 MB) -> Exceeds remaining 1000 MB
        # Should offload model_a (LRU) to host RAM and allocate model_c
        ok_c = budget.allocate_vram("model_c", 2500.0, ExecutionDevice.CUDA_FP16)
        assert ok_c is True

        # Model A should now be on CPU with 0 VRAM
        assert budget._models["model_a"].device == ExecutionDevice.CPU
        assert budget._models["model_a"].vram_mb == 0.0

        # VRAM should now be model_b (2000) + model_c (2500) = 4500 MB <= 5000 MB ceiling
        assert budget.current_vram_mb == 4500.0

    def test_graceful_cpu_fallback(self):
        """Verify fallback to CPU execution when model exceeds total VRAM budget."""
        budget = HardwareMemoryBudget(max_vram_mb=5200.0)
        orchestrator = HardwareExecutionOrchestrator(budget)

        # Huge model requiring 12,000 MB VRAM cannot fit on RTX 4050
        device, allocated = orchestrator.select_execution_strategy(
            task_name="giant_foundation_model",
            base_vram_requirement_mb=12000.0,
            supports_int8=False,
            supports_fp16=False,
        )

        assert device == ExecutionDevice.CPU
        assert allocated == 0.0
        assert budget.current_vram_mb == 0.0

    def test_precision_adaptation_fp16_and_int8(self):
        """Verify selection of FP16 or INT8 quantization under constrained VRAM."""
        budget = HardwareMemoryBudget(max_vram_mb=3000.0)
        orchestrator = HardwareExecutionOrchestrator(budget)

        # Base requirement: 4000 MB
        # FP16 (50% = 2000 MB) fits in 3000 MB
        device, allocated = orchestrator.select_execution_strategy(
            task_name="depth_estimator",
            base_vram_requirement_mb=4000.0,
            supports_fp16=True,
        )

        assert device == ExecutionDevice.CUDA_FP16
        assert allocated == 2000.0
        assert budget.current_vram_mb == 2000.0

    def test_spatial_point_cloud_tiling(self):
        """Verify spatial partitioning when point count exceeds 2,000,000."""
        # Create a large cloud of 2,100,000 points
        num_points = 2_100_000
        # Fast generation: synthetic grid across 30m x 30m domain
        pts = np.random.uniform(0.0, 30.0, (1000, 3))  # representative points for test

        # Force partition by passing low threshold or tile size
        tiles = TileBatchingScheduler.partition_point_cloud(
            points=pts,
            tile_size_m=10.0,
        )

        # Under 2M points, returns single tile
        assert len(tiles) == 1

        # Now test with explicit point array partitioned into spatial tiles
        pts_large = np.zeros((num_points, 3), dtype=np.float32)
        # Put points in 4 spatial quadrants
        pts_large[:500_000] = [5.0, 5.0, 0.0]
        pts_large[500_000:1_000_000] = [25.0, 5.0, 0.0]
        pts_large[1_000_000:1_500_000] = [5.0, 25.0, 0.0]
        pts_large[1_500_000:] = [25.0, 25.0, 0.0]

        large_tiles = TileBatchingScheduler.partition_point_cloud(
            points=pts_large,
            tile_size_m=10.0,
        )

        assert len(large_tiles) == 4
        total_p = sum(t.point_count for t in large_tiles)
        assert total_p == num_points

    def test_execution_guard_context_manager(self):
        """Verify context manager guarantees release of VRAM on exit."""
        budget = HardwareMemoryBudget(max_vram_mb=5200.0)
        orchestrator = HardwareExecutionOrchestrator(budget)

        with orchestrator.execution_guard("feature_matching", requested_vram_mb=1500.0) as dev:
            assert dev == ExecutionDevice.CUDA_FP16
            assert budget.current_vram_mb == 750.0  # FP16 = 50%

        # After exiting block, memory must be automatically reclaimed
        assert budget.current_vram_mb == 0.0
