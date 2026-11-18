"""Regression tests for RTX 4050 6GB VRAM hardware profiling."""

from __future__ import annotations

import pytest

from benchmarks.hardware.vram_rtx4050_profile import (
    SimulatedModelBackend,
    profile_rtx4050_pipeline,
)
from engine.hardware.device_manager import (
    RTX4050_MAX_USABLE_VRAM_MB,
    HardwareMemoryBudget,
)


def test_rtx4050_hardware_profile_bounds_and_prevented_oom():
    """Verify RTX 4050 physical bounds, concurrent OOM trigger, and sequential safety."""
    report = profile_rtx4050_pipeline()

    assert report.total_physical_vram_mb == 6144.0
    assert report.os_reserve_mb == 944.0
    assert report.usable_budget_mb == 5200.0

    # Concurrent loading would require 6,950 MB and exceed 5,200 MB budget
    assert report.concurrent_demand_mb == 6950.0
    assert report.concurrent_would_oom is True

    # Sequential execution peak must stay <= 2,100 MB (under SAM 2 model peak)
    assert report.sequential_peak_vram_mb <= 2100.0
    assert report.sequential_safety_headroom_mb >= 3100.0

    # Zero memory leak after all 5 stages complete
    assert report.zero_leak_verified is True
    assert len(report.stages) == 5

    for s in report.stages:
        assert s.peak_vram_mb <= RTX4050_MAX_USABLE_VRAM_MB
        assert s.budget_headroom_mb >= 3100.0
        assert s.oom_prevented is True


def test_simulated_backend_raises_oom_when_budget_exceeded():
    """Verify that attempting to allocate a model exceeding the 5,200 MB VRAM ceiling raises an explicit OOM error."""
    budget = HardwareMemoryBudget(max_vram_mb=5200.0)

    # Attempting to allocate 6,000 MB (exceeding total 5,200 MB usable budget) must raise RuntimeError
    b_giant = SimulatedModelBackend("model_giant", 6000.0, budget)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        b_giant.load()

    assert budget.current_vram_mb == 0.0
