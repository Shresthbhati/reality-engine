"""Unit tests for Performance and Resource Governor (Wave 6, Block 38)."""

import time
import pytest
import numpy as np

from engine.core.resource_governor import (
    GovernedExecution,
    GovernedExecutionReport,
    GovernorPolicy,
    ResourceSnapshot,
    chunk_point_cloud_spatially,
    compute_governor_policy,
    probe_system_resources,
)


def test_probe_system_resources():
    res = probe_system_resources()
    assert isinstance(res, ResourceSnapshot)
    assert res.total_ram_mb > 1024
    assert res.available_ram_mb > 0
    assert res.cpu_cores >= 1
    assert isinstance(res.gpu_name, str)


def test_compute_governor_policy_tiers():
    # 1. Abundant resources -> Tier 1
    high_res = ResourceSnapshot(
        total_ram_mb=32768,
        available_ram_mb=24576,
        cpu_cores=16,
        gpu_name="NVIDIA RTX 4090",
        free_vram_mb=16000,
    )
    p1 = compute_governor_policy(image_count=50, point_count=100_000, snapshot=high_res)
    assert p1.tier == 1
    assert p1.max_image_dim_px >= 3840
    assert p1.decimation_ratio == 1.0
    assert p1.enable_out_of_core is False

    # 2. Constrained resources -> Tier 3
    low_res = ResourceSnapshot(
        total_ram_mb=4096,
        available_ram_mb=1800,
        cpu_cores=2,
        gpu_name="cpu",
        free_vram_mb=None,
    )
    p3 = compute_governor_policy(image_count=200, point_count=2_000_000, snapshot=low_res)
    assert p3.tier == 3
    assert p3.max_image_dim_px <= 1024
    assert p3.decimation_ratio <= 0.25
    assert p3.enable_out_of_core is True
    assert p3.max_workers <= 2


def test_chunk_point_cloud_spatially():
    # 750 points spread along X axis
    pts = [(float(x) * 0.1, 0.0, 0.0) for x in range(750)]

    chunks = list(chunk_point_cloud_spatially(pts, max_chunk_size=200))
    # 750 / 200 = 4 chunks (200, 200, 200, 150)
    assert len(chunks) == 4
    assert len(chunks[0]) == 200
    assert len(chunks[1]) == 200
    assert len(chunks[2]) == 200
    assert len(chunks[3]) == 150

    # Total points preserved
    total_pts = sum(len(c) for c in chunks)
    assert total_pts == 750


def test_governed_execution_context():
    policy = GovernorPolicy(
        tier=2,
        max_image_dim_px=1920,
        max_points_in_memory=500_000,
        voxel_pitch_m=0.015,
        decimation_ratio=0.5,
        max_workers=4,
        enable_out_of_core=False,
        rationale="Test policy",
    )

    with GovernedExecution(task_name="test_filter_dense_cloud", budget_s=2.0, policy=policy) as gov:
        time.sleep(0.05)

    assert gov.report is not None
    assert isinstance(gov.report, GovernedExecutionReport)
    assert gov.report.task_name == "test_filter_dense_cloud"
    assert gov.report.elapsed_s >= 0.04
    assert gov.report.within_budget is True
    assert gov.report.policy.tier == 2
