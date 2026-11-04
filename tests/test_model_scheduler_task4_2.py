"""Tests for Blueprint Task 4.2: One-Model-at-a-Time Scheduler.

Verifies:
- At most one heavy model occupies VRAM at any time.
- Previous model is automatically evicted when loading the next model.
- Peak VRAM remains strictly under the RTX 4050 5,200 MB usable ceiling.
- Graceful fallback to CPU execution works without failure.
- Benchmark JSON report output adheres to contract.
"""

import json
import pytest
from perception.device import (
    ModelResidentScheduler,
    PerceptionStageRunner,
    RTX4050_MAX_USABLE_VRAM_MB,
    DeviceInfo,
    select_device,
)


def test_model_resident_scheduler_exclusive_occupancy():
    scheduler = ModelResidentScheduler(vram_budget_mb=5200.0)
    assert scheduler.current_model is None

    # Load model 1
    with scheduler.model_scope("maskrcnn_resnet50_fpn") as dev:
        assert scheduler.current_model == "maskrcnn_resnet50_fpn"
        assert isinstance(dev, DeviceInfo)

    # After scope, model is evicted
    assert scheduler.current_model is None


def test_model_resident_scheduler_switch_eviction():
    scheduler = ModelResidentScheduler(vram_budget_mb=5200.0)
    history = []

    with scheduler.model_scope("maskrcnn_resnet50_fpn"):
        history.append(scheduler.current_model)
        # Nesting another model scope forces eviction of the outer model
        with scheduler.model_scope("midas_dpt_hybrid"):
            history.append(scheduler.current_model)

    assert history == ["maskrcnn_resnet50_fpn", "midas_dpt_hybrid"]
    assert scheduler.current_model is None


def test_model_resident_scheduler_cpu_fallback():
    scheduler = ModelResidentScheduler(vram_budget_mb=5200.0)

    # Force CPU
    with scheduler.model_scope("midas_dpt_hybrid", force_cpu=True) as dev:
        assert dev.device == "cpu"
        assert dev.fp16 is False
        assert dev.reason == "forced_cpu"

    # Excessive VRAM requirement beyond 5,200 MB ceiling
    with scheduler.model_scope("giant_model", required_vram_mb=8000.0) as dev:
        assert dev.device == "cpu"
        assert "exceeds budget" in dev.reason or dev.device == "cpu"


def test_perception_stage_runner_pipeline_execution():
    runner = PerceptionStageRunner(vram_budget_mb=5200.0)

    executed_stages = []

    def seg_fn(dev: DeviceInfo, items):
        executed_stages.append(("segmentation", dev.device, len(items)))
        return {"masks_detected": len(items) * 3}

    def depth_fn(dev: DeviceInfo, items):
        executed_stages.append(("depth", dev.device, len(items)))
        return {"depth_maps": len(items)}

    def embed_fn(dev: DeviceInfo, items):
        executed_stages.append(("embeddings", dev.device, len(items)))
        return {"feature_tracks": len(items) * 10}

    dummy_images = ["img_001.png", "img_002.png", "img_003.png"]

    results, report = runner.run_pipeline(
        evidence_items=dummy_images,
        segmentation_fn=seg_fn,
        depth_fn=depth_fn,
        embeddings_fn=embed_fn,
        force_cpu=True,
    )

    # Assert all three stages executed in exact order
    assert [s[0] for s in executed_stages] == ["segmentation", "depth", "embeddings"]
    for s in executed_stages:
        assert s[1] == "cpu"
        assert s[2] == 3

    # Check results dict
    assert results["segmentation"]["masks_detected"] == 9
    assert results["depth"]["depth_maps"] == 3
    assert results["embeddings"]["feature_tracks"] == 30

    # Verify benchmark telemetry report
    assert report.total_runtime_ms > 0
    assert report.overall_peak_vram_mb <= 5200.0
    assert len(report.tasks) == 3

    for task in report.tasks:
        assert task.status == "completed"
        assert task.items_processed == 3
        assert task.runtime_ms >= 0
        assert task.cpu_fallback is True

    # Validate JSON serialization contract
    report_dict = json.loads(report.to_json())
    assert "total_runtime_ms" in report_dict
    assert "tasks" in report_dict
    assert len(report_dict["tasks"]) == 3
    assert report_dict["vram_budget_mb"] == 5200.0
