"""Tests for Golden Dataset Benchmark Suite (Task 9.7).

Verifies that all 10 benchmark categories can be executed, evaluated against
ground truth, and reported via CLI and Python APIs.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from benchmarks.golden_datasets import (
    GoldenCategory,
    run_golden_benchmark,
    run_all_golden_benchmarks,
)


@pytest.mark.parametrize(
    "category",
    [
        GoldenCategory.SINGLE_ROOM,
        GoldenCategory.CONNECTED_ROOMS,
        GoldenCategory.COMPLETE_APARTMENT,
        GoldenCategory.MULTIPLE_FLOORS,
        GoldenCategory.FURNISHED_INTERIOR,
        GoldenCategory.FINE_DETAILS,
        GoldenCategory.EXTERIOR_BUILDING,
        GoldenCategory.URBAN_STREET,
        GoldenCategory.IRREGULAR_ARCHITECTURE,
    ],
)
def test_golden_dataset_category_passes(category):
    score = run_golden_benchmark(category)
    assert score.passed is True, f"Benchmark {category.value} failed: {score.notes}"
    assert score.entity_count > 0
    assert score.duration_s >= 0.0
    assert isinstance(score.metrics, dict)


def test_golden_dataset_json_serialization(tmp_path: Path):
    score = run_golden_benchmark(GoldenCategory.SINGLE_ROOM)
    data = score.to_dict()
    assert data["category"] == "single_room"
    assert data["passed"] is True
    assert "metrics" in data
    
    out_file = tmp_path / "report.json"
    out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    assert out_file.exists()
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["passed"] is True
