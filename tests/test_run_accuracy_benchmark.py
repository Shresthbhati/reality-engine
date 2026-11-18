"""Tests for Measured-Room Benchmark Accuracy Runner (Mandate Phase 2 & 4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.measured_rooms.run_accuracy_benchmark import (
    DEV_ROOMS,
    HOLDOUT_ROOMS,
    build_synthetic_room_world,
    generate_markdown_report,
    run_all_benchmarks,
    run_room_benchmark,
)


def test_room_benchmark_reports_blocked_when_photos_missing():
    """Rule 13: Missing physical datasets must be marked BLOCKED, never fabricated."""
    res = run_room_benchmark("room_01_living_studio", synthetic_fallback=False)
    assert res.status == "BLOCKED"
    assert res.data_type == "physical"
    assert "missing physical camera photography" in (res.reason or "").lower()
    assert res.metrics is None


def test_holdout_room_requires_explicit_authorization():
    """Rule 10: Holdout room cannot be evaluated without allow_holdout=True."""
    res = run_room_benchmark("room_06_holdout_conference", allow_holdout=False)
    assert res.status == "BLOCKED"
    assert res.is_holdout is True
    assert "access restricted" in (res.reason or "").lower()


def test_synthetic_fallback_evaluates_and_marks_data_type():
    """Mandate Phase 2: Synthetic validation is strictly labelled as synthetic."""
    res = run_room_benchmark(
        "room_01_living_studio",
        synthetic_fallback=True,
    )
    assert res.status == "EVALUATED"
    assert res.data_type == "synthetic"
    assert res.metrics is not None
    assert res.metrics.meets_tolerances is True
    assert res.metrics.mean_wall_error_m <= 0.015  # <= 1.5 cm tolerance
    assert res.metrics.ceiling_height_error_m <= 0.010  # <= 1.0 cm tolerance
    assert res.metrics.room_area_error_pct <= 0.50  # <= 0.5% tolerance
    assert res.metrics.topology_precision == 1.0
    assert res.metrics.topology_recall == 1.0
    assert res.metrics.same_camera_exclusion_adherence == 1.0


def test_aggregate_benchmarks_and_markdown_generation():
    """Mandate Phase 4: Full aggregate evaluation across dev rooms."""
    report = run_all_benchmarks(
        allow_holdout=False,
        synthetic_fallback=True,
        target_rooms=list(DEV_ROOMS),
    )
    assert report.total_rooms == len(DEV_ROOMS)
    assert report.evaluated_rooms == len(DEV_ROOMS)
    assert report.blocked_rooms == 0
    assert report.synthetic_count == len(DEV_ROOMS)
    assert report.overall_meets_tolerances is True
    assert report.mean_wall_error_m <= 0.015
    assert report.walls_within_2cm_pct >= 95.0
    assert report.same_camera_violations == 0
    assert report.idf1 >= 0.90
    assert report.mean_electrical_f1 >= 0.95

    # Check Markdown formatting
    md = generate_markdown_report(report)
    assert "# Reality Engine — Measured Room Benchmark Accuracy Report" in md
    assert "Wall Length Mean Error" in md
    assert "Ceiling Height Error" in md
    assert "Same-Camera Mutual Exclusion" in md
    assert "PASS" in md
