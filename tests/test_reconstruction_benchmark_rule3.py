"""Unit and integration tests for Universal Spatial Intelligence Directive Rule 3:
Real Reconstruction Benchmarks & Reproducible Before-and-After Comparisons.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from benchmarks.reconstruction_benchmark import (
    BenchmarkDomain,
    DomainBenchmarkScore,
    UniversalBenchmarkReport,
    UniversalReconstructionBenchmark,
)


def test_all_eleven_reconstruction_domains():
    """Verify all 11 core real-world reconstruction domains execute and pass."""
    bench = UniversalReconstructionBenchmark(system_version="test-v1.0")

    # Verify enum contains all 11 domains explicitly demanded by Rule 3
    assert len(BenchmarkDomain) == 11
    expected_domains = {
        "rooms",
        "apartments",
        "corridors",
        "multi_storey_buildings",
        "complex_architectural_forms",
        "electrical_installations",
        "plumbing_installations",
        "building_facades",
        "roads_and_footpaths",
        "urban_infrastructure",
        "vegetation_and_terrain",
    }
    assert {d.value for d in BenchmarkDomain} == expected_domains

    # Run individual domain evaluations
    score_elec = bench.evaluate_domain(BenchmarkDomain.ELECTRICAL_INSTALLATIONS)
    assert score_elec.passed is True
    assert score_elec.geometric_rmse_mm < 5.0  # Sub-5mm precision for electrical sockets/switches
    assert score_elec.topology_connectivity_pct == 100.0

    score_plumb = bench.evaluate_domain(BenchmarkDomain.PLUMBING_INSTALLATIONS)
    assert score_plumb.passed is True
    assert score_plumb.geometric_rmse_mm < 5.0  # Sub-5mm precision for plumbing runs & valves

    score_terr = bench.evaluate_domain(BenchmarkDomain.VEGETATION_AND_TERRAIN)
    assert score_terr.passed is True
    assert score_terr.entity_precision_pct == 100.0


def test_full_benchmark_report_and_markdown():
    """Verify full benchmark report aggregates all domains and renders clean markdown."""
    bench = UniversalReconstructionBenchmark(system_version="reality-engine-v2.0-bench")
    report = bench.run_all_domains()

    assert report.total_domains == 11
    assert report.passed_domains == 11
    assert report.overall_pass is True
    assert report.mean_geometric_rmse_mm < 25.0
    assert report.mean_f1_score > 0.90
    assert report.peak_memory_mb < 6144.0  # RTX 4050 6GB upper bound strictly respected

    # Verify markdown generation
    md = bench.generate_markdown_report(report)
    assert "# Reality Engine — Universal Spatial Reconstruction Benchmark (Rule 3)" in md
    assert "electrical_installations" in md
    assert "plumbing_installations" in md
    assert "vegetation_and_terrain" in md


def test_before_and_after_comparison_and_regression_detection():
    """Verify before-and-after comparison tracks improvement and flags regressions."""
    bench = UniversalReconstructionBenchmark()

    # Create baseline report
    report_v1 = bench.run_all_domains()
    report_v1.system_version = "v1.0-baseline"

    # Create simulated improved report v2
    report_v2 = bench.run_all_domains()
    report_v2.system_version = "v2.0-optimized"
    report_v2.mean_geometric_rmse_mm = report_v1.mean_geometric_rmse_mm - 2.5
    report_v2.total_runtime_ms = report_v1.total_runtime_ms * 0.85

    comp_improved = UniversalReconstructionBenchmark.compare_reconstruction_benchmarks(report_v1, report_v2)
    assert comp_improved.has_regressions is False
    assert comp_improved.rmse_delta_mm < 0.0  # Reduced error
    assert comp_improved.runtime_speedup_factor > 1.0  # Faster

    # Create simulated regressed report v3
    report_v3 = bench.run_all_domains()
    report_v3.system_version = "v3.0-buggy"
    # Artificially inject large geometric error in rooms domain
    report_v3.domain_scores["rooms"].geometric_rmse_mm += 10.0

    comp_regressed = UniversalReconstructionBenchmark.compare_reconstruction_benchmarks(report_v1, report_v3)
    assert comp_regressed.has_regressions is True
    assert any("rooms" in r and "Geometric error increased" in r for r in comp_regressed.regression_details)
