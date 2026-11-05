"""Honesty invariants for the technology benchmark (Rule 2)."""

from __future__ import annotations

import json

import pytest

from engine.technologies import benchmark as bm
from engine.technologies.adapters import AdapterExecutionResult
from engine.technologies.benchmark import TechnologyBenchmarkHarness, _resolve_installed_version
from engine.verification.classification import ExecutionClassification
from provenance import Provenance


@pytest.fixture(scope="module")
def harness():
    return TechnologyBenchmarkHarness()


def test_simulated_never_passes_or_recommended(harness):
    r = harness.benchmark_technology("detectron2")
    assert r.classification == "SIMULATED"
    assert r.summary_verdict == "SIMULATED"
    assert r.status == "SIMULATED"
    assert r.passes_threshold is None
    assert "simulated: no real upstream execution" in r.passes_threshold_reason
    assert all(m.passes_threshold is None for m in r.metrics.values())
    assert r.technology_version is None


def test_forced_simulated_with_passing_metric_is_not_recommended(harness, monkeypatch):
    class Fake:
        def execute(self, input_data, options=None):
            return AdapterExecutionResult(
                tech_id="kornia", subsystem=bm.TECHNOLOGIES_BY_ID["kornia"].subsystem, success=True,
                execution_time_ms=1.0, memory_peak_mb=1.0,
                output_data={"mean_gradient_magnitude": 5.0},
                provenance=Provenance.INFERRED, classification=ExecutionClassification.SIMULATED,
            )

        def promote_to_world_ir(self, result, world):
            return world

    monkeypatch.setattr(bm, "get_adapter_for_technology", lambda _id: Fake())
    r = harness.benchmark_technology("kornia")
    assert r.summary_verdict == "SIMULATED"
    assert r.passes_threshold is None
    assert r.metrics["mean_gradient_magnitude"].passes_threshold is None


def test_no_hardcoded_vram_or_memory_constants(harness):
    forbidden_vram = {1250.0, 180.0, 2400.0, 1850.0, 220.0, 350.0, 580.0}
    for tech in ("detectron2", "teaser_plus_plus", "manifold", "onnx_runtime"):
        r = harness.benchmark_technology(tech)
        if r.vram_peak_mb is None:
            assert r.vram_peak_reason.startswith("not measured")
        else:
            assert r.vram_peak_mb not in forbidden_vram
        # tracemalloc-measured: never an adapter constant such as 145.0 / 48.0 / 62.0
        assert r.memory_peak_mb not in {145.0, 48.0, 62.0, 310.0, 215.0, 128.0, 180.0}
        assert r.memory_peak_source == "tracemalloc"


def test_empty_metrics_is_no_measurement(harness, monkeypatch):
    class Fake:
        def execute(self, input_data, options=None):
            return AdapterExecutionResult(
                tech_id="teaser_plus_plus", subsystem=bm.TECHNOLOGIES_BY_ID["teaser_plus_plus"].subsystem,
                success=True, execution_time_ms=1.0, memory_peak_mb=1.0, output_data={},
                provenance=Provenance.RECONSTRUCTED,
                classification=ExecutionClassification.INTERNAL_IMPLEMENTATION,
            )

        def promote_to_world_ir(self, result, world):
            return world

    monkeypatch.setattr(bm, "get_adapter_for_technology", lambda _id: Fake())
    r = harness.benchmark_technology("teaser_plus_plus")
    assert r.metrics == {}
    assert r.status == "NO_MEASUREMENT"
    assert r.summary_verdict == "NO_MEASUREMENT"
    assert r.passes_threshold is None


def test_failure_rate_computed_from_actual_failures(harness, monkeypatch):
    class Boom:
        def execute(self, input_data, options=None):
            raise RuntimeError("boom")

        def promote_to_world_ir(self, result, world):
            return world

    monkeypatch.setattr(bm, "get_adapter_for_technology", lambda _id: Boom())
    r = harness.benchmark_technology("teaser_plus_plus")
    assert r.summary_verdict == "FAILED"
    assert r.failures == 1 and r.attempts == 1
    assert r.failure_rate_pct == 100.0
    assert "boom" in r.failure_info


def test_internal_implementation_is_labelled_not_upstream(harness):
    r = harness.benchmark_technology("teaser_plus_plus")
    assert r.classification == "INTERNAL_IMPLEMENTATION"
    assert "internal re-implementation" in r.evidence_label
    assert "not the upstream" in r.evidence_label
    assert r.failure_rate_pct == 0.0 and r.failures == 0
    assert r.technology_version is None  # upstream did not run
    assert "upstream not executed" in r.technology_version_source


def test_version_comes_from_metadata_not_invented():
    import importlib.metadata as md

    assert _resolve_installed_version(("numpy",)) == md.version("numpy")
    assert _resolve_installed_version(("definitely_not_a_real_pkg_xyz",)) is None
    assert _resolve_installed_version(()) is None


def test_report_has_no_unconditional_verification_claim(harness):
    report = harness.generate_markdown_report()
    assert "VRAM Safety Guarantee" not in report
    assert "Verified < 6,144" not in report
    data = harness.benchmark_all_50()
    # no simulated technology is counted as recommended/qualified
    for r in data["results"]:
        if r["classification"] in ("SIMULATED", "UNAVAILABLE"):
            assert r["summary_verdict"] not in ("RECOMMENDED", "QUALIFIED")
    json.dumps(data)  # still JSON-serialisable for API/CLI
