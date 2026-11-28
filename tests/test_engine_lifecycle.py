"""Tests for engine/multiengine/lifecycle.py (WS1 + WS5 status report)."""

from __future__ import annotations

from engine.multiengine.lifecycle import (
    EngineLifecycleState as S,
    LifecycleEvidence as E,
    build_status_report,
    classify_lifecycle,
)


class _Spec:
    def __init__(self, integration="registry-only"):
        self.integration = integration


def test_probe_alone_never_promotes_past_prototype():
    # A probe that passes + "wired" string is NOT an integration.
    rec = classify_lifecycle("eng", _Spec("wired"), E(), probe_ok=True)
    assert rec.state is S.PROTOTYPE
    assert any("no real-input execution evidence" in r for r in rec.reasons)
    assert any("not IMPLEMENTED" in r for r in rec.reasons)


def test_planned_and_research_required():
    assert classify_lifecycle("eng", _Spec(), E(), probe_ok=None).state is S.PLANNED
    rec = classify_lifecycle("eng", _Spec(), E(), probe_ok=False)
    assert rec.state is S.RESEARCH_REQUIRED
    assert any("BLOCKED" in r for r in rec.reasons)


def test_inference_alone_stays_prototype_not_implemented():
    # "executes" without normalized output is not IMPLEMENTED.
    rec = classify_lifecycle("eng", _Spec("wired"), E(executed_on_real_input=True), probe_ok=True)
    assert rec.state is S.PROTOTYPE


def test_implemented_requires_real_input_and_normalized_output():
    rec = classify_lifecycle("eng", _Spec(), E(executed_on_real_input=True, output_normalized=True))
    assert rec.state is S.IMPLEMENTED
    assert any("not INTEGRATED" in r for r in rec.reasons)


def test_integrated_requires_consumer_and_persistence():
    partial = classify_lifecycle("eng", _Spec(), E(
        executed_on_real_input=True, output_normalized=True,
        evidence_consumed_downstream="EvidenceMatrix"))
    assert partial.state is S.IMPLEMENTED  # persistence missing
    full = classify_lifecycle("eng", _Spec(), E(
        executed_on_real_input=True, output_normalized=True,
        evidence_consumed_downstream="EvidenceMatrix", contribution_persisted=True))
    assert full.state is S.INTEGRATED
    assert any("not BENCHMARKED" in r for r in full.reasons)


def test_benchmarked_and_production_ready_chain():
    base = dict(executed_on_real_input=True, output_normalized=True,
                evidence_consumed_downstream="WorldIR", contribution_persisted=True)
    bm = classify_lifecycle("eng", _Spec(), E(**base, benchmark_record="benches/holdout-v1"))
    assert bm.state is S.BENCHMARKED
    prod = classify_lifecycle("eng", _Spec(), E(
        **base, benchmark_record="benches/holdout-v1",
        contribution_measurable="474 observations", failure_observable=True, reproducible=True))
    assert prod.state is S.PRODUCTION_READY


def test_rejected_is_terminal_and_wins():
    rec = classify_lifecycle("eng", _Spec(), E(
        executed_on_real_input=True, output_normalized=True,
        evidence_consumed_downstream="WorldIR", contribution_persisted=True,
        benchmark_record="b", failure_observable=True, reproducible=True,
        contribution_measurable="x", rejected_reason="non-commercial licence"))
    assert rec.state is S.REJECTED
    assert any("explicitly rejected" in r for r in rec.reasons)


def test_missing_later_evidence_does_not_silently_upgrade():
    # benchmark present but persistence NOT proven -> cannot reach INTEGRATED.
    rec = classify_lifecycle("eng", _Spec(), E(
        executed_on_real_input=True, output_normalized=True,
        evidence_consumed_downstream="EvidenceMatrix", benchmark_record="b"))
    assert rec.state is S.IMPLEMENTED  # gated on missing persistence


def test_status_report_machine_readable():
    recs = {
        "a": classify_lifecycle("a", _Spec("wired"), E(), probe_ok=True),
        "b": classify_lifecycle("b", _Spec(), E(
            executed_on_real_input=True, output_normalized=True,
            evidence_consumed_downstream="WorldIR", contribution_persisted=True)),
    }
    report = build_status_report(recs)
    assert report["schema"] == "re.engine.integration_status.v1"
    assert report["engine_count"] == 2
    assert report["by_state"]["PROTOTYPE"] == ["a"]
    assert report["by_state"]["INTEGRATED"] == ["b"]
    assert report["engines"]["b"]["state"] == "INTEGRATED"
    assert report["engines"]["b"]["evidence"]["contribution_persisted"] is True
    assert "probe" in report["note"]
