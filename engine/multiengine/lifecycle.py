"""Evidence-derived lifecycle states for the engine registry.

WS1 of the Master Engine Integration Mandate. Additive: it does NOT change
``engine.multiengine.contract.EngineSpec`` (a shared contract other agents
review) or ``EngineRegistry``. It reads an ``EngineSpec`` + explicit
evidence records and classifies the engine into one of the eight mandated
states.

Golden rule: an engine is NEVER advanced on a successful probe, import, or
inference alone. Every promotion requires a concrete :class:`LifecycleEvidence`
record proving that step actually happened. Missing evidence => the engine
stays at its last proven state, never silently upgraded.

States (mandated order):
    PLANNED             declared only, nothing probed
    RESEARCH_REQUIRED   declared but a real probe showed BLOCKED / research needed
    PROTOTYPE           adapter/spec exists and probe passes, no real-input run yet
    IMPLEMENTED         executed on real input, output normalized to the contract
    INTEGRATED          evidence consumed downstream AND contribution persisted
    BENCHMARKED         measured against a held-out benchmark record
    PRODUCTION_READY    all of the above + honest failure handling + repeatable
    REJECTED            explicitly rejected with a reason (terminal)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

__all__ = [
    "EngineLifecycleState", "LifecycleEvidence", "LifecycleRecord",
    "classify_lifecycle", "build_status_report",
]


class EngineLifecycleState(str, Enum):
    PLANNED = "PLANNED"
    RESEARCH_REQUIRED = "RESEARCH_REQUIRED"
    PROTOTYPE = "PROTOTYPE"
    IMPLEMENTED = "IMPLEMENTED"
    INTEGRATED = "INTEGRATED"
    BENCHMARKED = "BENCHMARKED"
    PRODUCTION_READY = "PRODUCTION_READY"
    REJECTED = "REJECTED"


#: This is the mandated order. Any state not at least this proven is below it.
_RANK = {
    EngineLifecycleState.PLANNED: 0,
    EngineLifecycleState.RESEARCH_REQUIRED: 1,
    EngineLifecycleState.PROTOTYPE: 2,
    EngineLifecycleState.IMPLEMENTED: 3,
    EngineLifecycleState.INTEGRATED: 4,
    EngineLifecycleState.BENCHMARKED: 5,
    EngineLifecycleState.PRODUCTION_READY: 6,
}

_REJECTED = EngineLifecycleState.REJECTED


@dataclass(frozen=True)
class LifecycleEvidence:
    """Evidence records, each tied to a concrete source. Absent = not proven.

    ``executed_on_real_input``: a real (non-synthetic) capture was processed.
    ``output_normalized``: output validated against the canonical contract.
    ``evidence_consumed_downstream``: a named Evidence Matrix / WorldIR
        consumer ingested the engine's observations.
    ``contribution_persisted``: evidence survived a store/restart and reload.
    ``contribution_measurable``: the engine's contribution has a recorded
        metric (count, delta, score) attributable to this engine.
    ``failure_observable``: failures are recorded with a reason, not swallowed.
    ``benchmark_record``: id/path of a held-out benchmark record with metrics.
    ``reproducible``: commit sha + dataset split + config + hardware recorded.
    """
    executed_on_real_input: bool = False
    output_normalized: bool = False
    evidence_consumed_downstream: str | None = None   # e.g. "EvidenceMatrix", "WorldIR"
    contribution_persisted: bool = False
    contribution_measurable: str | None = None       # metric description
    failure_observable: bool = False
    benchmark_record: str | None = None
    reproducible: bool = False
    rejected_reason: str | None = None
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class LifecycleRecord:
    engine_id: str
    state: EngineLifecycleState
    evidence: LifecycleEvidence
    reasons: Tuple[str, ...] = ()  # why the state is what it is, for audit

    def to_dict(self) -> dict:
        return {
            "engine_id": self.engine_id,
            "state": self.state.value,
            "reasons": list(self.reasons),
            "evidence": {
                "executed_on_real_input": self.evidence.executed_on_real_input,
                "output_normalized": self.evidence.output_normalized,
                "evidence_consumed_downstream": self.evidence.evidence_consumed_downstream,
                "contribution_persisted": self.evidence.contribution_persisted,
                "contribution_measurable": self.evidence.contribution_measurable,
                "failure_observable": self.evidence.failure_observable,
                "benchmark_record": self.evidence.benchmark_record,
                "reproducible": self.evidence.reproducible,
                "rejected_reason": self.evidence.rejected_reason,
                "notes": list(self.evidence.notes),
            },
        }


def _spec_field(spec: Any, name: str, default: Any = None) -> Any:
    return getattr(spec, name, default)


def classify_lifecycle(
    engine_id: str,
    spec: Any,
    evidence: LifecycleEvidence,
    *,
    probe_ok: bool | None = None,
) -> LifecycleRecord:
    """Derive the lifecycle state from ``evidence``. Never trusts a probe or a
    declared ``integration`` string to promote an engine.

    ``probe_ok`` may be supplied from ``EngineRegistry.availability()``; it is
    used only to distinguish PLANNED/RESEARCH_REQUIRED/PROTOTYPE below
    IMPLEMENTED, never to promote past PROTOTYPE.
    """
    reasons: list[str] = []
    e = evidence

    # Explicit rejection is terminal and wins over everything.
    if e.rejected_reason:
        return LifecycleRecord(engine_id, _REJECTED, e, (f"explicitly rejected: {e.rejected_reason}",))

    # Every promotion is gated on the FULL chain of prior evidence.
    implemented = e.executed_on_real_input and e.output_normalized
    integrated = implemented and bool(e.evidence_consumed_downstream) and e.contribution_persisted
    benchmarked = integrated and bool(e.benchmark_record)
    production = (
        benchmarked
        and e.contribution_measurable
        and e.failure_observable
        and e.reproducible
    )

    if production:
        state = EngineLifecycleState.PRODUCTION_READY
        reasons.append("full evidence chain: real input, normalized, consumed, persisted, benchmarked, measured, failure-observable, reproducible")
    elif benchmarked:
        state = EngineLifecycleState.BENCHMARKED
        reasons.append(f"benchmarked via {e.benchmark_record}")
        if not e.failure_observable:
            reasons.append("not PRODUCTION_READY: failure handling not observable")
        if not e.reproducible:
            reasons.append("not PRODUCTION_READY: run not reproducible")
        if not e.contribution_measurable:
            reasons.append("not PRODUCTION_READY: contribution not measurable")
    elif integrated:
        state = EngineLifecycleState.INTEGRATED
        reasons.append(f"evidence consumed by {e.evidence_consumed_downstream} and persisted")
        if not e.benchmark_record:
            reasons.append("not BENCHMARKED: no held-out benchmark record")
    elif implemented:
        state = EngineLifecycleState.IMPLEMENTED
        reasons.append("executed on real input and output normalized")
        if not e.evidence_consumed_downstream:
            reasons.append("not INTEGRATED: no downstream consumer recorded")
    else:
        # Below IMPLEMENTED: probe/integration string may place PROTOTYPE,
        # but NEVER past it. Missing real-input evidence keeps it here.
        integration = str(_spec_field(spec, "integration", "registry-only") or "registry-only")
        if probe_ok is False:
            state = EngineLifecycleState.RESEARCH_REQUIRED
            reasons.append("real probe BLOCKED: research/environment required")
        elif probe_ok is True or integration in ("adapter", "wired"):
            state = EngineLifecycleState.PROTOTYPE
            reasons.append("declared/adapter present and probe passes, but no real-input execution evidence")
            reasons.append("not IMPLEMENTED: executed_on_real_input/output_normalized not proven")
        else:
            state = EngineLifecycleState.PLANNED
            reasons.append("declared only; no probe evidence")
    return LifecycleRecord(engine_id, state, e, tuple(reasons))


def build_status_report(
    classifications: Mapping[str, LifecycleRecord],
) -> dict:
    """Machine-readable integration status report (WS5 deliverable): per-engine
    lifecycle state plus counts. Read-only; no claims beyond recorded evidence."""
    by_state: dict[str, list[str]] = {}
    for eid, rec in classifications.items():
        by_state.setdefault(rec.state.value, []).append(eid)
    return {
        "schema": "re.engine.integration_status.v1",
        "engine_count": len(classifications),
        "by_state": {k: sorted(v) for k, v in sorted(by_state.items())},
        "engines": {eid: rec.to_dict() for eid, rec in sorted(classifications.items())},
        "note": "states derived from recorded evidence; a successful probe/inference alone does not advance an engine",
    }
