"""Truthful StageStatus mapping for the legacy ``*_stage`` facts dicts (P108).

The pipeline's optional stages report ``{"status": ..., "note": ...}`` facts. A stage must never say ``succeeded`` for
something that did not run, so the mapping is explicit and mirrors the grading rule of ``apps/api/jobs.py:
_stage_facts_degraded`` (ran / skipped / disabled / rejected / equivalent are honest outcomes, anything else is damage).
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional

from engine.pipeline.stages.base import StageStatus

#: facts statuses meaning "the stage did not run" (disabled by options, backend/weights/GPU/inputs unavailable)
NOT_RUN = frozenset({"skipped", "disabled", "unavailable"})
#: facts statuses meaning "ran and reported its result" (a judged dense candidate that was not adopted still ran;
#: "refused" is plane-sweep's honest "ran, no pixel had a clear winner")
RAN = frozenset({"ran", "rejected", "equivalent", "adopted", "kept_sparse", "refused", "accepted"})


def _reason(facts: Optional[Mapping], label: str) -> str:
    note = (facts or {}).get("note") or ""
    status = (facts or {}).get("status")
    return f"{label}: {status}" + (f" ({note})" if note else "")


def facts_status(facts: Optional[Mapping], label: str) -> tuple[StageStatus, list[str]]:
    """One facts dict -> (status, warnings). ``None`` facts mean the stage produced no facts at all: skipped."""
    if facts is None:
        return "skipped", [f"{label}: no facts reported"]
    status = facts.get("status")
    if status in NOT_RUN:
        return "skipped", [_reason(facts, label)]
    if status in RAN:
        return "succeeded", ([_reason(facts, label)] if status != "ran" else [])
    return "degraded", [_reason(facts, label)]


def combine(parts: Iterable[tuple[StageStatus, list[str]]]) -> tuple[StageStatus, list[str]]:
    """Several sub-steps of one stage: degraded if any degraded; skipped only if NOTHING ran; else succeeded.
    Every non-clean sub-step keeps its reason in the warnings, so a partly-skipped stage is never silently clean."""
    parts = list(parts)
    warnings = [w for _, ws in parts for w in ws]
    statuses = [s for s, _ in parts]
    if "degraded" in statuses or "failed" in statuses:
        return "degraded", warnings
    if statuses and all(s == "skipped" for s in statuses):
        return "skipped", warnings
    return "succeeded", warnings
