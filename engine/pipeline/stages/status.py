"""Stage execution lifecycle state machine.

Consolidated from Cline's uncommitted work in worktree `216df`
(`engine/pipeline/stages/base.py`), which extended the stage contract with
explicit lifecycle states. Recovered ADDITIVELY: this new module does not
change ``base.py`` (a shared contract) or the runner, so no other agent's
work is overwritten and current main is unaffected until a caller opts in.

Golden rules (mandate: "real failure-state reporting", "no fake success"):
  * A terminal state cannot transition again (no resurrection of a failed
    stage into succeeded).
  * ``cancellation_requested`` is non-terminal but only returns to
    ``running`` or ends in ``cancelled`` -- never straight to ``succeeded``.
  * Non-productive states (skipped/failed/cancelled/timed_out) mean no
    usable output; grading must downgrade accordingly, never upgrade.
  * Every illegal transition raises -- silent transitions are forbidden.
"""

from __future__ import annotations

from typing import FrozenSet

__all__ = [
    "TERMINAL_STAGES", "NON_TERMINAL_STAGES", "NON_PRODUCTIVE_STAGES",
    "LEGAL_TRANSITIONS", "is_terminal", "is_non_productive",
    "validate_transition", "can_transition", "grading_impact",
]

TERMINAL_STAGES: FrozenSet[str] = frozenset({
    "succeeded", "degraded", "partial", "skipped", "failed", "cancelled", "timed_out",
})
NON_TERMINAL_STAGES: FrozenSet[str] = frozenset({
    "queued", "running", "cancellation_requested",
})
NON_PRODUCTIVE_STAGES: FrozenSet[str] = frozenset({
    "skipped", "failed", "cancelled", "timed_out",
})

LEGAL_TRANSITIONS: dict[str, FrozenSet[str]] = {
    "queued": frozenset({"running", "cancelled", "failed"}),
    "running": frozenset({
        "succeeded", "degraded", "partial", "failed",
        "cancellation_requested", "timed_out",
    }),
    "cancellation_requested": frozenset({"running", "cancelled"}),
    "cancelled": frozenset(),
    "succeeded": frozenset(),
    "degraded": frozenset(),
    "partial": frozenset(),
    "skipped": frozenset(),
    "failed": frozenset(),
    "timed_out": frozenset(),
}


def is_terminal(status: str) -> bool:
    """True when ``status`` is terminal (cannot transition further)."""
    return status in TERMINAL_STAGES


def is_non_productive(status: str) -> bool:
    """True when ``status`` produced no usable output for downstream stages."""
    return status in NON_PRODUCTIVE_STAGES


def can_transition(from_status: str, to_status: str) -> bool:
    return to_status in LEGAL_TRANSITIONS.get(from_status, frozenset())


def validate_transition(from_status: str, to_status: str) -> None:
    """Raise ValueError for an illegal transition. Never silently allows one."""
    if not can_transition(from_status, to_status):
        allowed = sorted(LEGAL_TRANSITIONS.get(from_status, frozenset()))
        raise ValueError(
            f"illegal stage status transition {from_status!r} -> {to_status!r}; allowed: {allowed}"
        )


def grading_impact(status: str, *, required: bool) -> str:
    """How a stage status affects job grading: 'ok', 'partial', 'failed'.

    Required non-productive stages fail the job; optional ones downgrade to
    partial. ``degraded``/``partial`` downgrade to partial but never fail.
    """
    if status in ("failed", "cancelled", "timed_out", "skipped"):
        return "failed" if required else "partial"
    if status in ("degraded", "partial"):
        return "partial"
    return "ok"
