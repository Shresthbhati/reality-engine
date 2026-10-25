"""User-facing reconstruction states.

The user should understand what is happening without knowing backend
vocabulary. This maps (job status, job stage, failure kind) to one of:

    CAPTURING  ANALYZING  BUILDING_ROUGH_MODEL  RECONSTRUCTING  REFINING
    PARTIALLY_COMPLETE  READY_TO_INSPECT  NEEDS_MORE_EVIDENCE  FAILED  CANCELLED

Rules that keep the states honest:
  * a timeout or crash is FAILED, never READY_TO_INSPECT
  * a partial-graded job is PARTIALLY_COMPLETE, never READY_TO_INSPECT
  * "READY_TO_INSPECT" is reserved for a cleanly graded, adopted version
  * unknown statuses map to FAILED, never to something optimistic
"""

from __future__ import annotations

from typing import Optional

LABELS = {
    "CAPTURING": "Capturing",
    "ANALYZING": "Analyzing evidence",
    "BUILDING_ROUGH_MODEL": "Building rough model",
    "RECONSTRUCTING": "Reconstructing",
    "REFINING": "Refining",
    "PARTIALLY_COMPLETE": "Partially complete",
    "READY_TO_INSPECT": "Ready to inspect",
    "NEEDS_MORE_EVIDENCE": "Needs more evidence",
    "FAILED": "Failed",
    "CANCELLED": "Cancelled",
}

_STAGE_STATE = {
    "checking_reconstruction_backend": "ANALYZING",
    "resolving_evidence": "ANALYZING",
    "analyzing_evidence": "ANALYZING",
    "building_rough_model": "BUILDING_ROUGH_MODEL",
    "reconstructing": "RECONSTRUCTING",
    "refining": "REFINING",
    "committing_version": "RECONSTRUCTING",
}

ACTIVE_STATES = frozenset(
    {"CAPTURING", "ANALYZING", "BUILDING_ROUGH_MODEL", "RECONSTRUCTING", "REFINING"}
)


def derive_state(
    job_status: Optional[str],
    job_stage: Optional[str] = None,
    failure_kind: Optional[str] = None,
) -> str:
    if job_status is None:
        return "CAPTURING"
    if job_status == "queued":
        return "ANALYZING"
    if job_status == "running":
        return _STAGE_STATE.get(job_stage or "", "ANALYZING")
    if job_status == "succeeded":
        return "READY_TO_INSPECT"
    if job_status == "partial":
        return "PARTIALLY_COMPLETE"
    if job_status == "cancelled":
        return "CANCELLED"
    if job_status == "failed":
        return "NEEDS_MORE_EVIDENCE" if failure_kind == "insufficient_evidence" else "FAILED"
    return "FAILED"


def failure_message(error: Optional[str]) -> Optional[str]:
    """The exception message from a stored traceback (last non-empty line),
    with the exception class prefix removed."""
    if not error:
        return None
    lines = [ln for ln in error.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    last = lines[-1].strip()
    head, sep, tail = last.partition(": ")
    if sep and head.replace(".", "").replace("_", "").isalnum():
        return tail
    return last


def batch_headline(summary: Optional[dict]) -> Optional[str]:
    """One honest sentence about what the newest batch contributed, from
    the measured contribution summary. None when nothing was measured."""
    if not summary or not summary.get("new_images"):
        return None
    n = summary["new_images"]
    good = summary.get("new_view", 0)
    red = summary.get("redundant", 0)
    lost = summary.get("disconnected", 0)
    parts = [f"{n} new image{'s' if n != 1 else ''} added."]
    if good:
        parts.append(f"{good} add{'s' if good == 1 else ''} a new viewpoint.")
    if red:
        parts.append(f"{red} mostly repeat{'s' if red == 1 else ''} an existing view.")
    if lost:
        parts.append(f"{lost} share{'s' if lost == 1 else ''} nothing verifiable with the rest.")
    return " ".join(parts)
