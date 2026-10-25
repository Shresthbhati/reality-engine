"""Proposals pipeline for Reality Engine.

A Proposal is a suggested mutation to a WorldIR / WorldStore that must be
reviewed before application. Nothing is silently applied: every proposal
passes through submit -> review (approve/reject) -> apply, and rejected or
unreviewed proposals can never be applied.

This module is intentionally storage-backend agnostic (in-memory, thread-safe)
so it can be wired into the API and tests without a database. Persistence
can be layered on later.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ProposalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


class ProposalError(Exception):
    """Raised when a proposal operation is invalid."""


@dataclass
class Proposal:
    proposal_id: str
    kind: str
    payload: Dict[str, Any]
    status: ProposalStatus = ProposalStatus.PENDING
    created_by: str = "system"
    created_at: float = field(default_factory=time.time)
    reviewed_by: Optional[str] = None
    review_reason: str = ""
    applied_at: Optional[float] = None
    apply_result: Any = None


class ProposalPipeline:
    """Thread-safe in-memory proposal store + review/apply workflow."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proposals: Dict[str, Proposal] = {}
        self._order: List[str] = []

    def submit(self, kind: str, payload: Optional[Dict[str, Any]] = None,
               created_by: str = "system") -> Proposal:
        if not kind:
            raise ProposalError("proposal kind must be non-empty")
        p = Proposal(
            proposal_id=str(uuid.uuid4()),
            kind=kind,
            payload=dict(payload or {}),
            created_by=created_by,
        )
        with self._lock:
            self._proposals[p.proposal_id] = p
            self._order.append(p.proposal_id)
        return p

    def get(self, proposal_id: str) -> Optional[Proposal]:
        with self._lock:
            return self._proposals.get(proposal_id)

    def list(self, status: Optional[ProposalStatus] = None) -> List[Proposal]:
        with self._lock:
            out = [self._proposals[pid] for pid in self._order]
        if status is not None:
            out = [p for p in out if p.status == status]
        return out

    def approve(self, proposal_id: str, reviewer: str = "reviewer",
                reason: str = "") -> Proposal:
        with self._lock:
            p = self._require(proposal_id)
            if p.status != ProposalStatus.PENDING:
                raise ProposalError(
                    f"cannot approve proposal in status {p.status.value}"
                )
            p.status = ProposalStatus.APPROVED
            p.reviewed_by = reviewer
            p.review_reason = reason
        return p

    def reject(self, proposal_id: str, reviewer: str = "reviewer",
               reason: str = "") -> Proposal:
        with self._lock:
            p = self._require(proposal_id)
            if p.status != ProposalStatus.PENDING:
                raise ProposalError(
                    f"cannot reject proposal in status {p.status.value}"
                )
            p.status = ProposalStatus.REJECTED
            p.reviewed_by = reviewer
            p.review_reason = reason
        return p

    def apply(self, proposal_id: str,
              apply_fn: Callable[[Proposal], Any]) -> Any:
        with self._lock:
            p = self._require(proposal_id)
            if p.status == ProposalStatus.REJECTED:
                raise ProposalError("cannot apply a rejected proposal")
            if p.status == ProposalStatus.APPLIED:
                raise ProposalError("proposal already applied")
            if p.status != ProposalStatus.APPROVED:
                raise ProposalError(
                    "proposal must be approved before application"
                )
            # mark applied optimistically; apply_fn runs outside? Keep sync.
            p.status = ProposalStatus.APPLIED
            p.applied_at = time.time()
        try:
            result = apply_fn(p)
        except Exception:
            # revert on failure
            with self._lock:
                p.status = ProposalStatus.APPROVED
                p.applied_at = None
            raise
        with self._lock:
            p.apply_result = result
        return result

    def _require(self, proposal_id: str) -> Proposal:
        p = self._proposals.get(proposal_id)
        if p is None:
            raise ProposalError(f"unknown proposal: {proposal_id}")
        return p


# Module-level default pipeline for API use.
_default_pipeline: Optional[ProposalPipeline] = None
_default_lock = threading.Lock()


def get_default_pipeline() -> ProposalPipeline:
    global _default_pipeline
    with _default_lock:
        if _default_pipeline is None:
            _default_pipeline = ProposalPipeline()
        return _default_pipeline


def reset_default_pipeline() -> ProposalPipeline:
    global _default_pipeline
    with _default_lock:
        _default_pipeline = ProposalPipeline()
    return _default_pipeline