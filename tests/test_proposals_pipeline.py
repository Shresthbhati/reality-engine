"""Tests for the proposals pipeline (worldstore/proposals.py)."""

from __future__ import annotations

import pytest

from worldstore.proposals import (
    ProposalPipeline,
    ProposalStatus,
    ProposalError,
)


def test_submit_creates_pending():
    pipe = ProposalPipeline()
    p = pipe.submit("add_room", {"room_id": "r1"})
    assert p.status == ProposalStatus.PENDING
    assert pipe.get(p.proposal_id) is p


def test_approve_then_apply():
    pipe = ProposalPipeline()
    p = pipe.submit("add_room", {"room_id": "r1"})
    pipe.approve(p.proposal_id, reviewer="alice")
    assert p.status == ProposalStatus.APPROVED

    def applier(prop):
        return {"applied": prop.payload["room_id"]}

    res = pipe.apply(p.proposal_id, applier)
    assert res == {"applied": "r1"}
    assert p.status == ProposalStatus.APPLIED
    assert p.apply_result == res


def test_rejected_cannot_apply():
    pipe = ProposalPipeline()
    p = pipe.submit("add_room", {})
    pipe.reject(p.proposal_id, reviewer="bob", reason="no")
    assert p.status == ProposalStatus.REJECTED
    with pytest.raises(ProposalError):
        pipe.apply(p.proposal_id, lambda x: None)


def test_unapproved_cannot_apply():
    pipe = ProposalPipeline()
    p = pipe.submit("add_room", {})
    with pytest.raises(ProposalError):
        pipe.apply(p.proposal_id, lambda x: None)


def test_double_apply_rejected():
    pipe = ProposalPipeline()
    p = pipe.submit("add_room", {})
    pipe.approve(p.proposal_id)
    pipe.apply(p.proposal_id, lambda x: "ok")
    with pytest.raises(ProposalError):
        pipe.apply(p.proposal_id, lambda x: "again")


def test_apply_failure_reverts_to_approved():
    pipe = ProposalPipeline()
    p = pipe.submit("add_room", {})
    pipe.approve(p.proposal_id)

    def boom(prop):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        pipe.apply(p.proposal_id, boom)
    assert p.status == ProposalStatus.APPROVED


def test_list_filter_by_status():
    pipe = ProposalPipeline()
    p1 = pipe.submit("k", {})
    p2 = pipe.submit("k", {})
    pipe.approve(p1.proposal_id)
    pending = pipe.list(ProposalStatus.PENDING)
    approved = pipe.list(ProposalStatus.APPROVED)
    assert len(pending) == 1 and pending[0].proposal_id == p2.proposal_id
    assert len(approved) == 1 and approved[0].proposal_id == p1.proposal_id