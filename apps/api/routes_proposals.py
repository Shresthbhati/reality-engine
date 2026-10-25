"""Routes: proposals pipeline (submit / review / apply)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from worldstore.proposals import (
    ProposalStatus,
    get_default_pipeline,
    reset_default_pipeline,
)

router = APIRouter(prefix="/proposals", tags=["proposals"])


class ProposalIn(BaseModel):
    kind: str
    payload: Dict[str, Any] = {}
    created_by: str = "system"


class ReviewIn(BaseModel):
    reviewer: str = "reviewer"
    reason: str = ""


@router.post("")
def submit_proposal(body: ProposalIn):
    p = get_default_pipeline().submit(body.kind, body.payload, body.created_by)
    return {"proposal_id": p.proposal_id, "status": p.status.value}


@router.get("")
def list_proposals(status: Optional[str] = None):
    st = ProposalStatus(status) if status else None
    out = []
    for p in get_default_pipeline().list(st):
        out.append({"proposal_id": p.proposal_id, "kind": p.kind,
                    "status": p.status.value, "created_by": p.created_by})
    return {"proposals": out}


@router.get("/{proposal_id}")
def get_proposal(proposal_id: str):
    p = get_default_pipeline().get(proposal_id)
    if p is None:
        raise HTTPException(status_code=404, detail="proposal not found")
    return {"proposal_id": p.proposal_id, "kind": p.kind, "status": p.status.value,
            "payload": p.payload, "reviewed_by": p.reviewed_by,
            "review_reason": p.review_reason}


@router.post("/{proposal_id}/approve")
def approve_proposal(proposal_id: str, body: ReviewIn):
    try:
        p = get_default_pipeline().approve(proposal_id, body.reviewer, body.reason)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"proposal_id": p.proposal_id, "status": p.status.value}


@router.post("/{proposal_id}/reject")
def reject_proposal(proposal_id: str, body: ReviewIn):
    try:
        p = get_default_pipeline().reject(proposal_id, body.reviewer, body.reason)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"proposal_id": p.proposal_id, "status": p.status.value}


# Expose reset for tests
def _reset():
    reset_default_pipeline()