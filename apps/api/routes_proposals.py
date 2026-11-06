"""Routes: evidence-backed change proposals against a world's trusted HEAD.

A proposal compares a CANDIDATE world (e.g. from a new scan) with the world's current version and returns reviewable,
per-entity assessments (supported / needs_review / conflict). Nothing changes until a proposal is accepted, and
accepting creates a NEW version through the same DB-mirrored, HEAD-claiming commit every other adoption uses
(worldstore_service.commit_version): a moved HEAD is a 409 (the proposal is stale), never a silent rebase or overwrite.
Conflicting changes need an explicit override plus a reason; both are recorded on the proposal.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import worldstore_service
from apps.api.db import get_db
from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.tenancy import team_stamp, world_scope
from apps.api.models import World, WorldVersion
from world_ir.world_v1 import WorldIR
from worldstore.proposals import (
    ProposalError, ProposalStateError, UnknownProposalError, get_proposal, list_proposals, prepare_acceptance,
    propose_update, record_acceptance, reject_proposal,
)

proposals = APIRouter(prefix="/api/worlds/{world_id}/proposals", tags=["proposals"], dependencies=[Depends(world_scope)])


class ProposeIn(BaseModel):
    candidate: dict                      # a WorldIR document (WorldIR.to_dict() shape)
    source_session_ids: List[str] = Field(default_factory=list)


class AcceptIn(BaseModel):
    #: None = every `supported` change; otherwise exactly these (needs_review/conflict changes must be named)
    entity_ids: Optional[List[str]] = None
    override_conflicts: List[str] = Field(default_factory=list)
    reason: str = ""


class RejectIn(BaseModel):
    reason: str


async def _head(db: AsyncSession, world_id: str) -> World:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    return w


async def _owned(db: AsyncSession, world_id: str, proposal_id: str):
    store = worldstore_service.get_store()
    try:
        p = get_proposal(store, proposal_id)
    except UnknownProposalError as exc:
        raise HTTPException(404, str(exc)) from exc
    row = await db.get(WorldVersion, p.base_version_id)
    if row is None or row.world_id != world_id:           # another world's proposal is "not found" here
        raise HTTPException(404, f"unknown proposal: {proposal_id!r}")
    return store, p


@proposals.post("", status_code=201)
async def create_proposal(world_id: str, body: ProposeIn, db: AsyncSession = Depends(get_db)) -> dict:
    w = await _head(db, world_id)
    if not w.current_version_id:
        raise HTTPException(409, "World has no version to compare against")
    try:
        candidate = WorldIR.from_dict(body.candidate)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise HTTPException(422, f"candidate is not a valid WorldIR document: {exc}") from exc
    candidate.id = world_id
    return propose_update(worldstore_service.get_store(), w.current_version_id, candidate,
                          body.source_session_ids).to_dict()


@proposals.get("")
async def list_world_proposals(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    await _head(db, world_id)
    out = []
    for p in list_proposals(worldstore_service.get_store()):
        row = await db.get(WorldVersion, p.base_version_id)
        if row is not None and row.world_id == world_id:
            out.append({"proposal_id": p.proposal_id, "base_version_id": p.base_version_id, "status": p.status,
                        "summary": p.summary})
    return {"proposals": out}


@proposals.get("/{proposal_id}")
async def read_proposal(world_id: str, proposal_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    _, p = await _owned(db, world_id, proposal_id)
    return p.to_dict()


@proposals.post("/{proposal_id}/accept")
async def accept(world_id: str, proposal_id: str, body: AcceptIn, db: AsyncSession = Depends(get_db)) -> dict:
    store, p = await _owned(db, world_id, proposal_id)
    w = await _head(db, world_id)
    if w.current_version_id != p.base_version_id:
        raise HTTPException(409, f"proposal is stale: it was made against {p.base_version_id} but HEAD is "
                                 f"{w.current_version_id}; propose again against the current version")
    try:
        p, merged, chosen, overrides = prepare_acceptance(
            store, proposal_id, body.entity_ids, override_conflicts=body.override_conflicts, reason=body.reason)
        row = await worldstore_service.commit_version(
            db, world_id=world_id, world=merged, parent=p.base_version_id, expect_parent=p.base_version_id,
            source_session_ids=list(p.source_session_ids))
    except ProposalStateError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ProposalError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (worldstore_service.ConcurrentModificationError, worldstore_service.AdoptionVerificationError) as exc:
        raise HTTPException(409, str(exc)) from exc
    record_acceptance(store, p, row.id, chosen, overrides, body.reason)
    return {"version_id": row.id, "proposal": p.to_dict()}


@proposals.post("/{proposal_id}/reject")
async def reject(world_id: str, proposal_id: str, body: RejectIn, db: AsyncSession = Depends(get_db)) -> dict:
    store, _ = await _owned(db, world_id, proposal_id)
    try:
        return reject_proposal(store, proposal_id, body.reason).to_dict()
    except ProposalStateError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ProposalError as exc:
        raise HTTPException(422, str(exc)) from exc



# ---------------------------------------------------------------------------
# Standalone Proposals Pipeline Router (for direct proposals workflow / tests)
# ---------------------------------------------------------------------------

from types import SimpleNamespace
from typing import Any, Dict
from worldstore.proposals import (
    ProposalStatus,
    get_default_pipeline,
    reset_default_pipeline,
)

pipeline_router = APIRouter(prefix="/proposals", tags=["proposals_pipeline"])


class ProposalIn(BaseModel):
    kind: str
    payload: Dict[str, Any] = {}
    created_by: str = "system"


class ReviewIn(BaseModel):
    reviewer: str = "reviewer"
    reason: str = ""


# The default pipeline is process-global and tenant-blind, so ownership is recorded here at submit time (same lifetime
# as the pipeline itself). A proposal this route did not see submitted has no recorded owner and stays unowned.
_PIPELINE_OWNER: dict[str, str | None] = {}


def _pipeline_owned(proposal_id: str, auth: AuthContext) -> None:
    try:
        enforce_project_isolation(SimpleNamespace(team_id=_PIPELINE_OWNER.get(proposal_id)), auth)
    except HTTPException:
        raise HTTPException(status_code=404, detail="proposal not found") from None


@pipeline_router.post("")
def submit_pipeline_proposal(body: ProposalIn, auth: AuthContext = Depends(get_auth_context)):
    p = get_default_pipeline().submit(body.kind, body.payload, body.created_by)
    _PIPELINE_OWNER[p.proposal_id] = team_stamp(auth)
    return {"proposal_id": p.proposal_id, "status": p.status.value}


@pipeline_router.get("")
def list_pipeline_proposals(status: Optional[str] = None, auth: AuthContext = Depends(get_auth_context)):
    st = ProposalStatus(status) if status else None
    out = []
    for p in get_default_pipeline().list(st):
        try:
            _pipeline_owned(p.proposal_id, auth)
        except HTTPException:
            continue
        out.append({"proposal_id": p.proposal_id, "kind": p.kind,
                    "status": p.status.value, "created_by": p.created_by})
    return {"proposals": out}


@pipeline_router.get("/{proposal_id}")
def get_pipeline_proposal(proposal_id: str, auth: AuthContext = Depends(get_auth_context)):
    p = get_default_pipeline().get(proposal_id)
    if p is None:
        raise HTTPException(status_code=404, detail="proposal not found")
    _pipeline_owned(proposal_id, auth)
    return {"proposal_id": p.proposal_id, "kind": p.kind, "status": p.status.value,
            "payload": p.payload, "reviewed_by": p.reviewed_by,
            "review_reason": p.review_reason}


@pipeline_router.post("/{proposal_id}/approve")
def approve_pipeline_proposal(proposal_id: str, body: ReviewIn, auth: AuthContext = Depends(get_auth_context)):
    _pipeline_owned(proposal_id, auth)
    try:
        p = get_default_pipeline().approve(proposal_id, body.reviewer, body.reason)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"proposal_id": p.proposal_id, "status": p.status.value}


@pipeline_router.post("/{proposal_id}/reject")
def reject_pipeline_proposal(proposal_id: str, body: ReviewIn, auth: AuthContext = Depends(get_auth_context)):
    _pipeline_owned(proposal_id, auth)
    try:
        p = get_default_pipeline().reject(proposal_id, body.reviewer, body.reason)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"proposal_id": p.proposal_id, "status": p.status.value}
