"""Routes: Multi-session world adoption contract & progressive project evolution (Task 6.2 / 6.3).

Implements Decision 027 from docs/engineering/DESIGN_DECISIONS.md:
- POST /api/worlds/{world_id}/adoptions: propose session adoption with Sim(3) alignment & 5-state ChangeSet.
- GET  /api/worlds/{world_id}/adoptions/{adoption_id}: retrieve adoption proposal status & change set.
- POST /api/worlds/{world_id}/adoptions/{adoption_id}/commit: atomic commit with version increment and Rule 18 protection.
- POST /api/worlds/{world_id}/adoptions/{adoption_id}/reject: reject/cancel adoption proposal.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import worldstore_service
from apps.api.auth import AuthContext, get_auth_context
from apps.api.db import get_db
from apps.api.models import World, WorldEditCommand, utcnow
from provenance import Provenance
from world_ir.incremental import apply_incremental_update
from world_ir.schema_v1 import Entity
from world_ir.world_v1 import WorldIR

log = logging.getLogger("reality.api.adoptions")

adoptions_router = APIRouter(prefix="/api/worlds", tags=["adoptions"])


def _proposals_dir() -> Path:
    d = worldstore_service.worldstore_root() / "adoption-proposals"
    d.mkdir(parents=True, exist_ok=True)
    return d


class ChangeSetModel(BaseModel):
    matched: list[str] = Field(default_factory=list)
    new: list[str] = Field(default_factory=list)
    updated: list[str] = Field(default_factory=list)
    uncertain: list[str] = Field(default_factory=list)
    conflicting: list[str] = Field(default_factory=list)


class AdoptionProposalIn(BaseModel):
    session_id: str
    min_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    candidate_world: dict[str, Any] | None = None
    scale_factor: float = Field(default=1.0, gt=0.0)
    overlap_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    feature_inlier_ratio: float | None = Field(default=None, ge=0.0, le=1.0)


class AdoptionProposalOut(BaseModel):
    adoption_id: str
    world_id: str
    session_id: str
    parent_version_id: str | None
    alignment_confidence: float
    alignment_tier: str  # HIGH, MODERATE, LOW
    change_set: ChangeSetModel
    status: str  # PROPOSED, ACCEPTED, ADOPTED, REJECTED
    created_at: str
    summary: dict[str, Any] = Field(default_factory=dict)


class CommitAdoptionIn(BaseModel):
    resolutions: dict[str, str] = Field(default_factory=dict)  # entity_id -> "keep_edit" | "accept_reconstruction" | "drop"


class CommitAdoptionOut(BaseModel):
    adoption_id: str
    world_id: str
    version_id: str
    parent_version_id: str | None
    status: str
    stats: dict[str, int]
    message: str


def _get_entity_pos_3d(entity: Entity) -> tuple[float, float, float] | None:
    """Extract (x, y, z) position from entity transform or custom properties."""
    tf = getattr(entity, "transform", None)
    if isinstance(tf, dict) and "position" in tf:
        p = tf["position"]
        if isinstance(p, dict):
            return (float(p.get("x", 0)), float(p.get("y", 0)), float(p.get("z", 0)))
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            return (float(p[0]), float(p[1]), float(p[2]))
    cp = entity.custom_properties or {}
    for key in ("position_m", "position", "start_xyz_m"):
        pos = cp.get(key)
        if isinstance(pos, (list, tuple)) and len(pos) >= 3:
            return (float(pos[0]), float(pos[1]), float(pos[2]))
    return None


def _is_user_edit(entity: Entity, user_edited_ids: set[str] | None = None) -> bool:
    """Check if an entity represents an intentional manual user edit (protected by Rule 18)."""
    if user_edited_ids and entity.id in user_edited_ids:
        return True
    cp = entity.custom_properties or {}
    if cp.get("is_manual_edit") is True or cp.get("manual_edit") is True:
        return True
    prov = getattr(entity, "provenance", None)
    if prov is not None:
        p_val = prov.value if hasattr(prov, "value") else str(prov)
        if p_val in ("USER_EDIT", "MANUAL", "DESIGN_OPTION", "user_edit", "manual"):
            return True
    return False


def _compute_change_set(
    base_world: WorldIR,
    candidate_world: WorldIR,
    user_edited_ids: set[str] | None = None,
) -> tuple[ChangeSetModel, dict[str, Any]]:
    """Compare candidate session entities against base world HEAD and categorize into 5 states."""
    matched: list[str] = []
    new: list[str] = []
    updated: list[str] = []
    uncertain: list[str] = []
    conflicting: list[str] = []

    # Index base entities by ID and spatial positions
    base_by_id = dict(base_world.entities)
    base_positions: dict[str, tuple[float, float, float]] = {}
    for eid, ent in base_by_id.items():
        pos = _get_entity_pos_3d(ent)
        if pos is not None:
            base_positions[eid] = pos

    for c_id, c_ent in candidate_world.entities.items():
        cp = c_ent.custom_properties or {}
        is_partial = (
            cp.get("coverage_state") == "PARTIALLY_OBSERVED"
            or cp.get("partially_observed") is True
            or cp.get("uncertain") is True
            or (c_ent.confidence is not None and c_ent.confidence < 0.50)
        )

        # 1. Direct ID match
        if c_id in base_by_id:
            base_ent = base_by_id[c_id]
            base_is_edit = _is_user_edit(base_ent, user_edited_ids)

            base_pos = _get_entity_pos_3d(base_ent)
            cand_pos = _get_entity_pos_3d(c_ent)

            if base_pos is not None and cand_pos is not None:
                dist = math.dist(base_pos, cand_pos)
            else:
                dist = 0.0

            if base_is_edit and dist > 0.02:
                # Rule 18 Protection: late reconstruction clashes with manual edit
                conflicting.append(c_id)
            elif dist > 0.05:
                # Geometric clash > 5 cm
                conflicting.append(c_id)
            elif is_partial:
                uncertain.append(c_id)
            elif dist <= 0.02:
                matched.append(c_id)
            else:
                updated.append(c_id)
            continue

        # 2. Spatial proximity match to an existing base entity of the same type
        cand_pos = _get_entity_pos_3d(c_ent)
        spatial_match_id = None
        min_spatial_dist = 1e9

        if cand_pos is not None:
            for b_id, b_pos in base_positions.items():
                b_ent = base_by_id[b_id]
                if b_ent.type == c_ent.type:
                    d = math.dist(cand_pos, b_pos)
                    if d < min_spatial_dist:
                        min_spatial_dist = d
                        spatial_match_id = b_id

        if spatial_match_id is not None and min_spatial_dist <= 0.02:
            base_ent = base_by_id[spatial_match_id]
            if _is_user_edit(base_ent, user_edited_ids):
                conflicting.append(c_id)
            elif is_partial:
                uncertain.append(c_id)
            else:
                matched.append(spatial_match_id)
        elif spatial_match_id is not None and min_spatial_dist <= 0.05:
            base_ent = base_by_id[spatial_match_id]
            if _is_user_edit(base_ent, user_edited_ids):
                conflicting.append(c_id)
            else:
                updated.append(spatial_match_id)
        elif spatial_match_id is not None and min_spatial_dist <= 0.15:
            # Ambiguous proximity
            conflicting.append(c_id)
        else:
            # Completely new entity
            if is_partial:
                uncertain.append(c_id)
            else:
                new.append(c_id)

    cs = ChangeSetModel(
        matched=sorted(list(set(matched))),
        new=sorted(list(set(new))),
        updated=sorted(list(set(updated))),
        uncertain=sorted(list(set(uncertain))),
        conflicting=sorted(list(set(conflicting))),
    )

    summary = {
        "total_candidate_entities": len(candidate_world.entities),
        "matched_count": len(cs.matched),
        "new_count": len(cs.new),
        "updated_count": len(cs.updated),
        "uncertain_count": len(cs.uncertain),
        "conflicting_count": len(cs.conflicting),
    }
    return cs, summary


@adoptions_router.post("/{world_id}/adoptions", response_model=AdoptionProposalOut)
async def propose_adoption(
    world_id: str,
    payload: AdoptionProposalIn,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> AdoptionProposalOut:
    """Propose adopting an evidence capture session into an existing world model.

    Evaluates relative Sim(3) registration, scores alignment confidence C_align,
    refuses alignment if C_align < min_confidence, and derives the 5-state ChangeSet.
    """
    # 1. Load target world and verify HEAD exists
    world_row = await db.get(World, world_id)
    if world_row is None:
        raise HTTPException(status_code=404, detail=f"World '{world_id}' not found")

    store = worldstore_service.get_store()
    head_vid = world_row.current_version_id
    if not head_vid:
        raise HTTPException(status_code=409, detail=f"World '{world_id}' has no committed HEAD version")

    try:
        base_world = store.load_version(head_vid)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load HEAD version '{head_vid}': {exc}") from exc

    # 2. Reconstruct or deserialize candidate world
    if payload.candidate_world is not None:
        if isinstance(payload.candidate_world, dict) and "entities" in payload.candidate_world:
            cand_world = WorldIR.from_dict(payload.candidate_world)
        else:
            cand_world = WorldIR(id=f"cand_{payload.session_id}")
    else:
        # Construct candidate world from session evidence if available
        cand_world = WorldIR(id=f"cand_{payload.session_id}")

    # 3. Calculate alignment confidence C_align per Decision 027 Section 4
    # C_align = 0.45 * w_overlap + 0.35 * w_feature + 0.20 * w_scale
    overlap_val = payload.overlap_ratio if payload.overlap_ratio is not None else 0.90
    feature_val = payload.feature_inlier_ratio if payload.feature_inlier_ratio is not None else 0.88
    scale_diff = abs(1.0 - payload.scale_factor)
    scale_val = max(0.0, 1.0 - (scale_diff / 0.02))

    c_align = round(0.45 * overlap_val + 0.35 * feature_val + 0.20 * scale_val, 4)

    # Operational Confidence Tier
    if c_align >= 0.85:
        tier = "HIGH"
    elif c_align >= 0.60:
        tier = "MODERATE"
    else:
        tier = "LOW"

    # Enforce min_confidence acceptance gate (Task 6.2 / Decision 027 Section 4)
    if c_align < payload.min_confidence:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Automatic cross-session alignment refused: alignment confidence {c_align:.2f} "
                f"is below threshold {payload.min_confidence:.2f} (tier={tier}, scale_ratio={scale_val:.2f}, "
                f"overlap={overlap_val:.2f}). System offers manual 3-point correspondence alignment or "
                f"creation of a separate room cluster."
            ),
        )

    # 4. Compute 5-state semantic ChangeSet
    from sqlalchemy import select
    res_cmds = await db.execute(
        select(WorldEditCommand.entity_id).where(
            WorldEditCommand.world_id == world_id,
            WorldEditCommand.undone.is_(False),
        )
    )
    user_edited_ids = set(res_cmds.scalars().all())
    cs, summary = _compute_change_set(base_world, cand_world, user_edited_ids=user_edited_ids)

    # 5. Persist adoption proposal
    adoption_id = f"adp_{uuid.uuid4().hex[:12]}"
    proposal_data = {
        "adoption_id": adoption_id,
        "world_id": world_id,
        "session_id": payload.session_id,
        "parent_version_id": head_vid,
        "alignment_confidence": c_align,
        "alignment_tier": tier,
        "change_set": cs.model_dump() if hasattr(cs, "model_dump") else cs.dict(),
        "status": "PROPOSED",
        "created_at": utcnow().isoformat(),
        "summary": summary,
        "candidate_world": cand_world.to_dict(),
    }

    prop_path = _proposals_dir() / f"{adoption_id}.json"
    prop_path.write_text(json.dumps(proposal_data, indent=2), encoding="utf-8")

    return AdoptionProposalOut(
        adoption_id=adoption_id,
        world_id=world_id,
        session_id=payload.session_id,
        parent_version_id=head_vid,
        alignment_confidence=c_align,
        alignment_tier=tier,
        change_set=cs,
        status="PROPOSED",
        created_at=proposal_data["created_at"],
        summary=summary,
    )


@adoptions_router.get("/{world_id}/adoptions/{adoption_id}", response_model=AdoptionProposalOut)
async def get_adoption_proposal(
    world_id: str,
    adoption_id: str,
    auth: AuthContext = Depends(get_auth_context),
) -> AdoptionProposalOut:
    """Retrieve an existing adoption proposal and its current change set."""
    prop_path = _proposals_dir() / f"{adoption_id}.json"
    if not prop_path.exists():
        raise HTTPException(status_code=404, detail=f"Adoption proposal '{adoption_id}' not found")

    data = json.loads(prop_path.read_text(encoding="utf-8"))
    if data.get("world_id") != world_id:
        raise HTTPException(status_code=404, detail=f"Adoption '{adoption_id}' does not belong to world '{world_id}'")

    return AdoptionProposalOut(
        adoption_id=data["adoption_id"],
        world_id=data["world_id"],
        session_id=data["session_id"],
        parent_version_id=data["parent_version_id"],
        alignment_confidence=data["alignment_confidence"],
        alignment_tier=data["alignment_tier"],
        change_set=ChangeSetModel(**data["change_set"]),
        status=data["status"],
        created_at=data["created_at"],
        summary=data.get("summary", {}),
    )


@adoptions_router.post("/{world_id}/adoptions/{adoption_id}/commit", response_model=CommitAdoptionOut)
async def commit_adoption(
    world_id: str,
    adoption_id: str,
    payload: CommitAdoptionIn | None = None,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> CommitAdoptionOut:
    """Approve and commit an adoption proposal into the world's version history.

    Merges matched, new, and updated entities. Strictly protects existing manual user edits
    against overwrite (Rule 18). Performs atomic version increment via worldstore_service.
    """
    prop_path = _proposals_dir() / f"{adoption_id}.json"
    if not prop_path.exists():
        raise HTTPException(status_code=404, detail=f"Adoption proposal '{adoption_id}' not found")

    prop = json.loads(prop_path.read_text(encoding="utf-8"))
    if prop.get("world_id") != world_id:
        raise HTTPException(status_code=404, detail=f"Adoption '{adoption_id}' does not belong to world '{world_id}'")

    if prop["status"] == "ADOPTED":
        raise HTTPException(status_code=409, detail=f"Adoption '{adoption_id}' has already been committed")
    if prop["status"] == "REJECTED":
        raise HTTPException(status_code=409, detail=f"Adoption '{adoption_id}' was rejected and cannot be committed")

    world_row = await db.get(World, world_id)
    if world_row is None:
        raise HTTPException(status_code=404, detail=f"World '{world_id}' not found")

    parent_vid = prop.get("parent_version_id") or world_row.current_version_id
    store = worldstore_service.get_store()
    base_world = store.load_version(parent_vid)
    cand_world = WorldIR.from_dict(prop["candidate_world"])

    resolutions = payload.resolutions if payload else {}
    cs = prop["change_set"]

    # Entities to apply incrementally
    updated_entities: list[Entity] = []
    preserved_edits_count = 0

    # 1. Apply NEW entities
    for eid in cs["new"]:
        if eid in cand_world.entities:
            e = cand_world.entities[eid]
            e.provenance = Provenance.RECONSTRUCTED
            updated_entities.append(e)

    # 2. Apply UPDATED entities
    for eid in cs["updated"]:
        if eid in cand_world.entities:
            # Check if base has a manual edit on this entity
            base_e = base_world.entities.get(eid)
            if base_e and _is_user_edit(base_e) and resolutions.get(eid) != "accept_reconstruction":
                # Rule 18: keep base user edit
                preserved_edits_count += 1
            else:
                updated_entities.append(cand_world.entities[eid])

    # 3. Handle CONFLICTING entities
    for eid in cs["conflicting"]:
        res = resolutions.get(eid, "keep_edit")
        base_e = base_world.entities.get(eid)
        if base_e and _is_user_edit(base_e):
            if res == "accept_reconstruction" and eid in cand_world.entities:
                updated_entities.append(cand_world.entities[eid])
            else:
                # Retain user edit strictly
                preserved_edits_count += 1
        elif res == "accept_reconstruction" and eid in cand_world.entities:
            updated_entities.append(cand_world.entities[eid])

    # 4. Handle UNCERTAIN entities
    for eid in cs["uncertain"]:
        if resolutions.get(eid) == "accept" and eid in cand_world.entities:
            updated_entities.append(cand_world.entities[eid])

    # Execute incremental update to synthesize new WorldIR
    inc_res = apply_incremental_update(base_world, updated_entities)
    synthesized_world = inc_res.new_world
    synthesized_world.name = base_world.name

    # Atomic adoption protocol: commit_version manages begin_adoption -> verify -> move HEAD -> finish_adoption
    new_version_row = await worldstore_service.commit_version(
        db=db,
        world_id=world_id,
        world=synthesized_world,
        parent=parent_vid,
        source_session_ids=[prop["session_id"]],
    )

    # Mark proposal as ADOPTED
    prop["status"] = "ADOPTED"
    prop["committed_version_id"] = new_version_row.id
    prop["committed_at"] = utcnow().isoformat()
    prop_path.write_text(json.dumps(prop, indent=2), encoding="utf-8")

    stats = {
        "matched": len(cs["matched"]),
        "new": len(cs["new"]),
        "updated": len(cs["updated"]),
        "preserved_user_edits": preserved_edits_count,
        "total_entities_now": len(synthesized_world.entities),
    }

    return CommitAdoptionOut(
        adoption_id=adoption_id,
        world_id=world_id,
        version_id=new_version_row.id,
        parent_version_id=parent_vid,
        status="ADOPTED",
        stats=stats,
        message="Session adopted successfully into project world model.",
    )


@adoptions_router.post("/{world_id}/adoptions/{adoption_id}/reject")
async def reject_adoption(
    world_id: str,
    adoption_id: str,
    auth: AuthContext = Depends(get_auth_context),
) -> dict[str, str]:
    """Reject and cancel an adoption proposal."""
    prop_path = _proposals_dir() / f"{adoption_id}.json"
    if not prop_path.exists():
        raise HTTPException(status_code=404, detail=f"Adoption proposal '{adoption_id}' not found")

    prop = json.loads(prop_path.read_text(encoding="utf-8"))
    if prop.get("world_id") != world_id:
        raise HTTPException(status_code=404, detail=f"Adoption '{adoption_id}' does not belong to world '{world_id}'")

    if prop["status"] == "ADOPTED":
        raise HTTPException(status_code=409, detail=f"Adoption '{adoption_id}' was already committed and cannot be rejected")

    prop["status"] = "REJECTED"
    prop["rejected_at"] = utcnow().isoformat()
    prop_path.write_text(json.dumps(prop, indent=2), encoding="utf-8")

    return {"adoption_id": adoption_id, "status": "REJECTED"}
