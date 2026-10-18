"""Routes: Evidence Quality Firewall (screen photo evidence before reconstruction).

GET  /api/evidence/quality           every photo evidence item with its stored facts, derived
                                     check results, score, spatial contribution and decision
GET  /api/evidence/{id}/quality      the same record for one item (asset inspector)
PUT  /api/evidence/{id}/decision     persist the operator's decision

Everything shown is read from what the pipeline already stored (image facts,
the head version's contribution report); check results come from the pure
functions in ``evidence.quality_checks``. A ``rejected`` decision excludes the
item from reconstruction input (see apps/api/jobs.py); the other decisions are
advisory. This router MUST be included before ``routes_misc.evidence`` so that
``/api/evidence/quality`` is not captured by ``/api/evidence/{evidence_id}``.
"""

from __future__ import annotations

import asyncio
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.db import get_db
from apps.api.models import Evidence, EvidenceDecision, Session, World, WorldVersion, utcnow
from apps.api.storage import resolve_artifact
from evidence import quality_checks as qc
from evidence.image_check import inspect_image

quality = APIRouter(prefix="/api/evidence", tags=["evidence-quality"])


_POLICY = {
    "resolution_fail_mp": qc.RESOLUTION_FAIL_MP,
    "resolution_warn_mp": qc.RESOLUTION_WARN_MP,
    "sharpness_fail": qc.SHARPNESS_FAIL,
    "sharpness_warn": qc.SHARPNESS_WARN,
    "warn_penalty": qc.WARN_PENALTY,
    "fail_penalty": qc.FAIL_PENALTY,
}


class DecisionIn(BaseModel):
    decision: Literal["accepted", "accepted_with_warning", "needs_review", "rejected"]
    reason: Optional[str] = Field(default=None, max_length=2000)


def _decision_out(row: EvidenceDecision) -> dict:
    return {
        "decision": row.decision,
        "reason": row.reason,
        "decided_at": row.decided_at.isoformat() if row.decided_at else None,
    }


async def _contribution_maps(db: AsyncSession, world_ids: set[str]) -> dict[str, dict]:
    """world_id -> {"per_image": {evidence_id: dict}, "available": bool, "note": str, "reason": str|None}
    from the world's CURRENT version report (the measured SIFT/RANSAC contribution)."""
    out: dict[str, dict] = {}
    for wid in world_ids:
        world = await db.get(World, wid)
        head = await db.get(WorldVersion, world.current_version_id) if world and world.current_version_id else None
        contrib = ((head.report or {}).get("evidence") or {}).get("contribution") if head is not None else None
        if not contrib:
            out[wid] = {"reason": "No model has been built for this world yet", "per_image": {}}
        else:
            out[wid] = {
                "reason": None if contrib.get("available") else (contrib.get("note") or "Contribution was not measured"),
                "note": contrib.get("note"),
                "per_image": {p["evidence_id"]: p for p in contrib.get("per_image", [])},
            }
    return out


def _contribution_for(ev: Evidence, world_id: Optional[str], maps: dict, names: dict[str, str]) -> dict:
    if not world_id:
        return {"available": False, "reason": "Not attached to a world, so there is nothing to compare it with"}
    m = maps[world_id]
    if m["reason"]:
        return {"available": False, "reason": m["reason"]}
    p = m["per_image"].get(ev.id)
    if p is None:
        return {"available": False, "reason": "Added after the current model was built; rebuild the world to measure it"}
    partner = p.get("best_partner")
    return {
        "available": True,
        "label": p.get("label"),
        "status": p.get("status"),
        "verified_matches": p.get("verified_matches"),
        "median_shift_fraction": p.get("median_shift_fraction"),
        "best_partner_id": partner,
        "best_partner_name": names.get(partner) if partner else None,
        "basis": m.get("note"),
    }


async def _items(db: AsyncSession, only_id: Optional[str] = None, limit: Optional[int] = 500) -> list[dict]:
    rows = (await db.execute(
        select(Evidence).where(Evidence.type == "photo").order_by(Evidence.created_at.asc(), Evidence.id.asc())
    )).scalars().all()
    if not rows:
        return []
    sessions = {s.id: s for s in (await db.execute(select(Session))).scalars().all()}
    decisions = {d.evidence_id: d for d in (await db.execute(select(EvidenceDecision))).scalars().all()}
    names = {e.id: e.name for e in rows}

    # facts: stored at ingest; a photo uploaded via /api/uploads is inspected on first view and stored
    for ev in rows:
        if (ev.metadata_json or {}).get("image_facts") or not ev.artifact_uri:
            continue
        path = resolve_artifact(ev.artifact_uri)
        if path is None:
            continue
        facts = await asyncio.to_thread(inspect_image, path, (ev.metadata_json or {}).get("declared_evidence_class"))
        ev.metadata_json = {**(ev.metadata_json or {}), "image_facts": facts.to_dict()}
    await db.commit()

    world_ids = {sessions[e.session_id].world_id for e in rows if e.session_id in sessions and sessions[e.session_id].world_id}
    maps = await _contribution_maps(db, world_ids)

    first_by_sha: dict[str, Evidence] = {}
    items: list[dict] = []
    for ev in rows:  # ascending: the earliest copy is the original, later ones are the duplicates
        dup = first_by_sha.get(ev.checksum) if ev.checksum else None
        if ev.checksum and dup is None:
            first_by_sha[ev.checksum] = ev
        if only_id and ev.id != only_id:
            continue
        meta = ev.metadata_json or {}
        facts = meta.get("image_facts") or None
        checks = qc.evaluate_checks(facts, duplicate_of=dup.name if dup else None)
        auto = qc.auto_result(checks)
        dec = decisions.get(ev.id)
        world_id = sessions[ev.session_id].world_id if ev.session_id in sessions else None
        exif = (facts or {}).get("exif") or {}
        cam = " ".join(x for x in (exif.get("make"), exif.get("model")) if x)
        items.append({
            "id": ev.id,
            "name": ev.name,
            "session_id": ev.session_id,
            "world_id": world_id,
            "source": cam or ("Video frame" if meta.get("source_video") else "Upload"),
            "source_video": meta.get("source_video"),
            "added_at": ev.created_at.isoformat() if ev.created_at else None,
            "mime_type": ev.mime_type,
            "size": ev.size,
            "width": (facts or {}).get("width") or None,
            "height": (facts or {}).get("height") or None,
            "sharpness": (facts or {}).get("sharpness"),
            "evidence_class": (facts or {}).get("evidence_class"),
            "class_basis": (facts or {}).get("class_basis"),
            "gps": (facts or {}).get("gps"),
            "inspected": facts is not None,
            "checks": checks,
            "score": qc.quality_score(checks),
            "auto_result": auto,
            "key_issues": qc.key_issues(checks),
            "explanation": qc.explain(checks),
            "decision": _decision_out(dec) if dec else None,
            "result": dec.decision if dec else auto,
            "contribution": _contribution_for(ev, world_id, maps, names),
        })
    items.reverse()  # newest first
    return items[:limit] if limit else items


@quality.get("/quality")
async def list_quality(db: AsyncSession = Depends(get_db)) -> dict:
    items = await _items(db)
    counts = {k: 0 for k in (*qc.DECISIONS, "not_inspected")}
    for i in items:
        counts[i["result"] or "not_inspected"] += 1
    n_nonphoto = len((await db.execute(select(Evidence.id).where(Evidence.type != "photo"))).all())
    return {
        "items": items,
        "counts": {"total": len(items), **counts},
        "not_screened_non_photo": n_nonphoto,
        "policy": _POLICY,
    }


@quality.get("/{evidence_id}/quality")
async def get_quality(evidence_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    items = await _items(db, only_id=evidence_id, limit=None)
    if not items:
        raise HTTPException(404, "Photo evidence not found")
    return {**items[0], "policy": _POLICY}


@quality.put("/{evidence_id}/decision")
async def put_decision(evidence_id: str, body: DecisionIn, db: AsyncSession = Depends(get_db)) -> dict:
    if await db.get(Evidence, evidence_id) is None:
        raise HTTPException(404, "Evidence not found")
    row = await db.get(EvidenceDecision, evidence_id)
    if row is None:
        row = EvidenceDecision(evidence_id=evidence_id, decision=body.decision)
        db.add(row)
    row.decision, row.reason, row.decided_at = body.decision, body.reason, utcnow()
    await db.commit()
    return {"evidence_id": evidence_id, **_decision_out(row)}
