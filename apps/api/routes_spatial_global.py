"""Read-only, tenant-scoped inspection of a world's global spatial reconstruction (`spatial_global` stage output).

Every route authorizes through the WORLD (``enforce_project_isolation``), then reads only from that world's spatial-global WorldStore
(``<WORLDSTORE_ROOT>/spatial-global/<team>/<world>/``). Version ids are validated by the store before they touch a path. Nothing is
written and nothing is recomputed: what a reader sees is exactly what the stage persisted -- observations vs entities, the epistemic
status and sigma of every dimension, identity decisions (carried / split / merge / ambiguous / lost), the coordinate frame and scale
state, and the quality axes (never one score).

  GET  /api/worlds/{id}/spatial-global                      versions, newest last, with status and validation state
  GET  /api/worlds/{id}/spatial-global/latest               summary of the newest version
  GET  /api/worlds/{id}/spatial-global/versions/{vid}       summary of one version
  GET  /api/worlds/{id}/spatial-global/entities/{eid}       one entity with its evidence resolved to raw observations (?version=)
  GET  /api/worlds/{id}/spatial-global/changes/{vid}        what changed against the parent: identity decisions + added/removed entities
  GET  /api/worlds/{id}/spatial-global/export               the newest world and its evidence, as one JSON document (?version=)
"""

from __future__ import annotations

import os

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import World
from apps.api.spatial_global_stage import SpatialGlobalError, open_store

spatial_global_router = APIRouter(prefix="/api/worlds/{world_id}/spatial-global", tags=["spatial-global"])

_BELIEF_DIMS = ("existence", "position", "dimensions", "orientation", "geometry", "topology", "semantic")


async def _store(db: AsyncSession, world_id: str, auth: AuthContext):
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)
    try:
        return open_store(os.environ.get("WORLDSTORE_ROOT", "./data/worldstore"), w.team_id, w.id)[0]
    except SpatialGlobalError:
        raise HTTPException(404, "World not found") from None


def _load(store, version_id: str | None):
    versions = store.list_versions()
    if not versions:
        raise HTTPException(404, "No spatial reconstruction exists for this world")
    vid = version_id or versions[-1].version_id
    if vid not in {v.version_id for v in versions}:
        raise HTTPException(404, "Version not found")
    try:
        return vid, store.load_version(vid)
    except Exception as exc:  # noqa: BLE001 - a damaged version is an ERROR, never an empty successful world
        raise HTTPException(500, f"version is unreadable: {exc}") from None


def _entity_row(e) -> dict:
    cp = e.custom_properties or {}
    beliefs = cp.get("spatial_beliefs") or {}
    return {"id": e.id, "type": e.type.value, "storey_id": cp.get("storey_id"), "host_wall_id": cp.get("host_wall_id"),
            "plan_segment": cp.get("plan_segment"), "z_extent_known": cp.get("z_extent_known"),
            "provenance": e.provenance.value, "coordinate_frame": cp.get("coordinate_frame"), "scale_state": cp.get("scale_state"),
            "evidence_count": len(cp.get("evidence_ids") or ()), "views": len(cp.get("image_ids") or ()),
            "beliefs": {d: {"status": b.get("status"), "sigma": b.get("sigma"), "confidence": b.get("confidence")}
                        for d, b in beliefs.items() if d in _BELIEF_DIMS},
            "has_alternatives": bool(cp.get("alternatives"))}


def _summary(vid: str, w) -> dict:
    md = w.metadata or {}
    facts = md.get("spatial_facts") or {}
    report = md.get("spatial_compile_report") or {}
    ents = [_entity_row(e) for e in w.entities.values()]
    return {"version_id": vid, "frame": md.get("spatial_frame"), "input_hash": md.get("spatial_input_hash"),
            "status": facts.get("status"), "validation": facts.get("validation"), "topology_status": report.get("topology_status"),
            "quality": facts.get("quality"), "optimization": facts.get("optimization"), "warnings": facts.get("warnings"),
            "identity": (md.get("spatial_identity") or {}).get("carried"),
            "excluded_tentative": report.get("excluded_tentative"), "completeness": report.get("completeness"),
            "counts": {"entities": len(ents), "geometries": len(w.geometries)}, "entities": ents,
            "rooms": [{"id": e["id"]} for e in ents if e["type"] == "room"]}


@spatial_global_router.get("")
async def list_versions(world_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> dict:
    store = await _store(db, world_id, auth)
    out = []
    for v in store.list_versions():
        try:
            md = store.load_version(v.version_id).metadata or {}
            facts = md.get("spatial_facts") or {}
            out.append({"version_id": v.version_id, "parent": v.parent, "status": facts.get("status"),
                        "validation_state": (facts.get("validation") or {}).get("state"), "input_hash": md.get("spatial_input_hash")})
        except Exception:  # noqa: BLE001 - one damaged version is listed as such; it neither hides nor breaks the others
            out.append({"version_id": v.version_id, "parent": v.parent, "status": "unreadable"})
    return {"world_id": world_id, "versions": out}


@spatial_global_router.get("/latest")
async def latest(world_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> dict:
    vid, w = _load(await _store(db, world_id, auth), None)
    return _summary(vid, w)


@spatial_global_router.get("/versions/{version_id}")
async def version(world_id: str, version_id: str, db: AsyncSession = Depends(get_db),
                  auth: AuthContext = Depends(get_auth_context)) -> dict:
    vid, w = _load(await _store(db, world_id, auth), version_id)
    return _summary(vid, w)


@spatial_global_router.get("/entities/{entity_id}")
async def entity(world_id: str, entity_id: str, version: str | None = Query(default=None), db: AsyncSession = Depends(get_db),
                 auth: AuthContext = Depends(get_auth_context)) -> dict:
    vid, w = _load(await _store(db, world_id, auth), version)
    e = w.entities.get(entity_id)
    if e is None:
        raise HTTPException(404, "Entity not found")
    md = w.metadata or {}
    cp = e.custom_properties or {}
    evidence = {o["evidence_id"]: o for o in md.get("spatial_evidence", [])}
    raw = (md.get("spatial_optimization") or {}).get("raw") or {}
    poses = (md.get("spatial_optimization") or {}).get("poses") or {}
    cited = []
    for ev in cp.get("evidence_ids") or ():
        r = raw.get(ev) or {}
        cited.append({"evidence_id": ev, "observation": evidence.get(ev), "raw_image_frame": {k: r.get(k) for k in ("p0", "p1", "sigma_m")} if r else None,
                      "image_pose": poses.get(r.get("image_id")) if r else None})
    ident = (md.get("spatial_identity") or {})
    return {"version_id": vid, "entity": _entity_row(e), "beliefs": cp.get("spatial_beliefs"), "sigma_parts": cp.get("sigma_parts"),
            "alternatives": cp.get("alternatives"), "end_observed": {"a": cp.get("end_a_observed"), "b": cp.get("end_b_observed")},
            "evidence": cited, "identity_history": (ident.get("history") or {}).get(entity_id),
            "identity_decisions": [d for d in ident.get("decisions", []) if entity_id in (d.get("entity"), d.get("inherits"))],
            "relationships": [{"kind": r.kind.value if hasattr(r.kind, "value") else str(r.kind), "target": r.target_id} for r in e.relationships]}


@spatial_global_router.get("/changes/{version_id}")
async def changes(world_id: str, version_id: str, db: AsyncSession = Depends(get_db),
                  auth: AuthContext = Depends(get_auth_context)) -> dict:
    store = await _store(db, world_id, auth)
    vid, w = _load(store, version_id)
    parent = next((v.parent for v in store.list_versions() if v.version_id == vid), None)
    ident = (w.metadata or {}).get("spatial_identity") or {}
    out = {"version_id": vid, "parent": parent, "identity_decisions": ident.get("decisions", []), "added": [], "removed": []}
    if parent:
        _, pw = _load(store, parent)
        out["added"] = sorted(set(w.entities) - set(pw.entities))
        out["removed"] = sorted(set(pw.entities) - set(w.entities))
    return out


@spatial_global_router.get("/export")
async def export(world_id: str, version: str | None = Query(default=None), db: AsyncSession = Depends(get_db),
                 auth: AuthContext = Depends(get_auth_context)) -> dict:
    vid, w = _load(await _store(db, world_id, auth), version)
    return {"version_id": vid, "world": w.to_dict()}


class CorrectionBody(BaseModel):
    """One user judgment on the latest spatial version. kind: `dimension` (a stated real length/area, needs the world's metric scale),
    `reclassify` (entity class), `exclude` (this observation, or all evidence of this entity, is wrong)."""
    kind: str = Field(pattern="^(dimension|reclassify|exclude)$")
    entity_id: str | None = None
    evidence_id: str | None = None
    property: str | None = None
    value: float | None = None
    new_class: str | None = None
    note: str = ""


@spatial_global_router.post("/corrections")
async def add_correction(world_id: str, body: CorrectionBody, db: AsyncSession = Depends(get_db),
                         auth: AuthContext = Depends(get_auth_context)) -> dict:
    """Records the judgment as a NEW child version of the latest spatial version (history is never rewritten). Dimension and class
    corrections reuse world_ir.corrections and are replayed onto every later run; an exclusion removes the observation(s) from the
    next optimisation (the originals stay in the world's record) and changes the input hash, so the next job reprocesses."""
    from world_ir.corrections import (
        CorrectionError,
        make_correction,
        reclassify_entity,
        record_correction,
    )
    store = await _store(db, world_id, auth)
    vid, w = _load(store, None)
    try:
        if body.kind == "dimension":
            if not body.entity_id or body.property is None or body.value is None:
                raise CorrectionError("dimension needs entity_id, property and value")
            record_correction(w, make_correction(w, body.entity_id, body.property, body.value, body.note))
        elif body.kind == "reclassify":
            if not body.entity_id or not body.new_class:
                raise CorrectionError("reclassify needs entity_id and new_class")
            reclassify_entity(w, body.entity_id, body.new_class, body.note, user_id=auth.user_id if hasattr(auth, "user_id") else "")
        else:
            if bool(body.evidence_id) == bool(body.entity_id):
                raise CorrectionError("exclude needs exactly one of evidence_id or entity_id")
            if body.entity_id and body.entity_id not in w.entities:
                raise CorrectionError(f"entity '{body.entity_id}' is not in this world")
            known = {o["evidence_id"] for o in (w.metadata or {}).get("spatial_evidence", [])}
            if body.evidence_id and body.evidence_id not in known:
                raise CorrectionError(f"evidence '{body.evidence_id}' is not in this world")
            rec = {k: v for k, v in (("evidence_id", body.evidence_id), ("entity_id", body.entity_id)) if v}
            rec["note"] = body.note or None
            w.metadata["spatial_exclusions"] = list(w.metadata.get("spatial_exclusions") or []) + [rec]
    except CorrectionError as exc:
        raise HTTPException(422, str(exc)) from None
    new = store.save_version(w, parent=vid)
    return {"version_id": new.version_id, "parent": vid, "kind": body.kind,
            "note": "applied to this version" if body.kind != "exclude" else "takes effect on the next reconstruction run"}


@spatial_global_router.get("/corrections")
async def list_corrections(world_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> dict:
    _, w = _load(await _store(db, world_id, auth), None)
    md = w.metadata or {}
    return {"corrections": md.get("user_corrections") or [], "decisions": md.get("user_decisions") or [],
            "exclusions": md.get("spatial_exclusions") or [], "excluded_evidence": len(md.get("spatial_excluded_evidence") or [])}
