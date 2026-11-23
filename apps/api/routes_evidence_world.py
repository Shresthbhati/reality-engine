"""Read-only, tenant-scoped inspection of a world's evidence-to-world runs.

Every route authorizes through the WORLD (``enforce_project_isolation``: a non-admin tenant gets 404 for another tenant's world, and a
world-scoped token only reaches its own world), then reads only from that world's directory
``<WORLDSTORE_ROOT>/evidence-world/<team>/<world>/<run>/``. Run ids and artifact digests are validated before they touch a path.

Together these routes let a reader walk an entity back to the original images and the model predictions that support it:
``entities/{id}`` returns the supporting observation rows (engine, model version, checkpoint, source image, region, raw vs calibrated
confidence), the rejected / excluded ones with reasons, the association decisions, conflicts and the matrix re-verification.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.evidence_world_stage import EvidenceWorldError, _segment, world_root
from apps.api.models import World
from engine.multiengine.evidence_flow import SNAPSHOT_FILE
from engine.multiengine.evidence_matrix import EvidenceMatrix, MatrixError
from engine.multiengine.worldir_bridge import trace
from world_ir.schema_v1 import Entity

evidence_world_router = APIRouter(prefix="/api/worlds/{world_id}/evidence-world", tags=["evidence-world"])

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


async def _world_dir(db: AsyncSession, world_id: str, auth: AuthContext) -> Path:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)
    try:
        return world_root(w.team_id, w.id)
    except EvidenceWorldError:
        raise HTTPException(404, "World not found") from None


def _run_dir(base: Path, run_id: str) -> Path:
    try:
        d = base / _segment(run_id, "run id")
    except EvidenceWorldError:
        raise HTTPException(404, "Run not found") from None
    if not (d / SNAPSHOT_FILE).is_file():
        raise HTTPException(404, "Run not found")
    _snapshot(d)  # every route refuses a run whose snapshot is damaged, including those that only read the matrix
    return d


def _snapshot(d: Path) -> dict[str, Any]:
    try:
        s = json.loads((d / SNAPSHOT_FILE).read_text(encoding="utf-8"))
        if not isinstance(s, dict) or "status" not in s or "counts" not in s:
            raise ValueError("not an evidence-world snapshot")
        return s
    except (OSError, ValueError) as exc:  # a damaged snapshot is an ERROR, never an empty successful world
        raise HTTPException(500, f"run snapshot is unreadable: {exc}") from None


def _matrix(d: Path) -> EvidenceMatrix:
    try:
        return EvidenceMatrix(d)
    except (MatrixError, RuntimeError) as exc:
        raise HTTPException(500, f"evidence matrix could not be read: {exc}") from None


@evidence_world_router.get("")
async def list_runs(world_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> dict:
    base = await _world_dir(db, world_id, auth)
    runs = []
    if base.is_dir():
        for d in sorted(p for p in base.iterdir() if (p / SNAPSHOT_FILE).is_file()):
            try:
                s = _snapshot(d)
            except HTTPException:  # one damaged run is listed as such; it neither hides nor breaks the others
                runs.append({"run_id": d.name, "status": "unreadable", "counts": None})
                continue
            runs.append({"run_id": s["run_id"], "status": s["status"], "counts": s["counts"]})
    return {"world_id": world_id, "runs": runs}


@evidence_world_router.get("/{run_id}")
async def get_run(
    world_id: str, run_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    s = _snapshot(_run_dir(await _world_dir(db, world_id, auth), run_id))
    keys = ("run_id", "status", "sources", "counts", "stages", "limitations", "bridge_notes", "matrix_problems")
    return {k: s[k] for k in keys}


@evidence_world_router.get("/{run_id}/readiness")
async def get_readiness(
    world_id: str, run_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    """Honest product-facing state of one run, derived only from persisted data (see ``evidence_world_stage.readiness``)."""
    from apps.api.evidence_world_stage import readiness

    return readiness(_run_dir(await _world_dir(db, world_id, auth), run_id))


@evidence_world_router.get("/{run_id}/world")
async def get_persisted_world(
    world_id: str, run_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    """The persisted, verified canonical WorldIR of the run. No persisted world is a 404 -- never an empty successful world."""
    from apps.api.evidence_world_stage import WORLDSTORE_REL, head_version
    from worldstore.store import WorldStore, WorldStoreError

    d = _run_dir(await _world_dir(db, world_id, auth), run_id)
    vid = head_version(d)
    if not vid:
        raise HTTPException(404, "No persisted world for this run")
    store = WorldStore(d / WORLDSTORE_REL)
    try:
        problems = store.verify_version(vid)
        if problems:
            raise HTTPException(500, f"persisted world failed verification: {problems[:3]}")
        return {"version_id": vid, "world": store.load_version(vid).to_dict()}
    except WorldStoreError as exc:
        raise HTTPException(500, f"persisted world is unreadable: {exc}") from None


@evidence_world_router.get("/{run_id}/manifest")
async def get_manifest(
    world_id: str, run_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    """The run's reproducibility manifest plus a fresh integrity check against the disk (inputs, manifest hash, persisted world hash)."""
    from engine.multiengine import run_manifest

    d = _run_dir(await _world_dir(db, world_id, auth), run_id)
    f = d / run_manifest.MANIFEST_FILE
    if not f.is_file():
        raise HTTPException(404, "No manifest for this run")
    try:
        return {"manifest": json.loads(f.read_text(encoding="utf-8")), "verification": run_manifest.verify(d)}
    except ValueError:
        raise HTTPException(500, "manifest is unreadable") from None


class ExportBody(BaseModel):
    format: str = Field(min_length=1, max_length=40)
    acknowledge_unscaled: bool = False


_EXPORT_MEDIA = {"gltf": "model/gltf+json", "glb": "model/gltf-binary", "obj": "model/obj", "ply": "application/x-ply", "stl": "model/stl",
                 "usda": "text/plain", "geojson": "application/geo+json", "cityjson": "application/json"}


@evidence_world_router.post("/{run_id}/export")
async def export_run_world(
    world_id: str, run_id: str, body: ExportBody, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> Response:
    """Export the run's persisted world ONLY when it meets the export requirements, derived from persisted data (``readiness``):
    409 when no entity has a measured position (nothing to place), 409 when the geometry is not in metres unless the caller sets
    ``acknowledge_unscaled`` (the file then carries unitless/gauge values and the response says so in ``X-Export-Scale``). The exporter's
    own skipped entities are reported in headers, never hidden."""
    from apps.api.evidence_world_stage import WORLDSTORE_REL, head_version, readiness
    from sdk import reality
    from world_ir.artifact_store import FileArtifactStore
    from worldstore.store import WorldStore

    d = _run_dir(await _world_dir(db, world_id, auth), run_id)
    rd = readiness(d)
    q = rd.get("quality") or {}
    if q.get("verdict") in ("FAILED", "BLOCKED"):  # the measured gate decides; PARTIAL stays exportable but is labelled in the headers
        raise HTTPException(409, f"not exportable: quality verdict {q['verdict']}: {'; '.join(q.get('reasons') or ['no reason recorded'])}")
    if not rd.get("export_ready"):
        raise HTTPException(409, f"not exportable: {rd.get('export_not_ready_reason') or rd.get('reason') or rd['world'].get('reason') or 'readiness gate not met'}")
    if not rd.get("metric_export_ready") and not body.acknowledge_unscaled:
        raise HTTPException(409, f"{rd.get('export_scale_warning')} Re-send with acknowledge_unscaled=true to export gauge-unit geometry.")
    vid = head_version(d)
    world = WorldStore(d / WORLDSTORE_REL).load_version(vid)
    from engine.multiengine.world_scale import scale_block

    sb = scale_block(world)
    try:
        content, report = reality.export(world, body.format, artifact_store=FileArtifactStore(d / "export-artifacts"))
    except reality.UnsupportedExportFormatError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ImportError as exc:
        raise HTTPException(501, f"{body.format} export is unavailable on this server: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    data = json.dumps(content, sort_keys=True).encode("utf-8") if isinstance(content, dict) else content.encode("utf-8") if isinstance(content, str) else content
    return Response(content=data, media_type=_EXPORT_MEDIA.get(body.format, "application/octet-stream"), headers={
        "X-Export-Version": vid, "X-Export-Scale": "metric" if rd.get("metric_export_ready") else "sfm_units-NOT-metres",
        "X-Scale-Units": sb["units"], "X-Scale-Calibration": str(sb["calibration_status"]),
        "X-Entities-Exported": str(len(report.entities_exported)), "X-Entities-Skipped": str(len(report.entities_skipped)),
        "X-Physical-Accuracy": str(rd.get("physical_accuracy")), "X-Quality-Verdict": str(q.get("verdict", "UNKNOWN"))})


class DistanceAnchorBody(BaseModel):
    anchor_id: str = Field(min_length=1, max_length=120)
    point_a: list[float] = Field(min_length=3, max_length=3)  # reconstruction (gauge) coordinates of the first endpoint
    point_b: list[float] = Field(min_length=3, max_length=3)
    measured: float
    sigma: float
    unit: str = Field(max_length=4)
    source: str = Field(max_length=200)
    measured_at: str = Field(max_length=40)
    provenance: str = Field(max_length=500)
    endpoint_sigma_gauge: float = 0.0
    held_out: bool = False


class PointAnchorBody(BaseModel):
    anchor_id: str = Field(min_length=1, max_length=120)
    recon_xyz: list[float] = Field(min_length=3, max_length=3)
    physical_xyz: list[float] = Field(min_length=3, max_length=3)  # metres
    sigma_m: float
    source: str = Field(max_length=200)
    measured_at: str = Field(max_length=40)
    provenance: str = Field(max_length=500)
    held_out: bool = False


class CalibrateBody(BaseModel):
    distance_anchors: list[DistanceAnchorBody] = Field(default_factory=list, max_length=200)
    point_anchors: list[PointAnchorBody] = Field(default_factory=list, max_length=500)
    expected_version_id: str | None = Field(None, max_length=200)


@evidence_world_router.post("/{run_id}/calibrate")
async def calibrate_run_world(
    world_id: str, run_id: str, body: CalibrateBody, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    """Physically anchor the run's world from independent measurements. The anchors are fitted and tested on held-out anchors
    (``scale_anchors``); only a calibration that is ``validated`` or ``unvalidated`` (>= 2 consistent anchors, none held out) is applied, and
    it is applied as a NEW immutable version whose parent is the gauge version (``world_scale``). Anything else is refused (409) with the
    full calibration record -- insufficient or failed anchors never change the world. A world that is already calibrated is refused (409):
    scaling it again would multiply the scale twice."""
    from apps.api.evidence_world_stage import WORLDSTORE_REL, head_version
    from engine.multiengine.scale_anchors import AnchorError, DistanceAnchor, PointAnchor, calibrate_distances, calibrate_points
    from engine.multiengine.world_scale import AlreadyCalibrated, CalibrationRefused, CAL_KEY, commit_calibrated_version
    from worldstore.store import WorldStore, WorldStoreError

    d = _run_dir(await _world_dir(db, world_id, auth), run_id)
    head = head_version(d)
    if not head:
        raise HTTPException(404, "No persisted world for this run")
    _require_fresh(head, body.expected_version_id)
    if bool(body.distance_anchors) == bool(body.point_anchors):
        raise HTTPException(422, "send either distance_anchors or point_anchors (exactly one kind), not both and not neither")
    store = WorldStore(d / WORLDSTORE_REL)
    head_cal = (store.load_version(head).metadata or {}).get(CAL_KEY)
    try:
        if body.distance_anchors:
            cal = calibrate_distances([DistanceAnchor(**{**a.model_dump(), "point_a": tuple(a.point_a), "point_b": tuple(a.point_b)}) for a in body.distance_anchors])
        else:
            cal = calibrate_points([PointAnchor(**{**a.model_dump(), "recon_xyz": tuple(a.recon_xyz), "physical_xyz": tuple(a.physical_xyz)}) for a in body.point_anchors])
    except AnchorError as exc:
        raise HTTPException(422, f"invalid anchor: {exc}") from exc
    if head_cal:
        from engine.multiengine.world_scale import calibration_id

        if cal.applicable and calibration_id(cal, head_cal["parent_version"]) == head_cal["id"]:  # the same request again: a retry, not a second scaling
            return {"version_id": head, "parent_version_id": head_cal["parent_version"], "status": cal.status, "scale_m_per_gauge": cal.scale_m_per_gauge,
                    "scale_sigma": cal.scale_sigma, "calibration_id": head_cal["id"], "applied": head_cal["applied"], "warnings": cal.warnings,
                    "idempotent": True, "calibration": cal.to_dict()}
        raise HTTPException(409, "this world already carries a different physical calibration; it will not be scaled twice (calibrate the gauge version instead)")
    if not cal.applicable:
        raise HTTPException(409, {"message": f"calibration {cal.status}: {cal.reason}; the world was not changed", "calibration": cal.to_dict()})
    try:
        vid, record = commit_calibrated_version(store, head, cal, artifacts=_matrix(d).artifacts)
    except AlreadyCalibrated as exc:
        raise HTTPException(409, str(exc)) from exc
    except (CalibrationRefused, WorldStoreError) as exc:
        raise HTTPException(422, f"the calibration could not be applied and no version was saved: {exc}") from exc
    except OSError as exc:  # disk full / permission / I/O while persisting: nothing was adopted, the request is safe to retry
        raise HTTPException(503, f"the calibrated version could not be persisted and nothing was changed; retry: {exc}") from exc
    return {"version_id": vid, "parent_version_id": head, "status": cal.status, "scale_m_per_gauge": cal.scale_m_per_gauge,
            "scale_sigma": cal.scale_sigma, "calibration_id": record["id"], "applied": record["applied"], "warnings": cal.warnings,
            "idempotent": False, "calibration": cal.to_dict()}


class ProposeBody(BaseModel):
    note: str = Field("", max_length=1000)


@evidence_world_router.post("/{run_id}/propose", status_code=201)
async def propose_run_world(
    world_id: str, run_id: str, body: ProposeBody, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    """Propose the run's evidence entities to the world's committed HEAD through the adoption protocol (section 6 of the design
    decisions). NOTHING is committed here: the result is a pending change proposal (``/api/worlds/{id}/proposals``) whose per-entity
    assessments the user reviews and accepts, entity by entity. The candidate is HEAD plus the evidence entities, geometries and
    materials; HEAD's own entities are untouched.

    Refused (409) unless the evidence world's frame and units are PROVEN to be HEAD's: HEAD must have been committed by the very job
    that produced this run (one COLMAP model, one canonical frame, one scale state), the persisted evidence version must verify, and no
    id may collide with HEAD's. A HEAD from another job could be in another frame, and mixing frames would place objects wrongly."""
    from apps.api import worldstore_service
    from apps.api.evidence_world_stage import WORLDSTORE_REL, committed_version, head_version
    from apps.api.models import WorldVersion
    from worldstore.proposals import ProposalError, propose_update
    from worldstore.store import WorldStore
    from world_ir.world_v1 import WorldIR

    d = _run_dir(await _world_dir(db, world_id, auth), run_id)
    w = await db.get(World, world_id)
    if not w.current_version_id:
        raise HTTPException(409, "the world has no committed version to propose against")
    head_row = await db.get(WorldVersion, w.current_version_id)
    recorded = committed_version(d)
    if head_row is None or recorded != w.current_version_id:
        raise HTTPException(409, f"frame not proven: HEAD is {w.current_version_id}, but this run's evidence accompanied "
                                 f"{recorded or 'no recorded version'}; evidence from another solve (or a world that has evolved since) may be in a "
                                 "different frame or scale, so it is not mixed in")
    vid = head_version(d)
    if not vid:
        raise HTTPException(409, "this run has no persisted evidence world")
    ev_store = WorldStore(d / WORLDSTORE_REL)
    problems = ev_store.verify_version(vid)
    if problems:
        raise HTTPException(409, f"the persisted evidence world fails verification: {problems[:3]}")
    evidence = ev_store.load_version(vid)
    store = worldstore_service.get_store()
    candidate = WorldIR.from_dict(store.load_version(w.current_version_id).to_dict())
    clash = sorted((set(evidence.entities) & set(candidate.entities)) | (set(evidence.geometries) & set(candidate.geometries))
                   | (set(evidence.materials) & set(candidate.materials)))
    if clash:
        raise HTTPException(409, f"ids already exist in HEAD, nothing proposed: {clash[:5]}")
    scale_states = sorted({str((e.transform or {}).get("scale_state") or "undeclared") for e in evidence.entities.values()})
    for k, e in evidence.entities.items():
        ent_copy = Entity.from_dict(e.to_dict())
        ent_copy.custom_properties = {**(ent_copy.custom_properties or {}), "promoted_from_evidence_run": run_id, "evidence_world_version": vid}
        candidate.entities[k] = ent_copy
    candidate.geometries.update(evidence.geometries)
    candidate.materials.update(evidence.materials)
    try:
        p = propose_update(store, w.current_version_id, candidate, list(head_row.source_session_ids or []))
    except ProposalError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"proposal": p.to_dict(), "added_entities": len(evidence.entities), "evidence_version": vid,
            "frame_check": {"basis": "HEAD was committed by the same job as this evidence run (one COLMAP model and canonical frame)",
                            "head_version": w.current_version_id, "evidence_scale_states": scale_states, "note": body.note.strip() or None},
            "next": f"review, then POST /api/worlds/{world_id}/proposals/{p.proposal_id}/accept (entity_ids to pick a subset)"}


def _refuse_if_calibrated(d: Path, head: str) -> None:
    """A correction REBUILDS an entity from the raw (gauge-scale) evidence. On a metric-calibrated version that would put a gauge entity
    next to metric ones, so it is refused: correct the gauge version first, then calibrate (calibration is the last step)."""
    from apps.api.evidence_world_stage import WORLDSTORE_REL
    from worldstore.store import WorldStore

    if (WorldStore(d / WORLDSTORE_REL).load_version(head).metadata or {}).get("scale_calibration"):
        raise HTTPException(409, "this version is metric-calibrated and corrections are rebuilt from gauge-scale evidence; make corrections on the "
                                 "gauge version, then calibrate")


def _require_fresh(head: str, expected: str | None) -> None:
    """A correction made while looking at version X must not silently land on a newer version: when the caller says which version it
    saw (``expected_version_id``) and HEAD has moved, refuse (409) so the user reloads and re-decides."""
    if expected and expected != head:
        raise HTTPException(409, f"stale: this correction was made against version {expected} but the run's HEAD is {head}; reload and retry")


class MergeBody(BaseModel):
    other_entity_id: str = Field(min_length=1, max_length=200)
    note: str = Field("", max_length=1000)
    acknowledge_same_image: bool = False
    expected_version_id: str | None = Field(None, max_length=200)


@evidence_world_router.post("/{run_id}/entities/{entity_id}/merge")
async def merge_entities_route(
    world_id: str, run_id: str, entity_id: str, body: MergeBody, db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """User identity correction: "``other_entity_id`` is the same physical object as ``entity_id``". The merged entity is REBUILT from the
    raw evidence of both (position / dimensions / class / material are re-determined from the union of their observations in the Evidence
    Matrix, not copied), keeps ``entity_id`` (stable id), and is saved as a NEW immutable version; the absorbed entity's relationships are
    re-pointed, the user's decision is recorded, and the snapshot / matrix are never modified. Refused (422) when the two share an image
    (one object cannot appear twice in one photograph) unless ``acknowledge_same_image`` is set, and when the rebuilt world would be invalid."""
    import hashlib

    from apps.api.evidence_world_stage import WORLDSTORE_REL, head_version
    from engine.multiengine.association import AssociatedEntity
    from engine.multiengine.canonical import build_canonical_entity
    from engine.multiengine.worldir_bridge import convert
    from engine.multiengine.api_facade import get_default_ontology
    from world_ir.validation import validate_world_ir
    from worldstore.store import WorldStore, WorldStoreError

    d = _run_dir(await _world_dir(db, world_id, auth), run_id)
    head = head_version(d)
    if not head:
        raise HTTPException(404, "No persisted world for this run")
    _require_fresh(head, getattr(body, "expected_version_id", None))
    _refuse_if_calibrated(d, head)
    if body.other_entity_id == entity_id:
        raise HTTPException(422, "an entity cannot be merged with itself")
    store = WorldStore(d / WORLDSTORE_REL)
    try:
        world = store.load_version(head)
        ka, kb = f"ent-{entity_id}", f"ent-{body.other_entity_id}"
        ea, eb = world.entities.get(ka), world.entities.get(kb)
        if ea is None or eb is None:
            raise HTTPException(404, "Entity not found")
        ev_a, ev_b = (ea.custom_properties.get("evidence") or {}), (eb.custom_properties.get("evidence") or {})
        if not ev_a.get("members") or not ev_b.get("members"):
            raise HTTPException(422, "only entities built from evidence can be merged")
        matrix = _matrix(d)
        union = sorted(set(ev_a["members"]) | set(ev_b["members"]))
        bbox_ids = [m for m in union if matrix.store.get(m).quantity == "bbox"]
        by_source: dict[str, int] = {}
        for m in bbox_ids:
            src = matrix.store.get(m).source_id
            by_source[src] = by_source.get(src, 0) + 1
        shared = sorted({matrix.store.get(m).source_id for m in ev_a["members"] if matrix.store.get(m).quantity == "bbox"}
                        & {matrix.store.get(m).source_id for m in ev_b["members"] if matrix.store.get(m).quantity == "bbox"})
        if shared and not body.acknowledge_same_image:
            raise HTTPException(422, f"both entities were detected in the same photograph(s) {shared}: one object cannot appear twice in one image. "
                                     "Re-send with acknowledge_same_image=true if these are duplicate detections of one object.")
        labels = {m: str(matrix.store.get(m).predicted["label"]) for m in bbox_ids if isinstance(matrix.store.get(m).predicted.get("label"), str)}
        onto = get_default_ontology()
        ae = AssociatedEntity(entity_key=entity_id, members=tuple(union), labels=tuple(sorted(set(labels.values()))),
                              sources=tuple(sorted({matrix.store.get(m).source_id for m in union})))
        ce = build_canonical_entity(ae, matrix, target_frame="world", tolerances={}, labels=labels, registry=None, ontology=onto, association=None)
        rebuilt = convert([ce], matrix, ontology=onto)
        new_ent = rebuilt.entities[ka]
        decision = {"kind": "merge", "basis": "user", "entity_id": ka, "absorbed_entity_id": kb, "note": body.note.strip() or None,
                    "user_id": str(getattr(auth, "user_id", "") or "user"), "members_before": {ka: ev_a["members"], kb: ev_b["members"]},
                    "acknowledged_same_image": bool(shared)}
        new_ent.custom_properties["user_decision"] = decision
        new_ent.custom_properties["merged_from"] = [kb]
        for gid in list(ea.geometry_ids) + list(eb.geometry_ids):
            world.geometries.pop(gid, None)
        del world.entities[kb]
        world.entities[ka] = new_ent
        world.geometries.update(rebuilt.geometries)
        world.materials.update(rebuilt.materials)
        for e in world.entities.values():  # relationships that pointed at the absorbed entity now point at the survivor
            seen = set()
            fixed = []
            for r in e.relationships:
                if r.target_id == kb:
                    r.target_id = ka
                key = (r.kind, r.target_id)
                if e.id == r.target_id or key in seen:
                    continue
                seen.add(key)
                fixed.append(r)
            e.relationships = fixed
        world.metadata["user_decisions"] = list(world.metadata.get("user_decisions") or []) + [decision]
        errors = [str(i) for i in validate_world_ir(world).errors]
        if errors:
            raise HTTPException(422, f"the merged world would be invalid and was not saved: {errors[:3]}")
        vid = "v-" + hashlib.sha256(json.dumps(world.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:16]
        store.save_version(world, parent=head, version_id=vid)
        problems = store.verify_version(vid)
        if problems or store.load_version(vid).to_dict() != world.to_dict():
            raise HTTPException(500, f"the merged version failed verification: {problems[:3]}")
    except WorldStoreError as exc:
        raise HTTPException(409, f"could not save the merged version: {exc}") from None
    pos, dims = ce.determinations.get("position"), ce.determinations.get("dimensions")
    return {"version_id": vid, "parent_version_id": head, "entity_id": entity_id, "absorbed_entity_id": body.other_entity_id,
            "members": len(union), "rebuilt_from_evidence": True, "position_status": pos.status if pos else "absent",
            "dimensions_status": dims.status if dims else "absent", "decision": decision}


class SplitBody(BaseModel):
    detach_observation_ids: list[str] = Field(min_length=1, max_length=200)
    note: str = Field("", max_length=1000)
    expected_version_id: str | None = Field(None, max_length=200)


@evidence_world_router.post("/{run_id}/entities/{entity_id}/split")
async def split_entity_route(
    world_id: str, run_id: str, entity_id: str, body: SplitBody, db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """User identity correction: "these detections are NOT the same object as the rest". The listed detection observations move to a NEW
    entity; every observation derived from a moved detection (its lifted position / dimensions, its material claim: same image + same box)
    moves with it. BOTH entities are rebuilt from their own raw evidence, the original keeps its id, the decision is recorded, and a new
    immutable version is saved. Refused when nothing would stay behind, when an id is not one of the entity's detections, or when the rebuilt
    world would be invalid. The new entity starts without relationships: nothing is inferred about it."""
    import hashlib

    from apps.api.evidence_world_stage import WORLDSTORE_REL, head_version
    from engine.multiengine.association import AssociatedEntity
    from engine.multiengine.canonical import build_canonical_entity
    from engine.multiengine.worldir_bridge import convert
    from engine.multiengine.api_facade import get_default_ontology
    from world_ir.validation import validate_world_ir
    from worldstore.store import WorldStore, WorldStoreError

    d = _run_dir(await _world_dir(db, world_id, auth), run_id)
    head = head_version(d)
    if not head:
        raise HTTPException(404, "No persisted world for this run")
    _require_fresh(head, getattr(body, "expected_version_id", None))
    _refuse_if_calibrated(d, head)
    store = WorldStore(d / WORLDSTORE_REL)
    try:
        world = store.load_version(head)
        ka = f"ent-{entity_id}"
        ea = world.entities.get(ka)
        if ea is None:
            raise HTTPException(404, "Entity not found")
        members = list((ea.custom_properties.get("evidence") or {}).get("members") or [])
        if not members:
            raise HTTPException(422, "only entities built from evidence can be split")
        matrix = _matrix(d)
        bbox = [m for m in members if matrix.store.get(m).quantity == "bbox"]
        detach = set(body.detach_observation_ids)
        if not detach <= set(bbox):
            raise HTTPException(422, f"not detections of this entity: {sorted(detach - set(bbox))[:5]}")
        if len(detach) == len(bbox):
            raise HTTPException(422, "nothing would stay in the original entity; to re-label it use reclassify")
        region = {(matrix.store.get(m).source_id, tuple(matrix.store.get(m).image_region)): m for m in bbox if matrix.store.get(m).image_region is not None}
        moved = set(detach)
        for m in members:  # derived observations follow their detection (same image + same box)
            if m in bbox:
                continue
            o = matrix.store.get(m)
            owner = region.get((o.source_id, tuple(o.image_region))) if o.image_region is not None else None
            if owner in detach:
                moved.add(m)
        stay = [m for m in members if m not in moved]
        new_key = "ent_" + hashlib.sha256("|".join(sorted(moved)).encode("utf-8")).hexdigest()[:24]
        onto = get_default_ontology()

        def build(key: str, ids: list[str]):
            lab = {m: str(matrix.store.get(m).predicted["label"]) for m in ids
                   if matrix.store.get(m).quantity == "bbox" and isinstance(matrix.store.get(m).predicted.get("label"), str)}
            ae = AssociatedEntity(entity_key=key, members=tuple(sorted(ids)), labels=tuple(sorted(set(lab.values()))),
                                  sources=tuple(sorted({matrix.store.get(m).source_id for m in ids})))
            return build_canonical_entity(ae, matrix, target_frame="world", tolerances={}, labels=lab, registry=None, ontology=onto, association=None)

        ce_stay, ce_new = build(entity_id, stay), build(new_key, sorted(moved))
        rebuilt = convert([ce_stay, ce_new], matrix, ontology=onto)
        kn = f"ent-{new_key}"
        decision = {"kind": "split", "basis": "user", "entity_id": ka, "new_entity_id": kn, "note": body.note.strip() or None,
                    "user_id": str(getattr(auth, "user_id", "") or "user"), "moved_observation_ids": sorted(moved), "kept_observation_ids": sorted(stay)}
        old_rels = list(ea.relationships)
        for gid in ea.geometry_ids:
            world.geometries.pop(gid, None)
        stayed, born = rebuilt.entities[ka], rebuilt.entities[kn]
        stayed.relationships = [r for r in old_rels if r.target_id != kn]  # the survivor keeps its claims; the new entity inherits none
        stayed.custom_properties["user_decision"] = decision
        born.custom_properties["user_decision"] = decision
        born.custom_properties["split_from"] = ka
        world.entities[ka], world.entities[kn] = stayed, born
        world.geometries.update(rebuilt.geometries)
        world.materials.update(rebuilt.materials)
        world.metadata["user_decisions"] = list(world.metadata.get("user_decisions") or []) + [decision]
        errors = [str(i) for i in validate_world_ir(world).errors]
        if errors:
            raise HTTPException(422, f"the split world would be invalid and was not saved: {errors[:3]}")
        vid = "v-" + hashlib.sha256(json.dumps(world.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:16]
        store.save_version(world, parent=head, version_id=vid)
        problems = store.verify_version(vid)
        if problems or store.load_version(vid).to_dict() != world.to_dict():
            raise HTTPException(500, f"the split version failed verification: {problems[:3]}")
    except WorldStoreError as exc:
        raise HTTPException(409, f"could not save the split version: {exc}") from None
    return {"version_id": vid, "parent_version_id": head, "entity_id": entity_id, "new_entity_id": new_key, "kept_members": len(stay),
            "moved_members": len(moved), "rebuilt_from_evidence": True, "decision": decision}


class ReclassifyBody(BaseModel):
    new_class: str = Field(min_length=1, max_length=200)
    note: str = Field("", max_length=1000)
    expected_version_id: str | None = Field(None, max_length=200)


@evidence_world_router.post("/{run_id}/entities/{entity_id}/reclassify")
async def reclassify_entity_route(
    world_id: str, run_id: str, entity_id: str, body: ReclassifyBody, db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """User correction on the run's persisted world: a NEW immutable version whose parent is the previous head. The entity keeps its
    evidence block, observations and external references untouched (checked before returning); the old class and the user's decision are
    recorded on the entity and in the world's decisions. The snapshot and the Evidence Matrix are never modified."""
    import hashlib

    from apps.api.evidence_world_stage import WORLDSTORE_REL, head_version
    from world_ir.corrections import CorrectionError, reclassify_entity
    from worldstore.store import WorldStore, WorldStoreError

    d = _run_dir(await _world_dir(db, world_id, auth), run_id)
    head = head_version(d)
    if not head:
        raise HTTPException(404, "No persisted world for this run")
    _require_fresh(head, getattr(body, "expected_version_id", None))
    _refuse_if_calibrated(d, head)
    store = WorldStore(d / WORLDSTORE_REL)
    try:
        world = store.load_version(head)
        eid = f"ent-{entity_id}"
        ent = world.entities.get(eid)
        if ent is None:
            raise HTTPException(404, "Entity not found")
        before = (dict(ent.custom_properties.get("evidence") or {}), [o.id for o in ent.observations], list(ent.external_references))
        try:
            decision = reclassify_entity(world, eid, body.new_class, body.note, user_id=str(getattr(auth, "user_id", "") or ""))
        except CorrectionError as exc:
            raise HTTPException(422, str(exc)) from exc
        ent = world.entities[eid]
        preserved = before == (dict(ent.custom_properties.get("evidence") or {}), [o.id for o in ent.observations], list(ent.external_references))
        if not preserved:  # a correction must never touch the evidence: refuse rather than save a world that lost it
            raise HTTPException(500, "correction would have altered the entity's evidence; nothing was saved")
        vid = "v-" + hashlib.sha256(json.dumps(world.to_dict(), sort_keys=True).encode("utf-8")).hexdigest()[:16]
        store.save_version(world, parent=head, version_id=vid)
        problems = store.verify_version(vid)
        if problems or store.load_version(vid).to_dict() != world.to_dict():
            raise HTTPException(500, f"the corrected version failed verification: {problems[:3]}")
    except WorldStoreError as exc:
        raise HTTPException(409, f"could not save the corrected version: {exc}") from None
    return {"version_id": vid, "parent_version_id": head, "decision": decision, "original_evidence_preserved": True}


@evidence_world_router.get("/{run_id}/entities")
async def list_entities(
    world_id: str, run_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    s = _snapshot(_run_dir(await _world_dir(db, world_id, auth), run_id))
    return {
        "run_id": run_id,
        "entities": [
            {"entity_id": e["entity_id"], "ontology_path": e["ontology_path"], "members": e["members"],
             "uncertain": e["explain"]["uncertain"]}
            for e in s["entities"]
        ],
    }


@evidence_world_router.get("/{run_id}/entities/{entity_id}")
async def get_entity(
    world_id: str, run_id: str, entity_id: str, db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """One canonical entity traced to its original images and model predictions: supporting rows, rejected/excluded rows with
    their reasons, the association decisions involving it, the conflicts naming it, and a fresh matrix re-verification."""
    d = _run_dir(await _world_dir(db, world_id, auth), run_id)
    s = _snapshot(d)
    ent = next((e for e in s["entities"] if e["entity_id"] == entity_id), None)
    if ent is None:
        raise HTTPException(404, "Entity not found")
    m = _matrix(d)
    members = set(ent["members"])
    rows = [m.row(oid) for oid in sorted(members)]
    exp = ent["explain"]
    qs = [q for q in ("position", "dimensions") if q in exp]
    supporting = sorted({r["observation_id"] for q in qs for r in exp[q]["supported_by"]})
    return {
        "entity": ent["world_entity"],
        "explain": exp,
        "supporting_observation_ids": supporting,
        "observations": rows,  # every member row, with raw and calibrated confidence, run, checkpoint, camera, validation
        "rejected_observations": [r for r in rows if r["status"] == "rejected"],
        "excluded": [x for q in qs for x in exp[q]["excluded"]],
        "association_decisions": [x for x in s["associations"] if x["a"] in members or x["b"] in members],
        "conflicts": [c for c in s["conflicts"] if entity_id in c["entity_ids"] or members & set(c["entity_ids"])],
        "capture_requests": [c for c in s["capture_requests"] if (c.get("target_region") or {}).get("entity_id") == entity_id],
        "trace_problems": trace(Entity.from_dict(ent["world_entity"]), m),
    }


@evidence_world_router.get("/{run_id}/rows")
async def list_rows(
    world_id: str, run_id: str, engine_id: str | None = None, quantity: str | None = None, status: str | None = None,
    limit: int = Query(100, ge=1, le=1000), db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    """Evidence Matrix rows (rejected ones included): raw observation + context + raw/calibrated confidence + status."""
    m = _matrix(_run_dir(await _world_dir(db, world_id, auth), run_id))
    rows = [
        r for r in m.rows()
        if (engine_id is None or r["observation"]["engine_id"] == engine_id)
        and (quantity is None or r["observation"]["quantity"] == quantity)
        and (status is None or r["status"] == status)
    ]
    return {"total": len(rows), "rows": rows[:limit]}


@evidence_world_router.get("/{run_id}/runs")
async def list_engine_runs(
    world_id: str, run_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    """Every engine execution, failed / unavailable / cancelled ones included, with model, checkpoint and raw output artifact."""
    m = _matrix(_run_dir(await _world_dir(db, world_id, auth), run_id))
    return {"runs": [r.to_dict() for r in m.runs()]}


@evidence_world_router.get("/{run_id}/rejected")
async def list_rejected(
    world_id: str, run_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
) -> dict:
    """Predictions engines returned that failed observation validation: raw, with the reason. They never support an entity."""
    m = _matrix(_run_dir(await _world_dir(db, world_id, auth), run_id))
    return {"rejected": [r.to_dict() for r in m.rejected()]}


def _section(key: str):
    async def handler(
        world_id: str, run_id: str, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)
    ) -> dict:
        s = _snapshot(_run_dir(await _world_dir(db, world_id, auth), run_id))
        return {"run_id": run_id, key: s[key]}

    return handler


for _key, _path in (
    ("associations", "associations"), ("conflicts", "conflicts"), ("capture_requests", "capture-requests"),
    ("hierarchy", "hierarchy"), ("refinement_history", "refinement"),
):
    evidence_world_router.add_api_route(f"/{{run_id}}/{_path}", _section(_key), methods=["GET"])


@evidence_world_router.get("/{run_id}/artifacts/{digest}")
async def get_artifact(
    world_id: str, run_id: str, digest: str, db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> Response:
    """Raw bytes of an artifact (e.g. a run's complete raw model output) -- only artifacts THIS run's matrix references."""
    if not _DIGEST_RE.match(digest):
        raise HTTPException(404, "Artifact not found")
    m = _matrix(_run_dir(await _world_dir(db, world_id, auth), run_id))
    uri = f"artifact://{digest}"
    referenced = {u for r in m.rows() for u in r["context"]["artifact_refs"].values()}
    referenced |= {r.output_artifact for r in m.runs() if r.output_artifact}
    if uri not in referenced or not m.has_artifact(uri):
        raise HTTPException(404, "Artifact not found")
    return Response(content=m.artifacts.get(uri), media_type="application/octet-stream")
