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
