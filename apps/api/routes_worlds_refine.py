"""RERES entry point: bounded, evidence-driven refinement of a world's head version.

POST /api/worlds/{world_id}/refine runs the priority-driven refinement loop (engine/quality) over the head WorldIR with a
hard iteration and time budget, using whatever specialist engines are registered (engine/quality/refinement_engines.py) plus
the built-in model-free tasks. A new version is committed ONLY when the world actually changed; a run that changed nothing
(including "no engine available for the tasks that remain") commits nothing and says so. Previous versions are never
modified: a refinement is a new version whose report carries the loop's history.
"""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import worldstore_service as wss
from apps.api.auth import AuthContext, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import World
from engine.quality.autonomous_correction_loop import AutonomousCorrectionLoop
from engine.quality.refinement_engines import registered_executor, registered_kinds

refine_router = APIRouter(tags=["refine"])

MAX_ITERATIONS = 10
MAX_TIME_BUDGET_S = 120.0


class RefineIn(BaseModel):
    max_iterations: int = Field(5, ge=1, le=MAX_ITERATIONS)
    time_budget_s: float = Field(30.0, gt=0.0, le=MAX_TIME_BUDGET_S)


def _fingerprint(world) -> str:
    """What a refinement can change: entity types/attributes and the capture requests it files. The loop's own report in
    ``metadata`` is excluded, or every run would look like a change."""
    requests = json.dumps((world.metadata or {}).get("capture_requests", []), sort_keys=True)
    return AutonomousCorrectionLoop.hash_entities_state(world) + requests


@refine_router.post("/{world_id}/refine")
async def refine_world(
    world_id: str,
    body: RefineIn | None = None,
    db: AsyncSession = Depends(get_db),
    auth: AuthContext = Depends(get_auth_context),
) -> dict:
    body = body or RefineIn()
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    enforce_project_isolation(w, auth)

    worldir = wss.load_worldir_from_store(w)
    if worldir is None:
        raise HTTPException(404, "WorldIR not available for current version")
    worldir.id = w.id
    parent = w.current_version_id

    before = _fingerprint(worldir)
    report = await asyncio.to_thread(
        AutonomousCorrectionLoop.run_refinement_loop, worldir, body.max_iterations, body.time_budget_s, registered_executor)
    summary = report.to_dict()
    summary["engines_registered"] = registered_kinds()

    if _fingerprint(worldir) == before:
        return {"committed": False, "version_id": parent, "report": summary,
                "reason": "refinement changed nothing" if report.tasks_succeeded == 0 and report.tasks_unavailable == 0
                else "no net change to the world"}

    version = await wss.commit_version(
        db, world_id=w.id, world=worldir, parent=parent, report={"refinement": summary}, label="refinement")
    return {"committed": True, "version_id": version.id, "parent_version_id": parent, "report": summary}
