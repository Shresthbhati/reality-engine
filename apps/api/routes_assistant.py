"""Routes: assistant tool layer (Phase F, first slice). Deterministic, no LLM.

    GET  /api/assistant/tools                      JSON-schema style tool list
    POST /api/worlds/{id}/assistant/call           {tool, args}: read tools run; propose_edit stores a plan (no commit)
    POST /api/worlds/{id}/assistant/confirm        {plan_id}: executes via _generate_and_commit (the one write path)

Edits are restricted to the existing safe paths: update a generated object's params (PUT semantics) or delete one
(dependents check). Renames and anything else are 422. A plan is bound to the HEAD it was previewed against; if HEAD
moved (or the plan expired) confirm answers 409 and marks it stale. Table `assistant_plans` is created via create_all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ValidationError
from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from apps.api import worldstore_service
from apps.api.db import get_db
from apps.api.models import Base, World, WorldMeasurement, new_id, utcnow
from apps.api.routes_procedural import (
    _BUILDERS, _PATCH_FIX, ApartmentPatch, BuildingPatch, CityPatch, CorridorPatch, DistrictPatch, RoomPatch,
    _find_spec, _generate_and_commit,
)
from assistant import tools
from procedural import spec_store
from world_ir.diff import diff_worlds

assistant_tools = APIRouter(prefix="/api/assistant", tags=["assistant"])
assistant = APIRouter(prefix="/api/worlds/{world_id}/assistant", tags=["assistant"])

PLAN_TTL = timedelta(minutes=30)
_PATCHES = {"room": RoomPatch, "corridor": CorridorPatch, "city": CityPatch,
            "apartment": ApartmentPatch, "building": BuildingPatch, "district": DistrictPatch}
_ID_CAP = 50


class AssistantPlan(Base):
    __tablename__ = "assistant_plans"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    world_id: Mapped[str] = mapped_column(ForeignKey("worlds.id"), index=True)
    base_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    operation: Mapped[dict] = mapped_column(JSON)
    preview: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="proposed")  # proposed | confirmed | stale
    result_version_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class CallIn(BaseModel):
    tool: str
    args: dict = {}


class ConfirmIn(BaseModel):
    plan_id: str


# ---------------------------------------------------------------- edit planning

def _gen_for(op: dict, stats: dict):
    """Build gen(current) -> WorldIR for a validated operation, mirroring the PUT/DELETE routes. Raises
    HTTPException (from _find_spec / validation) or ValueError, which callers turn into 422."""
    kind, name = op["kind"], op["name"]

    if op["op"] == "delete_generated":
        def gen(current):
            sid, specs = _find_spec(current, kind, name)
            deps = spec_store.dependents(specs, sid, include_connects=True)
            if deps and not op.get("cascade"):
                raise HTTPException(409, f"{kind} {name!r} has dependents: "
                                         f"{', '.join(specs[d].name for d in deps)}; pass cascade=true to delete them too")
            world, removed = spec_store.delete_specs(current, [sid, *deps])
            stats.update(cascaded=[specs[d].name for d in deps])
            return world
        return gen

    try:
        patch = _PATCHES[kind].model_validate(op.get("patch") or {})
    except ValidationError as exc:
        raise HTTPException(422, exc.errors(include_url=False, include_context=False, include_input=False)) from exc
    if not patch.model_fields_set:
        raise HTTPException(422, "patch is empty; nothing to change")

    def gen(current):
        sid, specs = _find_spec(current, kind, name)
        old = specs[sid]
        body_cls, build = _BUILDERS[kind]
        fields = {**old.params, "origin": tuple(old.origin), "attach": old.attach}
        if kind == "corridor" and old.attach:
            fields["connects"] = [c for c in old.params.get("connects", []) if c != old.attach["to"]]
        edits = patch.model_dump(exclude_unset=True)
        if kind in _PATCH_FIX:
            _PATCH_FIX[kind](fields, edits)
        try:
            new = build(body_cls(name=name, **{**fields, **edits}))
        except ValidationError as exc:
            raise HTTPException(422, exc.errors(include_url=False, include_context=False, include_input=False)) from exc
        new_params = {**new.params, "origin": new.origin, "attach": new.attach}
        world, changed, removed, unchanged = spec_store.regenerate(current, sid, new_params)
        stats.update(unchanged=len(unchanged))
        return world
    return gen


def _check_operation(raw: Any) -> dict:
    op = raw if isinstance(raw, dict) else {}
    name = op.get("op")
    if name in ("rename", "rename_generated", "rename_entity"):
        raise HTTPException(422, "rename is not allowed: names are identities that dependents reference")
    if name not in tools.EDIT_OPS:
        raise HTTPException(422, f"unknown operation {name!r}; allowed: {', '.join(tools.EDIT_OPS)}")
    try:
        tools.validate_args(tools.TOOLS["propose_edit"]["parameters"]["properties"]["operation"], op, "operation")
    except tools.ToolError as exc:
        raise HTTPException(422, str(exc)) from exc
    if name == "delete_generated" and "patch" in op:
        raise HTTPException(422, "patch is only valid for update_generated")
    if name == "update_generated" and "cascade" in op:
        raise HTTPException(422, "cascade is only valid for delete_generated")
    return op


def _capped(ids) -> list:
    return sorted(ids)[:_ID_CAP]


def _preview(current, result, op: dict, stats: dict) -> dict:
    d = diff_worlds(current, result)
    warnings = []
    if stats.get("cascaded"):
        warnings.append(f"also deletes dependents: {', '.join(stats['cascaded'])}")
    if op["op"] == "update_generated" and len(d.modified_entity_ids) + len(d.added_entity_ids) + len(d.removed_entity_ids) == 0:
        warnings.append("this edit changes no entities")
    if op["op"] == "delete_generated":
        warnings.append("deletion is versioned: the previous version stays available")
    return {"entities_added": len(d.added_entity_ids), "entities_changed": len(d.modified_entity_ids),
            "entities_removed": len(d.removed_entity_ids),
            "added_ids": _capped(d.added_entity_ids), "changed_ids": _capped(d.modified_entity_ids),
            "removed_ids": _capped(d.removed_entity_ids), "warnings": warnings}


async def _propose(db: AsyncSession, w: World, world_id: str, args: dict) -> dict:
    op = _check_operation(args["operation"])
    current = await worldstore_service.get_current_worldir(db, world_id)
    if current is None:
        raise HTTPException(422, "world has no committed version to edit")
    stats: dict = {}
    try:
        result = _gen_for(op, stats)(current)
    except HTTPException as exc:
        raise HTTPException(422, exc.detail) from exc
    except (ValueError, spec_store.UnknownSpec) as exc:
        raise HTTPException(422, str(exc)) from exc
    preview = _preview(current, result, op, stats)
    now = utcnow()
    plan = AssistantPlan(id=new_id("plan"), world_id=world_id, base_version_id=w.current_version_id,
                         operation=op, preview=preview, status="proposed", expires_at=now + PLAN_TTL)
    db.add(plan)
    await db.commit()
    return {"plan_id": plan.id, "operation": op, "base_version_id": plan.base_version_id, "preview": preview,
            "warnings": preview["warnings"], "requires_confirmation": True, "expires_at": plan.expires_at.isoformat()}


# ---------------------------------------------------------------------- routes

@assistant_tools.get("/tools")
async def list_tools() -> dict:
    return {"tools": tools.tool_schemas()}


async def _scale_state(db: AsyncSession, world_id: str) -> str:
    rows = (await db.execute(select(WorldMeasurement.scale_state).where(
        WorldMeasurement.world_id == world_id))).scalars().all()
    return "metric" if any(s in ("metric", "authored") for s in rows) else "unknown"


@assistant.post("/call")
async def call(world_id: str, body: CallIn, db: AsyncSession = Depends(get_db)) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    spec = tools.TOOLS.get(body.tool)
    if spec is None:
        raise HTTPException(422, f"unknown tool {body.tool!r}")
    try:
        tools.validate_args(spec["parameters"], body.args)
    except tools.ToolError as exc:
        raise HTTPException(422, str(exc)) from exc
    if body.tool == "propose_edit":
        return {"tool": body.tool, "result": await _propose(db, w, world_id, body.args)}
    current = await worldstore_service.get_current_worldir(db, world_id)
    if current is None:
        raise HTTPException(422, "world has no committed version to query")
    ctx = tools.Context(world=current, scale_state=await _scale_state(db, world_id))
    try:
        return {"tool": body.tool, "result": tools.call_read_tool(body.tool, body.args, ctx)}
    except tools.ToolError as exc:
        raise HTTPException(422, str(exc)) from exc


def _aware(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@assistant.post("/confirm")
async def confirm(world_id: str, body: ConfirmIn, db: AsyncSession = Depends(get_db)) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    plan = await db.get(AssistantPlan, body.plan_id)
    if plan is None or plan.world_id != world_id:
        raise HTTPException(404, "plan not found")
    if plan.status == "confirmed":
        raise HTTPException(409, f"plan already confirmed as version {plan.result_version_id}")
    reason = None
    if plan.status == "stale":
        reason = "plan is stale"
    elif _aware(plan.expires_at) < datetime.now(timezone.utc):
        reason = "plan expired"
    elif plan.base_version_id != w.current_version_id:
        reason = "the world changed since this plan was previewed"
    if reason:
        plan.status = "stale"
        await db.commit()
        raise HTTPException(409, f"{reason}; propose the edit again")
    op = plan.operation
    stats: dict = {}
    result = await _generate_and_commit(db, w, world_id, op["kind"], op["name"], _gen_for(op, stats),
                                        verb="updated" if op["op"] == "update_generated" else "deleted")
    plan.status, plan.result_version_id = "confirmed", result["version_id"]
    await db.commit()
    return {"plan_id": plan.id, "status": "confirmed", "version_id": result["version_id"],
            "entity_ids_added": result["entity_ids_added"], "preview": plan.preview}
