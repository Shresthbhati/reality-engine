"""Tenant isolation for every route module except routes_worlds (which applies the same check inline).

Everything here is built on ``auth.enforce_project_isolation`` / ``get_auth_context``: ADMIN is unrestricted, any other
role only reaches rows whose ``team_id`` is NULL (unowned) or equals its tenant, and a foreign row is a 404 so
existence is never revealed. Nothing here is a second policy.

* ``world_scope`` / ``session_scope`` / ... are router-level dependencies: put one on an ``APIRouter`` and every
  route whose path carries that id is checked before the handler runs. A missing row is left to the handler's own 404.
* ``assert_visible`` covers ids that arrive in a body or form instead of the path.
* ``visible`` / ``entity_clause`` are the SQL form of the same rule, for list endpoints.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.auth import AuthContext, Role, enforce_project_isolation, get_auth_context
from apps.api.db import get_db
from apps.api.models import Evidence, Job, Session, Site, Upload, World

#: Job / Notification / ActivityEvent point at their subject with (entity_type, entity_id).
ENTITY_MODELS = {"world": World, "session": Session, "evidence": Evidence, "upload": Upload, "site": Site}


def team_stamp(auth: AuthContext) -> str | None:
    """The team_id a row created by this caller carries (same rule as create_world: the default tenant is unowned)."""
    return auth.tenant_id if auth.tenant_id != "default" else None


async def team_for_session(db: AsyncSession, session_id: str | None, auth: AuthContext) -> str | None:
    """Evidence/uploads belong to their session's tenant; with no (known) session, to the caller's."""
    s = await db.get(Session, session_id) if session_id else None
    return (s.team_id if s is not None else None) or team_stamp(auth)


def _check(row, auth: AuthContext, message: str) -> None:
    """enforce_project_isolation, but the 404 reads exactly like the handler's own "<thing> not found", so a foreign
    id cannot be told apart from an unknown one by its message."""
    try:
        enforce_project_isolation(row, auth)
    except HTTPException as exc:
        raise HTTPException(exc.status_code, message) from None


async def assert_visible(db: AsyncSession, model, row_id: str | None, auth: AuthContext) -> None:
    """404 if the row exists and belongs to another tenant. A row that does not exist passes: the caller keeps its
    existing behaviour for unknown ids."""
    row = await db.get(model, row_id) if row_id else None
    if row is not None:
        _check(row, auth, f"{model.__name__} not found")


def visible(model, auth: AuthContext):
    """SQL predicate for 'this caller may see the row'; None = no restriction (ADMIN)."""
    if auth.role == Role.ADMIN:
        return None
    return or_(model.team_id.is_(None), model.team_id == auth.tenant_id)


def visible_ids(model, auth: AuthContext):
    clause = visible(model, auth)
    return select(model.id) if clause is None else select(model.id).where(clause)


def entity_clause(type_col, id_col, auth: AuthContext):
    """SQL predicate over an (entity_type, entity_id) pair. Rows with no entity are global; a pair this module cannot
    resolve to an owner is hidden from non-admins (fail closed). None = no restriction (ADMIN)."""
    if auth.role == Role.ADMIN:
        return None
    return or_(
        type_col.is_(None),
        *(and_(type_col == name, id_col.in_(visible_ids(m, auth))) for name, m in ENTITY_MODELS.items()),
    )


async def assert_entity_visible(db: AsyncSession, entity_type: str | None, entity_id: str | None, auth: AuthContext) -> None:
    """Strict form of the above for one job's subject: an unresolvable or vanished entity is a 404 for non-admins."""
    if auth.role == Role.ADMIN:
        return
    model = ENTITY_MODELS.get(entity_type or "")
    row = await db.get(model, entity_id) if model is not None and entity_id else None
    if row is None:
        raise HTTPException(404, "Job not found")
    _check(row, auth, "Job not found")


def _path_scope(model, param: str):
    async def dep(request: Request, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> None:
        await assert_visible(db, model, request.path_params.get(param), auth)

    dep.__name__ = f"{param}_scope"
    return dep


world_scope = _path_scope(World, "world_id")
session_scope = _path_scope(Session, "session_id")
evidence_scope = _path_scope(Evidence, "evidence_id")
upload_scope = _path_scope(Upload, "upload_id")
site_scope = _path_scope(Site, "site_id")


async def job_scope(request: Request, db: AsyncSession = Depends(get_db), auth: AuthContext = Depends(get_auth_context)) -> None:
    job_id = request.path_params.get("job_id")
    if job_id is None or auth.role == Role.ADMIN:
        return
    job = await db.get(Job, job_id)
    if job is not None:
        await assert_entity_visible(db, job.entity_type, job.entity_id, auth)
