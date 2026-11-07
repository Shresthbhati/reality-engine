"""Persistent per-world undo/redo (plan T-6.7).

Every committed edit is an immutable WorldStore version whose parent is the version it edited. Undo therefore only moves the
world's HEAD pointer to the parent, and redo moves it back to the version that was undone; nothing is rewritten, the redo list is
stored in the database, and both survive a reload or a restart. A world's history is its own version chain, so undoing world A
cannot touch world B.

Only EDIT versions are undoable: a version made by a reconstruction job (``created_by_job_id``) or the first version of a world is
a floor that undo never crosses, so undo can never discard a reconstruction. The redo list is valid only while its newest entry's
parent is the current HEAD; a new edit after an undo makes it stale, and it is dropped on next use.
"""
from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import ActivityEvent, Job, Session, World, WorldEditState, WorldVersion, new_id, utcnow


class HistoryError(Exception):
    """A refused undo/redo; ``status`` is the HTTP status the route should answer with."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


def _is_edit(v: WorldVersion | None) -> bool:
    return v is not None and v.created_by_job_id is None and v.parent_version_id is not None


async def _redo_list(db: AsyncSession, world_id: str) -> list[str]:
    row = await db.get(WorldEditState, world_id)
    return list(row.redo or []) if row is not None else []


async def _save_redo(db: AsyncSession, world_id: str, redo: list[str]) -> None:
    row = await db.get(WorldEditState, world_id)
    if row is None:
        db.add(WorldEditState(world_id=world_id, redo=redo, updated_at=utcnow()))
    else:
        row.redo, row.updated_at = redo, utcnow()


async def _valid_redo(db: AsyncSession, world: World) -> list[str]:
    """The redo list if it still applies to the current HEAD, else empty."""
    redo = await _redo_list(db, world.id)
    if not redo:
        return []
    top = await db.get(WorldVersion, redo[-1])
    return redo if top is not None and top.world_id == world.id and top.parent_version_id == world.current_version_id else []


async def state(db: AsyncSession, world: World) -> dict:
    head = await db.get(WorldVersion, world.current_version_id) if world.current_version_id else None
    redo = await _valid_redo(db, world)
    top = await db.get(WorldVersion, redo[-1]) if redo else None
    depth = 0
    cur = head
    while _is_edit(cur):
        depth += 1
        cur = await db.get(WorldVersion, cur.parent_version_id)
    return {"can_undo": _is_edit(head), "can_redo": bool(redo), "undo_depth": depth, "redo_depth": len(redo),
            "head": world.current_version_id, "undo_label": head.label if _is_edit(head) else None,
            "redo_label": top.label if top is not None else None}


async def _refuse_if_busy(db: AsyncSession, world_id: str) -> None:
    sids = (await db.execute(select(Session.id).where(Session.world_id == world_id))).scalars().all()
    busy = (await db.execute(select(Job.id).where(Job.entity_type == "session", Job.entity_id.in_(sids),
                                                  Job.status.in_(("queued", "running"))).limit(1))).scalars().first() if sids else None
    if busy:
        raise HistoryError(f"a reconstruction is in progress (job {busy}); wait for it before undoing or redoing")


async def _move_head(db: AsyncSession, world: World, target: str) -> None:
    from apps.api.worldstore_service import get_store

    failures = get_store().verify_version(target)
    if failures:
        raise HistoryError("that version failed integrity verification: " + "; ".join(f["reason"] for f in failures), 422)
    moved = await db.execute(update(World).where(World.id == world.id, World.current_version_id == world.current_version_id)
                             .values(current_version_id=target, updated_at=utcnow()))
    if moved.rowcount == 0:
        raise HistoryError("the world changed while undoing; reload and try again")


async def undo(db: AsyncSession, world: World) -> dict:
    await _refuse_if_busy(db, world.id)
    head = await db.get(WorldVersion, world.current_version_id) if world.current_version_id else None
    if not _is_edit(head):
        raise HistoryError("nothing to undo: the current version was not made by an edit", 409)
    redo = await _valid_redo(db, world)
    await _move_head(db, world, head.parent_version_id)
    await _save_redo(db, world.id, redo + [head.id])
    db.add(ActivityEvent(id=new_id("act"), type="world.undo", entity_type="world", entity_id=world.id,
                         summary=f"Undid version {head.id}", metadata_json={"undone": head.id, "head": head.parent_version_id}))
    await db.commit()
    await db.refresh(world)
    return await state(db, world)


async def redo(db: AsyncSession, world: World) -> dict:
    await _refuse_if_busy(db, world.id)
    stack = await _valid_redo(db, world)
    if not stack:
        raise HistoryError("nothing to redo", 409)
    target = stack[-1]
    await _move_head(db, world, target)
    await _save_redo(db, world.id, stack[:-1])
    db.add(ActivityEvent(id=new_id("act"), type="world.redo", entity_type="world", entity_id=world.id,
                         summary=f"Redid version {target}", metadata_json={"redone": target, "head": target}))
    await db.commit()
    await db.refresh(world)
    return await state(db, world)
