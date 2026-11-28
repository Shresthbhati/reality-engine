"""Named design options (plan T-4b.3).

An option is a plain-language name for one immutable saved version of a world ("Option A", "Client's preference"). Saving copies
nothing; opening an option moves the world's HEAD to that version (history is never rewritten, and edits made afterwards branch
from it). The AS-BUILT baseline is the newest reconstruction in the lineage of HEAD, so a design option is always compared with the
survey it was designed on, never with another option by accident.

``summarise`` describes the difference between two versions in design words (furniture moved / replaced / removed / added, surfaces
refinished) from the two worlds themselves, so it does not depend on which commands were used to get there.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import ActivityEvent, World, WorldOption, WorldVersion, new_id, utcnow

AS_BUILT = "as-built"
MAX_OPTIONS = 20
MAX_NAME = 80
MOVE_TOL_M = 0.01
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class OptionError(Exception):
    """A refused option operation; ``status`` is the HTTP status the route answers with."""

    def __init__(self, message: str, status: int = 409):
        super().__init__(message)
        self.status = status


def clean_name(name: Any) -> str:
    n = " ".join(str(name or "").split())
    if not n or len(n) > MAX_NAME or _CONTROL.search(str(name or "")):
        raise OptionError(f"An option needs a name of 1 to {MAX_NAME} characters", 422)
    if n.lower() == AS_BUILT:
        raise OptionError(f"'{AS_BUILT}' is reserved for the survey itself", 422)
    return n


def _is_edit(v: Optional[WorldVersion]) -> bool:
    return v is not None and v.created_by_job_id is None and v.parent_version_id is not None


async def as_built_version(db: AsyncSession, version_id: Optional[str]) -> Optional[str]:
    """The newest reconstruction (not an edit) at or above ``version_id``."""
    cur = await db.get(WorldVersion, version_id) if version_id else None
    seen = 0
    while _is_edit(cur) and seen < 10_000:
        cur = await db.get(WorldVersion, cur.parent_version_id)
        seen += 1
    return cur.id if cur is not None else None


async def edits_between(db: AsyncSession, version_id: str, base_id: Optional[str]) -> int:
    n, cur = 0, await db.get(WorldVersion, version_id)
    while cur is not None and cur.id != base_id and _is_edit(cur) and n < 10_000:
        n += 1
        cur = await db.get(WorldVersion, cur.parent_version_id)
    return n


def _row(o: WorldOption, *, current: Optional[str], base: Optional[str], edits: int) -> dict:
    return {"id": o.id, "name": o.name, "version_id": o.version_id, "created_at": o.created_at.isoformat() if o.created_at else None,
            "created_by": o.created_by, "is_current": o.version_id == current, "edits_since_as_built": edits,
            "as_built_version_id": base}


async def list_options(db: AsyncSession, world: World) -> dict:
    rows = (await db.execute(select(WorldOption).where(WorldOption.world_id == world.id).order_by(WorldOption.created_at))).scalars().all()
    head_base = await as_built_version(db, world.current_version_id)
    out = []
    for o in rows:
        base = await as_built_version(db, o.version_id)
        out.append(_row(o, current=world.current_version_id, base=base, edits=await edits_between(db, o.version_id, base)))
    return {"world_id": world.id, "current_version_id": world.current_version_id, "as_built_version_id": head_base,
            "current_is_as_built": world.current_version_id == head_base, "items": out}


async def _get(db: AsyncSession, world: World, option_id: str) -> WorldOption:
    o = await db.get(WorldOption, option_id)
    if o is None or o.world_id != world.id:
        raise OptionError("Option not found", 404)
    return o


async def _name_taken(db: AsyncSession, world_id: str, name: str, except_id: Optional[str] = None) -> bool:
    q = select(WorldOption.id).where(WorldOption.world_id == world_id, func.lower(WorldOption.name) == name.lower())
    if except_id:
        q = q.where(WorldOption.id != except_id)
    return (await db.execute(q.limit(1))).first() is not None


async def save_option(db: AsyncSession, world: World, name: Any, user_id: Optional[str]) -> dict:
    n = clean_name(name)
    head = await db.get(WorldVersion, world.current_version_id) if world.current_version_id else None
    if not _is_edit(head):
        raise OptionError("Nothing has been designed yet: the current version is the survey itself. Make a design change first.")
    if await _name_taken(db, world.id, n):
        raise OptionError(f"There is already an option called '{n}'")
    count = (await db.execute(select(func.count()).select_from(WorldOption).where(WorldOption.world_id == world.id))).scalar_one()
    if count >= MAX_OPTIONS:
        raise OptionError(f"A world can hold {MAX_OPTIONS} options; delete one first")
    existing = (await db.execute(select(WorldOption).where(WorldOption.world_id == world.id, WorldOption.version_id == head.id))).scalars().first()
    if existing is not None:
        raise OptionError(f"This exact design is already saved as '{existing.name}'")
    o = WorldOption(id=new_id("opt"), world_id=world.id, name=n, version_id=head.id, created_by=user_id, created_at=utcnow())
    db.add(o)
    db.add(ActivityEvent(id=new_id("act"), type="world.option_saved", entity_type="world", entity_id=world.id,
                         summary=f"Saved design option '{n}'", metadata_json={"option_id": o.id, "version_id": head.id}))
    await db.commit()
    base = await as_built_version(db, head.id)
    return _row(o, current=head.id, base=base, edits=await edits_between(db, head.id, base))


async def rename_option(db: AsyncSession, world: World, option_id: str, name: Any) -> dict:
    o = await _get(db, world, option_id)
    n = clean_name(name)
    if await _name_taken(db, world.id, n, except_id=o.id):
        raise OptionError(f"There is already an option called '{n}'")
    o.name = n
    await db.commit()
    base = await as_built_version(db, o.version_id)
    return _row(o, current=world.current_version_id, base=base, edits=await edits_between(db, o.version_id, base))


async def delete_option(db: AsyncSession, world: World, option_id: str) -> dict:
    o = await _get(db, world, option_id)
    await db.delete(o)
    await db.commit()
    return {"deleted": option_id}


async def open_option(db: AsyncSession, world: World, option_id: str) -> dict:
    """Make this option the version the world is on. Edits afterwards branch from it; Undo/Redo follow the option's own history."""
    from apps.api import world_history

    o = await _get(db, world, option_id)
    if world.current_version_id == o.version_id:
        return await list_options(db, world)
    try:
        await world_history._refuse_if_busy(db, world.id)
        await world_history._move_head(db, world, o.version_id)
    except world_history.HistoryError as exc:
        raise OptionError(str(exc), exc.status) from None
    await world_history._save_redo(db, world.id, [])          # the redo list belonged to the previous design
    db.add(ActivityEvent(id=new_id("act"), type="world.option_opened", entity_type="world", entity_id=world.id,
                         summary=f"Opened design option '{o.name}'", metadata_json={"option_id": o.id, "version_id": o.version_id}))
    await db.commit()
    await db.refresh(world)
    return await list_options(db, world)


async def resolve(db: AsyncSession, world: World, ref: str) -> tuple[str, str]:
    """``'as-built'``, ``'current'`` or an option id -> (display name, version id)."""
    if ref == AS_BUILT:
        v = await as_built_version(db, world.current_version_id)
        if v is None:
            raise OptionError("This world has no survey version yet", 404)
        return "As-built survey", v
    if ref == "current":
        if not world.current_version_id:
            raise OptionError("This world has no version yet", 404)
        return "Current", world.current_version_id
    o = await _get(db, world, ref)
    return o.name, o.version_id


# ---- design-language comparison ---------------------------------------------------------------------------------------

def _pos(e) -> Optional[tuple]:
    p = (getattr(e, "transform", None) or {}).get("position")
    return (float(p["x"]), float(p["y"]), float(p["z"])) if p else None


def _cat(e) -> str:
    cp = e.custom_properties or {}
    return str(cp.get("category") or cp.get("entity_class") or "furniture")


def _finish(e):
    f = (e.custom_properties or {}).get("design_finish")
    return None if not f else (f.get("name"), tuple(f.get("colour_rgb") or ()))


def _label(e) -> str:
    return e.name or e.id


def summarise(a, b) -> dict:
    """What changed going from world ``a`` to world ``b``, in design words. Both are WorldIR objects."""
    from world_ir import EntityType

    fa = {i: e for i, e in a.entities.items() if e.type == EntityType.FURNITURE}
    fb = {i: e for i, e in b.entities.items() if e.type == EntityType.FURNITURE}
    moved, replaced, removed, added = [], [], [], []
    for i, e in fb.items():
        if i not in fa:
            added.append({"id": i, "name": _label(e)})
            continue
        old = fa[i]
        if _cat(old) != _cat(e):
            replaced.append({"id": i, "from": _label(old), "to": _label(e)})
        pa, pb = _pos(old), _pos(e)
        if pa and pb and math.dist(pa, pb) > MOVE_TOL_M:
            moved.append({"id": i, "name": _label(e), "distance_m": round(math.dist(pa, pb), 2)})
        elif (old.custom_properties or {}).get("yaw_rad") != (e.custom_properties or {}).get("yaw_rad") and _cat(old) == _cat(e):
            moved.append({"id": i, "name": _label(e), "distance_m": 0.0, "turned": True})
    for i, e in fa.items():
        if i not in fb:
            removed.append({"id": i, "name": _label(e)})
    finishes = []
    for i, e in b.entities.items():
        if e.type not in (EntityType.WALL, EntityType.FLOOR, EntityType.CEILING) or i not in a.entities:
            continue
        fo, fn = _finish(a.entities[i]), _finish(e)
        if fo != fn:
            finishes.append({"id": i, "type": getattr(e.type, "value", str(e.type)), "from": fo[0] if fo else None, "to": fn[0] if fn else None})
    same = not (moved or replaced or removed or added or finishes)
    return {"moved": moved, "replaced": replaced, "removed": removed, "added": added, "refinished": finishes, "identical_design": same,
            "sentences": _sentences(moved, replaced, removed, added, finishes, same)}


def _sentences(moved, replaced, removed, added, finishes, same) -> List[str]:
    if same:
        return ["No difference in furniture or surface finishes."]
    out = []
    if replaced:
        out.append(f"{len(replaced)} piece{'s' if len(replaced) != 1 else ''} swapped: " + ", ".join(f"{r['from']} to {r['to']}" for r in replaced[:5]))
    if moved:
        out.append(f"{len(moved)} piece{'s' if len(moved) != 1 else ''} moved or turned")
    if removed:
        out.append(f"{len(removed)} piece{'s' if len(removed) != 1 else ''} removed: " + ", ".join(r["name"] for r in removed[:5]))
    if added:
        out.append(f"{len(added)} piece{'s' if len(added) != 1 else ''} added: " + ", ".join(r["name"] for r in added[:5]))
    if finishes:
        out.append(f"{len(finishes)} surface{'s' if len(finishes) != 1 else ''} refinished: " + ", ".join(
            f"{f['type']} {f['from'] or 'survey colour'} to {f['to'] or 'survey colour'}" for f in finishes[:5]))
    return out
