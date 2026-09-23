"""Bridge between the application database and the real, computational
WorldStore (worldstore/store.py).

WorldStore is the source of truth for version *content and lineage* -- it
already persists immutable, content-addressed, hash-verified WorldIR
snapshots with real structural diffs. The application database's
WorldVersion table is a queryable *mirror* of that lineage, kept in sync
by `commit_version` (the only write path) and `resync_versions` (the read
path, guarding against the CLI writing into the same store root outside
this process).

World.current_version_id is the one mutable "HEAD" pointer per world --
WorldStore itself has no such concept, only a DAG via parents/ancestors.
It only ever moves through `commit_version`.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import World, WorldVersion, utcnow
from apps.api.storage import store_bytes
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore

_store: WorldStore | None = None


def worldstore_root() -> Path:
    root = os.environ.get("WORLDSTORE_ROOT")
    return Path(root) if root else Path("./data/worldstore")


def get_store() -> WorldStore:
    global _store
    if _store is None:
        _store = WorldStore(worldstore_root())
    return _store


def _mirror_row(stored, *, report: dict | None = None,
                 points_artifact_uri: str | None = None,
                 cameras_artifact_uri: str | None = None) -> WorldVersion:
    return WorldVersion(
        id=stored.version_id,
        world_id=stored.world_id,
        parent_version_id=stored.parent,
        artifact_uri=stored.artifact_uri,
        artifact_hash=stored.artifact_hash,
        source_session_ids=list(stored.source_session_ids),
        changed_entity_ids=list(stored.changed_entity_ids),
        changed_geometry_ids=list(stored.changed_geometry_ids),
        report=report,
        points_artifact_uri=points_artifact_uri,
        cameras_artifact_uri=cameras_artifact_uri,
    )


async def commit_version(
    db: AsyncSession,
    *,
    world_id: str,
    world: WorldIR,
    parent: str | None,
    source_session_ids: list[str] | None = None,
    points: bytes | None = None,
    cameras: bytes | None = None,
    report: dict | None = None,
) -> WorldVersion:
    """Save a new WorldStore version and advance this World's HEAD.

    WorldStore (filesystem) is written first -- immutable and safe to
    leave orphaned on a crash. The DB mirror row + HEAD pointer commit
    second, so a crash between the two steps never leaves a DB row
    pointing at a version that doesn't exist on disk.

    WorldStore scopes StoredVersion.world_id by the WorldIR's own `id`
    field, not the caller's application `world_id` -- these are two
    different identifier spaces unless explicitly unified. commit_version
    is the single place that enforces `world.id == world_id`, so every
    later lookup (resync_versions, the CLI's `reality store list`) can
    filter WorldStore's on-disk versions by the application world_id
    directly.
    """
    world.id = world_id
    store = get_store()
    stored = store.save_version(
        world, parent=parent, source_session_ids=source_session_ids
    )

    points_uri = None
    if points is not None:
        digest, _ = store_bytes(points)
        points_uri = f"sha256://{digest}"
    cameras_uri = None
    if cameras is not None:
        digest, _ = store_bytes(cameras)
        cameras_uri = f"sha256://{digest}"

    row = _mirror_row(
        stored, report=report,
        points_artifact_uri=points_uri, cameras_artifact_uri=cameras_uri,
    )
    db.add(row)

    w = await db.get(World, world_id)
    if w is not None:
        w.current_version_id = stored.version_id
        w.updated_at = utcnow()

    await db.commit()
    await db.refresh(row)
    return row


async def get_current_worldir(db: AsyncSession, world_id: str) -> WorldIR | None:
    w = await db.get(World, world_id)
    if w is None or not w.current_version_id:
        return None
    return get_store().load_version(w.current_version_id)


async def get_current_version_row(db: AsyncSession, world_id: str) -> WorldVersion | None:
    w = await db.get(World, world_id)
    if w is None or not w.current_version_id:
        return None
    return await db.get(WorldVersion, w.current_version_id)


async def resync_versions(db: AsyncSession, world_id: str) -> None:
    """Reconcile the DB mirror against WorldStore's on-disk versions for
    this world. The CLI (`reality store save`, `reality compile`) can
    write directly into the same store root outside this API process, so
    the mirror must never silently drift from what's actually on disk.

    ponytail: full-store list_versions() scan on every read, O(all
    versions across all worlds) -- fine at this milestone's scale; if the
    store grows large, add a world-scoped index instead of scanning here.
    """
    store = get_store()
    stored = [v for v in store.list_versions() if v.world_id == world_id]
    if not stored:
        return
    existing_ids = set(
        (
            await db.execute(
                select(WorldVersion.id).where(WorldVersion.world_id == world_id)
            )
        ).scalars().all()
    )
    missing = [v for v in stored if v.version_id not in existing_ids]
    if not missing:
        return
    for v in missing:
        db.add(_mirror_row(v))
    await db.commit()
