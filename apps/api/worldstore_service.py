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

import json
import os
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import World, WorldVersion, utcnow
from apps.api.storage import store_bytes
from world_ir.world_v1 import WorldIR
from worldstore.store import WorldStore


class ConcurrentModificationError(RuntimeError):
    """HEAD moved between a writer's base read and its adopt attempt.

    Carries the orphaned (saved but never adopted) version id so callers
    can report honestly. Retrying after reloading HEAD converges: the
    retry either dedups (identical content) or chains onto the new HEAD.
    """

    def __init__(self, *, expected_parent: str | None, current_head: str | None, orphan_version_id: str):
        self.expected_parent = expected_parent
        self.current_head = current_head
        self.orphan_version_id = orphan_version_id
        super().__init__(
            f"HEAD moved during commit (expected '{expected_parent}', "
            f"now '{current_head}'); version '{orphan_version_id}' was "
            "saved but not adopted -- reload HEAD and retry"
        )


def _canonical_bytes(world: WorldIR) -> bytes:
    """Byte-identical to what WorldStore.save_version persists."""
    return json.dumps(world.to_dict(), sort_keys=True).encode("utf-8")

_stores: dict[str, WorldStore] = {}


def worldstore_root() -> Path:
    root = os.environ.get("WORLDSTORE_ROOT")
    return Path(root) if root else Path("./data/worldstore")


def get_store() -> WorldStore:
    """One cached WorldStore per root. A single global silently served a
    stale root after WORLDSTORE_ROOT changed (multi-tenant processes,
    tests), making reads/reconciliations hit the wrong store -- a
    cross-world data leak by misdirection. Keying by resolved root keeps
    the single-entry fast path for production while staying correct."""
    key = str(worldstore_root().resolve())
    store = _stores.get(key)
    if store is None:
        store = WorldStore(worldstore_root())
        _stores[key] = store
    return store


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
    expect_parent: str | None = None,
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

    Duplicate requests with byte-identical content return the already
    adopted parent version instead of minting a duplicate. HEAD advances
    only if it still holds the claimed base (one atomic conditional
    UPDATE, including an IS NULL claim for first versions); a concurrent
    mover wins and this call raises ConcurrentModificationError -- never
    a silent last-writer-wins overwrite. When `expect_parent` is omitted
    the current HEAD is claimed, so concurrent first-writers converge
    through retry (dedup if identical, chain otherwise) instead of
    racing blind adopts.
    """
    world.id = world_id
    store = get_store()
    new_bytes = _canonical_bytes(world)
    if expect_parent is None:
        head_row = await db.get(World, world_id)
        head_now = head_row.current_version_id if head_row is not None else None
        if head_now is not None:
            if parent is None:
                parent = head_now
            expect_parent = head_now
    if parent is not None:
        try:
            parent_world = store.load_version(parent)
        except Exception:
            parent_world = None
        if parent_world is not None and _canonical_bytes(parent_world) == new_bytes:
            row = await db.get(WorldVersion, parent)
            if row is not None and row.world_id == world_id:
                return row
            # Mirror missing or diverged: fall through and save (safe).

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

    if expect_parent is not None:
        head_predicate = World.current_version_id == expect_parent
    else:
        # No base claimed (first version racing another first version):
        # adopt only onto an empty HEAD.
        head_predicate = World.current_version_id.is_(None)
    adopted = await db.execute(
        update(World)
        .where(World.id == world_id, head_predicate)
        .values(current_version_id=stored.version_id, updated_at=utcnow())
    )
    if adopted.rowcount == 0:
        current = await db.get(World, world_id)
        raise ConcurrentModificationError(
            expected_parent=expect_parent,
            current_head=current.current_version_id if current else None,
            orphan_version_id=stored.version_id,
        )

    row = _mirror_row(
        stored, report=report,
        points_artifact_uri=points_uri, cameras_artifact_uri=cameras_uri,
    )
    db.add(row)

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
    row = await db.get(WorldVersion, w.current_version_id)
    # A HEAD pointer naming another world's version must never serve
    # foreign content: treat as absent (callers answer 404/409), the same
    # ownership rule the direct version routes enforce.
    if row is not None and row.world_id != world_id:
        return None
    return row


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
