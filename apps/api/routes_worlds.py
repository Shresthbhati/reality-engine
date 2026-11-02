"""Routes: worlds (WorldStore remains authoritative for computational data)."""

from __future__ import annotations

import json
import logging
import math
import os
import struct
from pathlib import Path

log = logging.getLogger("reality.api.worlds")

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import place_summary
from apps.api.db import get_db
from apps.api.models import (
    ActivityEvent,
    Evidence,
    Job,
    Location,
    Session,
    Upload,
    World,
    WorldVersion,
    new_id,
    utcnow,
)
from world_ir.diff import diff_worlds
from world_ir.schema_v1 import EntityType, Relationship, RelationshipKind, Provenance
from world_ir.world_v1 import WorldIR
from world_ir.artifact_store import FileArtifactStore
from worldstore.store import WorldStore, WorldStoreError


def _worldstore_root() -> Path:
    """Same root the reconstruct job persists WorldStore versions under."""
    return Path(os.environ.get("WORLDSTORE_ROOT", "./data/worldstore"))


worlds = APIRouter(prefix="/api/worlds", tags=["worlds"])

WORLD_STATUSES = ("active", "archived", "trashed")


class WorldIn(BaseModel):
    name: str
    description: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    coverage: dict | None = None


@worlds.post("", status_code=201)
async def create_world(body: WorldIn, db: AsyncSession = Depends(get_db)) -> dict:
    from apps.api.validation import (
        require_json_size,
        require_latitude,
        require_longitude,
        require_name,
    )

    w = World(
        id=new_id("wld"),
        name=require_name(body.name, "world name"),
        description=require_json_size(body.description, "description"),
        latitude=require_latitude(body.latitude) if body.latitude is not None else None,
        longitude=require_longitude(body.longitude) if body.longitude is not None else None,
        coverage=require_json_size(body.coverage, "coverage"),
    )
    db.add(w)
    db.add(
        ActivityEvent(
            id=new_id("act"),
            type="world.created",
            entity_type="world",
            entity_id=w.id,
            summary=f"World '{w.name}' created",
        )
    )
    await db.commit()
    await db.refresh(w)
    return {"id": w.id, "name": w.name}


def _world_out(w: World, session_count: int | None = None) -> dict:
    out = {
        "id": w.id,
        "name": w.name,
        "description": w.description,
        "status": w.status,
        "latitude": w.latitude,
        "longitude": w.longitude,
        "current_version_id": w.current_version_id,
        "archived_at": w.archived_at.isoformat() if w.archived_at else None,
        "deleted_at": w.deleted_at.isoformat() if w.deleted_at else None,
        "thumbnail_uri": w.thumbnail_uri,
        "scale_state": w.scale_state,
        "created_at": w.created_at.isoformat() if w.created_at else None,
    }
    if session_count is not None:
        out["session_count"] = session_count
    return out


@worlds.get("")
async def list_worlds(status: str = "active", db: AsyncSession = Depends(get_db)) -> dict:
    """``?status=active|archived|trashed`` (default active): archived and trashed worlds are hidden from the default list."""
    if status not in WORLD_STATUSES:
        raise HTTPException(422, f"status must be one of {', '.join(WORLD_STATUSES)}")
    res = await db.execute(
        select(World).where(World.status == status).order_by(World.updated_at.desc()).limit(200))
    ses_counts = dict(
        (await db.execute(
            select(Session.world_id, func.count()).group_by(Session.world_id)
        )).all()
    )
    return {"items": [_world_out(w, ses_counts.get(w.id, 0)) for w in res.scalars().all()]}


class WorldRename(BaseModel):
    name: str


@worlds.patch("/{world_id}")
async def rename_world(world_id: str, body: WorldRename, db: AsyncSession = Depends(get_db)) -> dict:
    from apps.api.validation import require_name

    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    old = w.name
    w.name = require_name(body.name, "world name")
    db.add(ActivityEvent(id=new_id("act"), type="world.renamed", entity_type="world", entity_id=w.id,
                         summary=f"World '{old}' renamed to '{w.name}'"))
    await db.commit()
    return {"id": w.id, "name": w.name}


@worlds.delete("/{world_id}", status_code=204)
async def delete_world(world_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    """Delete a place: its application records, its quarantined version records, its derived reconstruction state.

    Refused (409) while a build is queued or running. Shared content-addressed artifacts are NOT removed (see
    worldstore_service.delete_world_state)."""
    from apps.api import worldstore_service

    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    session_ids = list((await db.execute(select(Session.id).where(Session.world_id == world_id))).scalars().all())
    if session_ids:
        busy = (await db.execute(
            select(Job.id).where(Job.entity_id.in_(session_ids), Job.status.in_(("queued", "running"))).limit(1)
        )).scalar_one_or_none()
        if busy:
            raise HTTPException(409, f"a build is in progress (job {busy}); wait for it to finish or cancel it first")
    version_ids = list((await db.execute(select(WorldVersion.id).where(WorldVersion.world_id == world_id))).scalars().all())

    name = w.name
    from sqlalchemy import delete

    ev_rows = (await db.execute(
        select(Evidence.id, Evidence.upload_id).where(Evidence.session_id.in_(session_ids))
    )).all() if session_ids else []
    evidence_ids = [r[0] for r in ev_rows]
    upload_ids = [r[1] for r in ev_rows if r[1]]
    for entity_type, ids in (("session", session_ids), ("evidence", evidence_ids), ("world", [world_id])):
        if ids:
            await db.execute(delete(Location).where(Location.entity_type == entity_type, Location.entity_id.in_(ids)))
    if session_ids:
        await db.execute(delete(Job).where(Job.entity_type == "session", Job.entity_id.in_(session_ids)))
        await db.execute(delete(Evidence).where(Evidence.session_id.in_(session_ids)))
        if upload_ids:
            await db.execute(delete(Upload).where(Upload.id.in_(upload_ids)))
        await db.execute(delete(Session).where(Session.id.in_(session_ids)))
    if version_ids:
        await db.execute(delete(WorldVersion).where(WorldVersion.world_id == world_id))
    await db.delete(w)
    db.add(ActivityEvent(id=new_id("act"), type="world.deleted", entity_type="world", entity_id=world_id,
                         summary=f"World '{name}' deleted"))
    await db.commit()
    # Disk state last: if it fails the place is already gone from the app, and the leftovers are inert.
    worldstore_service.delete_world_state(world_id, version_ids)
    return Response(status_code=204)


@worlds.get("/{world_id}/thumbnail")
async def world_thumbnail(world_id: str, v: str | None = None, db: AsyncSession = Depends(get_db)):
    """A 480 px JPEG of one of the place's own photographs (a placed one when the model has them). `v` is only a
    cache-buster the client sets to the current version id."""
    import asyncio

    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    ev = await place_summary.pick_thumbnail_source(db, w)
    if ev is None:
        raise HTTPException(404, "This place has no photographs yet, so there is nothing to show")
    path = place_summary.thumbnail_path(w, ev)
    rendered = await asyncio.to_thread(place_summary.render_thumbnail, ev, path)
    if rendered is None:
        raise HTTPException(404, "The photograph could not be read, so no thumbnail was made")
    return FileResponse(rendered, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@worlds.get("/{world_id}")
async def get_world(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    session_count = (await db.execute(
        select(func.count()).select_from(Session).where(Session.world_id == world_id)
    )).scalar_one()
    return _world_out(w, session_count)


class WorldPatch(BaseModel):
    name: str | None = None
    description: str | None = None


async def _world_or_404(db: AsyncSession, world_id: str) -> World:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    return w


@worlds.patch("/{world_id}")
async def patch_world(world_id: str, body: WorldPatch, db: AsyncSession = Depends(get_db)) -> dict:
    """Rename / edit the description. Identity (id, versions, sessions) is untouched. A trashed world must be restored first."""
    from apps.api.validation import require_json_size, require_name
    from apps.api.world_lifecycle import record_event

    w = await _world_or_404(db, world_id)
    if w.status == "trashed":
        raise HTTPException(409, "World is in the trash; restore it before editing")
    changed = {}
    if body.name is not None:
        new = require_name(body.name, "world name")
        if new != w.name:
            changed["name"] = {"from": w.name, "to": new}
            w.name = new
    if "description" in body.model_fields_set:
        new_d = require_json_size(body.description, "description")
        if new_d != w.description:
            changed["description"] = True
            w.description = new_d
    if not changed:
        raise HTTPException(422, "nothing to change: send a different name and/or description")
    record_event(db, "updated", w, f"World '{w.name}' edited ({', '.join(changed)})", {"changed": changed})
    await db.commit()
    return _world_out(w)


@worlds.post("/{world_id}/duplicate", status_code=201)
async def duplicate_world(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Copy-on-write duplicate: a new world carrying the source's metadata and its CURRENT version's content. The
    WorldIR JSON is re-saved under the new world id (version rows are keyed by world), but geometry, points and
    cameras are content-addressed blobs that both worlds reference -- no file copies. Sessions/evidence are not
    copied (the duplicate shares the reconstruction, not the capture history)."""
    from worldstore.store import WorldStoreError

    from apps.api import worldstore_service
    from apps.api.world_lifecycle import record_event

    src = await _world_or_404(db, world_id)
    if src.status == "trashed":
        raise HTTPException(409, "World is in the trash; restore it before duplicating")
    src_row = None
    if src.current_version_id:
        src_row = await worldstore_service.get_current_version_row(db, world_id)
        if src_row is None:
            raise HTTPException(409, f"Current version '{src.current_version_id}' has no usable version record")
    new = World(id=new_id("wld"), name=f"{src.name} (copy)"[:200], description=src.description, status="active",
                owner_id=src.owner_id, team_id=src.team_id, latitude=src.latitude, longitude=src.longitude,
                coverage=src.coverage, thumbnail_uri=src.thumbnail_uri, scale_state=src.scale_state)
    db.add(new)
    record_event(db, "duplicated", new, f"World '{src.name}' duplicated as '{new.name}'", {"source_world_id": src.id})
    try:
        if src_row is not None:
            wir = worldstore_service.get_store().load_version(src_row.id)
            row = await worldstore_service.commit_version(
                db, world_id=new.id, world=wir, parent=None,
                source_session_ids=list(src_row.source_session_ids or []), report=src_row.report)
            row.label = f"Duplicate of {src.name} @ {src_row.id}"[:200]
            row.points_artifact_uri, row.cameras_artifact_uri = src_row.points_artifact_uri, src_row.cameras_artifact_uri
        await db.commit()
    except (WorldStoreError, worldstore_service.ConcurrentModificationError,
            worldstore_service.AdoptionVerificationError) as exc:
        await db.rollback()
        raise HTTPException(409, f"Could not duplicate: {exc}") from exc
    except Exception:
        await db.rollback()
        raise
    await db.refresh(new)
    return {**_world_out(new), "duplicated_from": src.id}


@worlds.post("/{world_id}/archive")
async def archive_world(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    from apps.api.world_lifecycle import record_event

    w = await _world_or_404(db, world_id)
    if w.status == "trashed":
        raise HTTPException(409, "World is in the trash; restore it first")
    if w.status == "archived":
        return _world_out(w)                       # idempotent: already archived, no new event
    w.status, w.archived_at = "archived", utcnow()
    record_event(db, "archived", w, f"World '{w.name}' archived")
    await db.commit()
    return _world_out(w)


@worlds.post("/{world_id}/restore")
async def restore_world(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    from apps.api.world_lifecycle import record_event

    w = await _world_or_404(db, world_id)
    if w.status == "active":
        raise HTTPException(409, "World is already active")
    was = w.status
    w.status, w.archived_at, w.deleted_at = "active", None, None
    record_event(db, "restored", w, f"World '{w.name}' restored from {was}", {"from": was})
    await db.commit()
    return _world_out(w)


@worlds.delete("/{world_id}")
async def delete_world(world_id: str, permanent: bool = False, db: AsyncSession = Depends(get_db)):
    """Soft delete (default): cancel the world's queued/running jobs FIRST, then status=trashed, deleted_at=now; the
    trash sweep purges it after 30 days. ``?permanent=true`` purges now: rows, WorldStore versions and only the blobs
    nothing else references (409 while a job is still stopping)."""
    from apps.api.world_lifecycle import JobsStillRunning, cancel_world_jobs, purge_world, record_event

    w = await _world_or_404(db, world_id)
    if permanent:
        try:
            return {"permanently_deleted": True, **await purge_world(db, w)}
        except JobsStillRunning as exc:
            raise HTTPException(409, str(exc)) from exc
    if w.status == "trashed":
        raise HTTPException(409, "World is already in the trash; use ?permanent=true to delete it for good")
    jobs = await cancel_world_jobs(db, w.id)
    w.status, w.deleted_at = "trashed", utcnow()
    record_event(db, "trashed", w, f"World '{w.name}' moved to trash",
                 {"jobs_cancelled": jobs["cancelled"], "jobs_cancel_requested": jobs["cancel_requested"]})
    await db.commit()
    return {**_world_out(w), "jobs_cancelled": jobs["cancelled"], "jobs_cancel_requested": jobs["cancel_requested"]}


@worlds.get("/{world_id}/versions")
async def list_world_versions(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """Application mirror of WorldStore lineage. Reconciled against the
    on-disk store on every read (writers outside this process, or a
    version orphaned by a failed commit, must not drift silently from
    what is listed) -- never synthesized here. A corrupted store degrades
    to the DB mirror instead of failing the whole listing; the corrupt
    version itself still fails explicitly (409) when read."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    try:
        from apps.api import worldstore_service

        await worldstore_service.resync_versions(db, world_id)
    except Exception as exc:
        # A corrupted/unreadable store must not take the whole listing
        # down: fall back to the DB mirror. Per-version reads still fail
        # explicitly (409) via _load_world_or_409.
        log.warning("versions resync degraded for world %s: %s", world_id, exc)
    res = await db.execute(
        select(WorldVersion).where(WorldVersion.world_id == world_id).order_by(WorldVersion.created_at.desc())
    )
    return {
        "items": [
            {
                "id": v.id,
                "world_id": v.world_id,
                "parent_version_id": v.parent_version_id,
                "artifact_uri": v.artifact_uri,
                "artifact_hash": v.artifact_hash,
                "source_session_ids": v.source_session_ids or [],
                "changed_entity_ids": v.changed_entity_ids or [],
                "created_at": v.created_at.isoformat() if v.created_at else None,
                "is_current": v.id == w.current_version_id,
            }
            for v in res.scalars().all()
        ]
    }


@worlds.post("/{world_id}/attach/{session_id}")
async def attach_session(
    world_id: str, session_id: str, db: AsyncSession = Depends(get_db)
) -> dict:
    w = await db.get(World, world_id)
    s = await db.get(Session, session_id)
    if w is None or s is None:
        raise HTTPException(404, "World or Session not found")
    s.world_id = w.id
    await db.commit()
    return {"session_id": s.id, "world_id": w.id}


@worlds.get("/{world_id}/coverage")
async def world_coverage(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    res = await db.execute(
        select(Location).where(
            Location.entity_type == "session",
            Location.entity_id.in_(
                select(Session.id).where(Session.world_id == world_id)
            ),
        )
    )
    points = [
        {"lat": l.latitude, "lon": l.longitude, "accuracy": l.accuracy}
        for l in res.scalars().all()
    ]
    if w.coverage:
        return {"available": True, "coverage": w.coverage, "session_points": points}
    if points:
        return {"available": True, "coverage": None, "session_points": points}
    return {"available": False, "reason": "No spatial data captured for this world"}


# --------------------------------------------------------------------------
# Computational surfaces (WorldStore is authoritative)
#
# The Studio's spatial workstation fetches a world's actual reconstruction
# through these endpoints. Everything reads from the version lineage the
# reconstruct job created; a world with no versions has nothing to show and
# says so explicitly (no fabrication, no placeholder geometry). A corrupted
# artifact is a 409, never an empty success.
# --------------------------------------------------------------------------


async def _version_row(
    db: AsyncSession, world_id: str, version_id: str | None
) -> WorldVersion:
    """The mirror row for the world's effective version (current unless
    ?version= overrides). 404 world / 404 version / explicit None."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    if version_id is not None:
        row = await db.get(WorldVersion, version_id)
        if row is None or row.world_id != world_id:
            raise HTTPException(
                404, f"Version '{version_id}' not found for world {world_id}"
            )
        return row
    if not w.current_version_id:
        raise HTTPException(
            404, "World has no reconstruction versions yet - run reconstruction first"
        )
    row = await db.get(WorldVersion, w.current_version_id)
    if row is None:
        raise HTTPException(
            409,
            f"Current version '{w.current_version_id}' has no version record",
        )
    if row.world_id != world_id:
        raise HTTPException(
            409,
            f"Current version '{w.current_version_id}' does not belong to world {world_id}",
        )
    return row


def _load_world_or_409(row: WorldVersion):
    """Parse the version's WorldStore artifact. Corruption/absence is a 409
    with the store's own diagnosis -- never a silently empty world."""
    from worldstore.store import WorldStore, WorldStoreError

    store = WorldStore(_worldstore_root())
    try:
        return store.load_version(row.id)
    except (WorldStoreError, FileNotFoundError) as exc:
        raise HTTPException(409, f"Version '{row.id}' artifact unreadable: {exc}")


@worlds.get("/{world_id}/worldir")
async def world_worldir(
    world_id: str, version: str | None = None, db: AsyncSession = Depends(get_db)
) -> dict:
    """The world's compiled WorldIR (current version unless ?version=)."""
    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)
    return world.to_dict()


@worlds.get("/{world_id}/entities/{entity_id}/provenance")
async def entity_provenance(
    world_id: str, entity_id: str, version: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Trace an entity back to the Evidence it was built from.

    The compile pipeline (engine.pipeline.vertical_slice) does not yet
    stamp per-observation evidence links on every entity it produces --
    only the version's source_session_ids record which capture sessions
    fed it. So tracing has two honest precision levels, tried in order:

      1. observation: an Observation on the entity carries a data_hash
         that matches a real Evidence.checksum (or a data_uri matching
         Evidence.artifact_uri) -- exact, file-level provenance, for
         whichever pipeline stage populates it.
      2. session: no per-observation match exists, so this falls back to
         every Evidence row belonging to the version's source sessions --
         still real, just coarser (this entity came from *this session's*
         capture, not a specific frame within it).

    A PROCEDURAL entity (e.g. a generated room) has no source sessions at
    all -- trace_level is "none", never a fabricated session guess.
    """
    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)
    entity = world.entities.get(entity_id)
    if entity is None:
        raise HTTPException(404, f"Entity '{entity_id}' not found in version '{row.id}'")

    observation_matches: list[dict] = []
    for obs in getattr(entity, "observations", None) or []:
        ev = None
        if obs.data_hash:
            res = await db.execute(select(Evidence).where(Evidence.checksum == obs.data_hash))
            ev = res.scalar_one_or_none()
        if ev is None and obs.data_uri:
            res = await db.execute(select(Evidence).where(Evidence.artifact_uri == obs.data_uri))
            ev = res.scalar_one_or_none()
        if ev is not None:
            observation_matches.append({
                "observation_id": obs.id,
                "evidence_id": ev.id,
                "evidence_name": ev.name,
                "evidence_type": ev.type,
            })

    if observation_matches:
        return {
            "entity_id": entity_id,
            "version_id": row.id,
            "provenance": entity.provenance.value if entity.provenance else None,
            "trace_level": "observation",
            "evidence": observation_matches,
        }

    session_ids = row.source_session_ids or []
    if session_ids:
        res = await db.execute(select(Evidence).where(Evidence.session_id.in_(session_ids)))
        session_evidence = [
            {"evidence_id": e.id, "evidence_name": e.name, "evidence_type": e.type,
             "session_id": e.session_id}
            for e in res.scalars().all()
        ]
        return {
            "entity_id": entity_id,
            "version_id": row.id,
            "provenance": entity.provenance.value if entity.provenance else None,
            "trace_level": "session",
            "source_session_ids": session_ids,
            "evidence": session_evidence,
        }

    return {
        "entity_id": entity_id,
        "version_id": row.id,
        "provenance": entity.provenance.value if entity.provenance else None,
        "trace_level": "none",
        "evidence": [],
        "reason": "This version has no source capture sessions (procedurally generated, or predates session tracking).",
    }


@worlds.get("/{world_id}/quality")
async def world_quality(
    world_id: str, version: str | None = None, db: AsyncSession = Depends(get_db)
) -> dict:
    """Per-entity quality bands plus the place-level summary the viewer's quality card shows (B-07)."""
    row = await _version_row(db, world_id, version)
    world_row = await db.get(World, world_id)
    world = _load_world_or_409(row)
    card = (await place_summary.summaries(db, [world_row]))[world_id]
    out = place_summary.quality_view(world, card, row.report)
    out["version_id"] = row.id
    return out


@worlds.get("/{world_id}/report")
async def world_report(
    world_id: str, version: str | None = None, db: AsyncSession = Depends(get_db)
) -> dict:
    """The version's compile-pipeline report (current version unless
    ?version=) -- the real report.json stages, never fabricated."""
    row = await _version_row(db, world_id, version)
    if row.report is None:
        raise HTTPException(404, f"No pipeline report recorded for version '{row.id}'")
    return row.report


@worlds.get("/{world_id}/points")
async def world_points(
    world_id: str, version: str | None = None, db: AsyncSession = Depends(get_db)
) -> Response:
    """The version's reconstruction points as one PLY stream, rendered from
    the geometries' own artifact bytes (real data only)."""
    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)
    ply = _render_points_ply(world)
    if ply is None:
        raise HTTPException(
            404,
            "Version has no point geometry - reconstruction produced no renderable points",
        )
    return Response(
        content=ply,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{row.id}_points.ply"'},
    )


def _render_points_ply(world) -> bytes | None:
    """Collect point positions from the world's geometries into one ASCII
    PLY. Each geometry's data_uri artifact is decoded with the canonical
    PointCloudData reader (RESPC001) when it carries one; geometries
    without a resolvable payload are skipped and counted in a comment
    header. Returns None when nothing usable exists (honest empty, not
    zeros)."""
    geoms = getattr(world, "geometries", None) or {}
    positions: list[tuple[float, float, float]] = []
    skipped = 0
    stores = None
    for g in geoms.values():
        data_uri = getattr(g, "data_uri", None)
        pts: list[tuple[float, float, float]] = []
        if data_uri:
            if stores is None:
                from world_ir.artifact_store import FileArtifactStore

                # Geometry payloads live in the pipeline-artifacts store
                # the compile pipeline was given; the WorldStore artifacts
                # store is the legacy location. Consult both so a version
                # renders regardless of which store its geometries reference.
                stores = [
                    FileArtifactStore(_worldstore_root() / "pipeline-artifacts"),
                    FileArtifactStore(_worldstore_root() / "artifacts"),
                ]
            raw = b""
            for store in stores:
                try:
                    raw = store.get(data_uri)
                except Exception:
                    continue
                break
            pts = _decode_point_payload(raw)
        if pts:
            positions.extend(pts)
        else:
            skipped += 1
    if not positions:
        return None

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"comment reality-engine world points; geometries={len(geoms)} "
        f"skipped={skipped}\n"
        f"element vertex {len(positions)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    ).encode("ascii")
    body = "".join(
        f"{x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in positions
    ).encode("ascii")
    return header + body


def _decode_point_payload(raw: bytes) -> list[tuple[float, float, float]]:
    """Decode a geometry artifact's point positions. Canonical Reality
    Engine payloads (RESPC001 PointCloudData) take the exact reader; a
    PLY fallback covers backend-parsed artifacts. Empty for anything
    else -- bytes are never guessed into coordinates."""
    if raw[:8] == b"RESPC001":
        from world_ir.geometry_data import PointCloudData

        try:
            cloud = PointCloudData.from_bytes(raw)
        except ValueError:
            return []
        return [(p[0], p[1], p[2]) for p in cloud.points]
    # PLY fallback: parse the header-declared vertex layout (ascii and
    # binary_little_endian); anything else decodes to empty, never crash.
    head = raw[:4096]
    end = head.find(b"end_header")
    if end == -1:
        return []
    header = head[:end].decode("ascii", "replace")
    body = raw[end + len(b"end_header"):].lstrip(b"\r\n")
    binary = "format binary_little_endian" in header
    n = 0
    props: list[str] = []
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        elif line.startswith("property"):
            props.append(line.split()[-1])
    if n <= 0 or not {"x", "y", "z"} <= set(props):
        return []
    out: list[tuple[float, float, float]] = []
    if binary:
        stride = 4 * len(props)
        xi, yi, zi = props.index("x"), props.index("y"), props.index("z")
        for i in range(n):
            off = i * stride
            if off + stride > len(body):
                break
            vals = struct.unpack_from("<" + "f" * len(props), body, off)
            out.append((vals[xi], vals[yi], vals[zi]))
        return out
    count = 0
    for raw_line in body.splitlines():
        if count >= n:
            break
        parts = raw_line.split()
        if len(parts) < len(props):
            continue
        vals = dict(zip(props, parts))
        try:
            out.append((float(vals["x"]), float(vals["y"]), float(vals["z"])))
            count += 1
        except ValueError:
            continue
    return out


def _keep_registered_cameras(cameras: list, world) -> list:
    """Only photographs the version registered. The job records per photograph whether it registered
    (metadata.evidence.used[].registered); a level-0 fusion's per-entity camera poses are display layout and carry no
    registration. Worlds with no such record are returned unchanged rather than emptied."""
    used = ((getattr(world, "metadata", None) or {}).get("evidence") or {}).get("used")
    if not used:
        return cameras
    registered = {u.get("evidence_id") for u in used if u.get("registered")}
    return [c for c in cameras if c.get("evidence_id") in registered]


@worlds.get("/{world_id}/cameras")
async def world_cameras(
    world_id: str, version: str | None = None, db: AsyncSession = Depends(get_db)
) -> dict:
    """Registered camera poses for the version, in the field shape the
    Studio's camera layer consumes (position_m / rotation_wxyz, world
    frame, +Y up, camera-to-world quaternions)."""
    row = await _version_row(db, world_id, version)
    world = _load_world_or_409(row)

    cameras = []
    image_size = None
    for ent in (world.entities or {}).values():
        obs = getattr(ent, "observations", None) or []
        for ob in obs:
            meta = getattr(ob, "metadata", None)
            if not isinstance(meta, dict):
                continue
            pose = meta.get("camera_pose")
            if isinstance(pose, dict) and "position" in pose:
                cameras.append({
                    "evidence_id": meta.get("evidence_id") or ent.id,
                    "position_m": [float(c) for c in pose["position"]],
                    "rotation_wxyz": [
                        float(c) for c in pose.get("rotation", [1.0, 0.0, 0.0, 0.0])
                    ],
                })
            if image_size is None and isinstance(meta.get("image_size"), (list, tuple)):
                image_size = [int(v) for v in meta["image_size"]]
    if not cameras and row.cameras_artifact_uri:
        # A COLMAP-reconstructed world keeps its registered poses in the version's cameras artifact (written by the
        # job at adoption), not on entity observations (only the single-image bootstrap puts a pose there). Without
        # this the Studio's camera layer was empty for every real reconstruction.
        from apps.api.storage import resolve_artifact

        path = resolve_artifact(row.cameras_artifact_uri)
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path is not None else {}
        except (OSError, ValueError):
            payload = {}
        for cam in payload.get("cameras") or []:
            if isinstance(cam, dict) and "position_m" in cam:
                cameras.append({"evidence_id": cam.get("evidence_id"),
                                "position_m": [float(c) for c in cam["position_m"]],
                                "rotation_wxyz": [float(c) for c in cam.get("rotation_wxyz", [1.0, 0.0, 0.0, 0.0])]})
        if image_size is None and isinstance(payload.get("image_size"), (list, tuple)):
            image_size = [int(v) for v in payload["image_size"]]
    cameras = _keep_registered_cameras(cameras, world)
    return {
        "frame": "world (meters, +Y up after frame canonicalization)",
        "rotation_convention": "camera-to-world quaternion (w, x, y, z)",
        "image_size": image_size,
        "version_id": row.id,
        "cameras": cameras,
    }


class DiffRequest(BaseModel):
    base_version: str | None = None
    head_version: str | None = None


@worlds.get("/{world_id}/diff")
async def get_world_diff(world_id: str, base: str | None = None, head: str | None = None, db: AsyncSession = Depends(get_db)):
    """Compute structural diff between two WorldIR versions."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    
    store_path = _worldstore_root()
    if not store_path.exists():
        raise HTTPException(404, "WorldStore not found")
    store = WorldStore(str(store_path))
    
    # Default to current version as head
    head_version = head or w.current_version_id
    if not head_version:
        raise HTTPException(404, "No current version to diff")
    
    # Default to parent of head as base
    if base is None:
        head_record = store._record(head_version)
        base = head_record.get("parent")
    
    if not base:
        raise HTTPException(404, "No base version available for diff")
    
    try:
        before = store.load_version(base)
        after = store.load_version(head_version)
    except Exception as e:
        raise HTTPException(404, f"Version not found: {e}")
    
    world_diff = diff_worlds(before, after)
    return world_diff.to_dict()


# Fields a correction commit may change on an entity. Everything else
# (id, geometry/material/surface/component linkage, provenance,
# observations, relationships, ...) is rejected: silently ignoring an
# unsupported field while still minting a version made failed corrections
# look successful, and setattr on an arbitrary attribute could corrupt
# entity invariants (e.g. overwriting `id` or `type` with a plain string).
_COMMIT_MUTABLE_FIELDS = (
    "name",
    "type",
    "confidence",
    "semantic_labels",
    "custom_properties",
    "transform",
    "relationships",
    "boundary_element_ids",
    "connected_level_ids",
    "connected_room_ids",
    # F6: a typed real dimension. Applied to the world (entity + replayable registry + VERIFIED state), never to an
    # arbitrary entity attribute -- handled explicitly below, not by setattr.
    "user_correction",
)
_MAX_COMMIT_CHANGES = 32
_MAX_COMMIT_VALUE_BYTES = 16 * 1024


class CommitRequest(BaseModel):
    entity_id: str
    changes: dict
    parent_version_id: str | None = None
    commit_message: str | None = None


def _assert_all_finite(value, field: str) -> None:
    """A literal NaN/Infinity in a stored world is not valid JSON: the browser's JSON.parse rejects the whole
    /worldir response. Python's json accepts the tokens on the way in, so every number is checked here."""
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HTTPException(422, f"field '{field}' contains a non-finite number (NaN/Infinity)")
    elif isinstance(value, dict):
        for v in value.values():
            _assert_all_finite(v, field)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _assert_all_finite(v, field)


def _validate_transform(value: dict) -> dict:
    """{position: {x,y,z}, rotation: {w,x,y,z}} -- finite real numbers, a unit quaternion, nothing else."""
    if set(value) != {"position", "rotation"}:
        raise HTTPException(422, "field 'transform' must be exactly {position: {x,y,z}, rotation: {w,x,y,z}}")
    out: dict = {}
    for part, axes in (("position", "xyz"), ("rotation", "wxyz")):
        sub = value[part]
        if not isinstance(sub, dict) or set(sub) != set(axes):
            raise HTTPException(422, f"transform.{part} must have exactly the numeric keys {list(axes)}")
        for a in axes:
            v = sub[a]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise HTTPException(422, f"transform.{part}.{a} must be a finite number")
        out[part] = {a: float(sub[a]) for a in axes}
    norm = math.sqrt(sum(out["rotation"][a] ** 2 for a in "wxyz"))
    if abs(norm - 1.0) > 1e-3:
        raise HTTPException(422, f"transform.rotation must be a unit quaternion (norm {norm:.4g})")
    return out


def _validate_commit_changes(changes: dict) -> dict:
    """Validate a correction payload. Returns sanitized {field: value};
    raises 422/413 naming the offending field -- never silently drops."""
    if not isinstance(changes, dict) or not changes:
        raise HTTPException(422, "changes must be a non-empty object")
    if len(changes) > _MAX_COMMIT_CHANGES:
        raise HTTPException(
            413, f"too many changed fields ({len(changes)} > {_MAX_COMMIT_CHANGES})"
        )
    validated: dict = {}
    for key, value in changes.items():
        if key not in _COMMIT_MUTABLE_FIELDS:
            raise HTTPException(
                422,
                f"unsupported mutation field '{key}'; mutable fields are "
                f"{list(_COMMIT_MUTABLE_FIELDS)}",
            )
        try:
            size = len(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            raise HTTPException(422, f"field '{key}' is not JSON-serializable")
        if size > _MAX_COMMIT_VALUE_BYTES:
            raise HTTPException(
                413, f"field '{key}' exceeds {_MAX_COMMIT_VALUE_BYTES} bytes"
            )
        _assert_all_finite(value, key)
        if key == "name":
            if not isinstance(value, str) or not value or len(value) > 300:
                raise HTTPException(
                    422, "field 'name' must be a non-empty string of at most 300 chars"
                )
            validated[key] = value
        elif key == "type":
            try:
                validated[key] = EntityType(value)
            except ValueError:
                if value == "stair":
                    validated[key] = EntityType.STAIRS
                else:
                    raise HTTPException(
                        422,
                        f"field 'type' must be one of {[t.value for t in EntityType]}",
                    )
        elif key == "confidence":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise HTTPException(422, "field 'confidence' must be a number")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise HTTPException(
                    422, "field 'confidence' must be finite and within [0, 1]"
                )
            validated[key] = float(value)
        elif key == "semantic_labels":
            if (
                not isinstance(value, list)
                or len(value) > 100
                or not all(isinstance(v, str) for v in value)
            ):
                raise HTTPException(
                    422, "field 'semantic_labels' must be a list of at most 100 strings"
                )
            validated[key] = list(value)
        elif key in ("custom_properties", "transform"):
            if not isinstance(value, dict):
                raise HTTPException(422, f"field '{key}' must be an object")
            validated[key] = _validate_transform(value) if key == "transform" else dict(value)
        elif key == "relationships":
            if not isinstance(value, list) or len(value) > 100:
                raise HTTPException(
                    422, "field 'relationships' must be a list of at most 100 items"
                )
            rels = []
            for item in value:
                if not isinstance(item, dict):
                    raise HTTPException(422, "each relationship must be an object")
                target_id = item.get("target_id") or item.get("target_entity_id")
                if not isinstance(target_id, str) or not target_id or len(target_id) > 300:
                    raise HTTPException(
                        422, "relationship target_id must be a non-empty string <= 300 chars"
                    )
                kind_str = item.get("kind") or item.get("type", "connects")
                try:
                    kind = RelationshipKind(kind_str)
                except ValueError:
                    kind = RelationshipKind.CONNECTS
                # A human-stated correction with no confidence is the human's own
                # assertion (1.0); an INVALID confidence is an error, never
                # silently coerced into certainty.
                try:
                    rel_conf = float(item.get("confidence", 1.0))
                except (TypeError, ValueError):
                    raise HTTPException(422, "relationship confidence must be a number in [0, 1]")
                if not math.isfinite(rel_conf) or not 0.0 <= rel_conf <= 1.0:
                    raise HTTPException(422, "relationship confidence must be a number in [0, 1]")
                rel_meta = item.get("metadata", {})
                if not isinstance(rel_meta, dict):
                    rel_meta = {}
                rels.append({
                    "kind": kind,
                    "target_id": target_id,
                    "confidence": rel_conf,
                    "metadata": rel_meta,
                })
            validated[key] = rels
        elif key in ("boundary_element_ids", "connected_level_ids", "connected_room_ids"):
            if (
                not isinstance(value, list)
                or len(value) > 100
                or not all(isinstance(v, str) and v and len(v) <= 300 for v in value)
            ):
                raise HTTPException(
                    422, f"field '{key}' must be a list of at most 100 non-empty strings <= 300 chars"
                )
            validated[key] = list(value)
    return validated


@worlds.post("/{world_id}/commit")
async def commit_world_correction(world_id: str, body: CommitRequest, db: AsyncSession = Depends(get_db)):
    """Apply a correction to a world entity and persist as new WorldStore version.

    Failure contract: validation failures (422/413), stale parents (409),
    and store failures (404/409) all return BEFORE anything is persisted --
    a failed commit never moves HEAD, never writes a mirror row, and never
    looks successful. A byte-identical retry returns the current version
    (duplicate: true) instead of minting a duplicate version.
    """
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")

    # Load current WorldIR
    worldir = await _load_worldir_from_store(w)
    if worldir is None:
        raise HTTPException(404, "WorldIR not available for this world's current version")

    # Apply the entity change
    entity = worldir.entities.get(body.entity_id)
    if entity is None:
        raise HTTPException(404, f"Entity {body.entity_id} not found in world")

    validated = _validate_commit_changes(body.changes)
    if body.commit_message is not None and len(body.commit_message) > 500:
        raise HTTPException(422, "commit_message must be at most 500 chars")
    # Unify identifiers (same invariant worldstore_service.commit_version
    # enforces): the WorldIR's own id is the application world id, so
    # on-disk versions filter by world and the DB mirror stays truthful.
    worldir.id = w.id
    base_snapshot = worldir.to_dict()
    for key, value in validated.items():
        if key == "user_correction":
            from world_ir.corrections import CorrectionError, make_correction, record_correction

            if not isinstance(value, dict) or value.get("entity_id") != body.entity_id:
                raise HTTPException(422, "user_correction.entity_id must match the committed entity")
            try:
                record = make_correction(worldir, body.entity_id, str(value.get("property")),
                                         value.get("value"), str(value.get("note") or ""))
                record_correction(worldir, record)
            except CorrectionError as exc:
                raise HTTPException(422, str(exc)) from exc
            continue
        if key == "name":
            entity.name = value
        elif key == "type":
            entity.type = value
        elif key == "confidence":
            entity.confidence = value
        elif key == "semantic_labels":
            entity.semantic_labels = value
        elif key == "custom_properties":
            # `reconstruction` is the pipeline's own build trace (session, backend, level): a correction may add or
            # change anything else, but it may not erase or rewrite the provenance of what the engine built.
            new_props = dict(value)
            trace = (entity.custom_properties or {}).get("reconstruction")
            if trace is not None:
                if "reconstruction" in new_props and new_props["reconstruction"] != trace:
                    raise HTTPException(422, "custom_properties.reconstruction is pipeline-owned and cannot be rewritten")
                new_props["reconstruction"] = trace
            entity.custom_properties = new_props
        elif key == "transform":
            entity.transform = value
        elif key == "relationships":
            entity.relationships = [
                Relationship(
                    kind=r["kind"],
                    target_id=r["target_id"],
                    confidence=r["confidence"],
                    provenance=Provenance.INFERRED,
                    metadata=r["metadata"],
                )
                for r in value
            ]
        elif key == "boundary_element_ids":
            for bid in value:
                if bid not in worldir.entities:
                    raise HTTPException(422, f"boundary element '{bid}' not found in world")
            entity.custom_properties = dict(entity.custom_properties)
            entity.custom_properties["boundary_element_ids"] = list(value)
            # Remove old boundary containment edges and re-link
            entity.relationships = [
                r for r in entity.relationships
                if not (r.kind == RelationshipKind.CONTAINS and r.metadata.get("derived_from") == "room_boundary_membership")
            ]
            for bid in value:
                entity.relationships.append(Relationship(
                    kind=RelationshipKind.CONTAINS,
                    target_id=bid,
                    confidence=entity.confidence,
                    provenance=Provenance.INFERRED,
                    metadata={"derived_from": "room_boundary_membership"},
                ))
                b_ent = worldir.entities[bid]
                if not any(r.target_id == entity.id and r.kind == RelationshipKind.PART_OF for r in b_ent.relationships):
                    b_ent.relationships.append(Relationship(
                        kind=RelationshipKind.PART_OF,
                        target_id=entity.id,
                        confidence=entity.confidence,
                        provenance=Provenance.INFERRED,
                        metadata={"derived_from": "room_boundary_membership"},
                    ))
        elif key == "connected_level_ids":
            for lid in value:
                if lid not in worldir.entities:
                    raise HTTPException(422, f"level entity '{lid}' not found in world")
            entity.custom_properties = dict(entity.custom_properties)
            entity.custom_properties["connected_level_ids"] = list(value)
            for lid in value:
                if not any(r.target_id == lid for r in entity.relationships):
                    entity.relationships.append(Relationship(
                        kind=RelationshipKind.CONNECTS,
                        target_id=lid,
                        confidence=entity.confidence,
                        provenance=Provenance.INFERRED,
                        metadata={"derived_from": "stair_storey_connection"},
                    ))
                l_ent = worldir.entities[lid]
                if not any(r.target_id == entity.id for r in l_ent.relationships):
                    l_ent.relationships.append(Relationship(
                        kind=RelationshipKind.CONNECTS,
                        target_id=entity.id,
                        confidence=entity.confidence,
                        provenance=Provenance.INFERRED,
                        metadata={"derived_from": "stair_storey_connection"},
                    ))
        elif key == "connected_room_ids":
            for rid in value:
                if rid not in worldir.entities:
                    raise HTTPException(422, f"room entity '{rid}' not found in world")
            entity.custom_properties = dict(entity.custom_properties)
            entity.custom_properties["connected_room_ids"] = list(value)
            for rid in value:
                if not any(r.target_id == rid for r in entity.relationships):
                    entity.relationships.append(Relationship(
                        kind=RelationshipKind.CONNECTS,
                        target_id=rid,
                        confidence=entity.confidence,
                        provenance=Provenance.INFERRED,
                        metadata={"derived_from": "corridor_room_connection"},
                    ))
                r_ent = worldir.entities[rid]
                if not any(r.target_id == entity.id for r in r_ent.relationships):
                    r_ent.relationships.append(Relationship(
                        kind=RelationshipKind.CONNECTS,
                        target_id=entity.id,
                        confidence=entity.confidence,
                        provenance=Provenance.INFERRED,
                        metadata={"derived_from": "corridor_room_connection"},
                    ))

    # Malformed WorldIR must never be persisted by a correction: validate
    # the corrected world and refuse with the validator's own diagnosis.
    # Only ERRORs block (warnings stay readable, matching the recon path).
    from world_ir.validation import validate_world_ir as _validate_world

    _validation = _validate_world(worldir)
    if not _validation.is_valid():
        raise HTTPException(
            422,
            "corrected world failed validation: "
            + "; ".join(_validation.messages()[:5]),
        )

    # Stale-client guard: an explicit parent that is no longer HEAD is a
    # 409, not a silent fork -- the caller must rebase onto HEAD.
    effective_parent = body.parent_version_id or w.current_version_id
    if body.parent_version_id is not None and body.parent_version_id != w.current_version_id:
        raise HTTPException(
            409,
            f"parent version '{body.parent_version_id}' is stale; "
            f"current HEAD is '{w.current_version_id}' -- reload and reapply",
        )

    # Duplicate-request guard: a byte-identical retry (same content, same
    # parent) returns the adopted version instead of minting a duplicate.
    # modified_at is wall-clock bookkeeping, excluded from the comparison.
    import time
    new_snapshot = worldir.to_dict()
    base_cmp = dict(base_snapshot)
    new_cmp = dict(new_snapshot)
    base_cmp.pop("modified_at", None)
    new_cmp.pop("modified_at", None)
    if (
        effective_parent is not None
        and effective_parent == w.current_version_id
        and json.dumps(base_cmp, sort_keys=True) == json.dumps(new_cmp, sort_keys=True)
    ):
        return {
            "version_id": w.current_version_id,
            "world_id": w.id,
            "entity_id": body.entity_id,
            "changed_fields": [],
            "duplicate": True,
        }
    worldir.modified_at = time.time()

    # Save as new version
    store_path = _worldstore_root()
    if not store_path.exists():
        store_path.mkdir(parents=True, exist_ok=True)
    store = WorldStore(str(store_path))

    from apps.api import worldstore_service as _wss

    adoption_id = _wss.begin_adoption(w.id, effective_parent)
    try:
        stored = store.save_version(worldir, parent=effective_parent, version_id=adoption_id)
    except WorldStoreError as exc:
        _wss.abandon_adoption(adoption_id)
        msg = str(exc)
        if "unknown version" in msg:
            raise HTTPException(404, msg)
        raise HTTPException(409, msg)

    # Atomic adoption: advance HEAD only if it still holds the parent
    # this commit was based on. The conditional UPDATE is a single
    # atomic statement, so two concurrent commits cannot both win -- the
    # loser's rowcount is 0 and it gets an explicit 409 while its saved
    # version file stays on disk (harmless, never adopted). An in-memory
    # re-check could not close this race: concurrent requests use
    # separate sessions whose commits serialize after the check.
    if effective_parent is None:  # defensive: unreachable (404 above)
        raise HTTPException(404, "World has no HEAD version to commit onto")
    from sqlalchemy import update as _sa_update

    head_update = await db.execute(
        _sa_update(World)
        .where(World.id == w.id, World.current_version_id == effective_parent)
        .values(current_version_id=stored.version_id)
    )
    if head_update.rowcount == 0:
        current = (await db.get(World, w.id)).current_version_id
        _wss.abandon_adoption(adoption_id)
        raise HTTPException(
            409,
            f"HEAD moved during commit (now '{current}'); "
            f"version '{stored.version_id}' was saved but not adopted -- "
            "reload and reapply",
        )

    # Adopt atomically: HEAD pointer + DB mirror row + activity share one
    # commit, so a version is never HEAD without its mirror record (which
    # previously made committed worlds unreadable with 409).
    db.add(
        WorldVersion(
            id=stored.version_id,
            world_id=w.id,
            parent_version_id=stored.parent,
            artifact_uri=stored.artifact_uri,
            artifact_hash=stored.artifact_hash,
            source_session_ids=list(stored.source_session_ids),
            changed_entity_ids=list(stored.changed_entity_ids),
        )
    )
    db.add(
        ActivityEvent(
            id=new_id("act"),
            type="world.committed",
            entity_type="world",
            entity_id=w.id,
            summary=(f"Verified {validated['user_correction'].get('property')} on {body.entity_id}: "
                     f"{validated['user_correction'].get('value')}" if "user_correction" in validated
                     else f"Committed correction to {body.entity_id}: {body.commit_message or 'no message'}"),
        )
    )

    try:
        await db.commit()
    except Exception as exc:
        # HEAD never moved and no mirror row exists: the version just written is quarantined (never adopted,
        # never a phantom in the version list), and the failure is reported explicitly instead of a bare 500.
        _wss.abandon_adoption(adoption_id)
        raise HTTPException(
            503, f"database unavailable; commit not adopted: {exc}"
        )
    _wss.finish_adoption(adoption_id)

    return {
        "version_id": stored.version_id,
        "world_id": w.id,
        "entity_id": body.entity_id,
        "changed_fields": list(validated.keys()),
        "duplicate": False,
    }


async def _load_worldir_from_store(world: World, version_id: str | None = None) -> WorldIR | None:
    """Load WorldIR from WorldStore for a World (current or specified version)."""
    vid = version_id or world.current_version_id
    if not vid:
        return None
    store_path = _worldstore_root()
    if not store_path.exists():
        return None
    store = WorldStore(str(store_path))
    try:
        return store.load_version(vid)
    except Exception:
        return None


def _get_worldir_space_graph(worldir: WorldIR) -> dict:
    """Extract or derive interior space graph from WorldIR metadata and entities."""
    if "interior_space_graph" in worldir.metadata:
        return worldir.metadata["interior_space_graph"]

    levels = []
    rooms = []
    corridors = []
    openings = []
    stairs = []

    for eid, entity in worldir.entities.items():
        etype = entity.type.value if hasattr(entity.type, "value") else str(entity.type)
        if etype in ("level", "storey"):
            levels.append({
                "level_id": eid,
                "elevation_m": entity.custom_properties.get("elevation_m", 0.0),
                "room_ids": list(entity.custom_properties.get("room_ids", [])),
                "corridor_ids": list(entity.custom_properties.get("corridor_ids", [])),
                "stair_ids": list(entity.custom_properties.get("stair_ids", [])),
            })
        elif etype == "room":
            rooms.append({
                "room_id": eid,
                "floor_area_m2": entity.custom_properties.get("floor_area_m2", 0.0),
                "adjacent_room_ids": list(entity.custom_properties.get("adjacent_room_ids", [])),
                "corridor_ids": list(entity.custom_properties.get("corridor_ids", [])),
                "boundary_completeness": entity.custom_properties.get("boundary_completeness", 1.0),
                "notes": list(entity.custom_properties.get("notes", [])),
                "status": entity.custom_properties.get("status", "detected"),
            })
        elif etype == "corridor":
            corridors.append({
                "corridor_id": eid,
                "aspect_ratio": entity.custom_properties.get("aspect_ratio", 2.0),
                "width_m": entity.custom_properties.get("width_m", 1.2),
                "length_m": entity.custom_properties.get("length_m", 3.0),
                "height_m": entity.custom_properties.get("height_m", 2.4),
                "floor_area_m2": entity.custom_properties.get("floor_area_m2", 3.6),
                "connected_room_ids": list(entity.custom_properties.get("connected_room_ids", [])),
                "longitudinal_axis": list(entity.custom_properties.get("longitudinal_axis", [1.0, 0.0, 0.0])),
                "bounds_min": list(entity.custom_properties.get("bounds_min", [0.0, 0.0, 0.0])),
                "bounds_max": list(entity.custom_properties.get("bounds_max", [0.0, 0.0, 0.0])),
            })
        elif etype in ("door", "window"):
            openings.append({
                "opening_id": eid,
                "kind": "window" if etype == "window" else "doorway",
                "width_m": entity.custom_properties.get("width_m", 0.9),
                "height_m": entity.custom_properties.get("height_m", 2.1),
                "sill_height_m": entity.custom_properties.get("sill_height_m", 0.0),
                "connected_space_ids": list(entity.custom_properties.get("connected_space_ids", [])),
                "is_exterior": entity.custom_properties.get("is_exterior", etype == "window"),
            })
        elif etype in ("stairs", "stair"):
            stairs.append({
                "stair_id": eid,
                "step_count": entity.custom_properties.get("step_count", 0),
                "total_rise_m": entity.custom_properties.get("total_rise_m", 0.0),
                "total_run_m": entity.custom_properties.get("total_run_m", 0.0),
                "connected_level_ids": list(entity.custom_properties.get("connected_level_ids", [])),
            })

    # Traverse entity relationships
    for eid, entity in worldir.entities.items():
        for rel in (entity.relationships or []):
            rkind = rel.kind.value if hasattr(rel.kind, "value") else str(rel.kind)
            target_id = rel.target_id
            if rkind == "connects":
                for op in openings:
                    if op["opening_id"] == eid and target_id not in op["connected_space_ids"]:
                        op["connected_space_ids"].append(target_id)
            elif rkind == "part_of":
                for lvl in levels:
                    if lvl["level_id"] == target_id:
                        if ("room" in eid or entity.type == "room") and eid not in lvl["room_ids"]:
                            lvl["room_ids"].append(eid)
                        elif ("corridor" in eid or entity.type == "corridor") and eid not in lvl["corridor_ids"]:
                            lvl["corridor_ids"].append(eid)

    return {
        "building_id": worldir.id,
        "levels": levels,
        "rooms": rooms,
        "corridors": corridors,
        "openings": openings,
        "stairs": stairs,
        "summary": {
            "level_count": len(levels),
            "room_count": len(rooms),
            "corridor_count": len(corridors),
            "opening_count": len(openings),
            "stair_count": len(stairs),
        },
    }


@worlds.get("/{world_id}/space-graph")
async def get_world_space_graph(
    world_id: str,
    version_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Retrieve the authoritative InteriorSpaceGraph for a world."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")

    worldir = await _load_worldir_from_store(w, version_id)
    if worldir is None:
        raise HTTPException(404, "WorldIR not available for this world/version")

    return _get_worldir_space_graph(worldir)


@worlds.get("/{world_id}/query/spaces")
async def query_world_spaces(
    world_id: str,
    room_id: str | None = None,
    corridor_id: str | None = None,
    level_id: str | None = None,
    space_id: str | None = None,
    space_a: str | None = None,
    space_b: str | None = None,
    level_a: str | None = None,
    level_b: str | None = None,
    version_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Query topological connectivity and spatial metrics within a world."""
    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")

    worldir = await _load_worldir_from_store(w, version_id)
    if worldir is None:
        raise HTTPException(404, "WorldIR not available for this world/version")

    graph_dict = _get_worldir_space_graph(worldir)
    from perception.architecture.space_graph import InteriorSpaceGraph
    graph = InteriorSpaceGraph.from_dict(graph_dict)

    if corridor_id:
        return {
            "query": "corridor_reachability",
            "corridor_id": corridor_id,
            "reachable_rooms": graph.rooms_reachable_from_corridor(corridor_id),
        }
    if room_id:
        return {
            "query": "connected_rooms",
            "room_id": room_id,
            "connected_spaces": graph.rooms_connected_to(room_id),
        }
    if space_a and space_b:
        return {
            "query": "openings_connecting",
            "space_a": space_a,
            "space_b": space_b,
            "openings": graph.openings_connecting(space_a, space_b),
        }
    if level_a and level_b:
        return {
            "query": "stairs_connecting_levels",
            "level_a": level_a,
            "level_b": level_b,
            "stairs": graph.stairs_connecting_levels(level_a, level_b),
        }
    if space_id:
        return {
            "query": "space_info",
            "space_id": space_id,
            "level": graph.level_of_space(space_id),
            "evidence": graph.trace_space_evidence(space_id),
        }
    if level_id:
        return {
            "query": "level_spaces",
            "level_id": level_id,
            "level": graph.levels.get(level_id),
        }

    return {"query": "all", "space_graph": graph_dict}



class RollbackIn(BaseModel):
    version_id: str


@worlds.post("/{world_id}/rollback")
async def rollback_world(world_id: str, body: RollbackIn, db: AsyncSession = Depends(get_db)) -> dict:
    """Move HEAD back to an earlier version (history stays immutable; see worldstore_service.rollback_head)."""
    from apps.api import worldstore_service as wss
    from apps.api.models import Job, Session as _Session

    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    sids = (await db.execute(select(_Session.id).where(_Session.world_id == world_id))).scalars().all()
    busy = (await db.execute(
        select(Job.id).where(Job.entity_type == "session", Job.entity_id.in_(sids),
                             Job.status.in_(("queued", "running"))).limit(1)
    )).scalars().first() if sids else None
    if busy:
        raise HTTPException(409, f"a reconstruction is in progress (job {busy}); wait for it before rolling back")
    before = w.current_version_id
    try:
        row = await wss.rollback_head(db, world_id, body.version_id)
    except wss.ConcurrentModificationError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"world_id": world_id, "version_id": row.id, "previous_head": before,
            "note": "history is unchanged; evidence is kept and will be retried by the next reconstruction"}


class ScaleIn(BaseModel):
    """One scale fact. ``method`` selects the contract:

    * camera baseline (default ``manual_measurement`` or any free text): evidence_id_a/b + distance_m.
    * ``known_length``: distance_m (real length, metres) + either ``point_a``/``point_b`` (viewer/world
      coordinates of the two picked points) or ``model_length`` (the same length in model units).
      Evidence ids are optional (the server fingerprints the model unit itself).
    * ``prior_accepted_by_user``: ``source_entity_id`` from GET /scale/suggestion; ``distance_m`` optional
      (defaults to the suggested value; must lie in the suggestion's typical range). The server recomputes
      the model length from the world -- the client cannot supply it.
    """

    evidence_id_a: str | None = None
    evidence_id_b: str | None = None
    distance_m: float | None = None
    method: str = "manual_measurement"
    point_a: list[float] | None = None
    point_b: list[float] | None = None
    model_length: float | None = None
    source_entity_id: str | None = None


def _head_scale_context(row: WorldVersion) -> tuple[dict, float, str | None, str | None, float | None]:
    """(world metadata scale, metres-per-unit of the current version or 1.0, cam_a, cam_b, their pre-scale
    separation) -- the camera pair fingerprints the model unit for model-space references."""
    world = _load_world_or_409(row)
    scale = dict((world.metadata or {}).get("scale") or {})
    # Displayed units per solver unit. Metric: metres per unit. Relative: the working unit the structure was found in
    # (reconstruction.scene_scale), so a length picked in the viewer is converted back to the solver units references
    # are recorded and applied in.
    mpu = (float(scale.get("meters_per_unit") or 1.0) if scale.get("state") == "metric"
           else float(scale.get("provisional_unit_k") or 1.0))
    cams = (row.report or {}).get("cameras") or {}
    ids = sorted(k for k, v in cams.items() if isinstance(v, (list, tuple)) and len(v) == 3)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sep = math.dist(cams[a], cams[b]) / mpu
            if sep > 0.0:
                return scale, mpu, a, b, sep
    return scale, mpu, None, None, None


def _suggestion_key(sug) -> str:
    return f"{sug.source_entity_id}:{sug.measured_model_units:.6g}"


@worlds.get("/{world_id}/scale/suggestion")
async def get_scale_suggestion(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    """A typical-size scale suggestion from a detected standard object (a door), for a RELATIVE world.

    Nothing is applied. ``suggestion`` is null -- never a default -- when the world is already metric or no
    door with a measured height exists. Accept = POST /scale with method=prior_accepted_by_user; reject =
    POST /scale/suggestion/reject."""
    from apps.api import worldstore_service as wss
    from reconstruction.scale import suggest_scale_from_world

    row = await _version_row(db, world_id, None)
    world = _load_world_or_409(row)
    meta = (world.metadata or {}).get("scale") or {}
    state = meta.get("state") or "relative"
    sug = suggest_scale_from_world(world)
    out = {"world_id": world_id, "scale_state": state, "suggestion": None, "rejected": False,
           "dimension_state": meta.get("dimension_state"), "scale_basis": meta.get("scale_basis"), "reason": None}
    if sug is None:
        out["reason"] = ("the world already has a metric scale" if state == "metric"
                         else "no door with a measured height was detected, so there is nothing to suggest")
        return out
    out["suggestion"] = sug.to_dict()
    out["rejected"] = _suggestion_key(sug) in wss.list_rejected_suggestions(world_id)
    return out


class RejectIn(BaseModel):
    source_entity_id: str


@worlds.post("/{world_id}/scale/suggestion/reject")
async def reject_scale_suggestion(world_id: str, body: RejectIn, db: AsyncSession = Depends(get_db)) -> dict:
    """Decline the current suggestion. Changes nothing about the world: it stays RELATIVE."""
    from apps.api import worldstore_service as wss
    from reconstruction.scale import suggest_scale_from_world

    row = await _version_row(db, world_id, None)
    sug = suggest_scale_from_world(_load_world_or_409(row))
    if sug is None or sug.source_entity_id != body.source_entity_id:
        raise HTTPException(409, "that suggestion is no longer current")
    wss.reject_scale_suggestion(world_id, _suggestion_key(sug))
    return {"world_id": world_id, "rejected": True, "scale_state": "unchanged"}


@worlds.post("/{world_id}/scale", status_code=202)
async def set_world_scale(world_id: str, body: ScaleIn, db: AsyncSession = Depends(get_db)) -> dict:
    """Anchor the world's scale to a distance the user measured between two of its photographs, a length
    they know in the model (``known_length``), or a typical size they explicitly accepted
    (``prior_accepted_by_user``, always labelled ESTIMATED).

    Never assumes meters: the reference is recorded, and a reconstruction of the SAME world is queued that
    applies it through the existing scale anchoring. The result is a new version of the same world whose
    report states the scale source and ratio spread; if the reference cannot be applied, the world stays
    RELATIVE and the report says why."""
    from apps.api import jobs as jobrunner
    from apps.api import worldstore_service as wss
    from apps.api.models import Job, Session as _Session
    from reconstruction.scale import (
        KNOWN_LENGTH, MODEL_SPACE_METHODS, PRIOR_ACCEPTED, ScaleAnchoringError, ScaleReference, gate_ratios,
        suggest_scale_from_world,
    )

    w = await db.get(World, world_id)
    if w is None:
        raise HTTPException(404, "World not found")
    sessions = (await db.execute(
        select(_Session).where(_Session.world_id == world_id).order_by(_Session.created_at.desc()))).scalars().all()
    known = set()
    if sessions:
        known = set((await db.execute(
            select(Evidence.id).where(Evidence.session_id.in_([s.id for s in sessions]), Evidence.type == "photo")
        )).scalars().all())
    model_space = body.method in MODEL_SPACE_METHODS
    if not model_space:
        if body.evidence_id_a is None or body.evidence_id_b is None or body.distance_m is None:
            raise HTTPException(422, "evidence_id_a, evidence_id_b and distance_m are required")
        for eid in (body.evidence_id_a, body.evidence_id_b):
            if eid not in known:
                raise HTTPException(422, f"evidence '{eid}' is not a photograph of this world")
    busy = (await db.execute(
        select(Job.id).where(Job.entity_type == "session", Job.entity_id.in_([s.id for s in sessions]),
                             Job.status.in_(("queued", "running"))).limit(1)
    )).scalars().first()
    if busy:
        raise HTTPException(409, f"a reconstruction is in progress (job {busy}); apply the scale when it finishes")
    model_distance = unit_check = None
    a, b = body.evidence_id_a, body.evidence_id_b
    distance_m = body.distance_m
    if model_space:
        row = await _version_row(db, world_id, None)
        scale, mpu, ca, cb, csep = _head_scale_context(row)
        if ca is None:
            raise HTTPException(422, "this world has no registered camera pair to tie the length to")
        a, b, unit_check = ca, cb, csep
        if body.method == PRIOR_ACCEPTED:
            sug = suggest_scale_from_world(_load_world_or_409(row))
            if scale.get("state") == "metric":
                raise HTTPException(409, "the world already has a metric scale; a typical-size estimate "
                                         "never replaces it")
            if sug is None or sug.source_entity_id != body.source_entity_id:
                raise HTTPException(409, "that suggestion is not current (or none exists); fetch it again")
            distance_m = sug.suggested_value_m if distance_m is None else distance_m
            lo, hi = sug.typical_range_m
            if not (lo <= distance_m <= hi):
                raise HTTPException(422, f"accepted size must lie in the typical range {lo}-{hi} m")
            model_distance = sug.measured_model_units / mpu     # displayed (working) units -> solver units
        else:  # KNOWN_LENGTH
            if distance_m is None:
                raise HTTPException(422, "distance_m (the real length) is required")
            if body.point_a is not None and body.point_b is not None:
                if len(body.point_a) != 3 or len(body.point_b) != 3:
                    raise HTTPException(422, "points must be [x, y, z]")
                model_distance = math.dist(body.point_a, body.point_b) / mpu
            elif body.model_length is not None:
                model_distance = body.model_length / mpu
            else:
                raise HTTPException(422, "give point_a and point_b, or model_length")
        try:
            ScaleReference(a, b, float(distance_m), body.method, model_distance, unit_check)
            if body.method == KNOWN_LENGTH:          # the new length must agree with the ones already given
                earlier = [r for r in wss.list_scale_references(world_id)
                           if r.get("method") == KNOWN_LENGTH and r.get("model_distance")]
                gate_ratios([r["distance_m"] / r["model_distance"] for r in earlier]
                            + [float(distance_m) / model_distance])
        except ScaleAnchoringError as exc:
            raise HTTPException(422, str(exc)) from exc
    try:
        ref = wss.add_scale_reference(world_id, a, b, distance_m, body.method, model_distance, unit_check)
    except ScaleAnchoringError as exc:
        raise HTTPException(422, str(exc)) from exc
    job = jobrunner.enqueue_job(db, jobrunner.RECONSTRUCT_SESSION, "session", sessions[0].id)
    await db.commit()
    return {"world_id": world_id, "job_id": job.id, "reference": ref,
            "dimension_state_after": "ESTIMATED" if body.method == PRIOR_ACCEPTED else "MEASURED",
            "note": "the same world will be re-evaluated with this measured distance; a new version is created "
                    "only if the authoritative geometry changes"}
