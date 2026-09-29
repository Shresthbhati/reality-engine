"""Routes: the one-action product surface.

    POST /api/reconstructions        photos in -> (world, session, evidence, job) out
    GET  /api/worlds/{id}/status     everything the UI needs to show a world's state

The user never creates a World or Session or attaches anything: the first
evidence creates both, later evidence joins the SAME world, and the
reconstruction job rebuilds that world from the union of its evidence.
Worlds/Sessions/Jobs remain available as advanced routes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import jobs as jobrunner
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
from apps.api.reconstruction_state import (
    ACTIVE_STATES,
    LABELS,
    batch_headline,
    derive_state,
    failure_message,
)
from apps.api.routes_misc import _classify, _read_bounded_upload, _safe_filename
from apps.api.storage import store_bytes
from evidence.image_check import DECLARABLE_CLASSES, inspect_image

reconstructions = APIRouter(prefix="/api/reconstructions", tags=["reconstructions"])
world_status = APIRouter(prefix="/api/worlds", tags=["reconstructions"])

MAX_FILES_PER_REQUEST = 300
#: frames kept from one video (evenly spaced in time; the video itself is
#: also stored as evidence). "Never blindly process every frame."
VIDEO_FRAME_BUDGET = 24
_VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")


async def _video_frames(path, fname: str, evidence_class: str | None):
    """(frames, problem): sample the video with the repo's frame-selection policy
    and return decodable frames as (name, jpeg_bytes, digest, facts_dict, extra_meta).
    ``problem`` is a human reason when the video cannot yield usable frames."""
    from evidence.frames import UniformTimeSamplingStrategy
    from evidence.importers import _enumerate_video_candidates, _extract_selected_frames

    try:
        candidates, container = await asyncio.to_thread(_enumerate_video_candidates, str(path))
        selected = UniformTimeSamplingStrategy(target_count=VIDEO_FRAME_BUDGET).select(candidates)
        extracted = await asyncio.to_thread(_extract_selected_frames, str(path), selected)
    except Exception as exc:  # corrupt/unsupported container, no cv2, ...
        return [], f"video could not be read ({type(exc).__name__}: {exc})"
    stem = PurePosixPath(fname).stem
    by_index = {c.frame_index: c for c in selected}
    frames = []
    for idx, jpeg in extracted:
        digest, fpath = store_bytes(jpeg)
        facts = inspect_image(fpath, evidence_class)
        if not facts.ok:
            continue
        cand = by_index[idx]
        frames.append((f"{stem}#frame-{idx:06d}.jpg", jpeg, digest, facts.to_dict(), {
            "source_video": fname, "frame_index": idx, "video_timestamp_s": cand.timestamp_s,
            "fps": container.get("fps"), "selection": f"uniform_time_sampling(target={VIDEO_FRAME_BUDGET})",
        }))
    if not frames:
        return [], "no decodable frames could be extracted from the video"
    return frames, None


@reconstructions.post("", status_code=202)
async def create_reconstruction(
    files: list[UploadFile],
    world_id: str | None = Form(default=None),
    name: str | None = Form(default=None),
    evidence_class: str | None = Form(default=None),
    latitude: float | None = Form(default=None),
    longitude: float | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Build (or improve) a world from photographs.

    No world_id -> a new World is created. With world_id -> the photos join
    that World and its model is rebuilt from ALL of its evidence."""
    from apps.api.validation import require_latitude, require_longitude, require_name

    if not files:
        raise HTTPException(422, "no files were provided")
    if len(files) > MAX_FILES_PER_REQUEST:
        raise HTTPException(413, f"at most {MAX_FILES_PER_REQUEST} files per request")
    if evidence_class is not None and evidence_class not in DECLARABLE_CLASSES:
        raise HTTPException(422, f"evidence_class must be one of {sorted(DECLARABLE_CLASSES)}")
    if (latitude is None) != (longitude is None):
        raise HTTPException(422, "latitude and longitude must be given together")

    accepted: list[tuple[str, str | None, bytes, str, str, dict, dict]] = []
    rejected: list[dict] = []
    for f in files:
        data = await _read_bounded_upload(f)
        fname = _safe_filename(f.filename)
        if not data:
            rejected.append({"name": fname, "reason": "empty file"})
            continue
        kind = _classify(fname, f.content_type)
        if kind == "dataset" and fname.lower().endswith(_VIDEO_EXT):
            kind = "video"
        digest, path = store_bytes(data)
        if kind == "photo":
            facts = inspect_image(path, evidence_class)
            if not facts.ok:
                rejected.append({"name": fname, "reason": facts.reason})
                continue
            accepted.append((fname, f.content_type, data, kind, digest, facts.to_dict(), {}))
        elif kind == "video":
            frames, problem = await _video_frames(path, fname, evidence_class)
            if problem:
                rejected.append({"name": fname, "reason": problem})
                continue
            # the original video is kept as evidence (never re-encoded); its
            # sampled frames are DERIVED photo evidence that carry its identity
            accepted.append((fname, f.content_type, data, kind, digest, {}, {"frames_extracted": len(frames)}))
            for fr_name, jpeg, fr_digest, fr_facts, fr_meta in frames:
                accepted.append((fr_name, "image/jpeg", jpeg, "photo", fr_digest, fr_facts, fr_meta))
        else:
            accepted.append((fname, f.content_type, data, kind, digest, {}, {}))
    photos = [a for a in accepted if a[3] == "photo"]
    if not photos:
        raise HTTPException(422, {
            "message": "none of the files is a usable photograph",
            "rejected": rejected,
        })

    now = datetime.now(timezone.utc)
    if world_id:
        world = await db.get(World, world_id)
        if world is None:
            raise HTTPException(404, "World not found")
    else:
        world = World(
            id=new_id("wld"),
            name=require_name(name or f"Reconstruction {now:%Y-%m-%d %H:%M}", "world name"),
            latitude=require_latitude(latitude) if latitude is not None else None,
            longitude=require_longitude(longitude) if longitude is not None else None,
        )
        db.add(world)
        db.add(ActivityEvent(id=new_id("act"), type="world.created", entity_type="world",
                             entity_id=world.id, summary=f"World '{world.name}' created from photos"))
        await db.flush()

    session = Session(
        id=new_id("ses"), name=f"Capture {now:%Y-%m-%d %H:%M:%S}", status="uploaded",
        world_id=world.id, uploaded_at=utcnow(), captured_at=now,
    )
    db.add(session)
    await db.flush()

    gps = None
    ev_out = []
    for fname, mime, data, kind, digest, facts, extra_meta in accepted:
        up = Upload(id=new_id("upl"), session_id=session.id, status="completed", filename=fname,
                    mime_type=mime, size=len(data), checksum=digest,
                    artifact_uri=f"sha256://{digest}", completed_at=utcnow())
        db.add(up)
        await db.flush()
        meta = {"verified": True, "verified_at": now.isoformat(), "verified_by": "ingest sha256"}
        if facts:
            meta["image_facts"] = facts
            if facts.get("gps") and gps is None:
                gps = tuple(facts["gps"])
        if evidence_class:
            meta["declared_evidence_class"] = evidence_class
        meta.update(extra_meta)
        ev = Evidence(id=new_id("ev"), name=fname, type=kind, session_id=session.id, upload_id=up.id,
                      processing_state="processed", mime_type=mime, size=len(data), checksum=digest,
                      artifact_uri=up.artifact_uri, metadata_json=meta, processed_at=utcnow())
        db.add(ev)
        ev_out.append({"id": ev.id, "name": fname, "type": kind, "evidence_class": facts.get("evidence_class")})
    # location: only what a device / the file itself actually reported
    if latitude is not None and longitude is not None:
        db.add(Location(id=new_id("loc"), entity_type="session", entity_id=session.id,
                        latitude=require_latitude(latitude), longitude=require_longitude(longitude),
                        source="device", captured_at=utcnow()))
    elif gps is not None:
        db.add(Location(id=new_id("loc"), entity_type="session", entity_id=session.id,
                        latitude=gps[0], longitude=gps[1], source="exif", captured_at=utcnow()))
    db.add(ActivityEvent(id=new_id("act"), type="evidence.created", entity_type="session",
                         entity_id=session.id, summary=f"{len(accepted)} file(s) added to '{world.name}'"))
    await db.flush()

    # A reconstruction still waiting in the queue for this world will read the
    # union of evidence when it runs, so it already covers this batch.
    queued = (await db.execute(
        select(Job).where(Job.type == jobrunner.RECONSTRUCT_SESSION, Job.status == "queued",
                          Job.entity_id.in_(select(Session.id).where(Session.world_id == world.id,
                                                                     Session.id != session.id)))
    )).scalars().first()
    if queued is not None:
        job, coalesced = queued, True
    else:
        job = jobrunner.enqueue_job(db, jobrunner.RECONSTRUCT_SESSION, "session", session.id)
        coalesced = False
    session.status = "processing"
    session.processing_started_at = utcnow()
    await db.commit()
    return {
        "world_id": world.id, "session_id": session.id, "job_id": job.id,
        "evidence": ev_out, "rejected": rejected, "coalesced": coalesced,
        "created_world": world_id is None,
    }


def _rows_to_versions(rows: list[WorldVersion], head_id: str | None) -> list[dict]:
    out = []
    for n, v in enumerate(sorted(rows, key=lambda r: (r.created_at, r.id)), start=1):
        rep = v.report or {}
        out.append({
            "id": v.id, "number": n, "label": f"V{n}", "is_current": v.id == head_id,
            "parent_version_id": v.parent_version_id,
            "created_at": v.created_at.isoformat() if v.created_at else None,
            "level": rep.get("level"), "model_state": rep.get("model_state"),
            "images_used": len((rep.get("evidence") or {}).get("input_ids") or []),
        })
    return out


@world_status.get("/{world_id}/status")
async def get_world_status(world_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    world = await db.get(World, world_id)
    if world is None:
        raise HTTPException(404, "World not found")

    sessions = (await db.execute(select(Session).where(Session.world_id == world_id))).scalars().all()
    sid = [s.id for s in sessions]
    ev_rows = (await db.execute(
        select(Evidence).where(Evidence.session_id.in_(sid)).order_by(Evidence.created_at, Evidence.id)
    )).scalars().all() if sid else []
    job = (await db.execute(
        select(Job).where(Job.type == jobrunner.RECONSTRUCT_SESSION, Job.entity_type == "session",
                          Job.entity_id.in_(sid)).order_by(Job.created_at.desc()).limit(1)
    )).scalars().first() if sid else None
    versions = (await db.execute(
        select(WorldVersion).where(WorldVersion.world_id == world_id)
    )).scalars().all()
    head = next((v for v in versions if v.id == world.current_version_id), None)
    report = dict(head.report or {}) if head is not None else {}
    ev_report = report.get("evidence") or {}

    failure_kind = (job.payload or {}).get("failure_kind") if job is not None else None
    state = derive_state(job.status if job else None, job.stage if job else None, failure_kind)
    # Evidence uploaded but no job yet is still "capturing"; a settled world
    # with no job record but a version is inspectable.
    if job is None and head is not None:
        state = "READY_TO_INSPECT" if report.get("status") == "SUCCESS" else "PARTIALLY_COMPLETE"

    per_image = {p["evidence_id"]: p for p in (ev_report.get("contribution") or {}).get("per_image", [])}
    records = {r["evidence_id"]: r for r in ev_report.get("records", [])}
    evidence = []
    for ev in ev_rows:
        facts = (ev.metadata_json or {}).get("image_facts") or {}
        rec = records.get(ev.id, {})
        contrib = per_image.get(ev.id, {})
        evidence.append({
            "id": ev.id, "name": ev.name, "type": ev.type, "size": ev.size,
            "session_id": ev.session_id,
            "evidence_class": rec.get("evidence_class") or facts.get("evidence_class"),
            "class_basis": rec.get("class_basis") or facts.get("class_basis"),
            "used_for_geometry": rec.get("used_for_geometry", facts.get("geometry_eligible")),
            "registered": rec.get("registered"),
            "contribution": contrib.get("label"),
            "in_current_model": ev.id in set(ev_report.get("all_ids") or []),
            # attempts / waiting|registered / ever registered: a photo that could not be
            # placed is "waiting" and is retried on every rebuild, never discarded
            "registration": (ev.metadata_json or {}).get("registration"),
        })

    failure = None
    if job is not None and job.status == "failed":
        failure = {"kind": failure_kind or "error", "message": failure_message(job.error),
                   "job_id": job.id}
    elif job is not None and job.status == "cancelled":
        failure = {"kind": "cancelled", "message": failure_message(job.error), "job_id": job.id}

    in_progress = state in ACTIVE_STATES
    return {
        "world_id": world.id,
        "name": world.name,
        "state": state,
        "state_label": LABELS.get(state, state),
        "in_progress": in_progress,
        "job": None if job is None else {
            "id": job.id, "status": job.status, "stage": job.stage,
            "session_id": job.entity_id,
            "created_at": job.created_at.isoformat() if job.created_at else None,
        },
        "has_model": head is not None,
        "model": None if head is None else {
            "version_id": head.id,
            "level": report.get("level"),
            "level_name": report.get("level_name"),
            "model_state": report.get("model_state"),
            "outcome": report.get("status"),
            "attempts": report.get("attempts", []),
            "scale": ((report.get("stages") or {}).get("scale") or {}),
            "images_used": len(ev_report.get("input_ids") or []),
            "images_registered": len(ev_report.get("registered_ids") or []),
            # what this version changed relative to its predecessor (measured, see world_delta)
            "changes": report.get("changes", []),
            "verdict": (report.get("verdict") or {}).get("verdict"),
            "uncertainties": (report.get("verdict") or {}).get("uncertainties", []),
            # explicit conflicts (both hypotheses + provenance + history), never silently resolved
            "conflicts": report.get("conflicts", []),
        },
        # the newest run, including one whose candidate was NOT adopted (HEAD kept)
        "last_run": None if job is None or not isinstance(job.payload, dict) or "adopted" not in job.payload else {
            "adopted": job.payload.get("adopted"), "verdict": job.payload.get("verdict"),
            "reasons": job.payload.get("reasons", []), "changes": job.payload.get("changes", []),
            "kept_version_id": job.payload.get("kept_version_id"),
        },
        "evidence": evidence,
        "evidence_summary": {
            "count": len(ev_rows),
            "headline": batch_headline((ev_report.get("contribution") or {}).get("summary")),
            "quality": ev_report.get("quality", []),
            "excluded": ev_report.get("excluded", []),
        },
        "guidance": report.get("guidance", []),
        "versions": _rows_to_versions(list(versions), world.current_version_id),
        "failure": failure,
    }
