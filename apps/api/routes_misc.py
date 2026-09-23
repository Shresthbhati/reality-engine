"""Routes: uploads → evidence, worlds, jobs, notifications, activity."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api import jobs as jobrunner
from apps.api.db import get_db
from apps.api.models import (
    ActivityEvent,
    Evidence,
    Job,
    Location,
    Notification,
    Session,
    Upload,
    World,
    new_id,
    utcnow,
)
from apps.api.routes_sessions import _latest_location, _location_out
from apps.api.storage import resolve_artifact, store_bytes


def _evidence_dict(ev: Evidence) -> dict:
    return {
        "id": ev.id,
        "name": ev.name,
        "type": ev.type,
        "session_id": ev.session_id,
        "processing_state": ev.processing_state,
        "mime_type": ev.mime_type,
        "size": ev.size,
        "checksum": ev.checksum,
        "created_at": ev.created_at.isoformat() if ev.created_at else None,
        "processed_at": ev.processed_at.isoformat() if ev.processed_at else None,
        "metadata": ev.metadata_json or {},
    }


uploads = APIRouter(prefix="/api/uploads", tags=["uploads"])
evidence = APIRouter(prefix="/api/evidence", tags=["evidence"])

_TYPE_BY_MIME = {
    "image/jpeg": "photo", "image/png": "photo", "image/heic": "photo",
    "image/webp": "photo", "image/tiff": "photo",
    "video/mp4": "video", "video/quicktime": "video",
}


def _classify(filename: str, mime: str | None) -> str:
    name = (filename or "").lower()
    if mime in _TYPE_BY_MIME:
        return _TYPE_BY_MIME[mime]
    if name.endswith((".las", ".laz", ".ply", ".pcd", ".e57")):
        return "point_cloud"
    if name.endswith((".gpx", ".nmea")):
        return "gnss"
    if name.endswith((".csv", ".json")):
        return "sensor_log"
    return "dataset"


@uploads.post("", status_code=201)
async def create_upload(
    file: UploadFile,
    session_id: str | None = None,
    db: AsyncSession = Depends(get_db),
) -> dict:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    digest, _path = store_bytes(data)
    up = Upload(
        id=new_id("upl"),
        session_id=session_id,
        status="completed",
        filename=file.filename or "unnamed",
        mime_type=file.content_type,
        size=len(data),
        checksum=digest,
        artifact_uri=f"sha256://{digest}",
        completed_at=utcnow(),
    )
    db.add(up)
    await db.flush()
    ev = Evidence(
        id=new_id("ev"),
        name=up.filename,
        type=_classify(up.filename, up.mime_type),
        session_id=session_id,
        upload_id=up.id,
        processing_state="uploaded",
        mime_type=up.mime_type,
        size=up.size,
        checksum=digest,
        artifact_uri=up.artifact_uri,
    )
    db.add(ev)
    if session_id:
        s = await db.get(Session, session_id)
        if s is not None:
            s.uploaded_at = utcnow()
            if s.status == "created":
                s.status = "uploaded"
    db.add(
        ActivityEvent(
            id=new_id("act"),
            type="evidence.created",
            entity_type="evidence",
            entity_id=ev.id,
            summary=f"Evidence '{ev.name}' uploaded",
        )
    )
    await db.commit()
    job = jobrunner.enqueue_job(db, jobrunner.PROCESS_EVIDENCE, "evidence", ev.id)
    await db.commit()
    return {"upload_id": up.id, "evidence_id": ev.id, "job_id": job.id, "checksum": digest}


@evidence.get("")
async def list_evidence(
    session_id: str | None = None, db: AsyncSession = Depends(get_db)
) -> dict:
    q = select(Evidence).order_by(Evidence.created_at.desc()).limit(200)
    if session_id:
        q = q.where(Evidence.session_id == session_id)
    res = await db.execute(q)
    return {"items": [_evidence_dict(e) for e in res.scalars().all()]}


@evidence.get("/{evidence_id}")
async def get_evidence(evidence_id: str, db: AsyncSession = Depends(get_db)) -> dict:
    ev = await db.get(Evidence, evidence_id)
    if ev is None:
        raise HTTPException(404, "Evidence not found")
    loc = None
    if ev.session_id:
        loc = await _latest_location(db, "session", ev.session_id)
    loc_out = _location_out(loc).model_dump(mode="json") if loc else None
    return {**_evidence_dict(ev), "location": loc_out}


@evidence.delete("/{evidence_id}", status_code=204)
async def delete_evidence(evidence_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    """Remove the application Evidence record.

    The content-addressed artifact is deliberately left in the store: other
    records may reference the same sha256, and artifacts are immutable. Only
    the application record and its upload row are deleted.
    """
    ev = await db.get(Evidence, evidence_id)
    if ev is None:
        raise HTTPException(404, "Evidence not found")
    name, session_id, upload_id = ev.name, ev.session_id, ev.upload_id
    await db.delete(ev)
    if upload_id:
        up = await db.get(Upload, upload_id)
        if up is not None:
            await db.delete(up)
    db.add(
        ActivityEvent(
            id=new_id("act"),
            type="evidence.deleted",
            entity_type="evidence",
            entity_id=evidence_id,
            summary=f"Evidence '{name}' removed",
        )
    )
    await db.commit()
    return Response(status_code=204)


@evidence.get("/{evidence_id}/artifact")
async def evidence_artifact(evidence_id: str, db: AsyncSession = Depends(get_db)):
    ev = await db.get(Evidence, evidence_id)
    if ev is None or not ev.artifact_uri:
        raise HTTPException(404, "Artifact not found")
    path = resolve_artifact(ev.artifact_uri)
    if path is None:
        raise HTTPException(410, "Artifact missing from store")
    return FileResponse(
        path, media_type=ev.mime_type or "application/octet-stream", filename=ev.name
    )
