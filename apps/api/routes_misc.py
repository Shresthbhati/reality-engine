"""Routes: uploads → evidence, worlds, jobs, notifications, activity."""

from __future__ import annotations

import os

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
    EvidenceDecision,
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


def _safe_filename(raw: str | None) -> str:
    """Client filenames are untrusted: strip directory components (some
    browsers send full paths), control characters, and header-breaking
    quotes/CRLF before the name is stored or reflected in
    Content-Disposition. Never empty, never overlong."""
    name = (raw or "unnamed").replace("\\", "/").split("/")[-1]
    name = "".join(c for c in name if c.isprintable() and c not in '"\r\n').strip(" .")
    return name[:200] or "unnamed"


def _max_upload_bytes() -> int:
    default = 1024**3
    try:
        value = int(os.environ.get("UPLOAD_MAX_BYTES", str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _max_files_per_session() -> int:
    default = 2000
    try:
        value = int(os.environ.get("UPLOAD_MAX_FILES_PER_SESSION", str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


async def _read_bounded_upload(file) -> bytes:
    """Read an upload without loading an unbounded body into memory: one
    byte past the limit is enough to refuse with 413."""
    limit = _max_upload_bytes()
    chunks = []
    remaining = limit + 1
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
        if remaining <= 0:
            raise HTTPException(
                413, f"upload exceeds {limit} bytes; split the capture or raise UPLOAD_MAX_BYTES"
            )
    return b"".join(chunks)


_VIDEO_EXT = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")
_HEIC_BRANDS = (b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1")
#: Content that is never evidence: executables, scripts and markup. Refused for every non-image/video upload so a
#: renamed binary or an HTML/SVG page cannot sit in the store under a data-file name.
_ACTIVE_CONTENT_PREFIXES = (b"MZ", b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xca\xfe\xba\xbe", b"#!")
_MARKUP_PREFIXES = (b"<!doctype", b"<html", b"<svg", b"<script", b"<?php")


def _sniff_media(data: bytes) -> str | None:
    """'photo' / 'video' from the file's own magic bytes, or None. Declared MIME and extension are never trusted for
    these two kinds: they are what the reconstruction engine decodes."""
    head = data[:16]
    if head[:3] == b"\xff\xd8\xff" or head[:8] == b"\x89PNG\r\n\x1a\n" or head[:4] in (b"II*\x00", b"MM\x00*"):
        return "photo"
    if head[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "photo"
    if data[4:8] == b"ftyp":
        return "photo" if data[8:12] in _HEIC_BRANDS else "video"
    if head[:4] == b"\x1aE\xdf\xa3" or (head[:4] == b"RIFF" and data[8:12] == b"AVI "):
        return "video"
    return None


def _upload_allow_any_type() -> bool:
    return os.environ.get("UPLOAD_ALLOW_ANY_TYPE", "").strip().lower() in ("1", "true", "yes")


def vet_upload(filename: str, mime: str | None, data: bytes) -> str:
    """The evidence kind for this upload, or raise ValueError(reason).

    Bytes win over labels: a JPEG sent as application/octet-stream is a photo; an HTML page sent as image/jpeg is
    refused. Unknown types are refused unless UPLOAD_ALLOW_ANY_TYPE=1. The offline reconstruction test seam
    (REALITY_TEST_BACKEND, 'pixels not inspected') keeps its placeholder photographs."""
    declared = _classify(filename, mime)
    sniffed = _sniff_media(data)
    if sniffed:
        if declared in ("photo", "video") and declared != sniffed:
            raise ValueError(f"content is a {sniffed} but the file is declared as a {declared}")
        return sniffed
    if declared in ("photo", "video"):
        if os.environ.get("REALITY_TEST_BACKEND", "").strip():
            return declared
        raise ValueError(f"not a valid {declared}: the bytes are not a supported {declared} format")
    if declared == "dataset" and not _upload_allow_any_type():
        raise ValueError("unsupported file type (accepted: photos, videos, point clouds, GNSS tracks, sensor logs)")
    if data[:8].startswith(_ACTIVE_CONTENT_PREFIXES) or data[:64].lstrip().lower().startswith(_MARKUP_PREFIXES):
        raise ValueError("file content is not evidence data (executable, script or markup)")
    return declared


def _classify(filename: str, mime: str | None) -> str:
    name = (filename or "").lower()
    if mime in _TYPE_BY_MIME:
        return _TYPE_BY_MIME[mime]
    if name.endswith(_VIDEO_EXT):
        return "video"
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
    data = await _read_bounded_upload(file)
    if not data:
        raise HTTPException(400, "Empty upload")
    try:
        kind = vet_upload(_safe_filename(file.filename), file.content_type, data)
    except ValueError as exc:
        raise HTTPException(415, str(exc)) from exc
    if session_id:
        limit = _max_files_per_session()
        stored = (await db.execute(
            select(func.count()).select_from(Upload).where(Upload.session_id == session_id)
        )).scalar_one()
        if stored >= limit:
            raise HTTPException(
                413, f"session already holds {stored} uploads (limit {limit}); raise UPLOAD_MAX_FILES_PER_SESSION"
            )
    digest, _path = store_bytes(data)
    up = Upload(
        id=new_id("upl"),
        session_id=session_id,
        status="completed",
        filename=_safe_filename(file.filename),
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
        type=kind,
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
    decision = await db.get(EvidenceDecision, evidence_id)
    if decision is not None:
        await db.delete(decision)
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
        path,
        media_type=ev.mime_type or "application/octet-stream",
        # Re-sanitize: rows written before filename hygiene may hold raw
        # client bytes (paths, quotes, CRLF) that must never reach headers.
        filename=_safe_filename(ev.name),
        headers={"X-Content-Type-Options": "nosniff"},
    )


# ----------------------------------------------------------------------------------------------- pre-flight check
_CHECK_MAX_FILES = 300
_SOFT_RATIO = 0.4  # a photo is "relatively soft" below this fraction of the set's median sharpness
_MIN_GOOD_SET = 8  # fewer usable photographs than this -> suggest more angles


@uploads.post("/check")
async def check_uploads(files: list[UploadFile]) -> dict:
    """Tell the user what is wrong with a set of photographs BEFORE uploading it. Nothing is stored.

    Per file: decodable?, size, relative sharpness, exact duplicate of an earlier file in the set, GPS present, and
    whether it looks like a photograph at all. Sharpness is a ranking inside the set, not a calibrated blur score, so
    `relatively_soft` is only reported for sets of 3+ usable photographs and says so. Advisory: the caller never has
    to obey it."""
    import asyncio
    import hashlib
    import statistics
    import tempfile
    from pathlib import Path

    from evidence.image_check import GEOMETRY_CLASSES, inspect_image

    if not files:
        raise HTTPException(422, "no files to check")
    if len(files) > _CHECK_MAX_FILES:
        raise HTTPException(413, f"at most {_CHECK_MAX_FILES} files can be checked at once")

    rows: list[dict] = []
    first_seen: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix="re-check-") as tmp:
        for i, f in enumerate(files):
            name = _safe_filename(f.filename)
            data = await _read_bounded_upload(f)
            digest = hashlib.sha256(data).hexdigest()
            kind = _classify(name, f.content_type)
            row: dict = {"name": name, "size": len(data), "kind": kind, "sha256": digest,
                         "duplicate_of": first_seen.get(digest), "decodable": None, "width": None, "height": None,
                         "sharpness": None, "has_gps": False, "looks_like_photo": None, "too_small": False,
                         "problem": None}
            first_seen.setdefault(digest, name)
            if kind == "photo":
                path = Path(tmp) / f"{i}.bin"
                path.write_bytes(data)
                facts = await asyncio.to_thread(inspect_image, path, None)
                row.update(decodable=facts.ok or facts.width > 0, width=facts.width or None,
                           height=facts.height or None, sharpness=facts.sharpness, has_gps=facts.gps is not None)
                if facts.ok:
                    row["looks_like_photo"] = facts.evidence_class in GEOMETRY_CLASSES
                else:
                    row["too_small"] = facts.width > 0 and "too small" in (facts.reason or "")
                    row["decodable"] = facts.width > 0
                    row["problem"] = facts.reason
            rows.append(row)

    sharp = [r["sharpness"] for r in rows if r["sharpness"] is not None and r["duplicate_of"] is None]
    median = statistics.median(sharp) if len(sharp) >= 3 else None
    for r in rows:
        r["relatively_soft"] = bool(median and r["sharpness"] is not None and r["sharpness"] < _SOFT_RATIO * median)

    def usable(r: dict) -> bool:
        return r["kind"] == "photo" and r["decodable"] and not r["too_small"] and r["looks_like_photo"] is not False \
            and r["duplicate_of"] is None

    n_usable = sum(1 for r in rows if usable(r))
    n_flagged = sum(1 for r in rows if r["relatively_soft"] or r["duplicate_of"] or r["too_small"]
                    or r["decodable"] is False or r["looks_like_photo"] is False)
    verdict = "no_usable_photos" if n_usable == 0 else "add_angles" if n_usable < _MIN_GOOD_SET else \
        "has_problems" if n_flagged else "good"
    return {"files": rows, "summary": {"total": len(rows), "usable": n_usable, "flagged": n_flagged,
                                       "soft_threshold": None if median is None else _SOFT_RATIO * median,
                                       "verdict": verdict}}
