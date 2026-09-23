"""Durable job execution for the application API.

Jobs are database-backed and processed by a worker loop. Handlers invoke
the existing Reality Engine computational services — no computation is
duplicated here. Progress is reported only where the backend actually
measures it (stage names, never invented percentages).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Evidence, Job, Session, World, utcnow
from evidence.session import EvidenceItem, EvidenceKind

log = logging.getLogger("reality.api.jobs")

WORKER_ID = f"worker-{os.environ.get('REALITY_WORKER_NAME', uuid.uuid4().hex[:8])}"

INGEST_EVIDENCE = "INGEST_EVIDENCE"
PROCESS_EVIDENCE = "PROCESS_EVIDENCE"
RECONSTRUCT_SESSION = "RECONSTRUCT_SESSION"
REGISTER_SESSIONS = "REGISTER_SESSIONS"
COMPILE_WORLD = "COMPILE_WORLD"
RUN_ANALYSIS = "RUN_ANALYSIS"
GENERATE_REPORT = "GENERATE_REPORT"
EXPORT_ARTIFACT = "EXPORT_ARTIFACT"

# Evidence kind mapping for EvidenceItem conversion
mapping = {
    "PHOTO": "PHOTO",
    "VIDEO": "VIDEO",
    "POINT_CLOUD": "POINT_CLOUD",
    "GNSS": "GNSS",
    "IMU": "IMU",
    "SENSOR_LOG": "SENSOR_LOG",
    "DATASET": "DATASET",
}


def enqueue_job(
    db: AsyncSession,
    job_type: str,
    entity_type: str,
    entity_id: str,
    payload: dict | None = None,
    priority: int = 0,
) -> Job:
    job = Job(
        id=f"job_{uuid.uuid4().hex[:12]}",
        type=job_type,
        entity_type=entity_type,
        entity_id=entity_id,
        payload=payload or {},
        priority=priority,
    )
    db.add(job)
    return job


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


async def _run_process_evidence(db: AsyncSession, job: Job) -> None:
    """Verify + register an evidence artifact. Real work only: artifact
    resolution and checksum verification, recorded honestly in metadata."""
    evidence = await db.get(Evidence, job.entity_id)
    if evidence is None:
        raise RuntimeError(f"Evidence {job.entity_id} not found")
    job.stage = "verifying_artifact"
    evidence.processing_state = "processing"
    await db.commit()

    from apps.api.storage import resolve_artifact

    if not evidence.artifact_uri:
        raise RuntimeError("Evidence has no stored artifact")
    path = resolve_artifact(evidence.artifact_uri)
    if path is None:
        raise RuntimeError(f"Artifact missing from store: {evidence.artifact_uri}")

    digest = await asyncio.to_thread(_file_sha256, path)
    if evidence.checksum and digest != evidence.checksum:
        raise RuntimeError("Checksum mismatch between evidence record and stored artifact")

    job.stage = "registering"
    metadata = dict(evidence.metadata_json or {})
    metadata["verified"] = True
    metadata["verified_at"] = datetime.now(timezone.utc).isoformat()
    evidence.metadata_json = metadata
    evidence.processing_state = "processed"
    evidence.processed_at = utcnow()
    await db.commit()
async def _run_reconstruct_session(db: AsyncSession, job: Job) -> None:
    """Run the real capture-to-WorldIR vertical slice for a session's
    photo evidence, then commit the result as a new WorldStore version.

    Heavy dependencies (COLMAP etc.) are optional; unavailability, too
    little evidence, or a pipeline stage refusing are explicit, honest
    job failures -- never a silent pass or a fabricated world.
    """
    from apps.api import worldstore_service
    from apps.api.storage import resolve_artifact

    session = await db.get(Session, job.entity_id)
    if session is None:
        raise RuntimeError(f"Session {job.entity_id} not found")
    if not session.world_id:
        raise RuntimeError(
            "Session is not attached to a World; attach it before reconstructing "
            "(a compiled version has nowhere to be committed otherwise)."
        )
    world = await db.get(World, session.world_id)
    if world is None:
        raise RuntimeError(f"World {session.world_id} not found")

    job.stage = "checking_reconstruction_backend"
    await db.commit()
    try:
        from engine.pipeline.vertical_slice import (
            VerticalSliceError,
            VerticalSliceOptions,
            vertical_slice,
        )
        from world_ir.artifact_store import FileArtifactStore
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(f"Reconstruction backend unavailable: {exc}") from exc

    job.stage = "resolving_evidence"
    await db.commit()
    result = await db.execute(
        select(Evidence).where(Evidence.session_id == session.id, Evidence.type == "photo")
    )
    evidence_rows = list(result.scalars().all())
    items = []
    for ev in evidence_rows:
        if not ev.artifact_uri:
            continue
        path = resolve_artifact(ev.artifact_uri)
        if path is None:
            continue
        items.append(
            EvidenceItem(
                id=ev.id,
                kind=EvidenceKind.PHOTO,
                source_uri=path.resolve().as_uri(),
                sha256=ev.checksum,
            )
        )
    if len(items) < 2:
        raise RuntimeError(
            f"Session has {len(items)} usable photo evidence item(s) with stored "
            "artifacts; the reconstruction pipeline needs at least 2."
        )

    job.stage = "reconstructing"
    await db.commit()
    artifact_store = FileArtifactStore(worldstore_service.worldstore_root() / "pipeline-artifacts")
    options = VerticalSliceOptions(artifact_store=artifact_store)
    try:
        vs_result = await asyncio.to_thread(vertical_slice, items, options)
    except VerticalSliceError as exc:
        raise RuntimeError(f"Reconstruction failed: {exc}") from exc

    job.stage = "committing_version"
    await db.commit()
    report = {
        "status": "SUCCESS" if vs_result.registration_status == "success" else "PARTIAL_SUCCESS",
        "session_id": session.id,
        "images_ingested": len(items),
        "stages": {
            "reconstruction": {
                "backend": vs_result.stage_facts.get("backend"),
                "cameras_registered": vs_result.cameras_registered,
                "cameras_input": vs_result.cameras_input,
                "registration_status": vs_result.registration_status,
                "points": vs_result.points_total,
            },
            "scale": {"state": vs_result.scale_state, "meters_per_unit": vs_result.meters_per_unit},
            "depth": vs_result.stage_facts.get("depth"),
            "perception": vs_result.stage_facts.get("perception"),
            "mesh": vs_result.stage_facts.get("mesh"),
            "compile": {
                "entities": len(vs_result.world.entities),
                "measurements": vs_result.compile.measurements_count,
                "relationships": vs_result.compile.relationships_count,
            },
        },
    }

    points_bytes = await asyncio.to_thread(_render_points_ply, vs_result.points)
    cameras_bytes = await asyncio.to_thread(
        _render_cameras_json, vs_result.camera_poses, vs_result.scale_state, options.image_size
    )

    await worldstore_service.commit_version(
        db,
        world_id=world.id,
        world=vs_result.world,
        parent=world.current_version_id,
        source_session_ids=[session.id],
        points=points_bytes,
        cameras=cameras_bytes,
        report=report,
    )


def _render_points_ply(points) -> bytes:
    """Same binary PLY writer engine.pipeline.artifacts.write_points_ply
    uses, targeting an in-memory buffer instead of a file path -- the
    job commits bytes straight into content-addressed storage rather
    than round-tripping through a temp file on disk."""
    import struct

    n = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    ).encode("ascii")
    body = bytearray()
    for p in points:
        body += struct.pack("<3f", float(p[0]), float(p[1]), float(p[2]))
    return header + bytes(body)


def _render_cameras_json(camera_poses, scale_state: str, image_size) -> bytes:
    """Same JSON shape engine.pipeline.artifacts.write_cameras_json
    writes -- kept in sync with that module, in-memory."""
    import json as _json

    payload = {
        "frame": "world (meters, +Y up after frame canonicalization)",
        "scale_state": scale_state,
        "rotation_convention": "camera-to-world quaternion (w, x, y, z)",
        "image_size": list(image_size),
        "cameras": [
            {"evidence_id": eid, "position_m": list(pos), "rotation_wxyz": list(rot)}
            for eid, pos, rot in camera_poses
        ],
    }
    return _json.dumps(payload, indent=2).encode("utf-8")


_HANDLERS = {
    PROCESS_EVIDENCE: _run_process_evidence,
    RECONSTRUCT_SESSION: _run_reconstruct_session,
}


async def process_next_job(db: AsyncSession) -> Job | None:
    """Claim and run one queued job. Returns the job, or None if queue empty."""
    result = await db.execute(
        select(Job)
        .where(Job.status == "queued")
        .order_by(Job.priority.desc(), Job.created_at.asc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None
    job.status = "running"
    job.worker_id = WORKER_ID
    job.attempts += 1
    job.started_at = utcnow()
    job.heartbeat_at = utcnow()
    await db.commit()

    handler = _HANDLERS.get(job.type)
    try:
        if handler is None:
            raise RuntimeError(f"No handler for job type {job.type}")
        await handler(db, job)
        job.status = "completed"
        job.completed_at = utcnow()
        await _emit_completion(db, job)
    except Exception:
        log.exception("job %s failed", job.id)
        import traceback

        job.error = traceback.format_exc()[-2000:]
        if job.attempts >= job.max_attempts:
            job.status = "failed"
            job.completed_at = utcnow()
        else:
            job.status = "queued"
            job.stage = None
    await db.commit()
    return job


async def _emit_completion(db: AsyncSession, job: Job) -> None:
    from apps.api.models import Notification

    if job.entity_type == "evidence" and job.type == PROCESS_EVIDENCE:
        evidence = await db.get(Evidence, job.entity_id)
        if evidence is not None:
            db.add(
                Notification(
                    id=f"ntf_{uuid.uuid4().hex[:12]}",
                    type="evidence.processed",
                    title="Evidence processed",
                    body=evidence.name,
                    entity_type="evidence",
                    entity_id=evidence.id,
                )
            )


async def worker_loop(poll_seconds: float = 1.0) -> None:
    """Background worker. Started with the API process."""
    from apps.api.db import get_sessionmaker

    maker = get_sessionmaker()
    while True:
        try:
            async with maker() as db:
                job = await process_next_job(db)
            if job is None:
                await asyncio.sleep(poll_seconds)
        except Exception:
            log.exception("worker loop error")
            await asyncio.sleep(poll_seconds * 2)

