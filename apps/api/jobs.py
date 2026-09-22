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
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import Evidence, Job, Session, utcnow

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
    """Run the existing ReconstructionOrchestrator for a session.

    Heavy dependencies (COLMAP etc.) are optional; unavailability is an
    explicit, honest job failure, never a silent pass.
    """
    session = await db.get(Session, job.entity_id)
    if session is None:
        raise RuntimeError(f"Session {job.entity_id} not found")
    job.stage = "checking_reconstruction_backend"
    await db.commit()

    try:
        import reconstruction.orchestrator as orch  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(f"Reconstruction backend unavailable: {exc}") from exc

    job.stage = "reconstructing"
    await db.commit()
    # Full orchestration invocation is wired in the reconstruction follow-up
    # with attempt persistence; until then no fabricated results are emitted.
    raise RuntimeError(
        "Reconstruction orchestration for application sessions is not wired yet; "
        "no reconstruction was run."
    )


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

