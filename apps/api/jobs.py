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
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import ActivityEvent, Evidence, Job, Session, World, new_id, utcnow

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
        import reconstruction.orchestrator  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise RuntimeError(f"Reconstruction backend unavailable: {exc}") from exc

    job.stage = "reconstructing"
    await db.commit()

    from evidence.session import EvidenceItem, EvidenceKind

    # ---- application evidence -> canonical EvidenceItem (the same shape
    # the CLI vertical slice feeds the orchestrator). Only REAL stored
    # artifacts participate: an evidence row whose artifact cannot be
    # resolved from the content store is recorded as skipped, never
    # silently promoted into a fabricated input.
    res = await db.execute(
        select(Evidence).where(Evidence.session_id == session.id)
    )
    evidence_rows = res.scalars().all()
    items: list[EvidenceItem] = []
    skipped: list[dict] = []
    from apps.api.storage import resolve_artifact

    kind_by_type = {
        "photo": EvidenceKind.PHOTO,
        "video": EvidenceKind.VIDEO,
        "point_cloud": EvidenceKind.POINT_CLOUD,
        "depth": EvidenceKind.DEPTH,
        "gnss": EvidenceKind.GPS_TRACK,
        "imu": EvidenceKind.IMU,
        "sensor_log": EvidenceKind.OTHER,
        "dataset": EvidenceKind.OTHER,
    }
    for ev in evidence_rows:
        path = resolve_artifact(ev.artifact_uri) if ev.artifact_uri else None
        if path is None:
            skipped.append({
                "evidence_id": ev.id,
                "reason": f"artifact_uri {ev.artifact_uri!r} not resolvable in content store",
            })
            continue
        uri = Path(path).resolve().as_uri()
        metadata = dict(ev.metadata_json or {})
        metadata["application_evidence_id"] = ev.id
        items.append(EvidenceItem(
            id=ev.id,
            kind=kind_by_type.get(ev.type, EvidenceKind.OTHER),
            source_uri=uri,
            captured_at=ev.created_at.timestamp() if ev.created_at else None,
            sha256=ev.checksum,
            metadata=metadata,
        ))

    image_count = sum(
        1 for it in items if it.kind in (EvidenceKind.PHOTO, EvidenceKind.VIDEO)
    )
    if image_count < 2:
        detail = "; ".join(s["reason"] for s in skipped) if skipped else "none stored"
        raise RuntimeError(
            f"session {session.id} has {image_count} usable image evidence items "
            f"(need >= 2) -- no reconstruction was run. Unresolvable artifacts: {detail}"
        )

    # ---- orchestrator: the same backend chain the CLI uses. A missing
    # COLMAP binary is an explicit job failure carrying the backend's own
    # availability detail (never a fake success). REALITY_TEST_BACKEND
    # ("module:Class") injects a deterministic backend for offline
    # verification of this path, mirroring apps.cli.main._resolve_test_backend.
    backend = None
    spec = os.environ.get("REALITY_TEST_BACKEND", "").strip()
    if spec:
        import importlib

        module_name, _, class_name = spec.partition(":")
        if not module_name or not class_name:
            raise RuntimeError(
                f"REALITY_TEST_BACKEND must be 'module:Class', got {spec!r}"
            )
        backend = getattr(importlib.import_module(module_name), class_name)()
    else:
        from reconstruction.backend.colmap_backend import (
            ColmapReconstructionBackend,
        )

        backend = ColmapReconstructionBackend()

    from reconstruction.orchestrator import ReconstructionOrchestrator

    orchestrator = ReconstructionOrchestrator(backends=[backend])
    try:
        run = orchestrator.run(items)
    except Exception as exc:  # noqa: BLE001 - honest job failure with the stage's own error
        raise RuntimeError(f"reconstruction failed: {exc}") from exc

    result = run.result
    if result.registration_status == "failed" or not result.points:
        raise RuntimeError(
            f"reconstruction produced nothing usable "
            f"(status={result.registration_status!r}, {len(result.points)} points) "
            f"-- not compiling"
        )

    job.stage = "compiling"
    await db.commit()

    # ---- compile to WorldIR and persist a WorldStore version (real
    # artifacts only; a refusal here is an honest job failure).
    from sdk import reality
    from worldstore.store import WorldStore
    from world_ir.artifact_store import FileArtifactStore

    store_root = Path(os.environ.get("WORLDSTORE_ROOT", "./data/worldstore"))
    store_root.mkdir(parents=True, exist_ok=True)
    compile_options = reality.CompileOptions(
        artifact_store=FileArtifactStore(store_root / "artifacts")
    )
    try:
        world, diagnostics = reality.compile_world_from_reconstruction(
            result, compile_options
        )
    except (reality.CompileInputError, reality.WorldValidationGateError) as exc:
        raise RuntimeError(f"world compile refused: {exc}") from exc

    wstore = WorldStore(store_root)
    stored = wstore.save_version(
        world, parent=None, source_session_ids=[session.id]
    )

    job.stage = "linking"
    await db.commit()

    # ---- link the result into the application model: reuse the
    # session's existing World when attached, else create one. The
    # WorldVersion mirror row references the WorldStore artifact so the
    # versions/worldir surfaces read REAL computational state.
    world_row = None
    if session.world_id:
        world_row = await db.get(World, session.world_id)
    if world_row is None:
        world_row = World(
            id=new_id("wld"),
            name=f"World for {session.name}",
        )
        db.add(world_row)
    session.world_id = world_row.id
    session.status = "complete"
    session.processing_completed_at = utcnow()
    world_row.current_version_id = stored.version_id
    from apps.api.models import WorldVersion

    db.add(WorldVersion(
        id=stored.version_id,
        world_id=world_row.id,
        parent_version_id=stored.parent,
        artifact_uri=stored.artifact_uri,
        artifact_hash=stored.artifact_hash,
        source_session_ids=list(stored.source_session_ids),
        changed_entity_ids=list(stored.changed_entity_ids),
    ))
    db.add(ActivityEvent(
        id=new_id("act"),
        type="world.version_created",
        entity_type="world",
        entity_id=world_row.id,
        summary=(
            f"Version '{stored.version_id}' created from session "
            f"'{session.name}' ({len(result.points)} points, "
            f"{len(result.camera_poses)} cameras)"
        ),
    ))
    await db.commit()
    job.payload = {
        "world_id": world_row.id,
        "version_id": stored.version_id,
        "registration_status": result.registration_status,
        "points": len(result.points),
        "cameras_registered": len(result.camera_poses),
        "attempts": [a.to_dict() for a in run.diagnostics.attempts],
        "skipped_evidence": skipped,
    }
    await db.commit()
    _notify_world_version(db, job, world_row, stored.version_id)
    await db.commit()


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


def _notify_world_version(db, job, world_row, version_id) -> None:
    """A completed reconstruction is a real product event: surface it in
    notifications and activity so the UI's world surfaces see it."""
    from apps.api.models import Notification

    db.add(Notification(
        id=f"ntf_{uuid.uuid4().hex[:12]}",
        type="world.version_created",
        title="World reconstructed",
        body=f"Version {version_id} created for world {world_row.name}",
        entity_type="world",
        entity_id=world_row.id,
    ))
    db.add(ActivityEvent(
        id=new_id("act"),
        type="world.version_created",
        entity_type="world",
        entity_id=world_row.id,
        summary=f"World '{world_row.name}' version {version_id} created",
    ))


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

