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

# Canonical job lifecycle: queued -> running -> {succeeded | partial |
# failed | cancelled}. SUCCEEDED is gated (output exists, validates,
# persists, provenance present) -- a timeout, crash, or degraded run
# can never become SUCCEEDED. PARTIAL means a version was adopted but
# the run was degraded (partial registration, skipped evidence, failed
# optional stages); the degradation reasons ride on the payload.
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_PARTIAL = "partial"
JOB_SUCCEEDED = "succeeded"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"

TERMINAL_STATES = frozenset({JOB_PARTIAL, JOB_SUCCEEDED, JOB_FAILED, JOB_CANCELLED})


class _JobCancelled(RuntimeError):
    """Operator requested cancellation; the job stops at a stage boundary
    instead of running to completion."""


async def _throw_if_cancelled(db: AsyncSession, job: Job) -> None:
    """Stage-boundary cancellation point. Refreshes only the flag so no
    uncommitted handler state is disturbed."""
    await db.refresh(job, attribute_names=["cancel_requested"])
    if job.cancel_requested:
        raise _JobCancelled(f"job {job.id} cancelled by operator")


def _assert_reconstruction_traced(world) -> None:
    """Fail persistence when a RECONSTRUCTED entity lacks its build
    trace. Procedural/generated entities are out of scope (they trace
    to their generator, not a capture)."""
    from provenance import Provenance as _Provenance

    untraced = sorted(
        eid for eid, entity in world.entities.items()
        if entity.provenance is _Provenance.RECONSTRUCTED
        and not isinstance((entity.custom_properties or {}).get("reconstruction"), dict)
    )
    if untraced:
        raise RuntimeError(
            f"reconstruction produced {len(untraced)} untraced entity(ies) "
            f"(e.g. {untraced[0]}); refusing to persist without provenance"
        )


def _stage_facts_degraded(stage_facts: dict) -> list[str]:
    """Optional-stage outcomes that are neither clean runs nor honest
    config-skips: evidence of degradation for PARTIAL grading."""
    degraded = []
    for stage in ("depth", "perception", "mesh"):
        facts = stage_facts.get(stage)
        if isinstance(facts, dict):
            status = facts.get("status")
            if status is not None and status not in ("ran", "skipped"):
                degraded.append(f"{stage} stage status {status!r}")
    return degraded


def _stamp_reconstruction_provenance(world, *, session_id: str, backend: str | None,
                                     options, images_ingested: int) -> None:
    """Stamp every reconstructed entity with its build provenance
    (session, backend, pipeline configuration). setdefault: never
    overwrite an existing stamp, so retries stay idempotent. Stored in
    custom_properties -- the schema's designated extension slot -- so it
    survives WorldIR serialization, WorldStore persistence, and reload."""
    stamp = {
        "session_id": session_id,
        "backend": backend,
        "seed": getattr(options, "seed", None),
        "depth_model": getattr(options, "depth_model", None),
        "perception_model": getattr(options, "perception_model", None),
        "mesh_enabled": getattr(options, "mesh_enabled", None),
        "detail_enabled": getattr(options, "detail_enabled", None),
        "images_ingested": images_ingested,
    }
    for entity in world.entities.values():
        props = getattr(entity, "custom_properties", None)
        if isinstance(props, dict):
            props.setdefault("reconstruction", dict(stamp))

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


async def _run_process_evidence(db: AsyncSession, job: Job) -> str:
    """Verify + register an evidence artifact. Real work only: artifact
    resolution and checksum verification, recorded honestly in metadata.
    Returns the terminal outcome for process_next_job's grading."""
    await _throw_if_cancelled(db, job)
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
    return "succeeded"


async def _run_reconstruct_session(db: AsyncSession, job: Job) -> str:
    """Run the real capture-to-WorldIR vertical slice for a session's
    photo evidence, then commit the result as a new WorldStore version.

    Heavy dependencies (COLMAP etc.) are optional; unavailability, too
    little evidence, or a pipeline stage refusing are explicit, honest
    job failures -- never a silent pass or a fabricated world. Returns
    "succeeded" or "partial" for process_next_job's grading; a version
    is only adopted on those paths, never on failure or cancellation.
    """
    from apps.api import worldstore_service
    from apps.api.storage import resolve_artifact

    await _throw_if_cancelled(db, job)
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
    job.heartbeat_at = utcnow()
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

    # Deterministic-backend seam for offline verification (same
    # "module:Class" contract apps.cli uses): when set, the injected
    # backend drives vertical_slice and the heavyweight learned stages
    # are disabled, since they need real pixels, model weights, and GPU
    # -- none of which a synthetic backend provides.
    test_backend = None
    backend_spec = os.environ.get("REALITY_TEST_BACKEND", "").strip()
    if backend_spec:
        import importlib

        module_name, _, class_name = backend_spec.partition(":")
        if not module_name or not class_name:
            raise RuntimeError(
                f"REALITY_TEST_BACKEND must be 'module:Class', got {backend_spec!r}"
            )
        try:
            test_backend = getattr(importlib.import_module(module_name), class_name)()
        except Exception as exc:
            raise RuntimeError(
                f"Reconstruction failed: test backend {backend_spec!r} unavailable: {exc}"
            ) from exc

    job.stage = "resolving_evidence"
    job.heartbeat_at = utcnow()
    await db.commit()
    result = await db.execute(
        select(Evidence).where(Evidence.session_id == session.id, Evidence.type == "photo")
    )
    evidence_rows = list(result.scalars().all())
    items = []
    skipped_evidence = []
    for ev in evidence_rows:
        if not ev.artifact_uri:
            skipped_evidence.append({"evidence_id": ev.id, "reason": "no stored artifact"})
            continue
        path = resolve_artifact(ev.artifact_uri)
        if path is None:
            skipped_evidence.append({
                "evidence_id": ev.id,
                "reason": f"artifact_uri {ev.artifact_uri!r} not resolvable in content store",
            })
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

    await _throw_if_cancelled(db, job)
    job.stage = "reconstructing"
    job.heartbeat_at = utcnow()
    await db.commit()
    artifact_store = FileArtifactStore(worldstore_service.worldstore_root() / "pipeline-artifacts")
    if test_backend is not None:
        options = VerticalSliceOptions(
            artifact_store=artifact_store,
            reconstruction_backend=test_backend,
            depth_model=None,
            perception_model=None,
            mesh_enabled=False,
            detail_enabled=False,
        )
    else:
        options = VerticalSliceOptions(artifact_store=artifact_store)
    try:
        vs_result = await asyncio.to_thread(vertical_slice, items, options)
    except VerticalSliceError as exc:
        raise RuntimeError(f"Reconstruction failed: {exc}") from exc

    await _throw_if_cancelled(db, job)

    # Provenance + determinism record: stamp every reconstructed entity
    # with the session, backend, and pipeline configuration it was built
    # from, before anything is validated or persisted.
    _stamp_reconstruction_provenance(
        vs_result.world,
        session_id=session.id,
        backend=vs_result.stage_facts.get("backend"),
        options=options,
        images_ingested=len(items),
    )

    # Validation gate: an invalid WorldIR is never persisted or adopted.
    # The previous HEAD stays intact, so a failed refinement cannot
    # destroy valid earlier results.
    from world_ir.validation import validate_world_ir

    validation = validate_world_ir(vs_result.world)
    if not validation.is_valid():
        detail = "; ".join(validation.messages()[:5])
        raise RuntimeError(f"reconstructed WorldIR failed validation: {detail}")

    # Provenance hard gate: a reconstructed entity without its build
    # trace must never be persisted. The stamp above runs first, so a
    # missing trace here means stamping itself failed -- persisting
    # would create permanently untraceable state. Fail instead; HEAD
    # stays intact.
    _assert_reconstruction_traced(vs_result.world)

    job.stage = "committing_version"
    job.heartbeat_at = utcnow()
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

    base_head = world.current_version_id
    try:
        store = worldstore_service.get_store()
        version_row = await worldstore_service.commit_version(
            db,
            world_id=world.id,
            world=vs_result.world,
            parent=base_head,
            source_session_ids=[session.id],
            points=points_bytes,
            cameras=cameras_bytes,
            report=report,
            expect_parent=base_head,
        )
    except worldstore_service.ConcurrentModificationError as exc:
        # Retryable: the next attempt reloads HEAD and either dedups
        # (identical content) or chains onto the new HEAD.
        raise RuntimeError(f"reconstruction superseded: {exc}") from exc

    # Read-back verification: the adopted version must load with clean
    # hashes before anything is called a success. A version that cannot
    # be read back is not a persisted result, however healthy the
    # pipeline looked a moment ago.
    verify_failures = await asyncio.to_thread(store.verify_version, version_row.id)
    if verify_failures:
        reasons = "; ".join(f["reason"] for f in verify_failures)
        raise RuntimeError(
            f"adopted version '{version_row.id}' failed read-back verification: {reasons}"
        )

    # Outcome grading: SUCCEEDED only when the run is clean end to end
    # (successful registration, nothing skipped, no degraded optional
    # stages, no validation warnings). Anything less honest is PARTIAL:
    # a real adopted version with the degradation reasons on record.
    # Failed validation or missing output never reach here (raised above),
    # and a timeout can never become either (it raises before grading).
    degraded = []
    if vs_result.registration_status != "success":
        degraded.append(f"registration {vs_result.registration_status!r}")
    if skipped_evidence:
        degraded.append(f"{len(skipped_evidence)} skipped evidence item(s)")
    degraded.extend(_stage_facts_degraded(vs_result.stage_facts))
    for warning in validation.warnings:
        degraded.append(f"validation warning: {warning}")
    outcome = "partial" if degraded else "succeeded"

    # The completed reconstruction is a real product event: link it into
    # the application model (session complete + attached world), carry
    # the measured result on the job payload (never invented numbers),
    # and surface it in notifications/activity so world surfaces see it.
    session.world_id = world.id
    session.status = "complete"
    session.processing_completed_at = utcnow()
    job.payload = {
        "world_id": world.id,
        "version_id": version_row.id,
        "registration_status": vs_result.registration_status,
        "outcome": outcome,
        "degraded": degraded,
        "points": len(vs_result.points),
        "cameras_registered": vs_result.cameras_registered,
        "skipped_evidence": skipped_evidence,
    }
    await db.commit()
    _notify_world_version(db, job, world, version_row.id)
    await db.commit()
    return outcome


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


_HANDLERS = {
    PROCESS_EVIDENCE: _run_process_evidence,
    RECONSTRUCT_SESSION: _run_reconstruct_session,
}


def _stale_after_seconds() -> float:
    try:
        return max(30.0, float(os.environ.get("JOB_STALE_AFTER_SECONDS", "300")))
    except ValueError:
        return 300.0


def _job_timeout_seconds() -> float:
    try:
        return max(1.0, float(os.environ.get("JOB_TIMEOUT_SECONDS", "1800")))
    except ValueError:
        return 1800.0


async def reap_stale_jobs(db: AsyncSession) -> int:
    """Recover jobs stranded in 'running' by a crashed worker/restart.

    A claimed job heartbeats at claim time; one still 'running' past the
    staleness horizon belongs to a dead worker and is requeued for pickup
    instead of wedging its entity forever. Returns the requeued count.
    """
    from datetime import timedelta

    cutoff = utcnow() - timedelta(seconds=_stale_after_seconds())
    result = await db.execute(
        select(Job).where(Job.status == JOB_RUNNING, Job.heartbeat_at < cutoff)
    )
    reaped = 0
    for job in result.scalars().all():
        job.status = JOB_QUEUED
        job.stage = None
        job.worker_id = None
        reaped += 1
    if reaped:
        await db.commit()
        log.warning("reaped %d stale running job(s) back to queued", reaped)
    return reaped


async def process_next_job(db: AsyncSession) -> Job | None:
    """Claim and run one queued job. Returns the job, or None if queue empty."""
    result = await db.execute(
        select(Job)
        .where(Job.status == JOB_QUEUED)
        .order_by(Job.priority.desc(), Job.created_at.asc())
        .limit(1)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None
    job.status = JOB_RUNNING
    job.worker_id = WORKER_ID
    job.attempts += 1
    job.started_at = utcnow()
    job.heartbeat_at = utcnow()
    await db.commit()

    handler = _HANDLERS.get(job.type)
    try:
        if handler is None:
            raise RuntimeError(f"No handler for job type {job.type}")
        # Bound every handler: a wedged backend (or lost dependency)
        # must fail this job, never stall the whole queue behind it. A
        # timeout is always a failure -- it can never grade as success.
        try:
            outcome = await asyncio.wait_for(handler(db, job), timeout=_job_timeout_seconds())
        except TimeoutError as exc:
            raise RuntimeError(
                f"job handler timed out after {_job_timeout_seconds():.0f}s"
            ) from exc
        job.status = JOB_PARTIAL if outcome == "partial" else JOB_SUCCEEDED
        job.completed_at = utcnow()
        await _emit_completion(db, job)
    except _JobCancelled as exc:
        log.warning("job %s cancelled: %s", job.id, exc)
        job.status = JOB_CANCELLED
        job.error = f"cancelled by operator: {exc}"
        job.completed_at = utcnow()
    except Exception:
        log.exception("job %s failed", job.id)
        import traceback

        job.error = traceback.format_exc()[-2000:]
        if job.attempts >= job.max_attempts:
            job.status = JOB_FAILED
            job.completed_at = utcnow()
        else:
            job.status = JOB_QUEUED
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
    # A previous process may have died mid-job: reap once at startup so
    # stranded 'running' jobs become pickable again instead of wedging.
    try:
        async with maker() as db:
            await reap_stale_jobs(db)
    except Exception:
        log.exception("startup reap failed")
    while True:
        try:
            async with maker() as db:
                await reap_stale_jobs(db)
                job = await process_next_job(db)
            if job is None:
                await asyncio.sleep(poll_seconds)
        except Exception:
            log.exception("worker loop error")
            await asyncio.sleep(poll_seconds * 2)

