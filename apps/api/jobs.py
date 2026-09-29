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

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from apps.api.models import (
    ActivityEvent,
    Evidence,
    Job,
    Session,
    World,
    WorldVersion,
    new_id,
    utcnow,
)
from engine.pipeline import world_delta
from engine.pipeline.progressive import (
    EvidenceInput,
    InsufficientEvidence,
    run_progressive,
)
from evidence.image_check import ImageFacts, inspect_image
from reconstruction.proc import (
    discard_job,
    get_dead_worker_jobs,
    recover_orphaned_jobs,
    register_worker,
    set_current_job,
    terminate_job_procs,
    unregister_worker,
)
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
    uncommitted handler state is disturbed.

    Cancellation windows and their invariants:
      before claim (queued) ......... CANCELLED immediately, never runs
      after claim, before work ...... CANCELLED at the next boundary
      during reconstruction ......... CANCELLED at the pre-commit boundary
      during WorldStore write ....... no check inside the write; surfaces
                                      at the post-commit check below
      during DB mirror .............. same: post-commit check
      after HEAD adoption ........... version STANDS (durable work is not
                                      un-written); job grades CANCELLED
                                      with adopted_before_cancel recorded
    Core invariant: once a version is adopted, cancellation cannot erase
    or roll it back -- it only changes how the job itself is graded.
    """
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


async def _head_report(db: AsyncSession, world: World) -> dict:
    """Report of the world's current version ({} when none): carries the
    evidence ids the current model was built from, so a refinement can tell
    which images are new."""
    if not world.current_version_id:
        return {}
    row = await db.get(WorldVersion, world.current_version_id)
    return dict(row.report or {}) if row is not None else {}


def _record_registration(evidence_rows, job_id: str, registered: set, adopted: bool) -> None:
    """Per-photo registration history on the Evidence row: attempts, current state
    (registered | waiting), whether it EVER registered, and the last attempts."""
    for ev in evidence_rows:
        meta = dict(ev.metadata_json or {})
        reg = dict(meta.get("registration") or {})
        hist = list(reg.get("history") or [])
        is_reg = ev.id in registered
        hist.append({"job": job_id, "registered": is_reg, "adopted": adopted})
        reg.update(
            attempts=int(reg.get("attempts", 0)) + 1,
            state="registered" if is_reg else "waiting",
            ever_registered=bool(reg.get("ever_registered")) or is_reg,
            history=hist[-20:],
        )
        meta["registration"] = reg
        ev.metadata_json = meta


async def _contributing_session_ids(db: AsyncSession, world_id: str, trigger_session_id: str) -> list[str]:
    res = await db.execute(select(Session.id).where(Session.world_id == world_id))
    ids = sorted(set(res.scalars().all()) | {trigger_session_id})
    return ids


def _stamp_reconstruction_provenance(world, *, session_id: str, backend: str | None,
                                     options, images_ingested: int,
                                     extra: dict | None = None) -> None:
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
        **(extra or {}),
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
    # Progressive refinement: the model is built from ALL photo evidence of
    # every session attached to this world (compile(A+B) IS the definition
    # of V2), never just the newest session's batch.
    result = await db.execute(
        select(Evidence)
        .join(Session, Evidence.session_id == Session.id)
        .where(or_(Session.world_id == world.id, Session.id == session.id), Evidence.type == "photo")
        .order_by(Evidence.created_at.asc(), Evidence.id.asc())
    )
    evidence_rows = list(result.scalars().all())
    head_report = await _head_report(db, world)
    prior_ids = list((head_report.get("evidence") or {}).get("all_ids") or [])
    resolved = []
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
        resolved.append((ev, path))
    if not resolved:
        raise InsufficientEvidence(
            "World has 0 usable photo evidence item(s) with stored artifacts; "
            "at least 1 is required."
        )

    # Validate + classify every image BEFORE any expensive stage: bytes that
    # are not images must never reach COLMAP (measured: 3 garbage "photos"
    # wedged the pipeline for >90 s), and non-photographic material is kept
    # out of observed geometry.
    await _throw_if_cancelled(db, job)
    job.stage = "analyzing_evidence"
    job.heartbeat_at = utcnow()
    await db.commit()
    if test_backend is not None:
        # The offline deterministic-backend seam never reads pixels (its
        # placeholder "photos" are the point of the seam), so there is nothing
        # to decode. Say so on the record rather than pretending it was checked.
        facts_list = [
            ImageFacts(
                ok=True, evidence_class="photograph_unverified",
                class_basis="REALITY_TEST_BACKEND seam: pixels not inspected",
            )
            for _ in resolved
        ]
    else:
        facts_list = await asyncio.to_thread(
            lambda: [inspect_image(p, (ev.metadata_json or {}).get("declared_evidence_class"))
                     for ev, p in resolved]
        )
    eval_inputs = []
    for (ev, path), facts in zip(resolved, facts_list):
        meta = dict(ev.metadata_json or {})
        meta["image_facts"] = facts.to_dict()
        ev.metadata_json = meta
        eval_inputs.append(EvidenceInput(
            item=EvidenceItem(id=ev.id, kind=EvidenceKind.PHOTO,
                              source_uri=path.resolve().as_uri(), sha256=ev.checksum),
            facts=facts, name=ev.name, path=path,
        ))
    items = [ei.item for ei in eval_inputs if ei.facts.geometry_eligible]
    await db.commit()

    await _throw_if_cancelled(db, job)
    # user-facing stage: what the engine is actually about to do
    job.stage = ("building_rough_model" if len(items) <= 1
                 else "refining" if prior_ids else "reconstructing")
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
    # The ladder: multi-view reconstruction when the evidence supports it,
    # the strongest lower level otherwise (never an empty world).
    prog = await asyncio.to_thread(
        lambda: run_progressive(
            eval_inputs, vs_options=options, prior_ids=prior_ids, artifact_store=artifact_store,
            depth_model=(getattr(options, "depth_model", None) or "DPT_Hybrid"),
        )
    )
    world_ir = prog.world

    await _throw_if_cancelled(db, job)

    # Evidence classification survives into provenance: which images fed
    # geometry (and their class) versus which were kept as context only.
    evidence_records = [
        {"evidence_id": ei.item.id, "name": ei.name, "sha256": ei.item.sha256,
         "evidence_class": ei.facts.evidence_class, "class_basis": ei.facts.class_basis,
         "used_for_geometry": ei.facts.geometry_eligible,
         "registered": ei.item.id in set(prog.registered_ids)}
        for ei in eval_inputs
    ]
    world_ir.metadata["evidence"] = {"used": evidence_records, "excluded": prog.excluded}
    world_ir.metadata["progressive"] = {
        "level": prog.level, "level_name": prog.level_name, "model_state": prog.model_state,
        "attempts": prog.attempts,
    }

    # Provenance + determinism record: stamp every reconstructed entity
    # with the session, backend, and pipeline configuration it was built
    # from, before anything is validated or persisted.
    _stamp_reconstruction_provenance(
        world_ir,
        session_id=session.id,
        backend=prog.stage_facts.get("backend"),
        options=options,
        images_ingested=len(items),
        extra={"level": prog.level, "model_state": prog.model_state,
               "source_evidence_ids": list(prog.input_ids)},
    )

    # Validation gate: an invalid WorldIR is never persisted or adopted.
    # The previous HEAD stays intact, so a failed refinement cannot
    # destroy valid earlier results.
    from world_ir.validation import validate_world_ir

    validation = validate_world_ir(world_ir)
    if not validation.is_valid():
        detail = "; ".join(validation.messages()[:5])
        raise RuntimeError(f"reconstructed WorldIR failed validation: {detail}")

    # Provenance hard gate: a reconstructed entity without its build
    # trace must never be persisted. The stamp above runs first, so a
    # missing trace here means stamping itself failed -- persisting
    # would create permanently untraceable state. Fail instead; HEAD
    # stays intact.
    _assert_reconstruction_traced(world_ir)

    # World-level acceptance. A valid candidate is not automatically a BETTER
    # one: compare it with the current HEAD on frame-independent, measured
    # invariants (which photos stay registered, how the registered cameras
    # moved, structure counts) before it may replace the world's knowledge.
    cand_snap = world_delta.snapshot(
        registered_ids=prog.registered_ids, input_ids=prog.input_ids, level=prog.level,
        model_state=prog.model_state, points=len(prog.points), camera_poses=prog.camera_poses,
        world=world_ir,
    )
    delta = world_delta.compute_delta(world_delta.snapshot_from_report(head_report), cand_snap)
    # Explicit conflicts: prior ones are carried forward (never dropped silently), new
    # ones open when a previously registered photo lands somewhere materially different.
    conflicts = world_delta.reconcile_conflicts(head_report.get("conflicts"), delta, f"job:{job.id}")
    decision = world_delta.decide(delta, conflicts)
    adopt = decision["verdict"] != world_delta.REJECT
    changes = world_delta.describe_changes(delta, structure=adopt, conflicts=conflicts)
    if adopt:
        world_ir.metadata["conflicts"] = conflicts
    # Every geometry-eligible photo records THIS attempt, adopted or not: a photo
    # that could not be placed is "waiting", never discarded, and is retried on
    # the next rebuild (a later photo may be the missing bridge).
    _record_registration(
        [ev for (ev, _), ei in zip(resolved, eval_inputs) if ei.facts.geometry_eligible],
        job.id, set(prog.registered_ids), adopt,
    )
    if not adopt:
        session.world_id = world.id
        session.status = "complete"
        session.processing_completed_at = utcnow()
        included = {ev.session_id for ev, _ in resolved if ev.session_id} - {session.id}
        if included:
            others = await db.execute(
                select(Session).where(Session.id.in_(included), Session.status == "processing")
            )
            for other in others.scalars().all():
                other.status = "complete"
                other.processing_completed_at = utcnow()
        job.payload = {
            "world_id": world.id, "version_id": None, "adopted": False, "validated": True,
            "outcome": "partial", "verdict": decision["verdict"], "reasons": decision["reasons"],
            "uncertainties": decision["uncertainties"], "changes": changes, "delta": delta,
            "kept_version_id": world.current_version_id, "level": prog.level,
            "level_name": prog.level_name, "model_state": prog.model_state,
            "registration_status": prog.registration_status, "skipped_evidence": skipped_evidence,
        }
        await db.commit()
        return "partial"

    job.stage = "committing_version"
    job.heartbeat_at = utcnow()
    await db.commit()
    report = {
        "status": "SUCCESS" if (prog.level >= 2 and prog.registration_status == "success") else "PARTIAL_SUCCESS",
        "session_id": session.id,
        "images_ingested": len(items),
        "level": prog.level,
        "level_name": prog.level_name,
        "model_state": prog.model_state,
        "attempts": prog.attempts,
        "guidance": prog.guidance,
        "structure": {"entity_types": cand_snap["entity_types"], "provenance": cand_snap["provenance"]},
        "cameras": cand_snap["cameras"],
        "entities": cand_snap["entities"],
        "conflicts": conflicts,
        "delta": delta,
        "verdict": decision,
        "changes": changes,
        "evidence": {
            "all_ids": [ei.item.id for ei in eval_inputs],
            "input_ids": prog.input_ids,
            "registered_ids": prog.registered_ids,
            "records": evidence_records,
            "excluded": prog.excluded,
            "contribution": prog.contribution.to_dict(),
            "quality": prog.quality,
        },
        "stages": {
            "reconstruction": {
                "backend": prog.stage_facts.get("backend"),
                "cameras_registered": len(prog.registered_ids),
                "cameras_input": len(prog.input_ids),
                "registration_status": prog.registration_status,
                "points": len(prog.points),
            },
            "scale": {"state": prog.scale_state, "meters_per_unit": prog.meters_per_unit},
            "depth": prog.stage_facts.get("depth"),
            "perception": prog.stage_facts.get("perception"),
            "mesh": prog.stage_facts.get("mesh"),
            "compile": {
                "entities": prog.counts.get("entities", len(world_ir.entities)),
                "measurements": prog.counts.get("measurements", 0),
                "relationships": prog.counts.get("relationships", 0),
            },
        },
    }

    points_bytes = await asyncio.to_thread(_render_points_ply, prog.points)
    cameras_bytes = await asyncio.to_thread(
        _render_cameras_json, prog.camera_poses, prog.scale_state, options.image_size
    )

    base_head = world.current_version_id
    try:
        store = worldstore_service.get_store()
        version_row = await worldstore_service.commit_version(
            db,
            world_id=world.id,
            world=world_ir,
            parent=base_head,
            source_session_ids=await _contributing_session_ids(db, world.id, session.id),
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

    # Late cancellation: the flag may have landed while the version was
    # being committed. The adopted version stands (durable work is never
    # un-written), but the job grades CANCELLED -- never SUCCEEDED --
    # with the adopted version recorded for recovery instead of hidden.
    await db.refresh(job, attribute_names=["cancel_requested"])
    cancelled_late = bool(job.cancel_requested)

    # Outcome grading: SUCCEEDED only when the run is clean end to end
    # (successful registration, nothing skipped, no degraded optional
    # stages, no validation warnings). Anything less honest is PARTIAL:
    # a real adopted version with the degradation reasons on record.
    # Failed validation or missing output never reach here (raised above),
    # and a timeout can never become either (it raises before grading).
    # A rough (level <= 1) model is partial by construction: it can never
    # grade as a clean success.
    degraded = list(prog.degraded)
    if skipped_evidence:
        degraded.append(f"{len(skipped_evidence)} skipped evidence item(s)")
    degraded.extend(_stage_facts_degraded(prog.stage_facts))
    degraded.extend(f"changed with uncertainty: {u}" for u in decision["uncertainties"])
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
    # Every other session whose evidence went into this rebuild (a coalesced
    # upload rides on this job) is finished too -- and ONLY those: a session
    # created while this job ran was not part of the union.
    included = {ev.session_id for ev, _ in resolved if ev.session_id} - {session.id}
    if included:
        others = await db.execute(
            select(Session).where(Session.id.in_(included), Session.status == "processing")
        )
        for other in others.scalars().all():
            other.status = "complete"
            other.processing_completed_at = utcnow()
    job.payload = {
        "world_id": world.id,
        "version_id": version_row.id,
        "registration_status": prog.registration_status,
        "level": prog.level,
        "level_name": prog.level_name,
        "model_state": prog.model_state,
        "outcome": outcome,
        "degraded": degraded,
        "validated": True,
        "adopted": True,
        "verdict": decision["verdict"],
        "changes": changes,
        "points": len(prog.points),
        "points_bytes": len(points_bytes),
        "cameras_registered": len(prog.registered_ids),
        "cameras_bytes": len(cameras_bytes),
        "skipped_evidence": skipped_evidence,
    }
    await db.commit()
    _notify_world_version(db, job, world, version_row.id)
    await db.commit()
    if cancelled_late:
        # The operator cancelled while persistence was in flight. The
        # adopted version stands (durable side effects are not rolled
        # back), but the job must not report success it did not observe:
        # CANCELLED with the adopted version identifier for recovery.
        job.payload = {
            **job.payload,
            "outcome": "cancelled",
            "adopted_before_cancel": version_row.id,
        }
        await db.commit()
        return "cancelled"
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
    Also checks for dead workers and recovers their jobs.
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
    
    # Also check for dead workers
    dead_jobs = get_dead_worker_jobs()
    if dead_jobs:
        log.warning("Found %d dead worker jobs, recovering...", len(dead_jobs))
        for job_id in dead_jobs:
            # The actual recovery will be handled by recover_orphaned_jobs
            pass
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

    # Register this worker for crash detection
    register_worker(job.id)

    handler = _HANDLERS.get(job.type)
    # Bind subprocess ownership to this job for the handler's duration
    # (propagates into worker threads, so synchronous backends inherit
    # it transparently). Cleared in the finally below.
    set_current_job(job.id)
    try:
        if handler is None:
            raise RuntimeError(f"No handler for job type {job.type}")
        # Bound every handler: a wedged backend (or lost dependency)
        # must fail this job, never stall the whole queue behind it. A
        # timeout is always a failure -- it can never grade as success.
        # The timeout path tree-kills owned processes first, so a timed
        # out job leaves no live child behind.
        try:
            outcome = await asyncio.wait_for(handler(db, job), timeout=_job_timeout_seconds())
        except TimeoutError as exc:
            terminate_job_procs(job.id)
            await db.refresh(job, attribute_names=["cancel_requested"])
            if job.cancel_requested:
                # Timeout and cancel raced: the operator's explicit
                # cancel wins over the timeout failure.
                raise _JobCancelled(
                    f"job {job.id} cancelled by operator (timed out simultaneously)"
                ) from exc
            raise RuntimeError(
                f"job handler timed out after {_job_timeout_seconds():.0f}s"
            ) from exc
        if outcome == "partial":
            job.status = JOB_PARTIAL
        elif outcome == "cancelled":
            job.status = JOB_CANCELLED
            job.error = (
                f"cancelled by operator after version "
                f"{(job.payload or {}).get('version_id')} was adopted; "
                "version stands, see payload"
            )
        else:
            job.status = JOB_SUCCEEDED
        job.completed_at = utcnow()
        await _emit_completion(db, job)
    except _JobCancelled as exc:
        log.warning("job %s cancelled: %s", job.id, exc)
        terminate_job_procs(job.id)
        job.status = JOB_CANCELLED
        job.error = f"cancelled by operator: {exc}"
        job.completed_at = utcnow()
    except Exception as exc:
        log.exception("job %s failed", job.id)
        import traceback

        job.error = traceback.format_exc()[-2000:]
        # Failure classification for the user-facing state (NEEDS MORE
        # EVIDENCE vs FAILED): carried on the payload, never inferred from text.
        kind = getattr(exc, "failure_kind", None)
        if kind:
            job.payload = {**(job.payload or {}), "failure_kind": kind}
        # A failure that arrives with a pending cancel request grades as
        # CANCELLED (operator intent wins); otherwise failed/retry.
        # Refresh only the flag: nothing else pending here is disturbed,
        # and a refresh failure must not mask the original error.
        try:
            await db.refresh(job, attribute_names=["cancel_requested"])
            cancelled = bool(job.cancel_requested)
        except Exception:
            cancelled = False
        if cancelled:
            terminate_job_procs(job.id)
            job.status = JOB_CANCELLED
            job.error = f"cancelled by operator: {job.error}"
            job.completed_at = utcnow()
        elif job.attempts >= job.max_attempts or getattr(exc, "retryable", True) is False:
            # deterministic failures (no usable evidence, every level failed)
            # cannot succeed on a retry: fail now, don't loop the queue
            job.status = JOB_FAILED
            job.completed_at = utcnow()
        else:
            job.status = JOB_QUEUED
            job.stage = None
    try:
        await db.commit()
    finally:
        # Ownership cleanup is unconditional: even if the final commit
        # itself raises, the next claim rebinds context and the registry
        # never accumulates dead entries.
        set_current_job(None)
        unregister_worker(job.id)
        discard_job(job.id)
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
    # Also recover any jobs whose worker processes died.
    try:
        async with maker() as db:
            await reap_stale_jobs(db)
            # Recover any jobs whose worker processes died
            await recover_orphaned_jobs(db)
    except Exception:
        log.exception("startup recovery failed")
    while True:
        try:
            async with maker() as db:
                await reap_stale_jobs(db)
                # Check for dead workers periodically
                dead_jobs = get_dead_worker_jobs()
                if dead_jobs:
                    log.warning("Found %d dead worker jobs, recovering...", len(dead_jobs))
                    await recover_orphaned_jobs(db, None)
                job = await process_next_job(db)
            if job is None:
                await asyncio.sleep(poll_seconds)
        except Exception:
            log.exception("worker loop error")
            await asyncio.sleep(poll_seconds * 2)

