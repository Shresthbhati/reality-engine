"""Bridge multi-engine evidence into the canonical Evidence Matrix.

Two evidence representations exist in this repository:

  * ``perception.engine_evidence.UnifiedEngineOutput`` — what the perception
    adapters emit (detection, segmentation, depth, feature matching).
  * ``engine.multiengine.evidence_matrix.EvidenceMatrix`` — the append-only,
    content-addressed store the production evidence flow already writes and the
    WorldIR bridge reads.

Until now the multi-engine report only reached the job's JSON facts, so real
engine observations never entered the canonical store: they could not be traced
from a persisted world, and a fresh process could not recover their lineage.
This module closes that gap. Every observation an engine actually produced is
written as an ``EngineObservation`` (raw, immutable, provenance-stamped), and
every engine EXECUTION — including failures, timeouts, cancellations,
dependency skips and unavailable engines — is written as a ``RunRecord``.

What is deliberately NOT done here:

  * No calibrated confidence is claimed. ``EvidenceContext.calibrated_confidence``
    stays ``None``; the engine's raw number is kept on the observation and is
    labelled uncalibrated on import.
  * No observation is given a world position, metric scale or entity identity it
    did not carry itself. Image-frame evidence stays image-frame evidence.
  * Nothing is fabricated for an engine that did not run: it appears as a run
    record with its status and reason.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from perception.engine_evidence import ObservationRecord, UnifiedEngineOutput, _mask_descriptor

log = logging.getLogger("reality.engine_matrix_bridge")

MATRIX_SCHEMA_NOTE = "unified-engine evidence imported from the multi-engine orchestrator"


def _matrix(root: Path | str):
    from engine.multiengine.evidence_matrix import EvidenceMatrix

    return EvidenceMatrix(Path(root))


def _content_key(observation: Any) -> str:
    """The observation's content WITHOUT its processing timestamp.

    ``make_observation`` hashes ``processed_at`` into the id, which is right for a
    per-run log but wrong for idempotency: re-importing the same evidence in a
    later run would look like a new observation. Evidence identity is its content,
    so two records that differ only by when they were stamped are one observation.
    """
    from engine.multiengine.observation import _canonical

    core = _canonical({"engine_id": observation.engine_id, "model_version": observation.model_version,
                       "task": observation.task, "quantity": observation.quantity,
                       "source_id": observation.source_id,
                       "coordinate_system": observation.coordinate_system,
                       "predicted": observation.predicted, "frame": observation.frame,
                       "image_region": list(observation.image_region) if observation.image_region else None,
                       "confidence": observation.confidence, "uncertainty": observation.uncertainty,
                       "calibration": observation.calibration,
                       "execution_config": observation.execution_config, "fallback": observation.fallback})
    import json

    return json.dumps(core, sort_keys=True, separators=(",", ":"))


def _draft_for(obs: ObservationRecord):
    """One observation as a matrix draft, preserving frame, units and provenance."""
    from engine.multiengine.observation import ObservationDraft

    predicted: dict[str, Any] = {
        "label": obs.label,
        "category": obs.category,
        "scale_state": obs.coordinate_frame.scale_state,
        "units": obs.coordinate_frame.units,
        "bbox_px": list(obs.bbox_px) if obs.bbox_px else None,
        "world_pos": list(obs.world_pos) if obs.world_pos else None,
        "point_samples": [list(p) for p in obs.point_samples] if obs.point_samples else None,
        "material_hypothesis": obs.material_hypothesis,
        "color_rgb": list(obs.color_rgb) if obs.color_rgb else None,
        "surface_normal": list(obs.surface_normal) if obs.surface_normal else None,
        "sigma_m": obs.sigma_m,
        "calibration_quality": obs.calibration_quality,
        "timestamp_s": obs.timestamp_s,
        "source_evidence_id": obs.evidence_id,
        "depth_map_ref": obs.depth_map_ref,
        "raw_output_ref": obs.raw_output_ref,
        # Segmentation mask provenance: the row index into the run's MaskStore
        # and the compact descriptor of the in-memory mask. Previously the
        # bridge dropped both, so mask pixels reached no persistent consumer
        # (found by independent review of PR #397).
        "mask_ref": obs.mask_ref,
        "mask": _mask_descriptor(obs.mask) if hasattr(obs, "mask") else None,
        **{k: v for k, v in dict(obs.metadata).items() if k not in ("bbox_px", "world_pos")},
    }
    uncertainty: dict[str, Any] | None = None
    if obs.sigma_m is not None:
        uncertainty = {"sigma_m": obs.sigma_m, "basis": "reported by the engine; not calibrated against ground truth"}
    return ObservationDraft(
        task=obs.category or "perception",
        quantity=obs.label or "observation",
        source_id=obs.source_image_id,
        coordinate_system=obs.coordinate_frame.frame_id,
        predicted=predicted,
        image_region=tuple(obs.bbox_px) if obs.bbox_px else None,
        confidence=obs.confidence,
        uncertainty=uncertainty,
        # The engine's confidence/uncertainty is not calibrated by anyone in this
        # repository, so the record must not claim a calibration.
        calibration=None,
    )


def _execution_rows(report: Any) -> Iterable[Any]:
    """One ``RunRecord`` per engine execution, including the ones that did not run."""
    from engine.multiengine.evidence_matrix import RunRecord

    executions = getattr(report, "engine_executions", None) or []
    if executions:
        for ex in executions:
            yield RunRecord(
                run_id=str(getattr(report, "run_id", "") or ""),
                task=_task_of(ex.architecture_id),
                engine_id=ex.engine_id,
                model_version=ex.version,
                status=ex.status,
                architecture_id=ex.architecture_id,
                checkpoint_id=ex.checkpoint_id,
                device=ex.device,
                error_kind=ex.reason or None,
                error=ex.error,
                wall_s=ex.wall_time_s,
                peak_vram_mb=ex.peak_vram_mb,
                attempts=max(1, ex.attempts),
                sources=(),
                observation_ids=(),
                rejected_count=0,
            )
        return
    # Fall back to the summary statuses when no execution contract is present.
    for engine_id, status in sorted((getattr(report, "engines_run", {}) or {}).items()):
        yield RunRecord(run_id=str(getattr(report, "run_id", "") or ""), task="perception", engine_id=engine_id,
                        model_version="unknown", status=status)


_TASK_BY_ARCHITECTURE = {
    "sift": "matching",
    "lightglue_aliked": "matching",
    "orb": "matching",
    "grounding_dino": "detection",
    "sam2": "segmentation",
    "depth_anything_v2": "depth",
    "mi_da_s": "depth",
}


def _task_of(architecture_id: str) -> str:
    return _TASK_BY_ARCHITECTURE.get(architecture_id, "perception")


def import_multi_engine_report(
    report: Any,
    *,
    matrix_root: Path | str,
    processed_at: str | None = None,
) -> dict[str, Any]:
    """Write a ``MultiEngineReport`` into the run's Evidence Matrix.

    Returns a JSON-serialisable summary: what was imported, per-engine
    observation ids, run-record statuses, and a reload check proving the
    evidence is recoverable by a fresh reader of the same store.
    """
    from engine.multiengine.evidence_matrix import EvidenceContext, MatrixError
    from engine.multiengine.observation import ObservationError, make_observation

    stamp = processed_at or datetime.now(timezone.utc).isoformat()
    matrix = _matrix(matrix_root)
    run_id = str(getattr(report, "run_id", "") or "")
    if not run_id:
        return {"status": "refused", "reason": "the multi-engine report has no run id; evidence without a run cannot be attributed"}

    added: list[str] = []
    reused = 0
    rejected: list[dict[str, Any]] = []
    by_engine: dict[str, list[str]] = {}
    # Content keys already in the store, so re-importing the same evidence is a
    # no-op instead of creating a second observation for one engine output.
    seen: dict[str, str] = {_content_key(o): o.observation_id for o in matrix.store.query()}
    # Deterministic content ids for the world's lineage block: derived from the
    # evidence content only, so re-running the same job reproduces the same world
    # version (the recorded processed_at of a run must not change what the world IS).
    import hashlib as _hashlib

    stable: dict[str, list[str]] = {}

    for obs in getattr(report, "observations", []) or []:
        try:
            draft = _draft_for(obs)
            engine_obs = make_observation(
                draft,
                engine_id=obs.model_id.engine_id or obs.model_id.architecture_id,
                model_version=obs.model_id.version if obs.model_id.version != "unknown" else
                (obs.model_id.checkpoint_id or "unknown"),
                execution_config={
                    "architecture_id": obs.model_id.architecture_id,
                    "checkpoint_id": obs.model_id.checkpoint_id,
                    "license": obs.model_id.license,
                    "non_commercial": obs.model_id.non_commercial,
                    "hardware_class": obs.model_id.hardware_class,
                    "scale_state": obs.coordinate_frame.scale_state,
                    "units": obs.coordinate_frame.units,
                    "calibration_quality": obs.calibration_quality,
                    "imported_by": MATRIX_SCHEMA_NOTE,
                },
                processed_at=stamp,
            )
            key = _content_key(engine_obs)
            stable.setdefault(obs.model_id.architecture_id, []).append(
                "mev_" + _hashlib.sha256(key.encode("utf-8")).hexdigest()[:16])
            if key in seen:
                reused += 1
                by_engine.setdefault(obs.model_id.architecture_id, []).append(seen[key])
                continue
            ctx = EvidenceContext(observation_id=engine_obs.observation_id, run_id=run_id,
                                  architecture_id=obs.model_id.architecture_id,
                                  checkpoint_id=obs.model_id.checkpoint_id)
            if matrix.add(engine_obs, ctx):
                added.append(engine_obs.observation_id)
                seen[key] = engine_obs.observation_id
                by_engine.setdefault(obs.model_id.architecture_id, []).append(engine_obs.observation_id)
            else:
                reused += 1
        except (ObservationError, MatrixError, ValueError, TypeError) as exc:
            rejected.append({"evidence_id": getattr(obs, "evidence_id", ""),
                             "reason": f"{type(exc).__name__}: {exc}"})

    runs_written = 0
    run_statuses: dict[str, str] = {}
    for record in _execution_rows(report):
        try:
            matrix.record_run(record)
            runs_written += 1
            run_statuses[record.engine_id] = record.status
        except (MatrixError, ValueError, TypeError) as exc:
            rejected.append({"run": getattr(record, "engine_id", ""), "reason": f"{type(exc).__name__}: {exc}"})

    # Reload from disk through a NEW reader: the evidence must be recoverable by a
    # fresh process, not merely present in this object's memory.
    reloaded = _matrix(matrix_root)
    missing: list[str] = []
    for oid in added:
        try:
            reloaded.store.get(oid)
        except Exception:  # noqa: BLE001 -- absent from the reloaded store is exactly what is being checked
            missing.append(oid)

    return {
        "status": "imported" if not missing else "import_incomplete",
        "matrix_root": str(matrix_root),
        "n_observations_imported": len(added),
        "n_observations_already_present": reused,
        "n_rejected": len(rejected),
        "rejected": rejected[:20],
        "observation_ids_by_engine": {k: v for k, v in sorted(by_engine.items())},
        "observation_content_ids_by_engine": {k: sorted(set(v)) for k, v in sorted(stable.items())},
        "n_run_records": runs_written,
        "engine_statuses": dict(sorted(run_statuses.items())),
        "reload": {"checked": len(added), "missing": missing[:20]},
        "calibrated_confidence": "none: every imported observation is uncalibrated by construction",
        # Content-addressed URI of the run's RLE mask store (if masks were
        # produced); carried into world lineage so a fresh process can recover
        # mask pixels via mask_ref + this artifact.
        "mask_store_uri": getattr(report, "mask_store_uri", None),
    }


def import_engine_outputs(
    outputs: Sequence[UnifiedEngineOutput],
    *,
    matrix_root: Path | str,
    run_id: str,
    processed_at: str | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: merge raw outputs and import the resulting report."""
    from perception.engine_orchestrator import merge_engine_outputs

    report = merge_engine_outputs(list(outputs))
    if not report.run_id:
        report.run_id = run_id
    return import_multi_engine_report(report, matrix_root=matrix_root, processed_at=processed_at)


def world_lineage_block(summary: Mapping[str, Any]) -> dict[str, Any]:
    """The world-metadata digest of an import: which engines contributed which
    observation ids, and what did not run. Small, stable, and safe to persist."""
    return {
        "source": MATRIX_SCHEMA_NOTE,
        "status": summary.get("status"),
        "n_observations_imported": summary.get("n_observations_imported", 0),
        "n_run_records": summary.get("n_run_records", 0),
        # Content ids, not the run-stamped matrix ids: the world's lineage must be
        # reproducible for identical evidence.
        "observation_content_ids_by_engine": dict(summary.get("observation_content_ids_by_engine") or {}),
        "engine_statuses": dict(summary.get("engine_statuses") or {}),
        "calibrated_confidence": summary.get("calibrated_confidence"),
        "mask_store_uri": summary.get("mask_store_uri"),
    }


__all__ = [
    "import_multi_engine_report",
    "import_engine_outputs",
    "world_lineage_block",
    "MATRIX_SCHEMA_NOTE",
]
