"""Multi-engine integration for the production evidence pipeline.

This module bridges the MultiEngineOrchestrator (perception/engine_orchestrator.py)
and the UnifiedEngineOutput contract (perception/engine_evidence.py) into the
production evidence-world flow (apps/api/evidence_world_stage.py).

Every registered engine is invoked through the orchestrator, and its
observations are imported into the run's canonical Evidence Matrix through
``perception.engine_matrix_bridge`` so they are traceable from the persisted
world and recoverable by a later process.

A failed, timed-out, cancelled, skipped or unavailable engine is reported with
that status and its reason. It is never replaced by a substituted result, and a
run in which any engine did not succeed is reported ``partial``, never
``succeeded``.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from perception.engine_adapters import register_all_engines
from perception.engine_orchestrator import MultiEngineOrchestrator

log = logging.getLogger("reality.multi_engine_stage")

#: Engine statuses that mean "this engine did what it set out to do".
_OK_STATUSES = frozenset({"succeeded", "partial"})


def _overall_status(engines_run: dict[str, str]) -> str:
    """``succeeded`` only when every engine succeeded; otherwise ``partial``/``failed``.

    ``partial`` also covers unavailable/skipped engines: a run that could not
    execute part of the registry must not read as a complete one.
    """
    if not engines_run:
        return "skipped"
    statuses = set(engines_run.values())
    if statuses <= _OK_STATUSES:
        return "succeeded"
    if statuses <= {"failed", "timed_out", "invalid_output", "unavailable", "skipped", "cancelled"}:
        return "failed"
    return "partial"


def run_multi_engine_for_job(
    *,
    sources: Sequence[str],
    run_id: str,
    cameras: dict | None = None,
    pairs: Sequence[tuple] | None = None,
    matrix_root: Path | str | None = None,
) -> dict[str, Any]:
    """Run all registered engines on the job's evidence and return facts.

    Returns a dict suitable for the job report under ``stages.multi_engine``.
    Engine failures are recorded but never sink the job. When ``matrix_root`` is
    given the engine evidence is imported into that run's Evidence Matrix and the
    import summary is returned under ``matrix_import``.
    """
    try:
        register_all_engines()
    except Exception as exc:
        return {"status": "failed", "error": f"engine registration failed: {exc}"}

    try:
        from perception.engine_orchestrator import get_unavailable_engines, list_available_engines

        available = [e.engine_id for e in list_available_engines()]
        unavailable = get_unavailable_engines()
    except Exception as exc:
        return {"status": "failed", "error": f"engine discovery failed: {type(exc).__name__}: {exc}"}

    if not available:
        return {"status": "skipped", "reason": "no engines available", "available": [],
                "unavailable": unavailable}

    try:
        orch = MultiEngineOrchestrator()
        context: dict[str, Any] = {"run_id": run_id}
        if pairs:
            context["pairs"] = list(pairs)
        if cameras:
            context["cameras"] = cameras
        report = orch.run_all(list(sources), context=context)

        # Persist segmentation masks (SAM 2 and any mask-emitting engine) into
        # the run's MaskStore and stamp each observation with its persisted
        # mask_ref. This is the connection the A4 run was missing: SAM 2
        # computed masks but the adapter stored only mask_area_px in metadata,
        # so mask pixels were discarded before reaching MaskStore, the
        # ObservationTable, or any downstream consumer.
        mask_store_path: str | None = None
        if report.observations:
            from world_ir.mask_store import MaskStore

            from perception.mask_persistence import persist_masks, save_mask_store

            mask_store = MaskStore()
            stamped = persist_masks(list(report.observations), mask_store)
            report = replace(report, observations=tuple(stamped))
            stage_dir = Path(tempfile.mkdtemp(prefix="reality_engine_masks_"))
            mask_store_path = str(stage_dir / "masks.npz")
            save_mask_store(mask_store, mask_store_path)

        facts: dict[str, Any] = {
            "status": _overall_status(dict(report.engines_run)),
            "available": available,
            "engines_run": report.engines_run,
            "n_observations": len(report.observations),
            "n_contradictions": len(report.contradictions),
            "n_duplicate_observations": report.duplicate_observations,
            "contradictions": report.contradictions[:200],
            "unavailable": list(unavailable) + list(report.unavailable),
            "engine_executions": [e.to_dict() for e in report.engine_executions],
            "total_wall_s": round(report.total_wall_s, 3),
            "vram_peak_mb": report.vram_peak_mb,
            "cross_engine_summary": report.cross_engine_summary,
            #: Artifact path of the persisted MaskStore (RLE-encoded masks).
            #: None when no observation carried a mask.
            "mask_store_path": mask_store_path,
        }

        # Multi-engine detection fusion & cross-view identity over detection observations
        det_by_img_eng: dict[str, dict[str, list[Any]]] = {}
        try:
            from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionBox
            from perception.detection.multi_engine_fusion import (
                associate_cross_view_detections,
                fuse_multi_engine_detections,
            )

            for obs in report.observations:
                if obs.bbox_px is None or obs.model_id.engine_id != "detection":
                    continue
                eng_id = obs.model_id.architecture_id or obs.model_id.engine_id
                det_by_img_eng.setdefault(obs.source_image_id, {}).setdefault(eng_id, []).append(
                    DetectionBox(
                        box_id=obs.evidence_id,
                        class_name=obs.label,
                        box_2d=tuple(float(v) for v in obs.bbox_px),
                        confidence=float(obs.confidence) if obs.confidence is not None else 0.5,
                        source_engine=eng_id,
                        source_image_id=obs.source_image_id,
                        raw_score=obs.confidence,
                        attributes={"confidence_emitted": obs.confidence is not None},
                    )
                )
            fused_total = 0
            corroborated = 0
            competing = 0
            for img_id, eng_map in det_by_img_eng.items():
                fused = fuse_multi_engine_detections(eng_map)
                fused_total += len(fused)
                corroborated += sum(1 for f in fused if f.independent_family_count >= 2)
                competing += sum(1 for f in fused if f.competing_hypotheses)
            facts["multi_engine_fusion"] = {
                "images_fused": len(det_by_img_eng),
                "fused_hypotheses": fused_total,
                "multi_family_corroborated": corroborated,
                "competing_label_clusters": competing,
            }
            src_list = [str(s) for s in sources]
            if len(src_list) >= 2:
                cv_counts = {"matched": 0, "ambiguous": 0, "unmatched": 0}
                for idx in range(len(src_list) - 1):
                    a_all = [b for lst in det_by_img_eng.get(src_list[idx], {}).values() for b in lst]
                    b_all = [b for lst in det_by_img_eng.get(src_list[idx + 1], {}).values() for b in lst]
                    if a_all and b_all:
                        for dec in associate_cross_view_detections(a_all, b_all):
                            cv_counts[dec.status] = cv_counts.get(dec.status, 0) + 1
                facts["cross_view_identity"] = cv_counts
        except Exception as exc:  # noqa: BLE001
            facts["multi_engine_fusion"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

        if matrix_root is not None:
            try:
                from perception.engine_matrix_bridge import import_multi_engine_report

                facts["matrix_import"] = import_multi_engine_report(report, matrix_root=matrix_root)
            except Exception as exc:  # noqa: BLE001 -- recorded, never swallowed
                facts["matrix_import"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}

        return facts
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "unavailable": unavailable}


__all__ = ["run_multi_engine_for_job"]
