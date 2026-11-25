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
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from perception.engine_adapters import register_all_engines
from perception.engine_orchestrator import MultiEngineOrchestrator

log = logging.getLogger("reality.multi_engine_stage")

#: Engine statuses that mean "this engine did what it set out to do".
_OK_STATUSES = frozenset({"succeeded", "partial"})


def _overall_status(engines_run: Dict[str, str]) -> str:
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
    cameras: Optional[dict] = None,
    pairs: Optional[Sequence[tuple]] = None,
    matrix_root: Optional[Path | str] = None,
) -> Dict[str, Any]:
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
        context: Dict[str, Any] = {"run_id": run_id}
        if pairs:
            context["pairs"] = list(pairs)
        if cameras:
            context["cameras"] = cameras
        report = orch.run_all(list(sources), context=context)

        facts: Dict[str, Any] = {
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
        }

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
