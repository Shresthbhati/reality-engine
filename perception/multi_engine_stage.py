"""Multi-engine integration for the production evidence pipeline.

This module bridges the MultiEngineOrchestrator (perception/engine_orchestrator.py)
and the UnifiedEngineOutput contract (perception/engine_evidence.py) into the
production evidence-world flow (apps/api/evidence_world_stage.py).

Every registered engine is invoked through the orchestrator, and its
observations are converted into the evidence matrix's observation format so
they feed into the same reconstruction pipeline as detections, depth, and
spatial inputs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from perception.engine_adapters import register_all_engines
from perception.engine_evidence import UnifiedEngineOutput
from perception.engine_orchestrator import MultiEngineOrchestrator, merge_engine_outputs


def run_multi_engine_for_job(
    *,
    sources: Sequence[str],
    run_id: str,
    cameras: Optional[dict] = None,
    pairs: Optional[Sequence[tuple]] = None,
) -> Dict[str, Any]:
    """Run all registered engines on the job's evidence and return facts.

    Returns a dict suitable for the job report under ``stages.multi_engine``.
    Engine failures are recorded but never sink the job.
    """
    try:
        register_all_engines()
    except Exception as exc:
        return {"status": "failed", "error": f"engine registration failed: {exc}"}

    try:
        from perception.engine_orchestrator import list_available_engines, get_unavailable_engines
        available = [e.engine_id for e in list_available_engines()]
        unavailable = get_unavailable_engines()
    except Exception as exc:
        available = []
        unavailable = [{"error": str(exc)}]

    if not available:
        return {"status": "skipped", "reason": "no engines available", "unavailable": unavailable}

    try:
        orch = MultiEngineOrchestrator()
        context = {"run_id": run_id}
        if pairs:
            context["pairs"] = list(pairs)
        report = orch.run_all(list(sources), context=context)
        return {
            "status": "succeeded" if report.engines_run and all(
                s in ("succeeded", "partial") for s in report.engines_run.values()
            ) else "partial",
            "engines_run": report.engines_run,
            "n_observations": len(report.observations),
            "n_contradictions": len(report.contradictions),
            "contradictions": report.contradictions,
            "unavailable": unavailable + report.unavailable,
            "total_wall_s": report.total_wall_s,
            "vram_peak_mb": report.vram_peak_mb,
            "cross_engine_summary": report.cross_engine_summary,
        }
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "unavailable": unavailable}


__all__ = ["run_multi_engine_for_job"]
