"""Read-only view of the multi-engine registry: which engines exist, which are REALLY available here and why not, how far each is
integrated, the execution policies, and the standard quality gates (with no thresholds -- none are benchmark-established yet).

Every availability value is produced by the engine's own probe at request time (engine/multiengine/registry.py); nothing is cached
or assumed. Nothing here is tenant data.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from engine.multiengine.executor import Policy
from engine.multiengine.quality_gates import STANDARD_GATES, default_gate_specs
from engine.multiengine.registry import RegistryError, default_registry

engines_router = APIRouter(prefix="/api/engines", tags=["engines"])

_POLICY_NOTES = {
    Policy.MAXIMUM_FIDELITY.value: "every approved engine for a task must run; any that cannot makes the result PARTIAL (default target)",
    Policy.ACCURATE.value: "the primary engine per task only; if it is unavailable the task fails -- no silent substitution",
    Policy.FAST.value: "the cheapest available engine only; always labelled reduced fidelity",
}


@engines_router.get("")
def list_engines() -> Dict[str, Any]:
    reg = default_registry()
    report = reg.report()
    avail = Counter(e["availability"]["status"] for e in report)
    integ = Counter(e["integration"] for e in report)
    return {
        "engines": report,
        "summary": {"total": len(report), "available": avail.get("available", 0), "blocked": avail.get("blocked", 0),
                    "integration": dict(integ),
                    "adapters_implemented": integ.get("adapter", 0),
                    "note": "'registry-only' engines are declared and probed but have no EngineAdapter; 'wired' engines are called "
                            "by the production job through the older direct path, not through the multi-engine manager"},
        "policies": [{"policy": k, "meaning": v} for k, v in _POLICY_NOTES.items()],
    }


@engines_router.get("/quality-gates")
def standard_quality_gates() -> Dict[str, Any]:
    return {"gates": [{"name": g.name, "direction": g.direction, "unit": g.unit, "threshold": g.threshold,
                       "threshold_source": g.threshold_source, "required": g.required} for g in default_gate_specs()],
            "note": "no threshold is benchmark-established yet, so every gate is UNVERIFIED until one is recorded from a documented benchmark",
            "count": len(STANDARD_GATES)}


@engines_router.get("/{engine_id}")
def get_engine(engine_id: str) -> Dict[str, Any]:
    reg = default_registry()
    try:
        spec = reg.get(engine_id)
    except RegistryError:
        raise HTTPException(404, "Engine not found") from None
    return spec.to_dict(reg.availability(engine_id))
