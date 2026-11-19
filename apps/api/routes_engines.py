"""Read-only view of the multi-engine registry: which engines exist, which are REALLY available here and why not, how far each is
integrated, the execution policies, and the standard quality gates (with no thresholds -- none are benchmark-established yet).

Every availability value is produced by the engine's own probe at request time (engine/multiengine/registry.py); nothing is cached
or assumed. Nothing here is tenant data.
"""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from apps.api.auth import AuthContext, get_auth_context, is_global_admin
from engine.multiengine.evidence_store import EvidenceStore
from engine.multiengine.executor import Policy
from engine.multiengine.quality_gates import STANDARD_GATES, default_gate_specs
from engine.multiengine.registry import RegistryError, default_registry
from engine.multiengine.sequential_pipeline import SequentialMaximumFidelityPipeline
from engine.multiengine.stages import STAGE_DEFINITIONS

engines_router = APIRouter(prefix="/api/engines", tags=["engines"])

_SHARED_STORE = EvidenceStore()
_SHARED_PIPELINE = SequentialMaximumFidelityPipeline(store=_SHARED_STORE)

_POLICY_NOTES = {
    Policy.MAXIMUM_FIDELITY.value: "every approved engine for a task must run; any that cannot makes the result PARTIAL (default target)",
    Policy.ACCURATE.value: "the primary engine per task only; if it is unavailable the task fails -- no silent substitution",
    Policy.FAST.value: "the cheapest available engine only; always labelled reduced fidelity",
}


class ReconstructRequest(BaseModel):
    sources: list[str] = Field(..., description="List of source image paths or virtual identifiers")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Capture and sensor metadata"
    )
    allow_unestablished_licence: bool = Field(
        default=False, description="Whether non-commercial or unestablished licences are permitted"
    )


@engines_router.get("")
def list_engines(all: bool = False) -> dict[str, Any]:
    from engine.multiengine.registry import maximum_fidelity_registry

    reg = maximum_fidelity_registry() if all else default_registry()
    report = reg.report()
    avail = Counter(e["availability"]["status"] for e in report)
    integ = Counter(e["integration"] for e in report)
    return {
        "engines": report,
        "summary": {
            "total": len(report),
            "available": avail.get("available", 0),
            "blocked": avail.get("blocked", 0),
            "integration": dict(integ),
            "adapters_implemented": integ.get("adapter", 0),
            "note": "'registry-only' engines are declared and probed but have no EngineAdapter; 'adapter' engines have an "
            "implemented EngineAdapter conforming to the 10-step lifecycle; 'wired' engines are called by production job",
        },
        "policies": [{"policy": k, "meaning": v} for k, v in _POLICY_NOTES.items()],
    }


@engines_router.get("/stages")
def list_pipeline_stages() -> dict[str, Any]:
    """Lists the 13 defined stages of the Sequential Maximum-Fidelity Pipeline."""
    return {
        "stages": [
            {
                "stage_id": s.stage_id.value,
                "name": s.name,
                "task": s.task.value if s.task else None,
                "required_engines": list(s.required_engines),
                "dependencies": [d.value for d in s.dependencies],
                "description": s.description,
            }
            for s in STAGE_DEFINITIONS
        ],
        "count": len(STAGE_DEFINITIONS),
        "policy": Policy.MAXIMUM_FIDELITY.value,
    }


@engines_router.get("/quality-gates")
def standard_quality_gates() -> dict[str, Any]:
    return {
        "gates": [
            {
                "name": g.name,
                "direction": g.direction,
                "unit": g.unit,
                "threshold": g.threshold,
                "threshold_source": g.threshold_source,
                "required": g.required,
            }
            for g in default_gate_specs()
        ],
        "note": "no threshold is benchmark-established yet, so every gate is UNVERIFIED until one is recorded from a documented benchmark",
        "count": len(STANDARD_GATES),
    }


def _require_global_admin(auth: AuthContext, what: str) -> None:
    """These two routes use ONE process-wide store and take server file paths: they are operator tools, not tenant APIs.
    A tenant's own evidence is served, tenant-scoped, under /api/worlds/{id}/evidence-world."""
    if not is_global_admin(auth):
        raise HTTPException(403, f"{what} is restricted to global administrators; use /api/worlds/{{id}}/evidence-world")


@engines_router.post("/reconstruct")
def execute_reconstruction(req: ReconstructRequest, auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    """Execute sequential Maximum-Fidelity multi-model reconstruction session (global administrators only)."""
    _require_global_admin(auth, "running the engines on arbitrary server paths")
    if not req.sources:
        raise HTTPException(400, "At least one source image is required.")

    job_id = f"job_{uuid.uuid4().hex[:12]}"
    pipeline = (
        _SHARED_PIPELINE
        if not req.allow_unestablished_licence
        else SequentialMaximumFidelityPipeline(
            store=_SHARED_STORE, allow_unestablished_licence=True
        )
    )
    job_report = pipeline.execute_session(job_id=job_id, sources=req.sources, metadata=req.metadata)
    return job_report.to_dict()


@engines_router.get("/evidence")
def list_evidence(limit: int = 50, auth: AuthContext = Depends(get_auth_context)) -> dict[str, Any]:
    """Query the process-wide Evidence Store (global administrators only: it mixes every caller's observations)."""
    _require_global_admin(auth, "the shared evidence store")
    all_obs = _SHARED_STORE.query()
    return {
        "total": len(all_obs),
        "observations": [o.to_dict() for o in all_obs[-limit:]],
    }


@engines_router.get("/{engine_id}")
def get_engine(engine_id: str) -> dict[str, Any]:
    reg = default_registry()
    try:
        spec = reg.get(engine_id)
    except RegistryError:
        raise HTTPException(404, "Engine not found") from None
    return spec.to_dict(reg.availability(engine_id))
