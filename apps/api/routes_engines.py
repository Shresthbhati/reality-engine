"""Read-only view of the multi-engine registry: which engines exist, which are REALLY available here and why not, how far each is
integrated, the execution policies, and the standard quality gates (with no thresholds -- none are benchmark-established yet).

Also exposes the 125-slot Target Model Ecosystem Catalogue, Universal Object & Part Ontology,
Shared Evidence Matrix, and 14-Stage Universal Fusion Pipeline.
"""

from __future__ import annotations

import tempfile
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from apps.api.auth import AuthContext, get_auth_context, is_global_admin
from engine.multiengine.catalogue import (
    IntegrationStatus,
    ModelDomain,
    get_catalogue,
)
from engine.multiengine.evidence_matrix import (
    EvidenceMatrixEntry,
    EvidenceMatrixStore,
)
from engine.multiengine.evidence_store import EvidenceStore
from engine.multiengine.executor import Policy
from engine.multiengine.quality_gates import STANDARD_GATES, default_gate_specs
from engine.multiengine.registry import RegistryError, default_registry
from engine.multiengine.sequential_pipeline import SequentialMaximumFidelityPipeline
from engine.multiengine.stages import STAGE_DEFINITIONS
from engine.multiengine.universal_fusion import (
    UniversalFusionPipeline,
)
from world.hierarchy.spatial_hierarchy import SpatialHierarchyLevel
from world.ontology.universal_object_ontology import (
    ObjectCategory,
    SpatialRelationType,
    get_universal_ontology,
)

engines_router = APIRouter(prefix="/api/engines", tags=["engines"])

_SHARED_STORE = EvidenceStore()
_SHARED_PIPELINE = SequentialMaximumFidelityPipeline(store=_SHARED_STORE)

_TMP_MATRIX_ROOT = Path(tempfile.gettempdir()) / "reality_evidence_matrix"
_SHARED_MATRIX_STORE = EvidenceMatrixStore(storage_root=_TMP_MATRIX_ROOT)
_SHARED_UNIVERSAL_PIPELINE = UniversalFusionPipeline(evidence_store=_SHARED_MATRIX_STORE)

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


class UniversalReconstructRequest(BaseModel):
    session_id: str | None = Field(default=None, description="Reconstruction session ID")
    sources: list[str] = Field(..., description="List of source image paths or virtual identifiers")
    camera_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Camera calibration, intrinsics, and extrinsics"
    )
    target_hierarchy_level: str = Field(
        default="room", description="Target spatial scale (city, building, room, object)"
    )
    evidence_ids: list[str] = Field(
        default_factory=list, description="Specific pre-ingested evidence IDs in Evidence Matrix to fuse"
    )
    run_sequential_engines: bool = Field(
        default=True, description="Whether to execute available sequential engines to produce real observations"
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


@engines_router.get("/catalogue")
def get_target_model_catalogue(
    domain: str | None = None,
    commercial_only: bool = False,
    status: str | None = None,
) -> dict[str, Any]:
    """Queries the 125-slot target model catalogue across the 10 reconstruction domains."""
    cat = get_catalogue()
    domain_enum = ModelDomain(domain) if domain else None
    status_enum = IntegrationStatus(status) if status else None

    entries = cat.find(
        domain=domain_enum,
        commercial_viable_only=commercial_only,
        status=status_enum,
    )
    summary = cat.summary()
    return {
        "total_slots": len(entries),
        "catalogue_summary": summary,
        "entries": [e.to_dict() for e in entries],
    }


@engines_router.get("/catalogue/{slot_id}")
def get_catalogue_slot(slot_id: str) -> dict[str, Any]:
    cat = get_catalogue()
    entry = cat.get(slot_id)
    if not entry:
        raise HTTPException(404, f"Slot {slot_id} not found in 125-slot catalogue")
    return entry.to_dict()


@engines_router.get("/ontology")
def get_ontology_summary() -> dict[str, Any]:
    """Exposes the Universal Object & Part Ontology schema."""
    ont = get_universal_ontology()
    classes = []
    for name, c in ont._classes.items():
        parts = [p.part_name for p in ont.get_parts(name)]
        classes.append({
            "name": c.class_name,
            "category": c.category.value,
            "synonyms": c.synonyms,
            "allowed_parts": parts,
            "expected_relations": [r.value for r in c.expected_relations],
            "typical_dimensions_m": list(c.typical_dimensions_m),
            "is_container": c.is_container,
            "is_structural": c.is_structural,
        })
    return {
        "total_classes": len(classes),
        "categories": [cat.value for cat in ObjectCategory],
        "spatial_relations": [rel.value for rel in SpatialRelationType],
        "classes": sorted(classes, key=lambda x: x["name"]),
    }


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


@engines_router.post("/universal-reconstruct")
def execute_universal_reconstruction(req: UniversalReconstructRequest) -> dict[str, Any]:
    """Execute complete 14-stage universal multi-engine reconstruction & fusion on real evidence."""
    if not req.sources and not req.evidence_ids:
        raise HTTPException(400, "At least one source image or pre-ingested evidence ID is required.")

    session_id = req.session_id or f"univ_sess_{uuid.uuid4().hex[:12]}"
    level = SpatialHierarchyLevel(req.target_hierarchy_level)

    evidence_entries: list[EvidenceMatrixEntry] = []

    # 1. Gather explicitly requested pre-ingested evidence
    if req.evidence_ids:
        for eid in req.evidence_ids:
            entry = _SHARED_MATRIX_STORE.get_entry(eid)
            if entry:
                evidence_entries.append(entry)

    # 2. Gather existing entries for this session or sources
    existing = _SHARED_MATRIX_STORE.query(run_id=session_id)
    for entry in existing:
        if entry not in evidence_entries:
            evidence_entries.append(entry)

    # 3. If requested and sources provided, run sequential pipeline to obtain real observations
    if req.run_sequential_engines and req.sources and not evidence_entries:
        try:
            _SHARED_PIPELINE.execute_session(
                job_id=session_id,
                sources=req.sources,
                metadata=req.camera_metadata,
            )
            # Ingest observations produced by the real pipeline into EvidenceMatrixStore
            new_obs = _SHARED_STORE.query()
            for obs in new_obs:
                if obs.source_id in req.sources or any(s in req.sources for s in obs.source_id.split(",")):
                    eid = f"ev_obs_{obs.observation_id}"
                    entry = EvidenceMatrixEntry(
                        evidence_id=eid,
                        model_id=obs.engine_id,
                        architecture_id=obs.engine_id,
                        checkpoint_id=obs.engine_id,
                        run_id=session_id,
                        source_ids=[obs.source_id],
                        input_parameters=dict(obs.predicted),
                        prediction_type=obs.quantity,
                        raw_output_reference=obs.observation_id,
                        coordinate_frame=obs.coordinate_system,
                        confidence=obs.confidence,
                    )
                    if eid not in _SHARED_MATRIX_STORE._entries:
                        _SHARED_MATRIX_STORE.add_entry(entry)
                    evidence_entries.append(entry)
        except Exception:
            # Fall through if engines fail or are unavailable
            pass

    # 4. If no real evidence could be retrieved or produced, report honest BLOCKED state
    if not evidence_entries:
        return {
            "status": "BLOCKED",
            "success": False,
            "session_id": session_id,
            "message": "No evidence entries available for reconstruction. Provide valid pre-ingested evidence_ids or ensure approved engines are installed and available to process source images.",
            "fidelity_score": 0.0,
            "quality_gate_passed": False,
            "total_fused_entities": 0,
            "hierarchy_tree": {},
            "stage_results": [],
            "active_capture_requests": [],
            "summary_report": {
                "reason": "Missing evidence; sequential engine execution produced 0 observations or required checkpoints are blocked",
                "status": "BLOCKED",
            },
        }

    # 5. Run the 14-stage universal fusion pipeline
    result = _SHARED_UNIVERSAL_PIPELINE.run(
        session_id=session_id,
        evidence_entries=evidence_entries,
        camera_metadata=req.camera_metadata,
        target_hierarchy_level=level,
    )

    # Determine honest status
    if not result.success:
        status = "FAILED"
    elif result.quality_gate_passed and all(ent.bounding_box_3d is not None for ent in result.fused_entities.values()):
        status = "COMPLETE"
    else:
        status = "PARTIAL"

    return {
        "status": status,
        "success": result.success,
        "session_id": result.session_id,
        "fidelity_score": result.fidelity_score,
        "quality_gate_passed": result.quality_gate_passed,
        "total_fused_entities": len(result.fused_entities),
        "hierarchy_tree": result.hierarchy_tree.to_dict(),
        "stage_results": [
            {
                "stage": sr.stage.value,
                "success": sr.success,
                "duration_s": round(sr.duration_s, 4),
                "message": sr.message,
                "metrics": sr.metrics,
                "errors": sr.errors,
                "warnings": sr.warnings,
            }
            for sr in result.stage_results
        ],
        "active_capture_requests": [
            {
                "request_id": cr.request_id,
                "target_region": cr.target_region,
                "recommended_pose": cr.recommended_pose,
                "rationale": cr.rationale,
                "priority": cr.priority,
            }
            for cr in result.capture_requests
        ],
        "summary_report": result.summary_report,
    }


@engines_router.get("/evidence-matrix")
def list_evidence_matrix(limit: int = 50) -> dict[str, Any]:
    """Query Shared Evidence Matrix and verify immutable artifact integrity."""
    entries = _SHARED_MATRIX_STORE.query()
    all_valid, errors = _SHARED_MATRIX_STORE.verify_immutable()
    return {
        "total_entries": len(entries),
        "immutable_verification": {
            "all_valid": all_valid,
            "errors": errors,
        },
        "entries": [e.to_dict() for e in entries[-limit:]],
    }


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
