"""REST API endpoints for inspecting, filtering, and benchmarking the 50 Open-Source Technologies."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from apps.api.auth import AuthContext, get_auth_context
from engine.technologies.benchmark import TechnologyBenchmarkHarness
from engine.technologies.catalog import (
    ALL_50_TECHNOLOGIES,
    TECHNOLOGIES_BY_ID,
    TechnologySubsystem,
)
from engine.technologies.registry import get_technology_registry

technologies_router = APIRouter(prefix="/api/technologies", tags=["technologies"])


class BenchmarkRequest(BaseModel):
    tech_id: Optional[str] = Field(None, description="Specific technology ID to evaluate")
    subsystem: Optional[str] = Field(None, description="Subsystem category to evaluate")


@technologies_router.get("")
def list_technologies(
    subsystem: Optional[str] = Query(None, description="Filter by subsystem enum"),
    available_only: bool = Query(False, description="Filter to only host-available tools"),
    min_priority: Optional[int] = Query(None, description="Filter by maximum priority tier (1, 2, 3)"),
    auth: Optional[AuthContext] = Depends(get_auth_context),
) -> Dict[str, Any]:
    """List evaluated open-source technologies with subsystem and availability metadata."""
    registry = get_technology_registry()
    sub_enum: Optional[TechnologySubsystem] = None
    if subsystem:
        try:
            sub_enum = TechnologySubsystem(subsystem)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid subsystem '{subsystem}'. Valid options: {[s.value for s in TechnologySubsystem]}",
            )

    statuses = registry.filter_technologies(
        subsystem=sub_enum,
        available_only=available_only,
        min_priority=min_priority,
    )
    summary = registry.get_summary_statistics()

    return {
        "total_count": len(statuses),
        "summary": summary,
        "items": [
            {
                "id": s.descriptor.id,
                "name": s.descriptor.name,
                "subsystem": s.descriptor.subsystem.value,
                "license": s.descriptor.license,
                "license_notes": s.descriptor.license_notes,
                "evaluation_priority": s.descriptor.evaluation_priority,
                "hardware_tier": s.descriptor.hardware_tier.value,
                "repo_url": s.descriptor.repo_url,
                "description": s.descriptor.description,
                "reality_engine_role": s.descriptor.reality_engine_role,
                "available": s.available,
                "classification": s.classification.value,
                "verification_status": s.verification_status.value,
                "binary_found": s.binary_found,
                "module_found": s.module_found,
                "internal_module": s.internal_module,
                "notes": s.notes,
                "target_metrics": list(s.descriptor.target_metrics),
            }
            for s in statuses
        ],
    }


@technologies_router.get("/{tech_id}")
def get_technology(
    tech_id: str,
    auth: Optional[AuthContext] = Depends(get_auth_context),
) -> Dict[str, Any]:
    """Inspect full descriptor and live discovery state for a single technology."""
    registry = get_technology_registry()
    desc = registry.get_descriptor(tech_id)
    if not desc:
        raise HTTPException(status_code=404, detail=f"Technology '{tech_id}' not found in catalog")

    status = registry.check_technology_availability(tech_id)

    return {
        "id": desc.id,
        "name": desc.name,
        "subsystem": desc.subsystem.value,
        "license": desc.license,
        "license_notes": desc.license_notes,
        "repo_url": desc.repo_url,
        "description": desc.description,
        "reality_engine_role": desc.reality_engine_role,
        "hardware_tier": desc.hardware_tier.value,
        "evaluation_priority": desc.evaluation_priority,
        "binary_names": list(desc.binary_names),
        "python_modules": list(desc.python_modules),
        "discovery": {
            "available": status.available,
            "classification": status.classification.value,
            "verification_status": status.verification_status.value,
            "binary_found": status.binary_found,
            "module_found": status.module_found,
            "internal_module": status.internal_module,
            "notes": status.notes,
            "has_adapter": status.has_adapter,
        },
        "target_metrics": list(desc.target_metrics),
    }


@technologies_router.post("/benchmark")
def run_benchmark(
    body: Optional[BenchmarkRequest] = None,
    auth: Optional[AuthContext] = Depends(get_auth_context),
) -> Dict[str, Any]:
    """Execute benchmark harness and return quantitative performance metrics."""
    harness = TechnologyBenchmarkHarness()
    tech_id = body.tech_id if body else None

    if tech_id:
        try:
            result = harness.benchmark_technology(tech_id)
            return asdict(result)
        except KeyError as err:
            raise HTTPException(status_code=404, detail=str(err))

    # Benchmark all 50
    return harness.benchmark_all_50()
