"""WorldIR integration for the detail spine (universal-perception
directive sections 8/11; the open item recorded in the handoff after
PR #41): refined ROI outcomes become evidence-grounded WorldIR
statements -- refused ROIs never fabricate geometry.

Position in the universal pipeline:

    quality -> budget -> discovery -> ROI -> refinement
      -> (this module) WorldIR integration
      -> WorldStore / exporters / applications

Design rules:
  - REUSE the established WorldIR conventions unchanged; no parallel
    schema, no parallel ingestion path:
      * Geometry + provenance-carrying Observation records exactly
        like evidence/promote_planes.py and the mesh stage;
      * RECONSTRUCTED provenance (the pipeline produced these
        statements from measured evidence, it did not observe them
        directly);
      * Entity.statement_state via the existing classifier
        (world_ir/statement_state.py): RECONSTRUCTED -> DERIVED --
        never a fabricated OBSERVED;
      * measured quality (the refinement's documented-map quality
        from the winning rms) as the confidence on every record;
      * the mesh stage's gate: integrate -> validate_world_ir ->
        roll EVERYTHING back rather than emit an invalid world.
  - A REFUSAL is a recorded FACT, not geometry: nothing is added to
    world.entities / world.geometries for a refused ROI; its
    diagnostic is preserved on the report so downstream consumers
    can see WHERE evidence was insufficient.
  - Deterministic: ids derive from the ROI id, no wall clock, no
    RNG; same input -> byte-identical report and world additions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from perception.detail.refinement import RefinementOutcome
from perception.detail.roi import RegionOfInterest
from provenance import Provenance as ProvenanceEnum
from world_ir import Entity as _Entity
from world_ir import EntityType as _EntityType
from world_ir import Geometry as _Geometry
from world_ir import GeometryType as _GeometryType
from world_ir import Observation as _Observation
from world_ir import Vector3 as _Vector3
from world_ir.statement_state import classify_statement_state
from world_ir.validation import validate_world_ir

__all__ = ["DetailIntegrationReport", "integrate_detail_outcomes"]


#: Observed geometry types this integration emits, one per winning
#: refinement backend. The parameter payload is recorded on the
#: geometry's Observation metadata (the repo's parametric convention:
#: SPHERE/CYLINDER/PLANE parameters live beside the measurement facts
#: that produced them).
_BACKEND_TO_GEOMETRY_TYPE = {
    "plane": _GeometryType.PLANE,
    "cylinder": _GeometryType.CYLINDER,
    "sphere": _GeometryType.SPHERE,
}


@dataclass
class DetailIntegrationReport:
    """What the integration added to the world, and what it recorded
    instead of adding (refusals). Deterministic; serializable."""

    entity_ids: List[str] = field(default_factory=list)
    geometry_ids: List[str] = field(default_factory=list)
    refusals: List[dict] = field(default_factory=list)

    @property
    def refusal_count(self) -> int:
        return len(self.refusals)

    def to_dict(self) -> dict:
        return {
            "entity_ids": list(self.entity_ids),
            "geometry_ids": list(self.geometry_ids),
            "refusals": [dict(r) for r in self.refusals],
            "refusal_count": self.refusal_count,
        }


def _geometry_params(roi: RegionOfInterest, outcome: RefinementOutcome) -> dict:
    """Measured parameters of the winning fit, recorded in the
    observation metadata (never silently dropped)."""
    fit = outcome.refinement.fit
    md = {
        "rms_residual_m": outcome.refinement.rms_residual_m,
        "max_residual_m": outcome.refinement.max_residual_m,
        "n_points": outcome.refinement.n_points,
    }
    name = type(fit).__name__
    if name == "CylinderFit":
        span = fit.height_max_m - fit.height_min_m
        mid = (fit.height_min_m + fit.height_max_m) / 2.0
        md["cylinder"] = {
            "axis": list(fit.axis),
            "axis_point": list(fit.axis_point),
            "radius_m": fit.radius_m,
            "height_min_m": fit.height_min_m,
            "height_max_m": fit.height_max_m,
            "center_m": [
                fit.axis_point[i] + fit.axis[i] * mid for i in range(3)
            ],
            "height_m": span,
        }
    elif name == "SphereFit":
        md["sphere"] = {
            "center": list(fit.center),
            "radius_m": fit.radius_m,
        }
    elif name == "PlaneFit":
        md["plane"] = {
            "normal": list(fit.normal),
            "point": list(fit.point),
        }
    return md


def integrate_detail_outcomes(
    rois: Sequence[RegionOfInterest],
    outcomes: Sequence[RefinementOutcome],
    world,
) -> DetailIntegrationReport:
    """Integrate refinement outcomes into a WorldIR world.

    Refined outcomes become one Geometry + one Entity record each
    (typed by the winning backend, carrying the measured residuals,
    the algorithm's identity, and the ROI's provenance). Refused
    outcomes add NOTHING to the world -- they are recorded on the
    report as facts. After insertion the world is validated; any
    validation error rolls back every created record (the mesh
    stage's gate) and raises ValueError.
    """
    report = DetailIntegrationReport()
    created_entities: List[str] = []
    created_geometries: List[str] = []

    for roi, outcome in zip(rois, outcomes):
        if outcome.roi_id != roi.roi_id:
            raise ValueError(
                f"rois/outcomes misaligned: {roi.roi_id!r} vs "
                f"{outcome.roi_id!r}"
            )
        if outcome.status != "refined" or outcome.refinement is None:
            # A refusal is a recorded fact, not geometry.
            report.refusals.append({
                "roi_id": roi.roi_id,
                "reason": outcome.reason or "refused without a reason",
            })
            continue

        ref = outcome.refinement
        geom_type = _BACKEND_TO_GEOMETRY_TYPE[ref.backend]
        geometry_id = f"geom-detail-{roi.roi_id}"
        entity_id = f"entity-detail-{roi.roi_id}"

        observation = _Observation(
            id=f"obs-detail-{roi.roi_id}",
            sensor_type="detail_refinement",
            confidence=ref.quality,
            metadata={
                "backend": ref.backend,
                "roi_id": roi.roi_id,
                "detail_cells": list(roi.detail_cells),
                "point_ids": list(roi.point_ids),
                "voxel_size": roi.provenance.get("voxel_size"),
                "discovery": roi.provenance.get("discovery", ""),
                "compute_tier": roi.budget.compute_tier,
                "justified_level": roi.budget.justified_level,
                "quality": ref.quality,
                "fits_attempted": outcome.fits_attempted,
                "fits_succeeded": outcome.fits_succeeded,
                **_geometry_params(roi, outcome),
                "note": (
                    "locally refined detail statement: winning fit is "
                    "the measured-rms minimum over the ROI's resolvable "
                    "evidence; residuals are measured, never claimed"
                ),
            },
        )

        geometry = _Geometry(
            id=geometry_id,
            type=geom_type,
            lod_level=int(roi.budget.justified_level[1]),
            vertex_count=ref.n_points,
            bounds_min=_Vector3(
                x=roi.bounds[0], y=roi.bounds[1], z=roi.bounds[2]
            ),
            bounds_max=_Vector3(
                x=roi.bounds[3], y=roi.bounds[4], z=roi.bounds[5]
            ),
            provenance=ProvenanceEnum.RECONSTRUCTED,
            confidence=ref.quality,
            quality_metrics=ref.to_dict(),
            observations=[observation],
        )
        entity = _Entity(
            id=entity_id,
            type=_EntityType.STRUCTURE,
            name=f"Detail refinement {roi.roi_id}",
            geometry_ids=[geometry_id],
            semantic_labels=["detail_refinement"],
            provenance=ProvenanceEnum.RECONSTRUCTED,
            confidence=ref.quality,
            custom_properties={
                "roi_id": roi.roi_id,
                "detail_cells": list(roi.detail_cells),
            },
            statement_state=classify_statement_state(
                ProvenanceEnum.RECONSTRUCTED
            ),
        )

        world.geometries[geometry_id] = geometry
        world.entities[entity_id] = entity
        created_geometries.append(geometry_id)
        created_entities.append(entity_id)

    report.entity_ids = created_entities
    report.geometry_ids = created_geometries

    # The mesh stage's gate: never emit an invalid world.
    validation = validate_world_ir(world)
    if validation.errors:
        for eid in created_entities:
            world.entities.pop(eid, None)
        for gid in created_geometries:
            world.geometries.pop(gid, None)
        raise ValueError(
            f"detail integration produced an invalid world: "
            f"{validation.errors[:3]}"
        )

    return report
