"""Reconstruction -> WorldIR promotion (Goal C/D closing the loop).

Turns a ReconstructionResult (from any IReconstructionBackend) into a real
WorldIR Entity + point-cloud Geometry, with provenance set to RECONSTRUCTED
(never OBSERVED -- these points were computed, not measured) and per-point
uncertainty preserved rather than collapsed into one number.

Mirrors evidence/promote.py's honesty rules: refuses to promote a failed
or empty reconstruction instead of inventing an entity for it.
"""

from __future__ import annotations

from provenance import Provenance
from world_ir import Entity, EntityType, Geometry, GeometryType, WorldIR

from reconstruction.backend.interface import ReconstructionResult


class EmptyReconstructionError(ValueError):
    pass


def promote_reconstruction_to_entity(
    result: ReconstructionResult,
    world: WorldIR,
    entity_id: str,
    entity_name: str = "",
    entity_type: EntityType = EntityType.UNKNOWN,
) -> Entity:
    """Turn a successful/partial ReconstructionResult into a real WorldIR Entity.

    Raises EmptyReconstructionError if registration_status == "failed" or
    there are no points -- promoting nothing would either crash later or
    silently create an empty, meaningless entity.
    """
    if result.registration_status == "failed" or not result.points:
        raise EmptyReconstructionError(
            f"reconstruction for entity '{entity_id}' has status "
            f"'{result.registration_status}' with {len(result.points)} points -- refusing to promote"
        )

    confidences = [p.uncertainty.confidence for p in result.points]
    avg_confidence = sum(confidences) / len(confidences)

    geometry = Geometry(
        id=f"geom-{entity_id}",
        type=GeometryType.POINTCLOUD,
        vertex_count=len(result.points),
        provenance=Provenance.RECONSTRUCTED,
        confidence=avg_confidence,
    )
    world.geometries[geometry.id] = geometry

    entity = Entity(
        id=entity_id,
        name=entity_name,
        type=entity_type,
        geometry_ids=[geometry.id],
        provenance=Provenance.RECONSTRUCTED,
        confidence=avg_confidence,
    )
    world.entities[entity_id] = entity
    return entity
