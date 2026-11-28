"""Reconstruction -> WorldIR promotion (Goal C/D closing the loop).

Turns a ReconstructionResult (from any IReconstructionBackend) into a real
WorldIR Entity + point-cloud Geometry, with provenance set to RECONSTRUCTED
(never OBSERVED -- these points were computed, not measured) and per-point
uncertainty preserved rather than collapsed into one number.

Mirrors evidence/promote.py's honesty rules: refuses to promote a failed
or empty reconstruction instead of inventing an entity for it.
"""

from __future__ import annotations

from typing import Tuple

from provenance import Provenance
from world_ir import Entity, EntityType, Geometry, GeometryType, Observation, WorldIR

from reconstruction.backend.interface import ReconstructionResult
from reconstruction.validation import ValidationReport, validate_reconstructions


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


def promote_validated_reconstruction_to_entity(
    result: ReconstructionResult,
    reference_result: ReconstructionResult,
    world: WorldIR,
    entity_id: str,
    distance_threshold: float,
    entity_name: str = "",
    entity_type: EntityType = EntityType.UNKNOWN,
) -> Tuple[Entity, ValidationReport]:
    """Promote `result` to WorldIR only after checking it against an
    independent reconstruction of the same/overlapping evidence
    (`reference_result`) via `reconstruction.validation.validate_reconstructions()`.

    This is the wiring Goal F was missing: validation stops being a
    standalone function nothing calls and starts changing real WorldIR
    state. If any point-level disagreement is found, the promoted
    entity's and geometry's provenance is downgraded from RECONSTRUCTED
    to CONFLICT (Provenance.CONFLICT is not canonical, per
    Provenanced.is_canonical() -- disagreement must not silently look
    like a trustworthy reconstruction), confidence is capped at the
    agreement rate, and the disagreement counts are recorded as a real
    Observation on the geometry rather than discarded after this
    function returns.

    Both ReconstructionResults are otherwise validated/promoted exactly
    as promote_reconstruction_to_entity() does -- refuses a failed/empty
    `result` via EmptyReconstructionError, raised before validation runs
    so that error takes precedence over a validator complaint about the
    same failed input.
    """
    entity = promote_reconstruction_to_entity(result, world, entity_id, entity_name, entity_type)
    report = validate_reconstructions(result, reference_result, distance_threshold)

    if report.has_disagreements():
        geometry = world.geometries[entity.geometry_ids[0]]
        total = report.agreements + len(report.disagreements)
        agreement_rate = report.agreements / total if total else 0.0

        geometry.provenance = Provenance.CONFLICT
        geometry.confidence = min(geometry.confidence, agreement_rate)
        geometry.observations.append(Observation(
            id=f"obs-validation-{entity_id}",
            sensor_type="reconstruction_validation",
            confidence=agreement_rate,
            metadata={
                "agreements": report.agreements,
                "disagreements": len(report.disagreements),
                "unmatched_a": report.unmatched_a,
                "unmatched_b": report.unmatched_b,
            },
        ))

        entity.provenance = Provenance.CONFLICT
        entity.confidence = min(entity.confidence, agreement_rate)

    return entity, report
