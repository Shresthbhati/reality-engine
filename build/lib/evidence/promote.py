"""Evidence -> WorldIR promotion (spec §7-8/§16, minimal real slice).

The smallest honest link between the evidence layer and WorldIR that
doesn't require a reconstruction backend: a MANUAL_MEASUREMENT evidence
item (a human-entered dimension -- "this wall is 3.2m", not imagery)
becomes a real WorldIR Entity carrying a real Measurement, with
provenance traced back to the originating Session/evidence id rather
than appearing as an unexplained number.

This is deliberately narrow. It does NOT attempt photo/video ->
geometry (that needs an actual CV backend, a real dependency decision,
not something to fake here) -- see docs/CAPABILITY_MATRIX.md, Goal C.
"""

from __future__ import annotations

from typing import Optional

from provenance import Provenance
from world_ir import Entity, EntityType, Material, Measurement, Observation, PhysicalProperties, WorldIR

from .session import EvidenceKind, Session, UnknownEvidenceError


class UnsupportedEvidenceKindError(ValueError):
    pass


def promote_measurement_to_entity(
    session: Session,
    evidence_id: str,
    world: WorldIR,
    entity_id: str,
    entity_name: str = "",
    entity_type: EntityType = EntityType.UNKNOWN,
) -> Entity:
    """Turn one MANUAL_MEASUREMENT evidence item into a real WorldIR Entity.

    The evidence item's metadata must carry {"property": <PhysicalProperties
    field name>, "value": float, "unit": str}. Confidence/provenance on the
    resulting Measurement come from the evidence item itself, not invented.

    Raises UnsupportedEvidenceKindError if the evidence isn't a manual
    measurement -- promoting a photo/video this way would silently
    invent geometry, which is exactly what §31 forbids.
    """
    item = session.get_evidence(evidence_id)  # raises UnknownEvidenceError if missing
    if item.kind != EvidenceKind.MANUAL_MEASUREMENT:
        raise UnsupportedEvidenceKindError(
            f"evidence '{evidence_id}' is {item.kind.value}, not manual_measurement -- "
            "no reconstruction backend exists to derive geometry from it (see Goal C)"
        )

    prop_name = item.metadata["property"]
    measurement = Measurement(
        value=item.metadata["value"],
        unit=item.metadata["unit"],
        provenance=item.provenance,
        confidence=item.uncertainty.confidence,
    )

    material = Material(
        id=f"mat-{entity_id}",
        name=f"{entity_name} material" if entity_name else "measured material",
        properties=PhysicalProperties(**{prop_name: measurement}),
        provenance=item.provenance,
        confidence=measurement.confidence,
        observations=[Observation(
            id=f"obs-{evidence_id}",
            sensor_type="manual",
            timestamp=item.captured_at or 0.0,
            confidence=measurement.confidence,
            metadata={"source_session_id": session.id, "source_evidence_id": evidence_id},
        )],
    )
    world.materials[material.id] = material

    entity = Entity(
        id=entity_id,
        name=entity_name,
        type=entity_type,
        material_ids=[material.id],
        provenance=item.provenance,
        confidence=measurement.confidence,
    )
    world.entities[entity_id] = entity

    session.record_processing(
        "promoted_to_worldir",
        detail={"evidence_id": evidence_id, "entity_id": entity_id, "world_id": world.id},
    )
    return entity
