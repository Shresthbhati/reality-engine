"""Tests for evidence -> WorldIR promotion (spec §7-8/§16 minimal slice)."""

import pytest

from evidence import (
    Session, EvidenceItem, EvidenceKind, Dataset,
    promote_measurement_to_entity, UnsupportedEvidenceKindError,
)
from evidence.session import UnknownEvidenceError
from world_ir import WorldIR, EntityType
from provenance import Provenance, Uncertainty


def _measurement_item(confidence=0.9):
    return EvidenceItem(
        id="ev-wall-height",
        kind=EvidenceKind.MANUAL_MEASUREMENT,
        source_uri="manual://surveyor-notes",
        captured_at=5.0,
        provenance=Provenance.OBSERVED,
        uncertainty=Uncertainty(confidence=confidence),
        metadata={"property": "mass", "value": 3.2, "unit": "m"},
    )


def test_promote_creates_real_entity_and_material():
    session = Session("sess1")
    session.add_evidence(_measurement_item())
    world = WorldIR(id="w1")

    entity = promote_measurement_to_entity(
        session, "ev-wall-height", world, entity_id="wall_01", entity_name="North Wall", entity_type=EntityType.STRUCTURE,
    )

    assert entity.id in world.entities
    assert entity.type == EntityType.STRUCTURE
    assert entity.material_ids
    material = world.materials[entity.material_ids[0]]
    assert material.properties.mass.value == 3.2
    assert material.properties.mass.unit == "m"


def test_promote_carries_provenance_and_confidence_from_evidence_not_invented():
    session = Session("sess1")
    session.add_evidence(_measurement_item(confidence=0.42))
    world = WorldIR(id="w1")

    entity = promote_measurement_to_entity(session, "ev-wall-height", world, entity_id="wall_01")

    assert entity.provenance == Provenance.OBSERVED
    assert entity.confidence == 0.42  # exactly the evidence's own confidence, not a made-up default


def test_promote_records_processing_history_traceable_to_source():
    session = Session("sess1")
    session.add_evidence(_measurement_item())
    world = WorldIR(id="w1")
    promote_measurement_to_entity(session, "ev-wall-height", world, entity_id="wall_01")

    record = session.processing_history[-1]
    assert record.operation == "promoted_to_worldir"
    assert record.detail["entity_id"] == "wall_01"

    material = world.materials["mat-wall_01"]
    assert material.observations[0].metadata["source_session_id"] == "sess1"
    assert material.observations[0].metadata["source_evidence_id"] == "ev-wall-height"


def test_promote_rejects_non_measurement_evidence():
    """Promoting a photo this way would silently invent geometry -- must refuse (spec §31)."""
    session = Session("sess1")
    session.add_evidence(EvidenceItem(id="ev-photo", kind=EvidenceKind.PHOTO, source_uri="file://a.jpg"))
    world = WorldIR(id="w1")

    with pytest.raises(UnsupportedEvidenceKindError):
        promote_measurement_to_entity(session, "ev-photo", world, entity_id="wall_01")


def test_promote_unknown_evidence_raises():
    session = Session("sess1")
    world = WorldIR(id="w1")
    with pytest.raises(UnknownEvidenceError):
        promote_measurement_to_entity(session, "nope", world, entity_id="wall_01")


def test_promote_then_worldir_roundtrips():
    """The promoted entity is a real part of WorldIR -- it must survive
    WorldIR's own serialization, not live only in memory."""
    session = Session("sess1")
    session.add_evidence(_measurement_item())
    world = WorldIR(id="w1")
    promote_measurement_to_entity(session, "ev-wall-height", world, entity_id="wall_01", entity_name="North Wall")

    restored = WorldIR.from_dict(world.to_dict())
    assert restored.entities["wall_01"].name == "North Wall"
    assert restored.materials["mat-wall_01"].properties.mass.value == 3.2
