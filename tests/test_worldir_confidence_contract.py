"""Tests for WorldIR Confidence Contract.

Verifies:
1. Entity with confidence=None represents unmeasured confidence, serializing and deserializing cleanly.
2. Validation accepts confidence=None without error.
3. Legacy worlds with confidence=0.5 load and preserve 0.5 without silent reinterpretation.
4. Calibrated entities with confidence in [0.0, 1.0] and confidence_basis preserve both across round-trips.
5. Out-of-bounds confidence (< 0.0 or > 1.0) is rejected by validation.
"""

from __future__ import annotations

import json
from world_ir.schema_v1 import Entity, EntityType, Provenance
from world_ir.validation import validate_world_ir
from world_ir.world_v1 import WorldIR


def test_entity_with_none_confidence_serializes_and_validates() -> None:
    """An entity with confidence=None represents unmeasured confidence and passes validation."""
    entity = Entity(
        id="ent-unmeasured",
        type=EntityType.FURNITURE,
        name="Table",
        confidence=None,
        confidence_basis=None,
        provenance=Provenance.ESTIMATED,
    )

    d = entity.to_dict()
    assert d["confidence"] is None
    assert "confidence_basis" not in d

    # Round trip
    reloaded = Entity.from_dict(d)
    assert reloaded.confidence is None
    assert reloaded.confidence_basis is None

    # WorldIR validation
    world = WorldIR()
    world.entities[entity.id] = entity
    report = validate_world_ir(world)
    confidence_issues = [i for i in report.issues if i.code == "confidence_out_of_range"]
    assert len(confidence_issues) == 0


def test_legacy_world_with_05_confidence_preserved() -> None:
    """Legacy worlds serialized with confidence=0.5 preserve 0.5 without alteration."""
    raw_json = json.dumps({
        "id": "ent-legacy",
        "type": "furniture",
        "name": "Chair",
        "confidence": 0.5,
        "provenance": Provenance.ESTIMATED.value,
    })

    data = json.loads(raw_json)
    entity = Entity.from_dict(data)
    assert entity.confidence == 0.5
    assert entity.confidence_basis is None

    d = entity.to_dict()
    assert d["confidence"] == 0.5


def test_calibrated_confidence_and_basis_preserved() -> None:
    """Calibrated entities preserve measured confidence and calibration basis."""
    entity = Entity(
        id="ent-calibrated",
        type=EntityType.DOOR,
        name="Exterior Door",
        confidence=0.88,
        confidence_basis="isotonic_residuals_v2",
        provenance=Provenance.RECONSTRUCTED,
    )

    d = entity.to_dict()
    assert d["confidence"] == 0.88
    assert d["confidence_basis"] == "isotonic_residuals_v2"

    reloaded = Entity.from_dict(d)
    assert reloaded.confidence == 0.88
    assert reloaded.confidence_basis == "isotonic_residuals_v2"

    world = WorldIR()
    world.entities[entity.id] = entity
    report = validate_world_ir(world)
    confidence_issues = [i for i in report.issues if i.code == "confidence_out_of_range"]
    assert len(confidence_issues) == 0


def test_invalid_confidence_rejected_by_validator() -> None:
    """Confidence values outside [0.0, 1.0] produce confidence_out_of_range validation errors."""
    world = WorldIR()
    bad_entity = Entity(
        id="ent-bad-conf",
        type=EntityType.FURNITURE,
        name="Bad Confidence Entity",
        confidence=1.45,  # Invalid: > 1.0
        provenance=Provenance.ESTIMATED,
    )
    world.entities[bad_entity.id] = bad_entity

    report = validate_world_ir(world)
    confidence_issues = [i for i in report.issues if i.code == "confidence_out_of_range"]
    assert len(confidence_issues) == 1
    assert "outside [0, 1]" in confidence_issues[0].message
