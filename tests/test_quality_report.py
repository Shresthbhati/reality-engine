"""Tests for the WorldQualityReport aggregate quality layer."""

from provenance import Provenance
from world_ir import Entity, EntityType, Relationship, RelationshipKind, WorldIR

from engine.quality import WorldQualityReport, compute_quality_report
from engine.quality.report import (
    canonical_fraction,
    low_confidence_entity_ids,
    mean_confidence,
    provenance_breakdown,
    unresolved_relationship_entity_ids,
)


def _build_world() -> WorldIR:
    world = WorldIR(id="w1", name="quality-test")

    world.entities["e-observed"] = Entity(
        id="e-observed", type=EntityType.STRUCTURE, name="Wall A",
        provenance=Provenance.OBSERVED, confidence=0.95,
    )
    world.entities["e-reconstructed"] = Entity(
        id="e-reconstructed", type=EntityType.STRUCTURE, name="Wall B",
        provenance=Provenance.RECONSTRUCTED, confidence=0.7,
    )
    world.entities["e-inferred-low"] = Entity(
        id="e-inferred-low", type=EntityType.DEBRIS, name="Chair",
        provenance=Provenance.INFERRED, confidence=0.3,
        relationships=[Relationship(kind=RelationshipKind.UNKNOWN, target_id="e-observed")],
    )
    world.entities["e-generated"] = Entity(
        id="e-generated", type=EntityType.DEBRIS, name="Filler Prop",
        provenance=Provenance.GENERATED, confidence=0.4,
    )
    world.entities["e-conflict"] = Entity(
        id="e-conflict", type=EntityType.STRUCTURE, name="Wall C",
        provenance=Provenance.CONFLICT, confidence=0.5,
    )

    return world


def test_provenance_breakdown_counts():
    world = _build_world()
    breakdown = provenance_breakdown(world)
    assert breakdown["OBSERVED"] == 1
    assert breakdown["RECONSTRUCTED"] == 1
    assert breakdown["INFERRED"] == 1
    assert breakdown["GENERATED"] == 1
    assert breakdown["CONFLICT"] == 1
    assert breakdown["ESTIMATED"] == 0
    assert breakdown["UNKNOWN"] == 0


def test_mean_confidence_is_correct_arithmetic_mean():
    world = _build_world()
    values = [0.95, 0.7, 0.3, 0.4, 0.5]
    expected = sum(values) / len(values)
    assert mean_confidence(world) == expected


def test_canonical_fraction_excludes_generated_unknown_conflict():
    world = _build_world()
    # 5 entities total; canonical = OBSERVED, RECONSTRUCTED, INFERRED = 3
    assert canonical_fraction(world) == 3 / 5


def test_empty_world_returns_none_no_crash():
    world = WorldIR(id="empty", name="empty")
    assert mean_confidence(world) is None
    assert canonical_fraction(world) is None


def test_low_confidence_entity_ids():
    world = _build_world()
    ids = low_confidence_entity_ids(world, threshold=0.5)
    assert set(ids) == {"e-inferred-low", "e-generated"}


def test_unresolved_relationship_entity_ids():
    world = _build_world()
    ids = unresolved_relationship_entity_ids(world)
    assert ids == ["e-inferred-low"]


def test_summary_text_non_empty_and_has_entity_count():
    world = _build_world()
    report = compute_quality_report(world)
    text = report.summary_text()
    assert text
    assert "Entities: 5" in text


def test_summary_text_on_empty_world_does_not_crash():
    world = WorldIR(id="empty", name="empty")
    report = compute_quality_report(world)
    text = report.summary_text()
    assert text
    assert "Entities: 0" in text


def test_compute_quality_report_to_dict_shape():
    world = _build_world()
    report = compute_quality_report(world)
    assert isinstance(report, WorldQualityReport)
    d = report.to_dict()
    assert d["entity_count"] == 5
    assert set(d["low_confidence_ids"]) == {"e-inferred-low", "e-generated"}
    assert set(d["unresolved_relationship_ids"]) == {"e-inferred-low"}
