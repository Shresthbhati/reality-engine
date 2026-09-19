"""Tests for evidence/promote_objects.py: MergedObjectCandidate -> WorldIR
Entity promotion. Closes the gap between the object-perception pipeline
(lifting -> object_resolution -> measurement) and WorldIR, mirroring
evidence/promote_planes.py's pattern for structural elements."""

from __future__ import annotations

import math

from engine.math import Vec3
from evidence.promote_objects import ObjectPromotionError, promote_object_to_entity
from perception.instances.object_resolution import MergedObjectCandidate
from provenance import Provenance
from world_ir import EntityType, WorldIR
from world_ir.validation import validate_world_ir


def _candidate(candidate_id="obj-chair-0001", label="chair", x=1.0, y=0.5, z=2.0,
                half_extents=(0.3, 0.5, 0.3), confidence=0.75, source_hypotheses=()) -> MergedObjectCandidate:
    hx, hy, hz = half_extents
    return MergedObjectCandidate(
        candidate_id=candidate_id, label=label,
        position=Vec3(x, y, z),
        bounds_min=Vec3(x - hx, y - hy, z - hz),
        bounds_max=Vec3(x + hx, y + hy, z + hz),
        source_hypotheses=source_hypotheses or _fake_hypotheses(),
        confidence=confidence,
    )


class _FakeHypothesis:
    def __init__(self, region_id, evidence_id):
        self.region_id = region_id
        self.evidence_id = evidence_id


def _fake_hypotheses():
    return (_FakeHypothesis("reg-1", "ev-1"), _FakeHypothesis("reg-2", "ev-2"))


def test_promoted_entity_has_correct_transform_and_type():
    world = WorldIR()
    candidate = _candidate()
    result = promote_object_to_entity(candidate, world, "ent-chair-1")

    assert result.entity.type is EntityType.UNKNOWN
    assert result.entity.name == "chair"
    assert result.entity.semantic_labels == ["chair"]
    assert result.entity.transform["position"] == {"x": 1.0, "y": 0.5, "z": 2.0}
    assert "ent-chair-1" in world.entities


def test_promoted_entity_provenance_is_inferred():
    world = WorldIR()
    result = promote_object_to_entity(_candidate(), world, "ent-1")
    assert result.entity.provenance is Provenance.INFERRED
    assert result.geometry.provenance is Provenance.INFERRED


def test_promoted_geometry_bounds_match_candidate_aabb():
    world = WorldIR()
    candidate = _candidate(half_extents=(0.3, 0.5, 0.3))
    result = promote_object_to_entity(candidate, world, "ent-1")

    assert result.geometry.bounds_min.to_dict() == {"x": 0.7, "y": 0.0, "z": 1.7}
    assert result.geometry.bounds_max.to_dict() == {"x": 1.3, "y": 1.0, "z": 2.3}


def test_measurements_are_written_to_custom_properties():
    world = WorldIR()
    candidate = _candidate(half_extents=(0.3, 0.5, 0.3))
    result = promote_object_to_entity(candidate, world, "ent-1")

    assert "width_m" in result.entity.custom_properties
    assert "height_m" in result.entity.custom_properties
    assert "depth_m" in result.entity.custom_properties
    assert "volume_m3" in result.entity.custom_properties
    assert math.isclose(result.entity.custom_properties["width_m"], 0.6, abs_tol=1e-9)


def test_observation_records_supporting_evidence():
    world = WorldIR()
    candidate = _candidate()
    result = promote_object_to_entity(candidate, world, "ent-1")

    promotion_obs = next(o for o in result.entity.observations if o.sensor_type == "object_promotion")
    assert promotion_obs.metadata["evidence_ids"] == ["ev-1", "ev-2"]
    assert promotion_obs.metadata["observation_count"] == 2


def test_zero_hypotheses_candidate_is_refused():
    world = WorldIR()
    candidate = _candidate(source_hypotheses=())
    # Force an empty tuple explicitly (the default fixture fills it in otherwise).
    candidate = MergedObjectCandidate(
        candidate_id="obj-empty", label="chair", position=Vec3(0, 0, 0),
        bounds_min=Vec3(-1, -1, -1), bounds_max=Vec3(1, 1, 1),
        source_hypotheses=(), confidence=0.5,
    )
    try:
        promote_object_to_entity(candidate, world, "ent-1")
        assert False, "expected ObjectPromotionError"
    except ObjectPromotionError:
        pass
    assert "ent-1" not in world.entities


def test_promoted_world_passes_structural_validation():
    world = WorldIR()
    promote_object_to_entity(_candidate(), world, "ent-1")
    report = validate_world_ir(world)
    assert report.is_valid(), [str(i) for i in report.issues]


def test_multiple_objects_promote_independently():
    world = WorldIR()
    promote_object_to_entity(_candidate(candidate_id="obj-chair", label="chair", x=0.0), world, "ent-chair")
    promote_object_to_entity(_candidate(candidate_id="obj-table", label="table", x=5.0), world, "ent-table")

    assert len(world.entities) == 2
    assert world.entities["ent-chair"].name == "chair"
    assert world.entities["ent-table"].name == "table"
