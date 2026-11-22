"""Measured coverage of a persisted world (evidence_world_stage._coverage): counts what exists, never defaults."""
from __future__ import annotations

from apps.api.evidence_world_stage import _coverage
from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import Entity, EntityType, Observation
from world_ir.v2_extensions import ExternalReference


def _cam(frame: str) -> Observation:
    return Observation(sensor_type="camera", frame_id=frame)


def _world(*ents: Entity) -> WorldIR:
    w = WorldIR()
    for e in ents:
        w.entities[e.id] = e
    return w


def test_multiview_counts_distinct_photographs_and_surfaces_are_traced_through_observations():
    two = Entity(id="a", type=EntityType.FURNITURE, provenance=Provenance.ESTIMATED, observations=[_cam("cam:1"), _cam("cam:2")])
    one = Entity(id="b", type=EntityType.FURNITURE, provenance=Provenance.ESTIMATED, observations=[_cam("cam:1")])
    surf = Entity(id="surf-floor-1", type=EntityType.FLOOR, provenance=Provenance.ESTIMATED, observations=[_cam("cam:1")])
    w = _world(two, one, surf)
    c = _coverage(w, placed=[two, one, surf], sized=[two])
    assert (c["multiview_association"]["covered"], c["multiview_association"]["of"]) == (1, 2)  # surfaces are not "objects"
    assert (c["source_traceability"]["covered"], c["source_traceability"]["of"]) == (3, 3)
    assert (c["geometry_dimensions"]["covered"], c["geometry_dimensions"]["of"]) == (1, 3)


def test_one_photograph_is_not_counted_twice_across_identifier_families():
    e = Entity(id="a", type=EntityType.FURNITURE, provenance=Provenance.ESTIMATED, observations=[_cam("cam:1")],
               external_references=(ExternalReference("image", "C:/photos/1.jpg", "supporting_evidence"),))
    c = _coverage(_world(e), placed=[e], sized=[])
    assert c["multiview_association"]["covered"] == 0  # same photo as a path and as a frame id


def test_untraceable_and_undeclared_entities_are_not_covered_and_empty_world_has_no_fraction():
    e = Entity(id="x", type=EntityType.FURNITURE, provenance=Provenance.UNKNOWN)
    c = _coverage(_world(e), placed=[], sized=[])
    assert c["source_traceability"]["covered"] == 0 and c["provenance_declared"]["covered"] == 0
    empty = _coverage(_world(), placed=[], sized=[])
    assert empty["spatial_positioning"]["fraction"] is None  # nothing to cover is not "0% covered"
