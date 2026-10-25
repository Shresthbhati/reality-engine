"""Regression tests: the WorldIR measurement gate must not reject
honest coordinate-like measurements.

A reconstruction's world frame is arbitrary (COLMAP places the origin
wherever the SfM solution lands): a real capture legitimately measured
its ground storey at floor_height_m = -1.42. Elevations are positions,
not magnitudes -- the negative sign carries the frame, not corruption.
Dimension-like keys keep their strict negative rejection, and
non-finite values are rejected everywhere.
"""
from __future__ import annotations

from provenance import Provenance
from world_ir import Entity, EntityType, WorldIR
from world_ir.validation import validate_world_ir


def _entity(entity_id, custom_properties):
    return Entity(
        id=entity_id,
        name=entity_id,
        type=EntityType.STOREY,
        provenance=Provenance.RECONSTRUCTED,
        confidence=0.9,
        custom_properties=custom_properties,
    )


def test_negative_floor_height_is_honest_measured_data():
    world = WorldIR(id="w-coord")
    world.entities["storey-01"] = _entity(
        "storey-01", {"floor_height_m": -1.41568})
    report = validate_world_ir(world)
    assert report.errors == [], [(i.code, i.message) for i in report.errors]


def test_negative_sill_and_head_heights_are_honest_measured_data():
    world = WorldIR(id="w-coord")
    world.entities["storey-01"] = _entity(
        "storey-01", {"sill_height_m": -0.53, "head_height_m": 1.47})
    report = validate_world_ir(world)
    assert report.errors == [], [(i.code, i.message) for i in report.errors]


def test_negative_area_dimension_still_rejected():
    world = WorldIR(id="w-coord")
    world.entities["room-01"] = _entity(
        "room-01", {"floor_area_m2": -9.0})
    report = validate_world_ir(world)
    assert any(
        i.code == "measurement_negative_dimension" and "floor_area_m2" in i.message
        for i in report.issues
    ), [i.code for i in report.issues]


def test_negative_height_dimension_still_rejected():
    world = WorldIR(id="w-coord")
    world.entities["wall-01"] = _entity(
        "wall-01", {"height_m": -2.4})
    report = validate_world_ir(world)
    assert any(
        i.code == "measurement_negative_dimension" and "height_m" in i.message
        for i in report.issues
    ), [i.code for i in report.issues]


def test_non_finite_always_rejected_even_for_coordinates():
    world = WorldIR(id="w-coord")
    world.entities["storey-01"] = _entity(
        "storey-01", {"floor_height_m": float("nan")})
    report = validate_world_ir(world)
    assert any(i.code == "measurement_non_finite" for i in report.issues), \
        [i.code for i in report.issues]
