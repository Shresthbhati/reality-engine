"""Tests for Multi-Storey Building Vertical Alignment Validator."""

from __future__ import annotations

from perception.architecture.multi_storey_validator import MultiStoreyValidator
from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Vector3,
)


def test_single_storey_validates_as_pass():
    world = WorldIR(id="w-single", name="Single Storey House")
    r_geom = Geometry(
        id="g-room-0",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(5.0, 5.0, 2.7),
        provenance=Provenance.RECONSTRUCTED,
    )
    world.geometries[r_geom.id] = r_geom
    r_ent = Entity(
        id="room-0",
        name="Living Room",
        type=EntityType.ROOM,
        geometry_ids=[r_geom.id],
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"floor_height_m": 0.0},
    )
    world.entities[r_ent.id] = r_ent

    report = MultiStoreyValidator.validate(world)
    assert report.verdict == "PASS"
    assert report.total_storeys == 1
    assert report.is_vertically_consistent is True


def test_two_storey_aligned_building_passes():
    world = WorldIR(id="w-multi", name="Two Storey Office")

    # Storey 1 (Ground: z = 0.0m)
    s1 = Entity(
        id="storey-1",
        name="Ground Floor",
        type=EntityType.STOREY,
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"floor_height_m": 0.0, "ceiling_height_m": 2.8},
    )
    # Storey 2 (Level 1: z = 3.0m)
    s2 = Entity(
        id="storey-2",
        name="First Floor",
        type=EntityType.STOREY,
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"floor_height_m": 3.0, "ceiling_height_m": 2.8},
    )
    world.entities[s1.id] = s1
    world.entities[s2.id] = s2

    # Stairs spanning ground to level 1
    st_geom = Geometry(
        id="g-stair",
        type=GeometryType.BOX,
        bounds_min=Vector3(1.0, 1.0, 0.0),
        bounds_max=Vector3(3.0, 2.0, 3.0),
        provenance=Provenance.RECONSTRUCTED,
    )
    world.geometries[st_geom.id] = st_geom
    stairs = Entity(
        id="stairs-main",
        name="Main Staircase",
        type=EntityType.STAIRS,
        geometry_ids=[st_geom.id],
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"spanned_storeys": ["storey-1", "storey-2"]},
    )
    world.entities[stairs.id] = stairs

    # Continuous column at (2.0, 2.0) across storeys
    col1_geom = Geometry(
        id="g-col1",
        type=GeometryType.CYLINDER,
        bounds_min=Vector3(1.85, 1.85, 0.0),
        bounds_max=Vector3(2.15, 2.15, 3.0),
        provenance=Provenance.RECONSTRUCTED,
    )
    world.geometries[col1_geom.id] = col1_geom
    col1 = Entity(
        id="col-lower",
        name="Column Lower",
        type=EntityType.COLUMN,
        geometry_ids=[col1_geom.id],
        provenance=Provenance.RECONSTRUCTED,
    )
    world.entities[col1.id] = col1

    col2_geom = Geometry(
        id="g-col2",
        type=GeometryType.CYLINDER,
        bounds_min=Vector3(1.85, 1.85, 3.0),
        bounds_max=Vector3(2.15, 2.15, 6.0),
        provenance=Provenance.RECONSTRUCTED,
    )
    world.geometries[col2_geom.id] = col2_geom
    col2 = Entity(
        id="col-upper",
        name="Column Upper",
        type=EntityType.COLUMN,
        geometry_ids=[col2_geom.id],
        provenance=Provenance.RECONSTRUCTED,
    )
    world.entities[col2.id] = col2

    report = MultiStoreyValidator.validate(world)
    assert report.verdict == "PASS"
    assert report.total_storeys == 2
    assert report.elevation_monotonic is True
    assert report.clearance_checks_passed is True
    assert report.stair_circulation_connected is True
    assert report.load_bearing_overlap_pct == 100.0


def test_insufficient_clearance_fails():
    world = WorldIR(id="w-bad-clearance", name="Bad Height")
    s1 = Entity(
        id="storey-1",
        name="Level 1",
        type=EntityType.STOREY,
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"floor_height_m": 0.0},
    )
    s2 = Entity(
        id="storey-2",
        name="Level 2",
        type=EntityType.STOREY,
        provenance=Provenance.RECONSTRUCTED,
        custom_properties={"floor_height_m": 1.50},  # Only 1.5m clearance!
    )
    world.entities[s1.id] = s1
    world.entities[s2.id] = s2

    report = MultiStoreyValidator.validate(world)
    assert report.verdict == "FAIL"
    assert report.clearance_checks_passed is False
    assert any("insufficient storey clearance" in v.lower() for v in report.violations)
