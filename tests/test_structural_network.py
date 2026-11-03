"""Tests for Structural Network and Load-Bearing Understanding Engine (Wave 2, Block 12)."""

from __future__ import annotations

import pytest

from perception.architecture.structural_network import (
    StructuralAnalysisReport,
    StructuralRole,
    analyze_structural_system,
)
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, RelationshipKind, Vector3
from world_ir.world_v1 import WorldIR


def test_structural_column_beam_load_path():
    world = WorldIR()

    # Column: x in [-0.2, 0.2], y in [0.0, 3.0], z in [-0.2, 0.2]
    col_geom = Geometry(
        id="geom-col-1",
        type=GeometryType.BOX,
        bounds_min=Vector3(x=-0.2, y=0.0, z=-0.2),
        bounds_max=Vector3(x=0.2, y=3.0, z=0.2),
    )
    col_ent = Entity(
        id="ent-col-1",
        name="Support Column C1",
        type=EntityType.COLUMN,
        geometry_ids=[col_geom.id],
    )
    world.geometries[col_geom.id] = col_geom
    world.entities[col_ent.id] = col_ent

    # Beam: x in [-0.3, 4.0], y in [3.0, 3.5], z in [-0.3, 0.3] (rests on top of column at y=3.0)
    beam_geom = Geometry(
        id="geom-beam-1",
        type=GeometryType.BOX,
        bounds_min=Vector3(x=-0.3, y=3.0, z=-0.3),
        bounds_max=Vector3(x=4.0, y=3.5, z=0.3),
    )
    beam_ent = Entity(
        id="ent-beam-1",
        name="Transfer Beam B1",
        type=EntityType.BEAM,
        geometry_ids=[beam_geom.id],
    )
    world.geometries[beam_geom.id] = beam_geom
    world.entities[beam_ent.id] = beam_ent

    report = analyze_structural_system(world)

    assert isinstance(report, StructuralAnalysisReport)
    assert report.column_count == 1
    assert report.beam_count == 1
    assert report.relationships_added >= 2

    # Verify column SUPPORTS beam
    col_supports = [r for r in col_ent.relationships if r.kind == RelationshipKind.SUPPORTS and r.target_id == beam_ent.id]
    assert len(col_supports) == 1
    assert col_supports[0].metadata["load_path"] == "vertical_axial_support"

    # Verify beam RESTS_ON column
    beam_rests = [r for r in beam_ent.relationships if r.kind == RelationshipKind.RESTS_ON and r.target_id == col_ent.id]
    assert len(beam_rests) == 1

    # Verify strict safety disclaimer
    assert col_ent.custom_properties["is_visually_inferred_structural_role"] is True
    assert col_ent.custom_properties["engineering_safety_certified"] is False


def test_structural_bay_span_calculation():
    world = WorldIR()

    # Column 1 at (0, 0, 0)
    g1 = Geometry(id="g1", type=GeometryType.BOX, bounds_min=Vector3(x=-0.2, y=0.0, z=-0.2), bounds_max=Vector3(x=0.2, y=3.0, z=0.2))
    e1 = Entity(id="col-1", name="C1", type=EntityType.COLUMN, geometry_ids=["g1"])
    world.geometries["g1"] = g1
    world.entities["col-1"] = e1

    # Column 2 at (5.0, 0, 0) (5m span along X)
    g2 = Geometry(id="g2", type=GeometryType.BOX, bounds_min=Vector3(x=4.8, y=0.0, z=-0.2), bounds_max=Vector3(x=5.2, y=3.0, z=0.2))
    e2 = Entity(id="col-2", name="C2", type=EntityType.COLUMN, geometry_ids=["g2"])
    world.geometries["g2"] = g2
    world.entities["col-2"] = e2

    report = analyze_structural_system(world)
    assert len(report.bays) >= 1
    assert report.max_unsupported_span_m == pytest.approx(5.0, abs=0.1)


def test_bearing_vs_partition_wall_discrimination():
    world = WorldIR()

    # Thick external wall (0.25m thick)
    g_thick = Geometry(id="gt", type=GeometryType.BOX, bounds_min=Vector3(x=0, y=0, z=0), bounds_max=Vector3(x=5, y=3, z=0.25))
    e_thick = Entity(id="w-thick", name="External Wall", type=EntityType.WALL, geometry_ids=["gt"], custom_properties={"thickness_m": 0.25})
    world.geometries["gt"] = g_thick
    world.entities["w-thick"] = e_thick

    # Thin interior partition wall (0.08m thick)
    g_thin = Geometry(id="gn", type=GeometryType.BOX, bounds_min=Vector3(x=0, y=0, z=0), bounds_max=Vector3(x=3, y=3, z=0.08))
    e_thin = Entity(id="w-thin", name="Partition Wall", type=EntityType.WALL, geometry_ids=["gn"], custom_properties={"thickness_m": 0.08})
    world.geometries["gn"] = g_thin
    world.entities["w-thin"] = e_thin

    report = analyze_structural_system(world)
    assert report.bearing_wall_count == 1
    assert e_thick.custom_properties["structural_role"] == StructuralRole.BEARING_WALL.value
    assert e_thin.custom_properties["structural_role"] == StructuralRole.PARTITION_NON_BEARING.value
