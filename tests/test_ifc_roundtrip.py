"""Zero-drift bidirectional IFC roundtrip tests (P8-03 / Phase 9).

Validates:
  1. WorldIR -> IFC -> WorldIR roundtrip with zero geometric drift (< 1 mm).
  2. Multi-storey building export with IfcBuildingStorey elevations and spatial containment.
  3. Non-convex L-shaped room profile preservation via true closed 2D polyline extrusion.
  4. Triangulated 3D mesh geometry export (IfcTriangulatedFaceSet) and roundtrip restoration.
"""

from __future__ import annotations

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from exporters.ifc_bridge import export_world_to_ifc, import_ifc_to_world  # noqa: E402
from provenance import Provenance  # noqa: E402
from world_ir.schema_v1 import (  # noqa: E402
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Relationship,
    RelationshipKind,
    Vector3,
)
from world_ir.world_v1 import WorldIR  # noqa: E402


def test_roundtrip_basic_world_zero_drift(tmp_path):
    world = WorldIR(id="w-basic-roundtrip")
    wall_geom = Geometry(
        id="geom-wall",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(5.0, 0.25, 2.70),
        provenance=Provenance.OBSERVED,
    )
    floor_geom = Geometry(
        id="geom-floor",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, -0.15),
        bounds_max=Vector3(5.0, 4.0, 0.0),
        provenance=Provenance.OBSERVED,
    )
    col_geom = Geometry(
        id="geom-col",
        type=GeometryType.BOX,
        bounds_min=Vector3(1.5, 1.5, 0.0),
        bounds_max=Vector3(1.8, 1.8, 2.70),
        provenance=Provenance.OBSERVED,
    )
    world.geometries["geom-wall"] = wall_geom
    world.geometries["geom-floor"] = floor_geom
    world.geometries["geom-col"] = col_geom

    world.entities["wall-1"] = Entity(
        id="wall-1",
        name="Front Wall",
        type=EntityType.WALL,
        geometry_ids=["geom-wall"],
        provenance=Provenance.OBSERVED,
    )
    world.entities["floor-1"] = Entity(
        id="floor-1",
        name="Floor Slab",
        type=EntityType.FLOOR,
        geometry_ids=["geom-floor"],
        provenance=Provenance.OBSERVED,
    )
    world.entities["col-1"] = Entity(
        id="col-1",
        name="Main Column",
        type=EntityType.COLUMN,
        geometry_ids=["geom-col"],
        provenance=Provenance.OBSERVED,
    )

    ifc_file = tmp_path / "basic.ifc"
    report = export_world_to_ifc(world, ifc_file)
    assert len(report["written"]) == 3

    reconstructed = import_ifc_to_world(ifc_file)
    assert "wall-1" in reconstructed.entities
    assert "floor-1" in reconstructed.entities
    assert "col-1" in reconstructed.entities

    # Zero geometric drift check: tolerances < 1 mm (0.001 m)
    for eid in ("wall-1", "floor-1", "col-1"):
        orig_ent = world.entities[eid]
        recon_ent = reconstructed.entities[eid]
        assert recon_ent.type == orig_ent.type

        orig_geom = world.geometries[orig_ent.geometry_ids[0]]
        recon_geom = reconstructed.geometries[recon_ent.geometry_ids[0]]

        assert recon_geom.bounds_min is not None and orig_geom.bounds_min is not None
        assert recon_geom.bounds_max is not None and orig_geom.bounds_max is not None

        assert abs(recon_geom.bounds_min.x - orig_geom.bounds_min.x) < 0.001
        assert abs(recon_geom.bounds_min.y - orig_geom.bounds_min.y) < 0.001
        assert abs(recon_geom.bounds_min.z - orig_geom.bounds_min.z) < 0.001

        assert abs(recon_geom.bounds_max.x - orig_geom.bounds_max.x) < 0.001
        assert abs(recon_geom.bounds_max.y - orig_geom.bounds_max.y) < 0.001
        assert abs(recon_geom.bounds_max.z - orig_geom.bounds_max.z) < 0.001


def test_roundtrip_multistorey_containment(tmp_path):
    world = WorldIR(id="w-multistorey")

    # Storey 0: Ground Floor at z=0.0
    s0 = Entity(
        id="storey-0",
        name="Ground Floor",
        type=EntityType.STOREY,
        custom_properties={"elevation_m": 0.0, "is_storey": True},
    )
    # Storey 1: First Floor at z=3.0
    s1 = Entity(
        id="storey-1",
        name="Level 1",
        type=EntityType.STOREY,
        custom_properties={"elevation_m": 3.0, "is_storey": True},
    )
    world.entities["storey-0"] = s0
    world.entities["storey-1"] = s1

    # Elements on Storey 0
    g_w0 = Geometry(
        id="gw-0",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(6.0, 0.2, 2.8),
    )
    world.geometries["gw-0"] = g_w0
    world.entities["wall-l0"] = Entity(
        id="wall-l0",
        name="Wall L0",
        type=EntityType.WALL,
        geometry_ids=["gw-0"],
        relationships=[Relationship(target_id="storey-0", kind=RelationshipKind.INSIDE)],
    )

    # Elements on Storey 1
    g_w1 = Geometry(
        id="gw-1",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 3.0),
        bounds_max=Vector3(6.0, 0.2, 5.8),
    )
    world.geometries["gw-1"] = g_w1
    world.entities["wall-l1"] = Entity(
        id="wall-l1",
        name="Wall L1",
        type=EntityType.WALL,
        geometry_ids=["gw-1"],
        relationships=[Relationship(target_id="storey-1", kind=RelationshipKind.INSIDE)],
    )

    ifc_file = tmp_path / "multistorey.ifc"
    export_world_to_ifc(world, ifc_file)

    # Open with ifcopenshell and check storeys
    f = ifcopenshell.open(str(ifc_file))
    storeys = f.by_type("IfcBuildingStorey")
    assert len(storeys) == 2
    elevs = {s.Name: float(s.Elevation) for s in storeys}
    assert any(abs(v - 0.0) < 0.001 for v in elevs.values())
    assert any(abs(v - 3.0) < 0.001 for v in elevs.values())

    # Reconstruct back to WorldIR
    reconstructed = import_ifc_to_world(ifc_file)
    recon_storeys = [e for e in reconstructed.entities.values() if e.type == EntityType.STOREY]
    assert len(recon_storeys) == 2

    # Check that walls are present and placed correctly in elevation
    assert "wall-l0" in reconstructed.entities
    assert "wall-l1" in reconstructed.entities

    geom_l0 = reconstructed.geometries[reconstructed.entities["wall-l0"].geometry_ids[0]]
    geom_l1 = reconstructed.geometries[reconstructed.entities["wall-l1"].geometry_ids[0]]

    assert abs(geom_l0.bounds_min.z - 0.0) < 0.001
    assert abs(geom_l1.bounds_min.z - 3.0) < 0.001


def test_roundtrip_non_convex_room_profile(tmp_path):
    world = WorldIR(id="w-lshaped")

    # L-shaped polygon: 6 vertices
    l_polygon = [
        (0.0, 0.0),
        (5.0, 0.0),
        (5.0, 2.5),
        (2.5, 2.5),
        (2.5, 6.0),
        (0.0, 6.0),
    ]

    r_geom = Geometry(
        id="geom-room-l",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(5.0, 6.0, 2.8),
    )
    world.geometries["geom-room-l"] = r_geom
    world.entities["room-l"] = Entity(
        id="room-l",
        name="L-Shaped Living Room",
        type=EntityType.ROOM,
        geometry_ids=["geom-room-l"],
        custom_properties={"floor_polygon_2d": l_polygon},
    )

    ifc_file = tmp_path / "lshaped_room.ifc"
    export_world_to_ifc(world, ifc_file)

    # Verify that IFC file has IfcSpace with non-rectangular polycurve
    f = ifcopenshell.open(str(ifc_file))
    spaces = f.by_type("IfcSpace")
    assert len(spaces) == 1
    solid = spaces[0].Representation.Representations[0].Items[0]
    polycurve = solid.SweptArea.OuterCurve
    coords = list(polycurve.Points.CoordList)
    assert len(coords) == 6  # exact 6 polygon vertices

    # Reconstruct back to WorldIR
    reconstructed = import_ifc_to_world(ifc_file)
    assert "room-l" in reconstructed.entities
    recon_room = reconstructed.entities["room-l"]
    assert "boundary_polygon_2d" in recon_room.custom_properties
    recon_poly = recon_room.custom_properties["boundary_polygon_2d"]
    assert len(recon_poly) == 6
    for (x_orig, y_orig), (x_recon, y_recon) in zip(l_polygon, recon_poly):
        assert abs(x_orig - x_recon) < 0.001
        assert abs(y_orig - y_recon) < 0.001


def test_roundtrip_triangulated_mesh(tmp_path):
    world = WorldIR(id="w-mesh")

    mesh_geom = Geometry(
        id="geom-sculpture",
        type=GeometryType.MESH,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(1.0, 1.0, 1.2),
        vertex_count=4,
        triangle_count=4,
    )
    world.geometries["geom-sculpture"] = mesh_geom
    world.entities["sculpture-1"] = Entity(
        id="sculpture-1",
        name="Geometric Sculpture",
        type=EntityType.STRUCTURE,
        geometry_ids=["geom-sculpture"],
    )

    ifc_file = tmp_path / "mesh_sculpture.ifc"
    export_world_to_ifc(world, ifc_file)

    reconstructed = import_ifc_to_world(ifc_file)
    assert "sculpture-1" in reconstructed.entities
    recon_ent = reconstructed.entities["sculpture-1"]
    recon_geom = reconstructed.geometries[recon_ent.geometry_ids[0]]
    assert abs(recon_geom.bounds_max.z - 1.2) < 0.001
