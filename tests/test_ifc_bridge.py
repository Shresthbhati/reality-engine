"""Tests for exporters/ifc_bridge.py (P8-03: WorldIR -> IFC via
IfcOpenShell -- the ledger's "do not rebuild BIM from scratch" row).

Contract:

  - Uses the real ifcopenshell API to write a valid IFC4 file; the
    output OPENS again with ifcopenshell (the roundtrip proof).
  - WorldIR walls/floors/ceilings/columns map to real IFC classes
    (IfcWall, IfcSlab, IfcColumn); the entity's real AABB bounds
    become an IfcExtrudedAreaSolid -- geometry from evidence, not
    placeholder placement.
  - Entity identity travels: WorldIR entity id + name are recorded in
    the IFC element's GlobalId-adjacent Name/Description so the
    reverse path (later) can trace elements back.
  - Entities without real bounds are skipped with a recorded reason
    (the same honesty as every other exporter), not invented.
  - ifcopenshell unavailability is an ENVIRONMENT failure reported by
    an explicit error -- never silently degraded to fake output.
"""

from __future__ import annotations

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR
from exporters.ifc_bridge import export_world_to_ifc, IFCBridgeError


def _world() -> WorldIR:
    world = WorldIR()
    wall_geom = Geometry(
        id="geom-wall",
        type=GeometryType.PLANE,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(4.0, 0.2, 2.5),
        provenance=Provenance.OBSERVED,
    )
    world.geometries["geom-wall"] = wall_geom
    world.entities["wall-1"] = Entity(
        id="wall-1", name="North Wall", type=EntityType.WALL,
        geometry_ids=["geom-wall"], provenance=Provenance.OBSERVED)
    floor_geom = Geometry(
        id="geom-floor",
        type=GeometryType.PLANE,
        bounds_min=Vector3(0.0, 0.0, -0.1),
        bounds_max=Vector3(4.0, 4.0, 0.0),
    )
    world.geometries["geom-floor"] = floor_geom
    world.entities["floor-1"] = Entity(
        id="floor-1", name="Ground Floor", type=EntityType.FLOOR,
        geometry_ids=["geom-floor"])
    col_geom = Geometry(
        id="geom-col",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.5, 0.5, 0.0),
        bounds_max=Vector3(0.7, 0.7, 2.5),
    )
    world.geometries["geom-col"] = col_geom
    world.entities["col-1"] = Entity(
        id="col-1", name="Column A", type=EntityType.COLUMN,
        geometry_ids=["geom-col"])
    # No-bounds geometry: skipped with a reason.
    world.geometries["geom-empty"] = Geometry(
        id="geom-empty", type=GeometryType.BOX)
    world.entities["mystery"] = Entity(
        id="mystery", name="Mystery Box", type=EntityType.DEBRIS,
        geometry_ids=["geom-empty"])
    return world


class TestIfcExport:
    def test_writes_openable_ifc(self, tmp_path):
        out = tmp_path / "world.ifc"
        export_world_to_ifc(_world(), out)
        assert out.exists()
        f = ifcopenshell.open(str(out))
        assert f.schema == "IFC4"
        products = f.by_type("IfcProduct")
        assert products  # real elements written

    def test_maps_semantic_classes(self, tmp_path):
        out = tmp_path / "world.ifc"
        export_world_to_ifc(_world(), out)
        f = ifcopenshell.open(str(out))
        assert f.by_type("IfcWall")
        assert f.by_type("IfcSlab")
        assert f.by_type("IfcColumn")

    def test_geometry_is_extruded_solid(self, tmp_path):
        out = tmp_path / "world.ifc"
        export_world_to_ifc(_world(), out)
        f = ifcopenshell.open(str(out))
        solids = f.by_type("IfcExtrudedAreaSolid")
        assert len(solids) == 3  # wall + floor + column

    def test_identity_recorded(self, tmp_path):
        out = tmp_path / "world.ifc"
        export_world_to_ifc(_world(), out)
        f = ifcopenshell.open(str(out))
        wall = f.by_type("IfcWall")[0]
        # The WorldIR id survives in the element name (traceability).
        assert "wall-1" in (wall.Name or "")
        assert "North Wall" in (wall.Name or "")

    def test_skip_reasons_returned(self, tmp_path):
        out = tmp_path / "world.ifc"
        report = export_world_to_ifc(_world(), out)
        assert "mystery" in report["skipped"]
        assert "bounds" in report["skipped"]["mystery"].lower()

    def test_rejects_no_geometry_world(self, tmp_path):
        world = WorldIR()
        world.entities["empty"] = Entity(id="empty", type=EntityType.ROOM)
        with pytest.raises(IFCBridgeError):
            export_world_to_ifc(world, tmp_path / "empty.ifc")

    def test_deterministic_bytes(self, tmp_path):
        """Same world -> byte-identical IFC (the determinism contract
        the other exporters honor). This guards against regressions to
        ifcopenshell's defaults: random uuid4 GlobalIds, a wall-clock
        header time_stamp, and hash-set unit ordering."""
        world = _world()
        out1 = tmp_path / "a" / "world.ifc"
        out2 = tmp_path / "b" / "world.ifc"
        export_world_to_ifc(world, out1)
        export_world_to_ifc(world, out2)
        assert out1.read_bytes() == out2.read_bytes()

    def test_global_ids_are_stable_and_valid(self, tmp_path):
        """GlobalIds are derived from the world id + entity id: stable
        across runs, distinct per element, and valid 22-char IFC GUIDs
        that expand back to real uuids."""
        import ifcopenshell.guid

        world = _world()
        out1 = tmp_path / "a" / "world.ifc"
        out2 = tmp_path / "b" / "world.ifc"
        export_world_to_ifc(world, out1)
        export_world_to_ifc(world, out2)
        ids1 = {e.Name: e.GlobalId for e in ifcopenshell.open(str(out1)).by_type("IfcElement")}
        ids2 = {e.Name: e.GlobalId for e in ifcopenshell.open(str(out2)).by_type("IfcElement")}
        assert ids1 == ids2
        assert len(set(ids1.values())) == len(ids1)  # no collisions
        for gid in ids1.values():
            assert len(gid) == 22
            ifcopenshell.guid.expand(gid)  # raises if malformed

    def test_exports_room_as_ifc_space(self, tmp_path):
        world = WorldIR()
        room_geom = Geometry(
            id="geom-room",
            type=GeometryType.BOX,
            bounds_min=Vector3(0.0, 0.0, 0.0),
            bounds_max=Vector3(5.0, 4.0, 2.8),
        )
        world.geometries["geom-room"] = room_geom
        world.entities["room-1"] = Entity(
            id="room-1", name="Master Bedroom", type=EntityType.ROOM,
            geometry_ids=["geom-room"],
        )
        out = tmp_path / "room.ifc"
        report = export_world_to_ifc(world, out)
        assert "room-1" in report["written"]
        f = ifcopenshell.open(str(out))
        spaces = f.by_type("IfcSpace")
        assert len(spaces) == 1
        assert spaces[0].Name == "room-1 Master Bedroom"
        assert len(f.by_type("IfcExtrudedAreaSolid")) == 1

    def test_exports_mesh_geometry(self, tmp_path):
        world = WorldIR()
        mesh_geom = Geometry(
            id="geom-table",
            type=GeometryType.MESH,
            bounds_min=Vector3(1.0, 1.0, 0.0),
            bounds_max=Vector3(2.5, 1.8, 0.75),
        )
        world.geometries["geom-table"] = mesh_geom
        world.entities["table-1"] = Entity(
            id="table-1", name="Dining Table", type=EntityType.STRUCTURE,
            geometry_ids=["geom-table"],
        )
        out = tmp_path / "mesh.ifc"
        report = export_world_to_ifc(world, out)
        assert "table-1" in report["written"]
        f = ifcopenshell.open(str(out))
        proxies = f.by_type("IfcBuildingElementProxy")
        assert len(proxies) == 1
        assert "Dining Table" in (proxies[0].Name or "")

    def test_pset_wall_common_enrichment(self, tmp_path):
        world = WorldIR()
        geom = Geometry(
            id="g-w",
            type=GeometryType.BOX,
            bounds_min=Vector3(0.0, 0.0, 0.0),
            bounds_max=Vector3(5.0, 0.2, 2.8),
            provenance=Provenance.OBSERVED,
        )
        world.geometries["g-w"] = geom
        world.entities["w-1"] = Entity(
            id="w-1",
            name="Exterior Wall",
            type=EntityType.WALL,
            geometry_ids=["g-w"],
            confidence=0.95,
            provenance=Provenance.OBSERVED,
            custom_properties={"is_external": True, "load_bearing": True, "fire_rating": "FR-60"},
        )
        out = tmp_path / "wall_pset.ifc"
        export_world_to_ifc(world, out)
        f = ifcopenshell.open(str(out))
        wall = f.by_type("IfcWall")[0]

        psets = {rel.RelatingPropertyDefinition.Name: rel.RelatingPropertyDefinition
                 for rel in getattr(wall, "IsDefinedBy", []) if rel.is_a("IfcRelDefinesByProperties")}
        assert "Pset_WallCommon" in psets
        p_wall = psets["Pset_WallCommon"]
        props = {p.Name: p.NominalValue.wrappedValue for p in p_wall.HasProperties}
        assert props["Reference"] == "Exterior Wall"
        assert props["IsExternal"] is True
        assert props["LoadBearing"] is True
        assert props["FireRating"] == "FR-60"
        assert props["NominalThickness"] == 0.2
        assert props["GrossSideArea"] == 14.0  # 5.0 * 2.8

    def test_pset_space_common_enrichment(self, tmp_path):
        world = WorldIR()
        geom = Geometry(
            id="g-s",
            type=GeometryType.BOX,
            bounds_min=Vector3(0.0, 0.0, 0.0),
            bounds_max=Vector3(4.0, 5.0, 2.7),
            provenance=Provenance.RECONSTRUCTED,
        )
        world.geometries["g-s"] = geom
        world.entities["r-1"] = Entity(
            id="r-1",
            name="Living Room",
            type=EntityType.ROOM,
            geometry_ids=["g-s"],
            confidence=0.91,
            provenance=Provenance.RECONSTRUCTED,
        )
        out = tmp_path / "space_pset.ifc"
        export_world_to_ifc(world, out)
        f = ifcopenshell.open(str(out))
        space = f.by_type("IfcSpace")[0]

        psets = {rel.RelatingPropertyDefinition.Name: rel.RelatingPropertyDefinition
                 for rel in getattr(space, "IsDefinedBy", []) if rel.is_a("IfcRelDefinesByProperties")}
        assert "Pset_SpaceCommon" in psets
        p_space = psets["Pset_SpaceCommon"]
        props = {p.Name: p.NominalValue.wrappedValue for p in p_space.HasProperties}
        assert props["Reference"] == "Living Room"
        assert props["GrossFloorArea"] == 20.0  # 4 * 5
        assert props["GrossVolume"] == 54.0     # 4 * 5 * 2.7
        assert props["Height"] == 2.7

    def test_mep_and_fixture_property_sets(self, tmp_path):
        world = WorldIR()
        # 1. Outlet
        g1 = Geometry(id="g-sock", type=GeometryType.BOX, bounds_min=Vector3(1, 0, 1), bounds_max=Vector3(1.1, 0.1, 1.1))
        world.geometries["g-sock"] = g1
        world.entities["sock-1"] = Entity(
            id="sock-1", name="Duplex Socket", type=EntityType.STRUCTURE,
            geometry_ids=["g-sock"], confidence=0.89,
            custom_properties={"entity_class": "electrical.socket", "gangs": 2, "voltage": 230.0},
        )
        # 2. Sink
        g2 = Geometry(id="g-sink", type=GeometryType.BOX, bounds_min=Vector3(2, 0, 1), bounds_max=Vector3(2.8, 0.6, 1.8))
        world.geometries["g-sink"] = g2
        world.entities["sink-1"] = Entity(
            id="sink-1", name="Kitchen Sink", type=EntityType.STRUCTURE,
            geometry_ids=["g-sink"], confidence=0.94,
            custom_properties={"entity_class": "plumbing.sink"},
        )
        out = tmp_path / "mep_pset.ifc"
        export_world_to_ifc(world, out)
        f = ifcopenshell.open(str(out))

        # Check IfcOutlet
        outlets = f.by_type("IfcOutlet")
        assert len(outlets) == 1
        psets_sock = {rel.RelatingPropertyDefinition.Name: rel.RelatingPropertyDefinition
                      for rel in getattr(outlets[0], "IsDefinedBy", []) if rel.is_a("IfcRelDefinesByProperties")}
        assert "Pset_OutletCommon" in psets_sock
        props_sock = {p.Name: p.NominalValue.wrappedValue for p in psets_sock["Pset_OutletCommon"].HasProperties}
        assert props_sock["NumberOfGangs"] == 2
        assert props_sock["NominalVoltage"] == 230.0

        # Check IfcSanitaryTerminal
        sinks = f.by_type("IfcSanitaryTerminal")
        assert len(sinks) == 1
        psets_sink = {rel.RelatingPropertyDefinition.Name: rel.RelatingPropertyDefinition
                      for rel in getattr(sinks[0], "IsDefinedBy", []) if rel.is_a("IfcRelDefinesByProperties")}
        assert "Pset_SanitaryTerminalTypeCommon" in psets_sink
        props_sink = {p.Name: p.NominalValue.wrappedValue for p in psets_sink["Pset_SanitaryTerminalTypeCommon"].HasProperties}
        assert props_sink["SanitaryTerminalType"] == "SINK"

    def test_property_sets_byte_determinism(self, tmp_path):
        world = _world()
        p1 = tmp_path / "run1.ifc"
        p2 = tmp_path / "run2.ifc"
        export_world_to_ifc(world, p1)
        export_world_to_ifc(world, p2)
        assert p1.read_bytes() == p2.read_bytes()

