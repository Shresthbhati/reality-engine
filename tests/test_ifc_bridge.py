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
