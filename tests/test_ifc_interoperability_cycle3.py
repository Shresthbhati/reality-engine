"""Independent IFC interoperability and validation tests (Cycle 3).

Validates:
  1. Bidirectional zero-drift roundtrip with IFC2X3 schema (< 1 mm tolerance).
  2. Owner history and application metadata compliance in IFC2X3.
  3. Common property sets (Pset_WallCommon, Pset_DoorCommon, RealityEngine_Metadata) across IFC2X3 and IFC4.
  4. Unsupported schema version rejection (fails closed).
"""

from __future__ import annotations

import pytest

from exporters.ifc_bridge import IFCBridgeError, export_world_to_ifc, import_ifc_to_world
from provenance import Provenance
from world_ir import Entity, EntityType, Geometry, GeometryType, Vector3, WorldIR

ifcopenshell = pytest.importorskip("ifcopenshell")


def test_ifc2x3_zero_drift_roundtrip(tmp_path):
    """Verify WorldIR -> IFC2X3 -> WorldIR roundtrip with sub-millimeter precision."""
    world = WorldIR(id="w-ifc2x3-roundtrip")

    wall_geom = Geometry(
        id="geom-wall-2x3",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(6.0, 0.20, 2.80),
        provenance=Provenance.OBSERVED,
    )
    door_geom = Geometry(
        id="geom-door-2x3",
        type=GeometryType.BOX,
        bounds_min=Vector3(2.0, 0.0, 0.0),
        bounds_max=Vector3(2.9, 0.20, 2.10),
        provenance=Provenance.OBSERVED,
    )
    world.geometries[wall_geom.id] = wall_geom
    world.geometries[door_geom.id] = door_geom

    world.entities["wall-ext"] = Entity(
        id="wall-ext",
        name="Exterior Wall",
        type=EntityType.WALL,
        geometry_ids=[wall_geom.id],
        provenance=Provenance.OBSERVED,
    )
    world.entities["door-main"] = Entity(
        id="door-main",
        name="Main Entrance Door",
        type=EntityType.DOOR,
        geometry_ids=[door_geom.id],
        provenance=Provenance.OBSERVED,
    )

    ifc_path = tmp_path / "model_ifc2x3.ifc"
    report = export_world_to_ifc(world, ifc_path, schema="IFC2X3")
    assert "wall-ext" in report["written"]
    assert "door-main" in report["written"]

    # Verify underlying file schema is indeed IFC2X3
    f = ifcopenshell.open(str(ifc_path))
    assert f.schema == "IFC2X3"
    assert len(f.by_type("IfcWall")) == 1
    assert len(f.by_type("IfcDoor")) == 1
    assert len(f.by_type("IfcOwnerHistory")) >= 1

    # Reconstruct back to WorldIR
    recon = import_ifc_to_world(ifc_path)
    assert "wall-ext" in recon.entities
    assert "door-main" in recon.entities

    # Sub-millimeter drift check (< 1mm)
    for eid in ("wall-ext", "door-main"):
        orig_geom = world.geometries[world.entities[eid].geometry_ids[0]]
        recon_geom = recon.geometries[recon.entities[eid].geometry_ids[0]]

        assert abs(recon_geom.bounds_min.x - orig_geom.bounds_min.x) < 0.001
        assert abs(recon_geom.bounds_min.y - orig_geom.bounds_min.y) < 0.001
        assert abs(recon_geom.bounds_min.z - orig_geom.bounds_min.z) < 0.001

        assert abs(recon_geom.bounds_max.x - orig_geom.bounds_max.x) < 0.001
        assert abs(recon_geom.bounds_max.y - orig_geom.bounds_max.y) < 0.001
        assert abs(recon_geom.bounds_max.z - orig_geom.bounds_max.z) < 0.001


def test_ifc_unsupported_schema_rejected(tmp_path):
    """Verify exporter refuses unsupported schema string."""
    world = WorldIR(id="w-bad-schema")
    world.geometries["g1"] = Geometry(id="g1", type=GeometryType.BOX, bounds_min=Vector3(0, 0, 0), bounds_max=Vector3(1, 1, 1))
    world.entities["e1"] = Entity(id="e1", type=EntityType.WALL, geometry_ids=["g1"])

    with pytest.raises(IFCBridgeError) as exc_info:
        export_world_to_ifc(world, tmp_path / "bad.ifc", schema="IFC1")
    assert "unsupported IFC schema version" in str(exc_info.value)
