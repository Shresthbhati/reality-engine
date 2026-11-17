"""Tests for B19: Exports carry the detail (BUILD_SPEC B19).

Verifies:
1. Parts export as child nodes in glTF and IfcRelAggregates in IFC.
2. Attributes export as glTF extras and an IFC property set Pset_RealityEngine
   with Basis and Confidence.
3. Unknown attributes export as absent with Basis = "unknown", never a default.
4. Round-trip through ifcopenshell.open and glTF JSON parsing.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from provenance import Provenance
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Relationship,
    RelationshipKind,
    Vector3,
)
from world_ir.world_v1 import WorldIR
from exporters.gltf.exporter import export_to_gltf
from exporters.ifc_bridge import export_world_to_ifc


def _build_test_world_with_parts_and_attributes() -> WorldIR:
    world = WorldIR()

    # Cabinet (whole)
    cab_geom = Geometry(
        id="geom-cab",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(1.0, 0.6, 2.0),
        provenance=Provenance.OBSERVED,
    )
    world.geometries["geom-cab"] = cab_geom
    cab = Entity(
        id="ent-cab",
        name="Kitchen Cabinet",
        type=EntityType.FURNITURE,
        geometry_ids=["geom-cab"],
        provenance=Provenance.OBSERVED,
        confidence=0.92,
        custom_properties={
            "attributes": {
                "material": {"value": "wood", "basis": "visual", "p": 0.88},
                "door_count": {"value": "2", "basis": "visual", "p": 0.95},
                "lock_type": {"value": None, "basis": "unknown", "p": 0.0},
            }
        },
    )
    world.entities["ent-cab"] = cab

    # Drawer (part of cabinet)
    drw_geom = Geometry(
        id="geom-drw",
        type=GeometryType.BOX,
        bounds_min=Vector3(0.05, 0.05, 0.2),
        bounds_max=Vector3(0.95, 0.55, 0.6),
        provenance=Provenance.OBSERVED,
    )
    world.geometries["geom-drw"] = drw_geom
    drw = Entity(
        id="ent-drw",
        name="Cabinet Drawer",
        type=EntityType.FURNITURE,
        geometry_ids=["geom-drw"],
        provenance=Provenance.OBSERVED,
        confidence=0.89,
        relationships=[
            Relationship(
                kind=RelationshipKind.PART_OF,
                target_id="ent-cab",
            )
        ],
        custom_properties={
            "attributes": {
                "handle_type": {"value": "bar", "basis": "visual", "p": 0.91},
                "contents": {"value": "cutlery", "basis": "user_stated", "p": 1.0},
                "finish": {"value": None, "basis": "unknown", "p": 0.0},
            }
        },
    )
    world.entities["ent-drw"] = drw

    return world


def test_gltf_export_carries_parts_and_attributes(tmp_path: Path):
    world = _build_test_world_with_parts_and_attributes()
    gltf = export_to_gltf(world)

    nodes = gltf["nodes"]
    cab_node = next(n for n in nodes if n["name"] == "Kitchen Cabinet")
    drw_node = next(n for n in nodes if n["name"] == "Cabinet Drawer")

    cab_idx = nodes.index(cab_node)
    drw_idx = nodes.index(drw_node)

    # 1. Drawer is a child node of Cabinet in glTF hierarchy
    assert "children" in cab_node
    assert drw_idx in cab_node["children"]

    # Only Cabinet is a root node in scenes
    root_node_indices = gltf["scenes"][0]["nodes"]
    assert cab_idx in root_node_indices
    assert drw_idx not in root_node_indices

    # 2. Attributes export to extras
    assert "extras" in cab_node
    cab_attrs = cab_node["extras"]["attributes"]
    assert cab_attrs["material"]["value"] == "wood"
    assert cab_attrs["material"]["basis"] == "visual"
    assert cab_attrs["material"]["confidence"] == 0.88

    # 3. Unknown attributes export as absent with Basis = "unknown", never a default
    assert "value" not in cab_attrs["lock_type"]
    assert cab_attrs["lock_type"]["basis"] == "unknown"

    drw_attrs = drw_node["extras"]["attributes"]
    assert drw_attrs["handle_type"]["value"] == "bar"
    assert drw_attrs["contents"]["value"] == "cutlery"
    assert drw_attrs["contents"]["basis"] == "user_stated"
    assert "value" not in drw_attrs["finish"]
    assert drw_attrs["finish"]["basis"] == "unknown"


def test_ifc_export_carries_parts_and_attributes(tmp_path: Path):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    import ifcopenshell.util.element as ifc_element

    world = _build_test_world_with_parts_and_attributes()
    out = tmp_path / "world.ifc"
    rep = export_world_to_ifc(world, out)
    assert out.exists()

    f = ifcopenshell.open(str(out))

    # 1. Verify parts relationship via IfcRelAggregates
    rel_aggregates = f.by_type("IfcRelAggregates")
    found_part_rel = False
    for rel in rel_aggregates:
        relating = rel.RelatingObject
        related_names = [p.Name for p in rel.RelatedObjects]
        if "ent-cab" in relating.Name and any("ent-drw" in name for name in related_names):
            found_part_rel = True
            break
    assert found_part_rel, "Cabinet should aggregate Drawer via IfcRelAggregates"

    # 2. Verify Pset_RealityEngine property set
    elements = {e.Name.split()[0]: e for e in f.by_type("IfcProduct") if e.Name}
    cab_elem = elements["ent-cab"]
    drw_elem = elements["ent-drw"]

    cab_psets = ifc_element.get_psets(cab_elem)
    assert "Pset_RealityEngine" in cab_psets
    cab_re = cab_psets["Pset_RealityEngine"]
    assert cab_re["material"] == "wood"
    assert cab_re["material_Basis"] == "visual"
    assert cab_re["material_Confidence"] == 0.88
    # Unknown lock_type is absent from values, but has Basis = "unknown"
    assert "lock_type" not in cab_re
    assert cab_re["lock_type_Basis"] == "unknown"

    drw_psets = ifc_element.get_psets(drw_elem)
    assert "Pset_RealityEngine" in drw_psets
    drw_re = drw_psets["Pset_RealityEngine"]
    assert drw_re["handle_type"] == "bar"
    assert drw_re["contents"] == "cutlery"
    assert drw_re["contents_Basis"] == "user_stated"
    assert "finish" not in drw_re
    assert drw_re["finish_Basis"] == "unknown"
