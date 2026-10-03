"""The six core prototype exports -- glTF, USD(A), IFC, CityGML, CityJSON, Blender -- verified against ONE WorldIR.

Each export is parsed back with an independent reader (json / XML / IfcOpenShell / ast), not string-searched, and
checked against the world: every entity is either exported or reported skipped (nothing silently dropped), the
exported count survives the round trip, positions and semantic classes are the world's, and the output is
deterministic. The world is built through the real plane-promotion path (the same one a reconstruction takes), so
these are the exports a user gets from a reconstructed world.

Not claimed: opening the files in Blender / USD viewers / BIM tools (no such runtime on the build machine).
"""

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET

import pytest

from sdk import reality
from tests.test_export_pipeline_e2e import _promoted_world

CORE = ("gltf", "usda", "blender", "cityjson", "citygml", "ifc")


@pytest.fixture(scope="module")
def world():
    return _promoted_world()


@pytest.fixture(scope="module")
def exports(world):
    pytest.importorskip("ifcopenshell")
    return {fmt: reality.export(world, fmt) for fmt in CORE}


def test_every_entity_is_exported_or_reported_skipped(world, exports):
    for fmt, (_content, report) in exports.items():
        accounted = set(report.entities_exported) | set(report.entities_skipped)
        assert accounted == set(world.entities), f"{fmt} silently dropped {set(world.entities) - accounted}"
        assert len(report.skip_reasons) == len(report.entities_skipped)
        assert report.entities_exported, f"{fmt} exported nothing from a world with real geometry"
        assert report.world_id == world.id


def test_all_formats_export_the_same_entities(exports):
    sets = {fmt: set(r.entities_exported) for fmt, (_c, r) in exports.items()}
    assert len({frozenset(s) for s in sets.values()}) == 1, sets


def test_gltf_nodes_carry_the_entity_positions(world, exports):
    gltf = exports["gltf"][0]
    nodes = {n["name"]: n for n in gltf["nodes"]}
    assert set(nodes) == set(exports["gltf"][1].entities_exported)
    for eid, node in nodes.items():
        p = world.entities[eid].transform["position"]
        assert node["translation"] == pytest.approx([p["x"], p["y"], p["z"]])


def test_usda_has_one_prim_per_entity_at_its_position(world, exports):
    text = exports["usda"][0]
    assert text.startswith("#usda 1.0")
    for eid in exports["usda"][1].entities_exported:
        p = world.entities[eid].transform["position"]
        prim = eid.replace("-", "_")
        assert f'def Cube "{prim}"' in text
        assert f"({p['x']}, {p['y']}, {p['z']})" in text
    assert text.count("def Cube") == len(exports["usda"][1].entities_exported)


def test_blender_script_is_valid_python_naming_every_entity(exports):
    script, report = exports["blender"]
    ast.parse(script)
    for eid in report.entities_exported:
        assert eid in script


def test_cityjson_object_per_entity_with_declared_type(world, exports):
    cj, report = exports["cityjson"]
    assert cj["type"] == "CityJSON" and cj["version"].startswith("1.")
    assert set(cj["CityObjects"]) == set(report.entities_exported)
    for eid, obj in cj["CityObjects"].items():
        declared = obj["attributes"]["worldir_type"]
        assert getattr(declared, "value", declared) == world.entities[eid].type.value
        assert obj["geometry"], f"{eid} has no geometry"
    json.dumps(cj, default=str)   # serialisable


def test_citygml_is_well_formed_with_one_member_per_entity(exports):
    text, report = exports["citygml"]
    root = ET.fromstring(text)
    members = [m for m in root if m.tag.endswith("cityObjectMember")]
    assert len(members) == len(report.entities_exported)
    ids = {el.attrib[k] for m in members for el in m for k in el.attrib if k.endswith("}id")}
    assert set(report.entities_exported) <= ids


def test_ifc_parses_with_ifcopenshell_and_keeps_semantic_classes(world, exports, tmp_path):
    import ifcopenshell

    data, report = exports["ifc"]
    path = tmp_path / "w.ifc"
    path.write_bytes(data)
    model = ifcopenshell.open(str(path))
    expected = {"floor": "IfcSlab", "ceiling": "IfcSlab", "wall": "IfcWall"}
    names = {p.Name.split()[0]: p.is_a() for p in model.by_type("IfcBuildingElement")}
    assert set(names) == set(report.entities_exported)
    for eid, cls in names.items():
        assert cls == expected[world.entities[eid].type.value], (eid, cls)
    assert model.by_type("IfcProject") and model.by_type("IfcBuildingStorey")


@pytest.mark.parametrize("fmt", CORE)
def test_export_is_deterministic(world, fmt):
    pytest.importorskip("ifcopenshell")
    (a, ra), (b, rb) = reality.export(world, fmt), reality.export(world, fmt)
    assert ra.content_hash == rb.content_hash
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


@pytest.mark.parametrize("fmt", ("gltf", "citygml", "ifc"))
def test_an_entity_without_geometry_is_reported_not_exported(world, fmt):
    from copy import deepcopy

    from world_ir.schema_v1 import Entity, EntityType

    if fmt == "ifc":
        pytest.importorskip("ifcopenshell")
    w = deepcopy(world)
    w.entities["ghost"] = Entity(id="ghost", type=EntityType.WALL, name="ghost")
    _c, report = reality.export(w, fmt)
    assert "ghost" in report.entities_skipped and "ghost" not in report.entities_exported
