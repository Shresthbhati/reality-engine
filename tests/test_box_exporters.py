"""P16-01: the box-based world compilers -- GeoJSON (GIS), Gazebo SDF (ROS), Godot .tscn, Unreal Python, SUMO
node/edge files, Habitat stage bundle.

What these tests prove: the output is well-formed for its format, carries the REAL positions and extents with the
target's axis convention applied, is deterministic, and skips (with a reason) what it cannot place. What they do NOT
prove: that Gazebo / Godot / Unreal / netconvert / habitat-sim accept the files -- none of those runtimes exists on
the build machine, so loading is UNVERIFIED (stated in every module docstring). The Unreal script is the exception to
"structure only": it is EXECUTED against a fake `unreal` module that records the spawn calls.
"""

from __future__ import annotations

import json
import re
import sys
import types
import xml.etree.ElementTree as ET

import pytest

from provenance import Provenance
from sdk import reality
from tests.test_cityjson_exporter import _world
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR

FORMATS = ["geojson", "sdf", "tscn", "unreal", "sumo", "habitat"]


def _road_world() -> WorldIR:
    """Two collinear roads that meet at x = 20, a square plot, and a building."""
    w = WorldIR()

    def add(eid, etype, pos, lo, hi):
        w.geometries[f"g-{eid}"] = Geometry(id=f"g-{eid}", type=GeometryType.BOX, bounds_min=Vector3(*lo),
                                            bounds_max=Vector3(*hi))
        w.entities[eid] = Entity(id=eid, name=eid, type=etype, geometry_ids=[f"g-{eid}"],
                                 transform={"position": {"x": pos[0], "y": pos[1], "z": pos[2]}},
                                 provenance=Provenance.OBSERVED)

    add("road-a", EntityType.ROAD, (10, 0, 0), (-10, -2, 0), (10, 2, 0.1))          # x 0..20, 4 m wide
    add("road-b", EntityType.ROAD, (30, 0, 0), (-10, -2, 0), (10, 2, 0.1))          # x 20..40
    add("road-c", EntityType.ROAD, (20, 20, 0), (-2, -10, 0), (2, 10, 0.1))         # north-south, y 10..30
    add("plot", EntityType.ROAD, (0, 50, 0), (-3, -3, 0), (3, 3, 0.1))              # square: no direction
    add("house", EntityType.BUILDING, (50, 50, 0), (-1, -1, 0), (1, 1, 4))
    return w


# ------------------------------------------------------------------------------------------------ common


@pytest.mark.parametrize("fmt", FORMATS)
def test_every_format_is_registered_deterministic_and_reports_what_it_skipped(fmt):
    world = _world()                       # wall + building exportable, one entity without a transform
    content, report = reality.export(world, fmt)
    again, report2 = reality.export(world, fmt)
    assert content == again and report.content_hash == report2.content_hash          # byte-identical
    assert report.format in (fmt, "sdf", "tscn", "geojson", "sumo", "habitat", "unreal")
    assert report.world_id == world.id and len(report.entities_skipped) == len(report.skip_reasons)
    assert "no transform position" in " ".join(report.skip_reasons) or fmt in ("sumo", "habitat")
    assert isinstance(content, str) and content


def test_an_unknown_format_still_raises():
    with pytest.raises(reality.UnsupportedExportFormatError):
        reality.export(_world(), "fbx")


# ------------------------------------------------------------------------------------------------ GeoJSON


def _signed_area(ring):
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(ring, ring[1:])) / 2.0


def test_geojson_footprints_are_real_closed_counter_clockwise_polygons_in_a_labelled_local_frame():
    doc = json.loads(reality.export(_world(), "geojson")[0])
    assert doc["type"] == "FeatureCollection" and "NOT georeferenced" in doc["worldir"]["frame"]
    assert doc["crs"]["properties"]["name"] == "urn:worldir:crs:local-metres"
    feats = {f["id"]: f for f in doc["features"]}
    ring = feats["bld-1"]["geometry"]["coordinates"][0]
    assert ring[0] == ring[-1] and _signed_area(ring) > 0
    assert sorted({p[0] for p in ring}) == [49.0, 51.0] and sorted({p[1] for p in ring}) == [49.0, 51.0]
    assert feats["bld-1"]["properties"]["height_m"] == 4.0 and feats["bld-1"]["properties"]["worldir_type"] == "building"


def test_geojson_with_an_origin_projects_to_wgs84_and_records_its_assumptions():
    from exporters.gis.exporter import export_to_geojson

    doc = export_to_geojson(_world(), origin_lonlat=(8.0, 47.0))
    assert "EPSG:4326" in doc["worldir"]["frame"] and "east" in doc["worldir"]["assumption"]
    lon, lat = doc["features"][[f["id"] for f in doc["features"]].index("bld-1")]["geometry"]["coordinates"][0][0]
    assert lat == pytest.approx(47.0 + 49.0 / 111_320.0, abs=2e-5)               # 49 m north of the origin
    assert lon == pytest.approx(8.0 + 49.0 / (111_320.0 * 0.6820), abs=3e-5)     # 49 m east, cos(47 deg) = 0.682
    with pytest.raises(ValueError):
        export_to_geojson(_world(), origin_lonlat=(200.0, 0.0))


# ------------------------------------------------------------------------------------------------ SDF (ROS)


def test_sdf_is_well_formed_with_real_poses_and_box_sizes():
    root = ET.fromstring(reality.export(_world(), "sdf")[0])
    assert root.tag == "sdf" and root.get("version") == "1.9"
    models = {m.get("name"): m for m in root.iter("model")}
    assert set(models) == {"wall_north", "bld_1"} or set(models) == {"wall-north".replace("-", "_"), "bld_1"}
    b = models["bld_1"]
    assert b.find("static").text == "true"
    assert [float(v) for v in b.find("pose").text.split()] == [50, 50, 2, 0, 0, 0]       # centre, Z-up like WorldIR
    for kind in ("collision", "visual"):
        assert [float(v) for v in b.find(f"link/{kind}/geometry/box/size").text.split()] == [2, 2, 4]


def test_sdf_model_names_are_valid_and_unique():
    w = _world()
    twin = w.entities["bld-1"]
    w.entities["bld.1"] = Entity(id="bld.1", name="twin", type=twin.type, transform=twin.transform,
                                 geometry_ids=twin.geometry_ids)
    names = [m.get("name") for m in ET.fromstring(reality.export(w, "sdf")[0]).iter("model")]
    assert len(names) == len(set(names)) and all(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", n) for n in names)


# ------------------------------------------------------------------------------------------------ Godot


def _parse_tscn(text):
    sub, nodes, refs = {}, [], re.findall(r'SubResource\("([^"]+)"\)', text)
    for m in re.finditer(r'^\[(sub_resource|node)([^\]]*)\]\n((?:[^\[\n].*\n?)*)', text, re.M):
        kind, attrs, body = m.groups()
        a = dict(re.findall(r'(\w+)="([^"]*)"', attrs))
        if kind == "sub_resource":
            sub[a["id"]] = (a["type"], body)
        else:
            nodes.append((a, body))
    return sub, nodes, refs


def test_tscn_is_a_consistent_godot4_scene_with_y_up_positions():
    text = reality.export(_world(), "tscn")[0]
    assert text.startswith("[gd_scene load_steps=") and "format=3]" in text.splitlines()[0]
    sub, nodes, refs = _parse_tscn(text)
    assert int(re.search(r"load_steps=(\d+)", text).group(1)) == 1 + len(sub)
    assert refs and all(r in sub for r in refs), "a node references a sub-resource that was never declared"
    names = {a["name"] for a, _ in nodes}
    assert all(a.get("parent") in (None, ".") or a["parent"] in names for a, _ in nodes)
    body = next(b for a, b in nodes if a["name"] == "bld-1".replace("-", "-") and a.get("type") == "StaticBody3D")
    # WorldIR (50, 50, 2) -> Godot (x, z, -y) = (50, 2, -50)
    assert "50, 2, -50)" in body
    sizes = [b for t, b in sub.values() if t == "BoxMesh"]
    assert any("Vector3(2, 4, 2)" in b for b in sizes)                           # (dx, dz, dy)


def test_tscn_shares_one_sub_resource_per_distinct_size():
    w = _world()
    for i in range(3):
        g = w.geometries["geom-bld"]
        w.entities[f"copy-{i}"] = Entity(id=f"copy-{i}", name="c", type=EntityType.BUILDING, geometry_ids=[g.id],
                                         transform={"position": {"x": float(i), "y": 0.0, "z": 0.0}})
    sub, _, _ = _parse_tscn(reality.export(w, "tscn")[0])
    assert sum(1 for t, _ in sub.values() if t == "BoxMesh") == 2                # wall size + building size


# ------------------------------------------------------------------------------------------------ Unreal


def test_the_unreal_script_executes_and_spawns_boxes_in_unreal_units_and_handedness():
    script = reality.export(_world(), "unreal")[0]
    compile(script, "<unreal-export>", "exec")
    spawned = []

    class _Vec:
        def __init__(self, x=0, y=0, z=0):
            self.v = (x, y, z)

    class _Comp:
        def set_static_mesh(self, mesh):
            self.mesh = mesh

    class _Actor:
        def __init__(self, loc):
            self.loc, self.static_mesh_component, self.label, self.scale = loc, _Comp(), None, None

        def set_actor_label(self, label):
            self.label = label

        def set_actor_scale3d(self, scale):
            self.scale = scale
            spawned.append(self)

    fake = types.ModuleType("unreal")
    fake.Vector = _Vec
    fake.StaticMeshActor = object
    fake.EditorAssetLibrary = types.SimpleNamespace(load_asset=lambda path: f"asset:{path}")
    fake.EditorLevelLibrary = types.SimpleNamespace(spawn_actor_from_class=lambda cls, loc: _Actor(loc))
    sys.modules["unreal"] = fake
    try:
        exec(compile(script, "<unreal-export>", "exec"), {"__name__": "__main__"})
    finally:
        del sys.modules["unreal"]
    by = {a.label: a for a in spawned}
    house = by["bld-1"]
    assert house.loc.v == (5000.0, -5000.0, 200.0)                  # (50, 50, 2) m -> (x, -y, z) * 100 cm
    assert house.scale.v == (2.0, 2.0, 4.0)                         # 100 uu cube scaled to the extents
    assert house.static_mesh_component.mesh == "asset:/Engine/BasicShapes/Cube.Cube"
    assert len(spawned) == 2


# ------------------------------------------------------------------------------------------------ SUMO


def _sumo(world):
    content, report = reality.export(world, "sumo")
    files = json.loads(content)
    return ET.fromstring(files["network.nod.xml"]), ET.fromstring(files["network.edg.xml"]), report, files


def test_sumo_connects_roads_that_meet_and_carries_the_measured_width():
    nod, edg, report, _ = _sumo(_road_world())
    edges = {e.get("id"): e for e in edg.iter("edge")}
    assert set(edges) == {"road-a", "road-b", "road-c"}
    nodes = {n.get("id"): (float(n.get("x")), float(n.get("y"))) for n in nod.iter("node")}
    a, b = edges["road-a"], edges["road-b"]
    assert nodes[a.get("from")] == (0, 0) and nodes[a.get("to")] == (20, 0)
    assert a.get("to") == b.get("from"), "roads that meet at x=20 must share a node"
    assert nodes[edges["road-c"].get("from")] == (20, 10) and nodes[edges["road-c"].get("to")] == (20, 30)
    assert {p.get("key"): p.get("value") for p in a.iter("param")}["worldir_width_m"] == "4"
    assert "speed" not in a.attrib and "numLanes" not in a.attrib, "no speed / lane count may be invented"


def test_sumo_skips_non_roads_and_directionless_footprints_with_a_reason():
    *_, report, _ = _sumo(_road_world())
    why = dict(zip(report.entities_skipped, report.skip_reasons))
    assert "no dominant direction" in why["plot"] and "not a road" in why["house"]
    assert set(report.entities_exported) == {"road-a", "road-b", "road-c"}


def test_a_world_with_no_roads_exports_an_empty_network_not_a_fabricated_one():
    nod, edg, report, _ = _sumo(_world())
    assert not list(nod.iter("node")) and not list(edg.iter("edge")) and not report.entities_exported


def test_sumo_files_are_written_to_disk(tmp_path):
    from exporters.sumo.exporter import write_sumo_files

    paths = write_sumo_files(reality.export(_road_world(), "sumo")[0], tmp_path / "net")
    assert sorted(p.name for p in paths) == ["network.edg.xml", "network.nod.xml"]
    ET.parse(paths[0]), ET.parse(paths[1])


# ------------------------------------------------------------------------------------------------ Habitat


def test_habitat_bundle_is_self_consistent_and_declares_the_asset_axes():
    content, report = reality.export(_world(), "habitat")
    files = json.loads(content)
    assert set(files) == {"scene.gltf", "scene.stage_config.json", "scene.scene_dataset_config.json"}
    stage = json.loads(files["scene.stage_config.json"])
    assert stage["render_asset"] in files and stage["up"] == [0, 0, 1]            # the glTF is written Z-up
    assert json.loads(files["scene.scene_dataset_config.json"])["stages"]["configs"] == ["scene.stage_config.json"]
    gltf = json.loads(files["scene.gltf"])
    assert gltf["asset"]["version"] == "2.0" and gltf["nodes"]
    assert set(report.entities_exported) == {"wall-north", "bld-1"}               # same entities as the glTF exporter


def test_habitat_files_are_written_to_disk(tmp_path):
    from exporters.habitat.exporter import write_habitat_files

    paths = write_habitat_files(reality.export(_world(), "habitat")[0], tmp_path / "stage")
    assert len(paths) == 3 and all(p.read_text(encoding="utf-8") for p in paths)


# ------------------------------------------------------------------------------------------------ CLI


@pytest.mark.parametrize("fmt,expected", [("sumo", {"network.nod.xml", "network.edg.xml"}),
                                          ("habitat", {"scene.gltf", "scene.stage_config.json",
                                                       "scene.scene_dataset_config.json"})])
def test_the_cli_writes_bundle_formats_as_a_directory_of_files(tmp_path, fmt, expected):
    from apps.cli.main import main

    world_path = tmp_path / "world.json"
    world_path.write_text(json.dumps(_road_world().to_dict()), encoding="utf-8")
    out = tmp_path / "out"
    assert main(["export", str(world_path), "--format", fmt, "-o", str(out)]) == 0
    assert {p.name for p in out.iterdir()} == expected


def test_the_cli_writes_a_single_file_format_as_a_file(tmp_path):
    from apps.cli.main import main

    world_path = tmp_path / "world.json"
    world_path.write_text(json.dumps(_world().to_dict()), encoding="utf-8")
    out = tmp_path / "scene.tscn"
    assert main(["export", str(world_path), "--format", "tscn", "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("[gd_scene")
