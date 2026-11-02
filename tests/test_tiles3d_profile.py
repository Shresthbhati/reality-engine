"""3D Tiles profile: structure, frames, georeference maths, determinism, and (when the official 3d-tiles clone is
present) validation against the real tileset JSON schema. NOT loaded in CesiumJS -- UNVERIFIED there."""

import json
import math
import os
import pathlib
import struct

import pytest

from exporters.profiles import PROFILES, export_profile
from exporters.tiles3d.exporter import enu_to_ecef_matrix
from procedural.building_grammar import BuildingSpec, generate_building
from procedural.room_grammar import RoomGrammarSpec, generate_room

SPEC_DIR = pathlib.Path(os.environ.get(
    "REALITY_3DTILES_SPEC", pathlib.Path.home() / "codes-research" / "3d-tiles" / "specification" / "schema"))


def read_glb(data: bytes):
    magic, version, total = struct.unpack_from("<III", data, 0)
    assert (magic, version, total) == (0x46546C67, 2, len(data))
    jlen, jtype = struct.unpack_from("<II", data, 12)
    assert jtype == 0x4E4F534A
    return json.loads(data[20:20 + jlen])


def building():
    return generate_building(BuildingSpec(name="tower", footprint=(10.0, 8.0), storeys=2))


def export(world, **kw):
    files, report = PROFILES["3dtiles"](world, None, **kw)
    return files, report, json.loads(files["tileset.json"])


def test_registered_profile():
    files, report = export_profile(building(), "3dtiles")
    assert "tileset.json" in files and report.format == "3dtiles"


@pytest.mark.skipif(not SPEC_DIR.is_dir(), reason="official 3d-tiles clone not present")
@pytest.mark.parametrize("kw", [{}, {"origin_lonlat": (77.2, 28.6), "origin_height_m": 200.0}, {"tile_size_m": 3.0}])
def test_tileset_validates_against_official_schema(kw):
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    base = "https://schemas.local/"

    def retrieve(uri):
        return Resource.from_contents(json.loads((SPEC_DIR / uri[len(base):]).read_text(encoding="utf-8")),
                                      default_specification=DRAFT202012)

    root = json.loads((SPEC_DIR / "tileset.schema.json").read_text(encoding="utf-8"))
    root["$id"] = base + "tileset.schema.json"
    validator = Draft202012Validator(root, registry=Registry(retrieve=retrieve))
    _, _, tileset = export(building(), **kw)
    assert [e.message for e in validator.iter_errors(tileset)] == []


def test_every_box_in_exactly_one_tile_with_yup_nodes_inside_tile_volume():
    world = building()
    files, report, tileset = export(world, tile_size_m=3.0)
    children = tileset["root"]["children"]
    assert len(children) > 1                                            # the grid really splits the building
    seen = {}
    for child in children:
        glb = read_glb(files[child["content"]["uri"]])
        cx, cy, cz, hx, _, _, _, hy, _, _, _, hz = child["boundingVolume"]["box"]
        for node in glb["nodes"]:
            x, y_up, neg_y = node["translation"]
            wx, wy, wz = x, -neg_y, y_up                                # undo (x, y, z) -> (x, z, -y)
            sx, sz_up, sy = node["scale"]
            assert abs(wx - cx) + sx / 2 <= hx + 1e-4
            assert abs(wy - cy) + sy / 2 <= hy + 1e-4
            assert abs(wz - cz) + sz_up / 2 <= hz + 1e-4
            seen[node["name"]] = seen.get(node["name"], 0) + 1
    assert sorted(seen) == sorted(report.entities_exported)
    assert set(seen.values()) == {1}


def test_no_transform_when_not_georeferenced_and_says_so():
    _, _, tileset = export(building())
    assert "transform" not in tileset["root"]
    assert tileset["asset"]["extras"]["worldir"]["georeferenced"] is False


def test_georeference_matrix_known_points():
    m = enu_to_ecef_matrix(0.0, 0.0, 0.0)                               # equator / prime meridian
    assert m[12:15] == pytest.approx([6378137.0, 0.0, 0.0])
    assert m[0:3] == pytest.approx([0, 1, 0]) and m[4:7] == pytest.approx([0, 0, 1])      # east=+Y, north=+Z
    assert m[8:11] == pytest.approx([1, 0, 0])                                            # up=+X
    pole = enu_to_ecef_matrix(0.0, 90.0, 0.0)
    assert pole[12:15] == pytest.approx([0.0, 0.0, 6356752.314], abs=1e-2)               # WGS84 semi-minor axis
    m = enu_to_ecef_matrix(77.2, 28.6, 200.0)
    assert math.hypot(*m[12:15]) == pytest.approx(6.37e6, rel=0.01)


def test_deterministic_and_validates_inputs():
    a = export(building())[1].content_hash
    assert a == export(building())[1].content_hash
    with pytest.raises(ValueError):
        export(building(), tile_size_m=0)
    with pytest.raises(ValueError):
        export(building(), origin_lonlat=(200.0, 0.0))


def test_room_exports_without_skips_hiding():
    world = generate_room(RoomGrammarSpec("r", 4.0, 5.0, 2.5))
    files, report, tileset = export(world)
    assert report.entities_exported and len(report.entities_exported) + len(report.entities_skipped) == len(world.entities)
    assert tileset["asset"]["extras"]["worldir"]["skipped"] == [
        {"id": i, "reason": r} for i, r in zip(report.entities_skipped, report.skip_reasons)]
