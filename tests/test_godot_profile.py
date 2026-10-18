"""Godot export profile: round-trip a procedural world through GLB + manifest (structure only; Godot is not
available here, so loading in the engine is UNVERIFIED)."""

import json
import struct

import pytest

from exporters.profiles import PROFILES, UnknownProfileError, export_profile
from procedural.building_grammar import BuildingSpec, generate_building
from procedural.room_grammar import RoomGrammarSpec, generate_room
from world_ir.schema_v1 import Provenance


def read_glb(data: bytes):
    magic, version, total = struct.unpack_from("<III", data, 0)
    assert (magic, version, total) == (0x46546C67, 2, len(data))
    jlen, jtype = struct.unpack_from("<II", data, 12)
    assert jtype == 0x4E4F534A
    gltf = json.loads(data[20:20 + jlen])
    blen, btype = struct.unpack_from("<II", data, 20 + jlen)
    assert btype == 0x004E4942 and 28 + jlen + blen == len(data)
    return gltf, data[28 + jlen:]


def node_bounds(node):
    return ([c - s / 2 for c, s in zip(node["translation"], node["scale"])],
            [c + s / 2 for c, s in zip(node["translation"], node["scale"])])


def expected_bounds(world):
    out = {}
    for e in world.entities.values():
        if e.geometry_ids:      # the exporter takes the first geometry (the envelope; stairs also carry per-step boxes)
            geo = world.geometries[e.geometry_ids[0]]
            out[e.id] = ([geo.bounds_min.x, geo.bounds_min.y, geo.bounds_min.z],
                         [geo.bounds_max.x, geo.bounds_max.y, geo.bounds_max.z])
    return out


def roundtrip(world):
    files, report = export_profile(world, "godot")
    gltf, blob = read_glb(files["world.glb"])
    manifest = json.loads(files["manifest.json"])
    return files, report, gltf, blob, manifest


@pytest.mark.parametrize("make", [
    lambda: generate_room(RoomGrammarSpec("r", 4.0, 5.0, 2.5)),
    lambda: generate_building(BuildingSpec(name="tower", footprint=(10.0, 8.0), storeys=2)),
], ids=["room", "building"])
def test_roundtrip_counts_names_bounds(make):
    world = make()
    want = expected_bounds(world)
    files, report, gltf, blob, manifest = roundtrip(world)

    assert set(files) == {"world.glb", "manifest.json", "scene.tscn"}
    assert len(gltf["nodes"]) == len(want) == len(manifest["entities"]) == len(report.entities_exported)
    assert gltf["buffers"][0]["byteLength"] <= len(blob)

    by_id = {n["extras"]["entity_id"]: n for n in gltf["nodes"]}
    assert set(by_id) == set(want)
    for eid, (lo, hi) in want.items():
        node = by_id[eid]
        nlo, nhi = node_bounds(node)
        assert nlo == pytest.approx(lo, abs=1e-6) and nhi == pytest.approx(hi, abs=1e-6)
        solid = node["extras"]["kind"] in {"wall", "floor", "ceiling", "roof", "column", "beam"}
        assert node["name"] == (f"{eid}-col" if solid else eid)
    assert any(n["name"].endswith("-col") for n in gltf["nodes"])
    assert any(not n["name"].endswith("-col") for n in gltf["nodes"]) == any(
        e["kind"] in {"door", "window", "stairs"} for e in manifest["entities"])

    for entry in manifest["entities"]:   # manifest agrees with the GLB and the world
        assert entry["bounds_min"] == pytest.approx(want[entry["id"]][0], abs=1e-6)
        assert entry["bounds_max"] == pytest.approx(want[entry["id"]][1], abs=1e-6)
        assert entry["collision"] == entry["node"].endswith("-col")
        assert entry["provenance"] == "GENERATED"


def test_axis_convention_y_up_north_is_minus_z():
    w, d, h = 4.0, 5.0, 2.5
    _, _, gltf, _, manifest = roundtrip(generate_room(RoomGrammarSpec("r", w, d, h)))
    lo = [min(node_bounds(n)[0][a] for n in gltf["nodes"]) for a in range(3)]
    hi = [max(node_bounds(n)[1][a] for n in gltf["nodes"]) for a in range(3)]
    assert hi[0] - lo[0] == pytest.approx(w, abs=0.5)      # x = east (width), walls add thickness
    assert hi[2] - lo[2] == pytest.approx(d, abs=0.5)      # depth lies on Z
    assert lo[2] < 0 and hi[2] <= 0.5                      # local +y (north) maps to -Z: room extends toward -Z
    assert lo[1] >= -0.5 and hi[1] == pytest.approx(h, abs=0.5)   # height on Y
    assert manifest["frame"]["up"] == "+Y" and manifest["frame"]["forward"] == "-Z"
    cube = gltf["accessors"][0]
    assert cube["min"] == [-0.5] * 3 and cube["max"] == [0.5] * 3   # unit cube scaled per node


def test_units_metres_only_when_justified_never_invented():
    world = generate_room(RoomGrammarSpec("r", 4.0, 5.0, 2.5))
    m = roundtrip(world)[4]
    assert m["unit"] == "metre" and m["scale_to_metres"] == 1.0

    for e in world.entities.values():
        e.provenance = Provenance.RECONSTRUCTED
    m = roundtrip(world)[4]
    assert m["unit"] == "model-unit" and m["scale_to_metres"] is None and "no conversion" in m["unit_basis"]

    world.metadata["units"] = "meters"
    assert roundtrip(world)[4]["unit"] == "metre"


def test_deterministic_and_report_lists_skips():
    world = generate_building(BuildingSpec(name="tower", footprint=(10.0, 8.0), storeys=2))
    a, ra = export_profile(world, "godot")
    b, rb = export_profile(world, "godot")
    assert a == b and ra.content_hash == rb.content_hash
    manifest = json.loads(a["manifest.json"])
    assert len(ra.entities_skipped) == len(manifest["skipped"]) > 0   # rooms/storeys/building: no solid geometry
    assert all(s["reason"] for s in manifest["skipped"])
    stairs = next(e for e in manifest["entities"] if e["kind"] == "stairs")
    assert stairs["geometries_dropped"] > 0 and not stairs["collision"]   # per-step boxes are reported, not hidden
    assert "res://world.glb" in a["scene.tscn"].decode()


def test_profile_registry():
    assert "godot" in PROFILES
    with pytest.raises(UnknownProfileError):
        export_profile(generate_room(RoomGrammarSpec("r", 3, 3, 2.5)), "nope")
