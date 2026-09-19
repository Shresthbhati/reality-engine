"""Tests for the minimal glTF 2.0 exporter (exporters/gltf/exporter.py)."""

import base64
import struct

from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType
from world_ir.world_v1 import WorldIR
from exporters.gltf.exporter import export_to_gltf


def _build_world() -> WorldIR:
    world = WorldIR()

    # 1. Has transform + BOX geometry -> should export.
    box_geom = Geometry(id="geom-box", type=GeometryType.BOX)
    box_entity = Entity(
        id="ent-box",
        name="Crate",
        type=EntityType.DEBRIS,
        transform={"position": {"x": 1.0, "y": 2.0, "z": 3.0}},
        geometry_ids=["geom-box"],
    )

    # 2. Has transform but POINTCLOUD geometry -> should be skipped
    #    (no real vertex data to export, not fabricated).
    pc_geom = Geometry(id="geom-pc", type=GeometryType.POINTCLOUD, vertex_count=1000)
    pc_entity = Entity(
        id="ent-pc",
        name="Scan",
        type=EntityType.UNKNOWN,
        transform={"position": {"x": 5.0, "y": 5.0, "z": 5.0}},
        geometry_ids=["geom-pc"],
    )

    # 3. Has BOX geometry but no transform -> should be skipped
    #    (nothing to place the node at).
    box_geom2 = Geometry(id="geom-box2", type=GeometryType.BOX)
    no_transform_entity = Entity(
        id="ent-no-transform",
        name="Untethered",
        type=EntityType.DEBRIS,
        transform=None,
        geometry_ids=["geom-box2"],
    )

    world.geometries[box_geom.id] = box_geom
    world.geometries[pc_geom.id] = pc_geom
    world.geometries[box_geom2.id] = box_geom2
    world.entities[box_entity.id] = box_entity
    world.entities[pc_entity.id] = pc_entity
    world.entities[no_transform_entity.id] = no_transform_entity

    return world


def test_only_valid_entity_exported():
    world = _build_world()
    gltf = export_to_gltf(world)

    assert len(gltf["nodes"]) == 1
    assert gltf["asset"]["version"] == "2.0"
    node = gltf["nodes"][0]
    assert node["translation"] == [1.0, 2.0, 3.0]
    assert node["mesh"] == 0
    assert gltf["scenes"][gltf["scene"]]["nodes"] == [0]


def test_accessor_and_buffer_correctness():
    world = _build_world()
    gltf = export_to_gltf(world)

    position_accessor = gltf["accessors"][0]
    assert position_accessor["count"] == 8  # unit cube has 8 vertices
    assert position_accessor["componentType"] == 5126  # FLOAT
    assert position_accessor["type"] == "VEC3"

    index_accessor = gltf["accessors"][1]
    assert index_accessor["componentType"] == 5123  # UNSIGNED_SHORT
    assert index_accessor["count"] == 36  # 12 triangles * 3

    buffer = gltf["buffers"][0]
    uri = buffer["uri"]
    assert uri.startswith("data:application/octet-stream;base64,")
    raw = base64.b64decode(uri.split(",", 1)[1])
    assert len(raw) == buffer["byteLength"]

    # Position bufferView must decode to exactly 8 float32 xyz triples.
    pos_view = gltf["bufferViews"][position_accessor["bufferView"]]
    pos_bytes = raw[pos_view["byteOffset"]: pos_view["byteOffset"] + pos_view["byteLength"]]
    assert len(pos_bytes) == 8 * 3 * struct.calcsize("<f")
    floats = struct.unpack(f"<{8 * 3}f", pos_bytes)
    assert len(floats) == 24


def test_gltf_minimum_required_top_level_shape():
    """Lightweight structural check against glTF 2.0's minimum requirements
    (no schema-validation dependency — just presence/type of required keys)."""
    world = _build_world()
    gltf = export_to_gltf(world)

    assert isinstance(gltf["asset"], dict) and isinstance(gltf["asset"]["version"], str)
    assert isinstance(gltf["scenes"], list) and all(isinstance(s, dict) for s in gltf["scenes"])
    assert isinstance(gltf["scene"], int)
    assert isinstance(gltf["nodes"], list) and all(isinstance(n, dict) for n in gltf["nodes"])
    assert isinstance(gltf["meshes"], list) and all(isinstance(m, dict) for m in gltf["meshes"])
    for mesh in gltf["meshes"]:
        assert isinstance(mesh["primitives"], list)
        for prim in mesh["primitives"]:
            assert "POSITION" in prim["attributes"]
    assert isinstance(gltf["accessors"], list)
    for accessor in gltf["accessors"]:
        assert "componentType" in accessor and "count" in accessor and "type" in accessor
    assert isinstance(gltf["bufferViews"], list)
    for bv in gltf["bufferViews"]:
        assert "buffer" in bv and "byteLength" in bv
    assert isinstance(gltf["buffers"], list)
    for buf in gltf["buffers"]:
        assert "byteLength" in buf
