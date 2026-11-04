"""Tests for glTF and GLB validation, 4-byte chunk alignment, and transform preservation."""

import json
import struct
import numpy as np
import pytest

from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType
from world_ir.world_v1 import WorldIR
from exporters.gltf.exporter import export_to_gltf, export_to_glb


def test_export_glb_4byte_alignment_and_transforms():
    """Verify GLB 2.0 header, 4-byte chunk alignments, and rotation/scale transforms."""
    world = WorldIR()
    geom = Geometry(id="geom-cube", type=GeometryType.BOX)
    world.geometries[geom.id] = geom

    # Entity with translation, quaternion rotation, and scale
    entity = Entity(
        id="ent-transformed",
        name="TransformedChair",
        type=EntityType.STRUCTURE,
        transform={
            "position": {"x": 2.5, "y": 1.0, "z": -3.0},
            "rotation": {"x": 0.0, "y": 0.7071068, "z": 0.0, "w": 0.7071068},
            "scale": {"x": 0.5, "y": 0.5, "z": 0.5},
        },
        geometry_ids=["geom-cube"],
    )
    world.entities[entity.id] = entity

    # 1. Test glTF JSON export
    gltf = export_to_gltf(world)
    assert len(gltf["nodes"]) == 1
    node = gltf["nodes"][0]
    assert node["name"] == "TransformedChair"
    assert node["translation"] == [2.5, 1.0, -3.0]
    assert np.allclose(node["rotation"], [0.0, 0.7071068, 0.0, 0.7071068])
    assert node["scale"] == [0.5, 0.5, 0.5]

    # 2. Test GLB binary export
    glb_bytes = export_to_glb(world)
    assert len(glb_bytes) >= 20
    assert len(glb_bytes) % 4 == 0, "Total GLB file length must be a multiple of 4 bytes"

    # Header: magic (4), version (4), length (4)
    magic, version, total_length = struct.unpack("<4sII", glb_bytes[:12])
    assert magic == b"glTF", "GLB magic header must be 'glTF'"
    assert version == 2, "GLB version must be 2"
    assert total_length == len(glb_bytes), "Header length must match exact byte size"

    # Chunk 0: JSON chunk
    chunk0_len, chunk0_type = struct.unpack("<II", glb_bytes[12:20])
    assert chunk0_type == 0x4E4F534A, "Chunk 0 must be JSON type (0x4E4F534A)"
    assert chunk0_len % 4 == 0, "JSON chunk length must be 4-byte aligned"
    json_bytes = glb_bytes[20:20 + chunk0_len]
    parsed_json = json.loads(json_bytes.decode("utf-8").strip())
    assert parsed_json["asset"]["version"] == "2.0"
    assert len(parsed_json["nodes"]) == 1

    # Chunk 1: BIN chunk
    chunk1_start = 20 + chunk0_len
    chunk1_len, chunk1_type = struct.unpack("<II", glb_bytes[chunk1_start:chunk1_start + 8])
    assert chunk1_type == 0x004E4942, "Chunk 1 must be BIN type (0x004E4942)"
    assert chunk1_len % 4 == 0, "BIN chunk length must be 4-byte aligned"
    assert chunk1_start + 8 + chunk1_len == total_length, "File must terminate at end of BIN chunk"
