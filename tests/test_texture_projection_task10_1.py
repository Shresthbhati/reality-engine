"""Unit tests for Task 10.1: Texture projection from photos onto meshes and GLB texture export."""

import base64
import struct
from pathlib import Path
import cv2
import numpy as np
import pytest

from engine.render.textures.projection import PinholeCamera
from exporters.gltf.exporter import export_to_glb, export_to_gltf
from reconstruction.meshing.mesh import MeshData
from reconstruction.texturing.projection import (
    MeshTextureProjector,
    TextureResult,
    apply_texture_projection,
)
from world_ir.artifact_store import MemoryArtifactStore
from world_ir.schema_v1 import Entity, Geometry, GeometryType
from world_ir.world_v1 import WorldIR


@pytest.fixture
def sample_wall_mesh() -> MeshData:
    # A 4m wide x 2.5m high vertical wall in the X-Z plane at Y=0
    # 2 triangles (quad): (0,0,0), (4,0,0), (4,0,2.5), (0,0,2.5)
    vertices = (
        (0.0, 0.0, 0.0),
        (4.0, 0.0, 0.0),
        (4.0, 0.0, 2.5),
        (0.0, 0.0, 2.5),
    )
    faces = (
        (0, 1, 2),
        (0, 2, 3),
    )
    normals = (
        (0.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, -1.0, 0.0),
    )
    return MeshData(vertices=vertices, faces=faces, normals=normals)


@pytest.fixture
def front_facing_camera() -> PinholeCamera:
    # Camera placed at (2.0, -3.0, 1.25) looking along +Y toward wall at Y=0
    # Image: 400x300 textured pattern
    h, w = 300, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (350, 250), (220, 140, 60), -1)
    cv2.circle(img, (200, 150), 40, (0, 200, 255), -1)

    fx, fy = 350.0, 350.0
    cx, cy = 200.0, 150.0
    intrinsics = [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ]
    # world_to_cam: Camera at (2, -3, 1.25), looking +Y.
    # In camera frame: +X is right, +Y is down, +Z is forward.
    # World +X -> Cam +X; World -Z -> Cam +Y; World +Y -> Cam +Z
    world_to_cam = [
        [1.0, 0.0, 0.0, -2.0],
        [0.0, 0.0, -1.0, 1.25],
        [0.0, 1.0, 0.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return PinholeCamera(
        camera_id="cam-front-01",
        intrinsics=intrinsics,
        world_to_cam=world_to_cam,
        image_rgb=img,
    )


def test_mesh_texture_projector_generates_uvs_and_texture_map(sample_wall_mesh, front_facing_camera):
    projector = MeshTextureProjector(texture_width=256, texture_height=256)
    res = projector.project_texture(sample_wall_mesh, [front_facing_camera])

    assert isinstance(res, TextureResult)
    # Per-vertex UVs
    assert len(res.uvs) == len(sample_wall_mesh.vertices)
    for u, v in res.uvs:
        assert 0.0 <= u <= 1.0
        assert 0.0 <= v <= 1.0

    # Image properties
    assert res.image_rgb.shape == (256, 256, 3)
    assert len(res.png_bytes) > 0
    # PNG magic bytes check
    assert res.png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    # Surface coverage
    assert res.coverage > 0.3
    assert "cam-front-01" in res.cameras_used


def test_apply_texture_projection_to_world(sample_wall_mesh, front_facing_camera):
    store = MemoryArtifactStore()
    mesh_uri, _ = store.put(sample_wall_mesh.to_bytes())

    geom = Geometry(id="geom-wall-1", type=GeometryType.MESH, data_uri=mesh_uri)
    entity = Entity(id="ent-wall-1", name="LivingRoomWall", geometry_ids=["geom-wall-1"])
    entity.transform = {"position": {"x": 0.0, "y": 0.0, "z": 0.0}}

    world = WorldIR(
        id="world-textured",
        entities={"ent-wall-1": entity},
        geometries={"geom-wall-1": geom},
    )

    projector = MeshTextureProjector(texture_width=128, texture_height=128)
    tex_res = projector.project_texture(sample_wall_mesh, [front_facing_camera])

    apply_texture_projection(world, "ent-wall-1", tex_res, artifact_store=store)

    # Material created and wired
    assert entity.material_id is not None
    assert entity.material_id in world.materials
    mat = world.materials[entity.material_id]
    assert "texture_bytes" in mat.properties
    assert "texture_base64" in mat.properties
    assert "texture_uri" in mat.properties
    assert store.get(mat.properties["texture_uri"]) == tex_res.png_bytes

    # Mesh in store updated with UVs
    updated_geom = world.geometries["geom-wall-1"]
    updated_mesh_bytes = store.get(updated_geom.data_uri)
    updated_mesh = MeshData.from_bytes(updated_mesh_bytes)
    assert updated_mesh.uvs is not None
    assert len(updated_mesh.uvs) == len(sample_wall_mesh.vertices)


def test_gltf_and_glb_texture_export(sample_wall_mesh, front_facing_camera):
    store = MemoryArtifactStore()
    mesh_uri, _ = store.put(sample_wall_mesh.to_bytes())

    geom = Geometry(id="geom-wall-1", type=GeometryType.MESH, data_uri=mesh_uri)
    entity = Entity(id="ent-wall-1", name="Wall", geometry_ids=["geom-wall-1"])
    entity.transform = {"position": {"x": 0.0, "y": 0.0, "z": 0.0}}

    world = WorldIR(
        id="world-gltf-test",
        entities={"ent-wall-1": entity},
        geometries={"geom-wall-1": geom},
    )

    projector = MeshTextureProjector(texture_width=128, texture_height=128)
    tex_res = projector.project_texture(sample_wall_mesh, [front_facing_camera])
    apply_texture_projection(world, "ent-wall-1", tex_res, artifact_store=store)

    # Export to glTF
    gltf = export_to_gltf(world, artifact_store=store)
    assert "images" in gltf
    assert len(gltf["images"]) >= 1
    assert "textures" in gltf
    assert len(gltf["textures"]) >= 1
    assert "samplers" in gltf

    # Verify material points to texture
    mat = gltf["materials"][0]
    assert "pbrMetallicRoughness" in mat
    assert "baseColorTexture" in mat["pbrMetallicRoughness"]
    assert mat["pbrMetallicRoughness"]["baseColorTexture"]["index"] == 0

    # Verify primitive has TEXCOORD_0
    mesh_prim = gltf["meshes"][1]["primitives"][0]  # mesh 0 is cube, mesh 1 is real mesh
    assert "TEXCOORD_0" in mesh_prim["attributes"]

    # Export to GLB binary
    glb_bytes = export_to_glb(world, artifact_store=store)
    assert len(glb_bytes) > 1000

    # Parse GLB header
    magic, version, total_len = struct.unpack_from("<4sII", glb_bytes, 0)
    assert magic == b"glTF"
    assert version == 2
    assert total_len == len(glb_bytes)

    # Chunk 0: JSON
    json_len, json_type = struct.unpack_from("<II", glb_bytes, 12)
    assert json_type == 0x4E4F534A
    json_bytes = glb_bytes[20:20 + json_len]
    import json
    parsed_json = json.loads(json_bytes.decode("utf-8"))
    assert "images" in parsed_json
    assert "textures" in parsed_json

    # Chunk 1: BIN containing embedded texture PNG
    bin_offset = 20 + json_len
    bin_len, bin_type = struct.unpack_from("<II", glb_bytes, bin_offset)
    assert bin_type == 0x004E4942
    bin_payload = glb_bytes[bin_offset + 8:bin_offset + 8 + bin_len]

    # Check that PNG magic exists within the GLB binary buffer
    assert b"\x89PNG\r\n\x1a\n" in bin_payload
