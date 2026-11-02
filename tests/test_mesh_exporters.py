"""Tests for standard 3D mesh exporters: OBJ, PLY, and STL (Task 9.3).
"""

from __future__ import annotations

import struct
from provenance import Provenance
from reconstruction.meshing.mesh import MeshData
from sdk import reality
from world_ir.artifact_store import MemoryArtifactStore
from world_ir import WorldIR
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Material,
    Vector3,
)
from exporters.mesh import (
    export_to_obj,
    export_to_obj_with_report,
    export_to_ply,
    export_to_ply_with_report,
    export_to_stl,
    export_to_stl_with_report,
)


def _build_test_world(store: MemoryArtifactStore) -> WorldIR:
    world = WorldIR(id="w_mesh", name="Mesh Test World", version="v1")

    # 1. Box entity
    g_box = Geometry(
        id="geom_box",
        type=GeometryType.BOX,
        bounds_min=Vector3(0, 0, 0),
        bounds_max=Vector3(1, 2, 3),
    )
    mat_wood = Material(
        id="mat_wood",
        name="Oak Wood",
        color_rgb=[200, 150, 100],
    )
    e_box = Entity(
        id="ent_box",
        name="Cabinet",
        type=EntityType.STRUCTURE,
        provenance=Provenance.OBSERVED,
        transform={"position": {"x": 5.0, "y": 0.0, "z": 0.0}},
        geometry_ids=["geom_box"],
        custom_properties={"material_id": "mat_wood"},
    )

    # 2. Reconstructed MeshData entity
    mesh_payload = MeshData(
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        faces=((0, 1, 2),),
    ).to_bytes()
    uri, _ = store.put(mesh_payload)

    g_mesh = Geometry(
        id="geom_mesh",
        type=GeometryType.MESH,
        data_uri=uri,
    )
    e_mesh = Entity(
        id="ent_mesh",
        name="Sculpture",
        type=EntityType.STRUCTURE,
        provenance=Provenance.OBSERVED,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        geometry_ids=["geom_mesh"],
        custom_properties={"appearance": {"color_rgb": [50, 100, 200]}},
    )

    world.geometries[g_box.id] = g_box
    world.geometries[g_mesh.id] = g_mesh
    world.materials[mat_wood.id] = mat_wood
    world.entities[e_box.id] = e_box
    world.entities[e_mesh.id] = e_mesh

    return world


def test_obj_export_with_mtl():
    store = MemoryArtifactStore()
    world = _build_test_world(store)

    obj_str, mtl_str = export_to_obj(world, store)

    assert "mtllib world.mtl" in obj_str
    assert "o Cabinet" in obj_str
    assert "o Sculpture" in obj_str
    assert "usemtl Oak Wood" in obj_str
    assert "usemtl mat_ent_mesh" in obj_str

    # Vertices and faces
    assert "v 5.0000 0.0000 0.0000" in obj_str
    assert "f " in obj_str

    # MTL definitions
    assert "newmtl Oak Wood" in mtl_str
    assert "Kd 0.7843 0.5882 0.3922" in mtl_str  # 200/255, 150/255, 100/255
    assert "newmtl mat_ent_mesh" in mtl_str

    # Report
    obj_out, report = export_to_obj_with_report(world, store)
    assert report.format == "obj"
    assert "ent_box" in report.entities_exported
    assert "ent_mesh" in report.entities_exported


def test_ply_export_binary_and_ascii():
    store = MemoryArtifactStore()
    world = _build_test_world(store)

    # 1. Binary PLY
    bin_ply = export_to_ply(world, store, binary=True)
    assert bin_ply.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert b"element vertex 11\n" in bin_ply  # 8 from box + 3 from mesh
    assert b"element face 13\n" in bin_ply    # 12 from box + 1 from mesh
    assert b"end_header\n" in bin_ply

    # 2. ASCII PLY
    ascii_ply = export_to_ply(world, store, binary=False).decode("utf-8")
    assert ascii_ply.startswith("ply\nformat ascii 1.0\n")
    assert "element vertex 11\n" in ascii_ply
    assert "element face 13\n" in ascii_ply
    assert "3 8 9 10\n" in ascii_ply or "3 0 1 2\n" in ascii_ply

    # Report
    ply_out, report = export_to_ply_with_report(world, store)
    assert report.format == "ply"
    assert len(report.entities_exported) == 2


def test_stl_export_binary_and_ascii():
    store = MemoryArtifactStore()
    world = _build_test_world(store)

    # 1. Binary STL
    bin_stl = export_to_stl(world, store, binary=True)
    assert len(bin_stl) >= 84
    header = bin_stl[:80]
    assert b"Reality Engine STL Export" in header
    tri_count = struct.unpack("<I", bin_stl[80:84])[0]
    assert tri_count == 13  # 12 from box + 1 from mesh
    assert len(bin_stl) == 84 + (13 * 50)

    # 2. ASCII STL
    ascii_stl = export_to_stl(world, store, binary=False).decode("utf-8")
    assert ascii_stl.startswith("solid ")
    assert "facet normal" in ascii_stl
    assert "outer loop" in ascii_stl
    assert "endsolid" in ascii_stl

    # Report
    stl_out, report = export_to_stl_with_report(world, store)
    assert report.format == "stl"
    assert len(report.entities_exported) == 2


def test_sdk_mesh_exports_integration():
    store = MemoryArtifactStore()
    world = _build_test_world(store)

    # Export via SDK for all 3 formats
    obj_data, r_obj = reality.export(world, "obj", store)
    assert r_obj.format == "obj"
    assert isinstance(obj_data, str)
    assert "v " in obj_data

    ply_data, r_ply = reality.export(world, "ply", store)
    assert r_ply.format == "ply"
    assert isinstance(ply_data, bytes)
    assert ply_data.startswith(b"ply\n")

    stl_data, r_stl = reality.export(world, "stl", store)
    assert r_stl.format == "stl"
    assert isinstance(stl_data, bytes)
    assert b"Reality Engine STL Export" in stl_data[:80]
