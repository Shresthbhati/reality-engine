"""Tests for Object-Level 3D Geometry Reconstruction (Wave 1, Block 4)."""

from __future__ import annotations

import numpy as np
import pytest

from perception.instances.bounding import OrientedBoundingBox
from perception.instances.object_geometry import (
    ObjectMesh,
    attach_object_mesh_to_world,
    build_chair_mesh,
    build_door_mesh,
    build_sofa_mesh,
    build_switchboard_mesh,
    build_table_mesh,
    reconstruct_object_mesh,
)
from world_ir.schema_v1 import Entity, EntityType, GeometryType
from world_ir.world_v1 import WorldIR


def test_sofa_mesh_generation_and_subparts():
    obb = OrientedBoundingBox(
        center=(0.0, 0.45, 0.0),
        extents=(2.10, 0.90, 0.85),
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        yaw_rad=0.0,
        class_id="furniture.sofa",
    )
    mesh = build_sofa_mesh(obb)
    assert isinstance(mesh, ObjectMesh)
    assert mesh.vertex_count == 32
    assert mesh.triangle_count == 48
    assert "seat_base" in mesh.subparts
    assert "backrest" in mesh.subparts
    assert "armrest_left" in mesh.subparts
    assert "armrest_right" in mesh.subparts

    # Verify normal vectors are unit length
    lengths = np.linalg.norm(mesh.normals, axis=1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


def test_chair_mesh_generation():
    obb = OrientedBoundingBox(
        center=(1.5, 0.45, -2.0),
        extents=(0.50, 0.90, 0.50),
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        yaw_rad=0.0,
        class_id="furniture.chair",
    )
    mesh = build_chair_mesh(obb)
    assert "seat" in mesh.subparts
    assert "backrest" in mesh.subparts
    assert "leg_fl" in mesh.subparts
    assert "leg_br" in mesh.subparts
    # 6 subparts * 8 vertices = 48 vertices
    assert mesh.vertex_count == 48
    assert mesh.triangle_count == 72


def test_table_mesh_generation():
    obb = OrientedBoundingBox(
        center=(0.0, 0.38, 0.0),
        extents=(1.60, 0.76, 0.90),
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        yaw_rad=0.0,
        class_id="furniture.table",
    )
    mesh = build_table_mesh(obb)
    assert "tabletop" in mesh.subparts
    assert len(mesh.subparts) == 5  # tabletop + 4 legs


def test_door_mesh_generation():
    obb = OrientedBoundingBox(
        center=(2.0, 1.05, 0.0),
        extents=(0.90, 2.10, 0.10),
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        yaw_rad=0.0,
        class_id="opening.door",
    )
    mesh = build_door_mesh(obb)
    assert "jamb_left" in mesh.subparts
    assert "jamb_right" in mesh.subparts
    assert "lintel" in mesh.subparts
    assert "leaf_panel" in mesh.subparts


def test_switchboard_mesh_generation():
    obb = OrientedBoundingBox(
        center=(0.0, 1.20, 0.0),
        extents=(0.16, 0.09, 0.02),
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        yaw_rad=0.0,
        class_id="electrical.switchboard",
    )
    mesh = build_switchboard_mesh(obb, module_count=4)
    assert "faceplate" in mesh.subparts
    assert "switch_1" in mesh.subparts
    assert "switch_4" in mesh.subparts


def test_attach_object_mesh_to_world_ir():
    world = WorldIR()
    ent = Entity(
        id="ent-chair-01",
        name="Study Chair",
        type=EntityType.STRUCTURE,
    )
    world.entities[ent.id] = ent

    obb = OrientedBoundingBox(
        center=(0.0, 0.45, 0.0),
        extents=(0.50, 0.90, 0.50),
        rotation=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        yaw_rad=0.0,
        class_id="furniture.chair",
    )
    mesh = reconstruct_object_mesh(obb)
    geom = attach_object_mesh_to_world(world, ent.id, mesh)

    assert geom.id in world.geometries
    assert geom.type == GeometryType.MESH
    assert geom.vertex_count == mesh.vertex_count
    assert geom.triangle_count == mesh.triangle_count
    assert geom.id in ent.geometry_ids
    assert ent.custom_properties["reconstruction_level"] == "category_parametric_mesh"
    assert "seat" in ent.custom_properties["mesh_subparts"]
