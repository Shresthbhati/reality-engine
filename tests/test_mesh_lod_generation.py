"""Tests for Multi-Resolution LOD Generation (Wave 5, Block 34)."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from reconstruction.meshing.lod import (
    DEFAULT_LOD_TIERS,
    LODTier,
    attach_lods_to_world,
    generate_mesh_lods,
)
from reconstruction.meshing.mesh import MeshData
from reconstruction.meshing.repair import trimesh_to_mesh
from world_ir.schema_v1 import Entity, EntityType, GeometryType
from world_ir.world_v1 import WorldIR


def test_mesh_lod_generation_decimation():
    # Create an icosphere with 320 faces (subdivisions=2)
    ico = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    mesh = trimesh_to_mesh(ico)
    orig_faces = len(mesh.faces)
    assert orig_faces >= 300

    tiers = (
        LODTier(lod_level=0, name="LOD0", target_reduction_ratio=1.0),
        LODTier(lod_level=1, name="LOD1", target_reduction_ratio=0.50),
        LODTier(lod_level=2, name="LOD2", target_reduction_ratio=0.20),
    )

    lods = generate_mesh_lods(mesh, tiers=tiers)

    assert 0 in lods
    assert 1 in lods
    assert 2 in lods

    # LOD0 is original
    assert len(lods[0].faces) == orig_faces
    # LOD1 has fewer faces than LOD0
    assert len(lods[1].faces) < orig_faces
    # LOD2 has fewer faces than LOD1
    assert len(lods[2].faces) < len(lods[1].faces)


def test_attach_lods_to_world_ir():
    world = WorldIR()
    ent = Entity(id="ent-model-1", name="Building Complex", type=EntityType.BUILDING)
    world.entities[ent.id] = ent

    ico = trimesh.creation.icosphere(subdivisions=2, radius=2.0)
    mesh = trimesh_to_mesh(ico)
    lods = generate_mesh_lods(mesh)

    geoms = attach_lods_to_world(world, ent.id, lods)

    assert len(geoms) == len(lods)
    for g in geoms:
        assert g.id in world.geometries
        assert g.id in ent.geometry_ids
        assert g.type == GeometryType.MESH
        assert g.bounds_min is not None
        assert g.bounds_max is not None

    assert ent.custom_properties["has_lods"] is True
    assert 0 in ent.custom_properties["lod_levels"]
    assert 1 in ent.custom_properties["lod_levels"]
