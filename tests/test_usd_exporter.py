"""Tests for the minimal USD ASCII (.usda) exporter (exporters/usd/exporter.py)."""

import os
import re
import tempfile

from world_ir.artifact_store import MemoryArtifactStore
from world_ir.geometry_data import PointCloudData
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType
from world_ir.world_v1 import WorldIR
from exporters.usd.exporter import export_to_usda, write_usda_file, _sanitize_prim_name


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
    #    (nothing to place the prim at).
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
    usda = export_to_usda(world)

    assert usda.startswith("#usda 1.0")
    assert usda.count("def Cube") == 1

    match = re.search(r'double3 xformOp:translate = \(([^)]+)\)', usda)
    assert match is not None
    values = [float(v.strip()) for v in match.group(1).split(",")]
    assert values == [1.0, 2.0, 3.0]


def test_sanitize_prim_name():
    assert _sanitize_prim_name("wall-017") == "wall_017"
    assert _sanitize_prim_name("wall_017")[0].isalpha() or _sanitize_prim_name("wall_017")[0] == "_"
    sanitized_digit_start = _sanitize_prim_name("017wall")
    assert not sanitized_digit_start[0].isdigit()
    assert sanitized_digit_start == "_017wall"


def test_write_usda_file_matches_export_string():
    world = _build_world()
    expected = export_to_usda(world)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "world.usda")
        write_usda_file(world, path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

    assert content == expected


# ---------------------------------------------------------------- real geometry (Points prim)


def _entity_with_real_points(entity_id: str, geom_id: str, points, origin, store):
    payload = PointCloudData(points=tuple(points)).to_bytes()
    uri, digest = store.put(payload)
    geom = Geometry(id=geom_id, type=GeometryType.BOX, data_uri=uri, data_hash=digest)
    entity = Entity(
        id=entity_id,
        name="Real",
        type=EntityType.DEBRIS,
        transform={"position": {"x": origin[0], "y": origin[1], "z": origin[2]}},
        geometry_ids=[geom_id],
    )
    return geom, entity


def test_resolvable_geometry_emits_a_real_points_prim_not_a_cube():
    store = MemoryArtifactStore()
    world = WorldIR()
    points = [(1.0, 1.0, 1.0), (2.0, 3.0, 4.0)]
    geom, entity = _entity_with_real_points("ent-real", "geom-real", points, (5.0, 0.0, 0.0), store)
    world.geometries[geom.id] = geom
    world.entities[entity.id] = entity

    usda = export_to_usda(world, artifact_store=store)

    assert "def Points" in usda
    assert "def Cube" not in usda
    assert "point3f[] points" in usda
    # local-space translation: points relative to entity origin (5,0,0)
    assert "(-4.0, 1.0, 1.0)" in usda
    assert "(-3.0, 3.0, 4.0)" in usda


def test_no_store_falls_back_to_cube():
    world = _build_world()
    usda = export_to_usda(world)  # artifact_store omitted
    assert "def Points" not in usda
    assert usda.count("def Cube") == 1


def test_geometry_without_data_uri_falls_back_to_cube_even_with_store():
    store = MemoryArtifactStore()
    world = _build_world()
    usda = export_to_usda(world, artifact_store=store)
    assert "def Points" not in usda
    assert usda.count("def Cube") == 1


def test_unresolvable_artifact_falls_back_to_cube():
    store = MemoryArtifactStore()
    world = WorldIR()
    geom = Geometry(id="geom-missing", type=GeometryType.BOX, data_uri="artifact://" + "0" * 64)
    entity = Entity(
        id="ent-missing",
        name="Missing",
        type=EntityType.DEBRIS,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        geometry_ids=["geom-missing"],
    )
    world.geometries[geom.id] = geom
    world.entities[entity.id] = entity

    usda = export_to_usda(world, artifact_store=store)
    assert "def Points" not in usda
    assert usda.count("def Cube") == 1


def test_backward_compat_no_artifact_store_arg_reproduces_exact_output():
    """Regression: callers that never pass artifact_store must see
    byte-identical output to before this parameter existed."""
    world = _build_world()
    assert export_to_usda(world) == export_to_usda(world, artifact_store=None)


def test_write_usda_file_threads_artifact_store():
    store = MemoryArtifactStore()
    world = WorldIR()
    points = [(0.0, 0.0, 0.0)]
    geom, entity = _entity_with_real_points("ent-real", "geom-real", points, (0.0, 0.0, 0.0), store)
    world.geometries[geom.id] = geom
    world.entities[entity.id] = entity

    expected = export_to_usda(world, artifact_store=store)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "world.usda")
        write_usda_file(world, path, artifact_store=store)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    assert content == expected
    assert "def Points" in content
