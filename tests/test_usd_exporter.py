"""Tests for the minimal USD ASCII (.usda) exporter (exporters/usd/exporter.py)."""

import os
import re
import tempfile

from world_ir.artifact_store import MemoryArtifactStore
from world_ir.geometry_data import PointCloudData
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType
from world_ir.world_v1 import WorldIR
from exporters.usd.exporter import export_to_usda, write_usda_file, _sanitize_prim_name, validate_usda


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


# ---------------------------------------------------------------- real geometry (Mesh prim)


def _entity_with_real_mesh(entity_id: str, geom_id: str, vertices, faces, origin, store):
    from reconstruction.meshing.mesh import MeshData

    payload = MeshData(vertices=tuple(vertices), faces=tuple(faces)).to_bytes()
    uri, digest = store.put(payload)
    geom = Geometry(id=geom_id, type=GeometryType.MESH, data_uri=uri, data_hash=digest)
    entity = Entity(
        id=entity_id,
        name="Meshed",
        type=EntityType.DEBRIS,
        transform={"position": {"x": origin[0], "y": origin[1], "z": origin[2]}},
        geometry_ids=[geom_id],
    )
    return geom, entity


def test_resolvable_mesh_geometry_emits_a_real_mesh_prim_not_a_cube():
    store = MemoryArtifactStore()
    world = WorldIR()
    vertices = [(1.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, 1.0, 1.0)]
    faces = [(0, 1, 2)]
    geom, entity = _entity_with_real_mesh(
        "ent-mesh", "geom-mesh", vertices, faces, (1.0, 0.0, 0.0), store
    )
    world.geometries[geom.id] = geom
    world.entities[entity.id] = entity

    usda = export_to_usda(world, artifact_store=store)

    assert 'def Mesh "ent_mesh"' in usda
    assert "def Cube" not in usda
    assert "def Points" not in usda
    assert "int[] faceVertexCounts = [3]" in usda
    assert "int[] faceVertexIndices = [0, 1, 2]" in usda
    # local-space translation: vertices relative to entity origin (1,0,0)
    assert "(0.0, 0.0, 0.0)" in usda
    assert "(1.0, 1.0, 0.0)" in usda
    assert "(0.0, 1.0, 1.0)" in usda


def test_mesh_geometry_without_resolvable_artifact_falls_back_to_cube():
    """A MESH-typed geometry is exportable, but without a resolvable
    artifact it must not fabricate a mesh -- placeholder cube, exactly
    like BOX/PLANE."""
    store = MemoryArtifactStore()  # empty store: data_uri won't resolve
    world = WorldIR()
    geom = Geometry(id="geom-mesh", type=GeometryType.MESH, data_uri="artifact://" + "0" * 64)
    entity = Entity(
        id="ent-mesh",
        name="Meshed",
        type=EntityType.DEBRIS,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        geometry_ids=["geom-mesh"],
    )
    world.geometries[geom.id] = geom
    world.entities[entity.id] = entity

    usda = export_to_usda(world, artifact_store=store)
    assert "def Mesh" not in usda
    assert "def Points" not in usda
    assert usda.count("def Cube") == 1


# ---------------------------------------------------------------- Phase 9 Task 9.6 tests


def test_usd_materials_and_usdshade_bindings():
    from world_ir.schema_v1 import Material

    world = WorldIR()
    mat = Material(
        id="mat-wood",
        name="OakWood",
        color_rgb=(0.6, 0.4, 0.2),
        properties={"roughness": 0.45, "metallic": 0.1},
    )
    world.materials[mat.id] = mat

    box_geom = Geometry(id="geom-box", type=GeometryType.BOX)
    box_entity = Entity(
        id="ent-box",
        name="OakTable",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 1.0, "y": 2.0, "z": 0.0}},
        geometry_ids=["geom-box"],
        material_ids=["mat-wood"],
    )
    world.geometries[box_geom.id] = box_geom
    world.entities[box_entity.id] = box_entity

    usda = export_to_usda(world)
    assert 'def Scope "Materials"' in usda
    assert 'def Material "OakWood"' in usda
    assert 'def Shader "PBRShader"' in usda
    assert 'uniform token info:id = "UsdPreviewSurface"' in usda
    assert "color3f inputs:diffuseColor = (0.6, 0.4, 0.2)" in usda
    assert "float inputs:roughness = 0.45" in usda
    assert "float inputs:metallic = 0.1" in usda
    assert "rel material:binding = </World/Materials/OakWood>" in usda

    valid, errors = validate_usda(usda)
    assert valid, f"Validation failed: {errors}"


def test_usd_stage_hierarchy_nesting():
    world = WorldIR()

    bldg = Entity(
        id="bldg-1",
        name="Building_A",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
    )
    story = Entity(
        id="story-1",
        name="GroundFloor",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        custom_properties={"parent_id": "bldg-1"},
    )
    room = Entity(
        id="room-101",
        name="Lobby",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 2.0, "y": 0.0, "z": 0.0}},
        custom_properties={"parent_id": "story-1"},
    )

    box_geom = Geometry(id="geom-wall", type=GeometryType.PLANE)
    wall = Entity(
        id="wall-south",
        name="SouthWall",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 2.0, "y": -3.0, "z": 1.5}},
        geometry_ids=["geom-wall"],
        custom_properties={"parent_id": "room-101"},
    )

    world.geometries[box_geom.id] = box_geom
    world.entities[bldg.id] = bldg
    world.entities[story.id] = story
    world.entities[room.id] = room
    world.entities[wall.id] = wall

    usda = export_to_usda(world)

    # Validate hierarchy nesting
    assert 'def Xform "bldg_1"' in usda
    assert 'def Xform "story_1"' in usda
    assert 'def Xform "room_101"' in usda
    assert 'def Cube "wall_south"' in usda

    # Verify structural ordering: Building opens before Story, Story before Room, Room before Wall
    idx_bldg = usda.index('def Xform "bldg_1"')
    idx_story = usda.index('def Xform "story_1"')
    idx_room = usda.index('def Xform "room_101"')
    idx_wall = usda.index('def Cube "wall_south"')
    assert idx_bldg < idx_story < idx_room < idx_wall

    valid, errors = validate_usda(usda)
    assert valid, f"Validation failed: {errors}"


def test_usd_mesh_with_normals_and_geom_subsets():
    from reconstruction.meshing.mesh import MeshData

    store = MemoryArtifactStore()
    world = WorldIR()

    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    faces = [(0, 1, 2)]
    normals = [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]

    mesh_payload = MeshData(
        vertices=tuple(vertices),
        faces=tuple(faces),
        normals=tuple(normals),
    ).to_bytes()

    uri, digest = store.put(mesh_payload)
    geom = Geometry(
        id="geom-mesh",
        type=GeometryType.MESH,
        data_uri=uri,
        data_hash=digest,
    )
    geom.subsets = [
        {"name": "front_faces", "indices": [0], "material_path": "/World/Materials/OakWood"}
    ]
    entity = Entity(
        id="ent-mesh",
        name="WallPanel",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 0.0}},
        geometry_ids=["geom-mesh"],
    )

    world.geometries[geom.id] = geom
    world.entities[entity.id] = entity

    usda = export_to_usda(world, artifact_store=store)
    assert 'def Mesh "ent_mesh"' in usda
    assert "normal3f[] normals = [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]" in usda
    assert 'def GeomSubset "front_faces"' in usda
    assert 'uniform token elementType = "face"' in usda
    assert "int[] indices = [0]" in usda
    assert "rel material:binding = </World/Materials/OakWood>" in usda

    valid, errors = validate_usda(usda)
    assert valid, f"Validation failed: {errors}"


def test_usd_physics_rigid_body_schema():
    world = WorldIR()
    box_geom = Geometry(id="geom-box", type=GeometryType.BOX)
    entity = Entity(
        id="ent-physics-crate",
        name="HeavyCrate",
        type=EntityType.STRUCTURE,
        transform={"position": {"x": 0.0, "y": 0.0, "z": 1.0}},
        geometry_ids=["geom-box"],
        custom_properties={"mass": 42.5, "rigid_body": True, "collision": True},
    )
    world.geometries[box_geom.id] = box_geom
    world.entities[entity.id] = entity

    usda = export_to_usda(world)
    assert 'prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]' in usda
    assert "bool physics:rigidBodyEnabled = true" in usda
    assert "bool physics:collisionEnabled = true" in usda
    assert "float physics:mass = 42.5" in usda

    valid, errors = validate_usda(usda)
    assert valid, f"Validation failed: {errors}"


def test_usd_camera_prim_export():
    world = WorldIR()
    cam_entity = Entity(
        id="cam-front-01",
        name="FrontCamera",
        type=EntityType.SENSOR,
        transform={
            "position": {"x": 0.0, "y": -5.0, "z": 1.7},
            "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
        },
        custom_properties={
            "sensor_type": "camera",
            "focal_length": 50.0,
            "horizontal_aperture": 36.0,
            "vertical_aperture": 24.0,
        },
    )
    world.entities[cam_entity.id] = cam_entity

    usda = export_to_usda(world)
    assert 'def Camera "cam_front_01"' in usda
    assert "float focalLength = 50.0" in usda
    assert "float horizontalAperture = 36.0" in usda
    assert "float verticalAperture = 24.0" in usda
    assert "double3 xformOp:translate = (0.0, -5.0, 1.7)" in usda
    assert "quatf xformOp:orient = (1.0, 0.0, 0.0, 0.0)" in usda
    assert 'uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]' in usda

    valid, errors = validate_usda(usda)
    assert valid, f"Validation failed: {errors}"


def test_validate_usda_syntax_and_catches_errors():
    valid_usda = '#usda 1.0\n\ndef Xform "World"\n{\n    def Cube "Box"\n    {\n        double size = 2\n    }\n}\n'
    valid, errors = validate_usda(valid_usda)
    assert valid
    assert len(errors) == 0

    # Test error 1: Missing header
    invalid_header = 'def Xform "World"\n{\n}\n'
    valid, errors = validate_usda(invalid_header)
    assert not valid
    assert any("Missing required header" in e for e in errors)

    # Test error 2: Unbalanced braces
    invalid_braces = '#usda 1.0\n\ndef Xform "World"\n{\n'
    valid, errors = validate_usda(invalid_braces)
    assert not valid
    assert any("Unbalanced braces" in e for e in errors)

    # Test error 3: Invalid prim identifier
    invalid_prim = '#usda 1.0\n\ndef Xform "123bad name"\n{\n}\n'
    valid, errors = validate_usda(invalid_prim)
    assert not valid
    assert any("Invalid prim" in e for e in errors)
