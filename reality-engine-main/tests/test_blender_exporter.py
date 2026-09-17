"""Tests for the Blender import-script exporter (exporters/blender/exporter.py)."""

import ast
import os
import tempfile

from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR
from provenance import Provenance
from exporters.blender.exporter import export_to_blender_script, write_blender_script


def _build_world() -> WorldIR:
    world = WorldIR()

    # 1. BOX geometry, no bounds -> falls back to the (2, 2, 2) default cube.
    box_geom = Geometry(id="geom-box", type=GeometryType.BOX)
    box_entity = Entity(
        id="ent-box",
        name="Crate",
        type=EntityType.DEBRIS,
        transform={"position": {"x": 1.0, "y": 2.0, "z": 3.0}},
        geometry_ids=["geom-box"],
        provenance=Provenance.OBSERVED,
        confidence=0.9,
    )

    # 2. PLANE geometry with real bounds (as evidence/promote_planes.py
    #    always sets) -> dimensions must come from the AABB, not a default.
    plane_geom = Geometry(
        id="geom-wall",
        type=GeometryType.PLANE,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(4.0, 2.5, 0.2),
    )
    wall_entity = Entity(
        id="ent-wall",
        name="Wall North",
        type=EntityType.WALL,
        transform={"position": {"x": 2.0, "y": 1.25, "z": 0.1}},
        geometry_ids=["geom-wall"],
        provenance=Provenance.INFERRED,
        confidence=0.8,
    )

    # 3. Degenerate (zero-thickness) plane bounds -> every axis floors to
    #    _MIN_DIMENSION_M rather than exporting an invisible mesh.
    thin_geom = Geometry(
        id="geom-floor",
        type=GeometryType.PLANE,
        bounds_min=Vector3(0.0, 0.0, 0.0),
        bounds_max=Vector3(3.0, 0.0, 3.0),
    )
    floor_entity = Entity(
        id="ent-floor",
        name="Floor",
        type=EntityType.FLOOR,
        transform={"position": {"x": 1.5, "y": 0.0, "z": 1.5}},
        geometry_ids=["geom-floor"],
    )

    # 4. POINTCLOUD geometry -> no real vertex data, must be skipped.
    pc_geom = Geometry(id="geom-pc", type=GeometryType.POINTCLOUD, vertex_count=500)
    pc_entity = Entity(
        id="ent-pc",
        name="Scan",
        transform={"position": {"x": 5.0, "y": 5.0, "z": 5.0}},
        geometry_ids=["geom-pc"],
    )

    # 5. BOX geometry but no transform -> nothing to place it at.
    box_geom2 = Geometry(id="geom-box2", type=GeometryType.BOX)
    no_transform_entity = Entity(
        id="ent-no-transform",
        name="Untethered",
        geometry_ids=["geom-box2"],
    )

    for g in (box_geom, plane_geom, thin_geom, pc_geom, box_geom2):
        world.geometries[g.id] = g
    for e in (box_entity, wall_entity, floor_entity, pc_entity, no_transform_entity):
        world.entities[e.id] = e

    return world


def test_script_is_valid_python():
    world = _build_world()
    script = export_to_blender_script(world)
    ast.parse(script)  # raises SyntaxError if the generated script is broken


def test_only_placeable_geometry_backed_entities_exported():
    world = _build_world()
    script = export_to_blender_script(world)

    assert script.count("primitive_cube_add") == 3  # box, wall, floor
    assert "ent-pc" not in script  # pointcloud: no real vertex data
    assert "ent-no-transform" not in script  # no placement


def test_box_without_bounds_uses_default_dimensions():
    world = _build_world()
    script = export_to_blender_script(world)

    assert "obj.name = 'Crate'" in script
    idx = script.index("obj.name = 'Crate'")
    segment = script[idx:idx + 400]
    assert "obj.dimensions = (2.0, 2.0, 2.0)" in segment


def test_plane_dimensions_come_from_real_bounds():
    world = _build_world()
    script = export_to_blender_script(world)

    assert "obj.name = 'Wall North'" in script
    idx = script.index("obj.name = 'Wall North'")
    segment = script[idx:idx + 400]
    assert "obj.dimensions = (4.0, 2.5, 0.2)" in segment
    assert "obj.location" not in segment  # location comes from primitive_cube_add, not a separate assignment


def test_degenerate_plane_axis_is_floored_not_zero():
    world = _build_world()
    script = export_to_blender_script(world)

    idx = script.index("obj.name = 'Floor'")
    segment = script[idx:idx + 400]
    assert "obj.dimensions = (3.0, 0.01, 3.0)" in segment


def test_custom_properties_carry_real_worldir_metadata():
    world = _build_world()
    script = export_to_blender_script(world)

    idx = script.index("obj.name = 'Wall North'")
    segment = script[idx:idx + 600]
    assert "obj['entity_id'] = 'ent-wall'" in segment
    assert "obj['entity_type'] = 'wall'" in segment
    assert "obj['provenance'] = 'INFERRED'" in segment
    assert "obj['confidence'] = 0.8" in segment


def test_write_blender_script_matches_export_string():
    world = _build_world()
    expected = export_to_blender_script(world)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "scene.py")
        write_blender_script(world, path)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

    assert content == expected


def test_empty_world_produces_valid_script_with_zero_entities():
    world = WorldIR()
    script = export_to_blender_script(world)

    ast.parse(script)
    assert "primitive_cube_add" not in script
    assert "# 0 entity object(s) exported." in script
