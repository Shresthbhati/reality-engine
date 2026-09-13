"""Tests for the WorldIR -> Physics compiler bridge (engine/compiler/physics_compiler.py)."""

from __future__ import annotations

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR

from engine.compiler.physics_compiler import (
    PhysicsCompileStatus,
    compile_entity_physics,
    compile_physics_world,
)
from engine.physics.materials.material import CANONICAL_MATERIALS


def _entity_with_bounds(entity_id, entity_type, bmin, bmax, **kwargs):
    geom = Geometry(id=f"geom-{entity_id}", type=GeometryType.BOX, bounds_min=Vector3(*bmin), bounds_max=Vector3(*bmax))
    entity = Entity(id=entity_id, type=entity_type, geometry_ids=[geom.id], **kwargs)
    return entity, geom


def test_static_wall_gets_zero_mass_and_inferred_provenance():
    entity, geom = _entity_with_bounds("ent-wall", EntityType.WALL, (0, 0, 0), (4, 2.5, 0.2), confidence=0.8)
    world = WorldIR()
    world.entities[entity.id] = entity
    world.geometries[geom.id] = geom

    result = compile_entity_physics(entity, world)
    assert result.status is PhysicsCompileStatus.COMPILED
    assert result.body.mass == 0.0
    assert result.body.is_static
    assert result.mass_provenance is Provenance.INFERRED


def test_dynamic_entity_mass_derived_from_density_and_volume():
    entity, geom = _entity_with_bounds(
        "ent-crate", EntityType.DEBRIS, (0, 0, 0), (1.0, 1.0, 1.0),
        confidence=0.9, custom_properties={"physics_material": "wood"},
    )
    world = WorldIR()
    world.entities[entity.id] = entity
    world.geometries[geom.id] = geom

    result = compile_entity_physics(entity, world)
    assert result.status is PhysicsCompileStatus.COMPILED
    assert not result.body.is_static
    wood_density = CANONICAL_MATERIALS["wood"].density
    assert result.body.mass == wood_density * 1.0  # 1m^3 cube
    assert result.mass_provenance is Provenance.ESTIMATED
    assert result.material_source == "custom_properties"
    assert result.confidence == 0.9


def test_missing_physics_material_falls_back_to_low_confidence_default():
    entity, geom = _entity_with_bounds("ent-obj", EntityType.DEBRIS, (0, 0, 0), (1, 1, 1), confidence=0.95)
    world = WorldIR()
    world.entities[entity.id] = entity
    world.geometries[geom.id] = geom

    result = compile_entity_physics(entity, world)
    assert result.material_source == "default"
    assert result.body.material.name == "concrete"
    assert result.confidence < 0.9  # capped by the low default-material confidence
    assert any("no 'physics_material' evidence" in n for n in result.notes)


def test_unknown_physics_material_name_also_falls_back():
    entity, geom = _entity_with_bounds(
        "ent-obj", EntityType.DEBRIS, (0, 0, 0), (1, 1, 1),
        custom_properties={"physics_material": "unobtainium"},
    )
    world = WorldIR()
    world.entities[entity.id] = entity
    world.geometries[geom.id] = geom

    result = compile_entity_physics(entity, world)
    assert result.material_source == "default"


def test_entity_with_no_geometry_is_skipped_not_fabricated():
    entity = Entity(id="ent-bare", type=EntityType.DEBRIS)
    world = WorldIR()
    world.entities[entity.id] = entity

    result = compile_entity_physics(entity, world)
    assert result.status is PhysicsCompileStatus.SKIPPED_NO_GEOMETRY
    assert result.body is None


def test_geometry_without_bounds_is_skipped_not_fabricated():
    geom = Geometry(id="geom-x", type=GeometryType.POINTCLOUD)  # no bounds_min/max
    entity = Entity(id="ent-scan", type=EntityType.DEBRIS, geometry_ids=[geom.id])
    world = WorldIR()
    world.entities[entity.id] = entity
    world.geometries[geom.id] = geom

    result = compile_entity_physics(entity, world)
    assert result.status is PhysicsCompileStatus.SKIPPED_NO_BOUNDS


def test_degenerate_bounds_axis_is_floored_not_zero():
    entity, geom = _entity_with_bounds("ent-floor", EntityType.FLOOR, (0, 0, 0), (3.0, 0.0, 3.0))
    world = WorldIR()
    world.entities[entity.id] = entity
    world.geometries[geom.id] = geom

    result = compile_entity_physics(entity, world)
    assert result.body.shape.half_extents.y == 0.005  # 0.01 floor / 2


def test_compile_physics_world_is_deterministic_and_sorted():
    e1, g1 = _entity_with_bounds("ent-b", EntityType.DEBRIS, (0, 0, 0), (1, 1, 1))
    e2, g2 = _entity_with_bounds("ent-a", EntityType.WALL, (0, 0, 0), (1, 1, 1))
    world = WorldIR()
    for e, g in ((e1, g1), (e2, g2)):
        world.entities[e.id] = e
        world.geometries[g.id] = g

    diagnostics = compile_physics_world(world)
    assert [r.entity_id for r in diagnostics.results] == ["ent-a", "ent-b"]
    assert len(diagnostics.compiled) == 2
    assert len(diagnostics.skipped) == 0
    bodies = diagnostics.bodies()
    assert set(bodies) == {"ent-a", "ent-b"}


def test_compile_physics_world_never_mutates_input_world():
    entity, geom = _entity_with_bounds("ent-1", EntityType.DEBRIS, (0, 0, 0), (1, 1, 1))
    world = WorldIR()
    world.entities[entity.id] = entity
    world.geometries[geom.id] = geom
    before = world.entities["ent-1"].to_dict()

    compile_physics_world(world)

    assert world.entities["ent-1"].to_dict() == before


def test_diagnostics_to_dict_is_plain_data():
    entity, geom = _entity_with_bounds("ent-1", EntityType.DEBRIS, (0, 0, 0), (1, 1, 1))
    world = WorldIR()
    world.entities[entity.id] = entity
    world.geometries[geom.id] = geom

    diagnostics = compile_physics_world(world)
    payload = diagnostics.to_dict()
    assert payload["compiled_count"] == 1
    assert payload["skipped_count"] == 0
    assert payload["results"][0]["body"]["mass"] > 0
