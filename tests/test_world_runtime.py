import pytest

from engine.world import WorldRuntime
from provenance import Provenance
from world_ir.coordinates import Frame, Transform
from world_ir.entity import Entity
from world_ir.world import WorldIR
from world_ir import WorldIR as WorldIRV1, Entity as EntityV1


def translation(dx, dy, dz):
    return (
        (1.0, 0.0, 0.0, dx),
        (0.0, 1.0, 0.0, dy),
        (0.0, 0.0, 1.0, dz),
        (0.0, 0.0, 0.0, 1.0),
    )


def test_runtime_builds_coordinate_registry_from_entities():
    world = WorldIR(id="w1")
    world.entities.add(Entity(
        id="building_01",
        type="building",
        transform=Transform(Frame.SESSION_LOCAL, Frame.WORLD, matrix=translation(10, 0, 0)),
        provenance=Provenance.RECONSTRUCTED,
    ))
    runtime = WorldRuntime(world)
    point = runtime.resolve_point((0.0, 0.0, 0.0), Frame.SESSION_LOCAL, Frame.WORLD)
    assert point == (10.0, 0.0, 0.0)


def test_runtime_save_and_load(tmp_path):
    world = WorldIR(id="w1")
    world.entities.add(Entity(id="e1", type="building"))
    runtime = WorldRuntime(world)
    package_dir = runtime.save(tmp_path / "w1")

    loaded = WorldRuntime.load(package_dir)
    # Verify the entity exists in the loaded world
    entity = loaded.get_entity_from_world("e1")
    assert entity is not None
    assert entity.type == "building"


def test_unresolvable_frame_raises():
    world = WorldIR(id="w1")
    runtime = WorldRuntime(world)
    with pytest.raises(ValueError):
        runtime.resolve_point((0, 0, 0), Frame.CAMERA, Frame.ECEF)


def test_runtime_accepts_v1_schema_world_with_entities():
    """Regression test: WorldRuntime is type-hinted for the V1 schema
    WorldIR (dict-based `entities`), but previously crashed with
    AttributeError the moment that world had any entities, because
    __init__ iterated the dict directly (yielding string keys) instead
    of its values. See docs/KNOWN_LIMITATIONS.md and docs/DECISIONS.md #12.
    """
    world = WorldIRV1(id="w1")
    entity = EntityV1(id="e1", name="v1-entity")
    world.entities[entity.id] = entity

    runtime = WorldRuntime(world)

    assert runtime.entity_exists("e1")
    fetched = runtime.get_entity_from_world("e1")
    assert fetched is not None
    assert fetched.name == "v1-entity"


def test_runtime_v1_schema_transform_resolves_through_coordinate_registry():
    """A V1-schema Entity.transform is an "externalized transform" --
    per its own docstring in world_ir/schema_v1.py, this means
    Transform.to_dict() output, not an arbitrary shape. WorldRuntime
    must round-trip it through Transform.from_dict() and register it
    with CoordinateRegistry so resolve_point/resolve_transform work for
    V1-schema entities exactly as they do for legacy ones. Previously
    this was a no-op (coordinate registration silently skipped for any
    dict transform) -- fixed as part of the P0 audit-repair pass."""
    world = WorldIRV1(id="w1")
    transform = Transform(Frame.BUILDING_LOCAL, Frame.WORLD, matrix=translation(5, 0, 0))
    entity = EntityV1(id="e1", transform=transform.to_dict())
    world.entities[entity.id] = entity

    runtime = WorldRuntime(world)

    point = runtime.resolve_point((0.0, 0.0, 0.0), Frame.BUILDING_LOCAL, Frame.WORLD)
    assert point == (5.0, 0.0, 0.0)


def test_runtime_v1_schema_nested_frame_chain_resolves():
    """Two V1-schema entities register two edges of a chain
    (session-local -> building-local -> world); resolve_point must
    compose them via CoordinateRegistry's BFS path search, exercising
    genuine nested/local/world transform resolution rather than a
    single direct edge."""
    world = WorldIRV1(id="w1")

    session_to_building = Transform(Frame.SESSION_LOCAL, Frame.BUILDING_LOCAL, matrix=translation(2, 0, 0))
    building_to_world = Transform(Frame.BUILDING_LOCAL, Frame.WORLD, matrix=translation(10, 0, 0))

    world.entities["sensor_01"] = EntityV1(id="sensor_01", transform=session_to_building.to_dict())
    world.entities["building_01"] = EntityV1(id="building_01", transform=building_to_world.to_dict())

    runtime = WorldRuntime(world)

    # session-local (0,0,0) -> +2 in building-local -> +10 in world -> (12, 0, 0)
    point = runtime.resolve_point((0.0, 0.0, 0.0), Frame.SESSION_LOCAL, Frame.WORLD)
    assert point == (12.0, 0.0, 0.0)

    # and the inverse direction, exercising Transform.inverse() composition
    reverse = runtime.resolve_point((12.0, 0.0, 0.0), Frame.WORLD, Frame.SESSION_LOCAL)
    assert reverse == (0.0, 0.0, 0.0)


def test_runtime_v1_schema_malformed_transform_dict_raises():
    """A V1-schema Entity.transform that is a dict but NOT valid
    Transform.to_dict() output (missing required keys) is a real data
    problem and must fail loudly at WorldRuntime construction time,
    not silently resolve to "no transform" -- that silent-skip was the
    previous (incorrect) no-op behavior."""
    world = WorldIRV1(id="w1")
    entity = EntityV1(id="e1", transform={"position": {"x": 1.0, "y": 2.0, "z": 3.0}})
    world.entities[entity.id] = entity

    with pytest.raises(ValueError, match="malformed transform"):
        WorldRuntime(world)


def test_runtime_v1_schema_unsupported_transform_type_raises():
    """A transform field that is neither None, a real Transform, nor a
    dict (e.g. a bare string) is unambiguously a bug in whatever
    produced the WorldIR and must raise, not silently no-op."""
    world = WorldIRV1(id="w1")
    entity = EntityV1(id="e1", transform="not-a-transform")
    world.entities[entity.id] = entity

    with pytest.raises(ValueError, match="unsupported transform type"):
        WorldRuntime(world)
