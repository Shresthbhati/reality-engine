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


def test_runtime_v1_schema_dict_transform_does_not_register_as_coordinate_frame():
    """A V1-schema Entity.transform is an externalized dict, not a real
    Transform object -- WorldRuntime must skip coordinate registration
    for it rather than crash, since CoordinateRegistry only understands
    real Transform instances."""
    world = WorldIRV1(id="w1")
    entity = EntityV1(id="e1", transform={"position": {"x": 1.0, "y": 2.0, "z": 3.0}})
    world.entities[entity.id] = entity

    runtime = WorldRuntime(world)  # must not raise

    assert runtime.entity_exists("e1")
