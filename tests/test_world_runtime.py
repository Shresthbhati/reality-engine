import pytest

from engine.world import WorldRuntime
from provenance import Provenance
from world_ir.coordinates import Frame, Transform
from world_ir.entity import Entity
from world_ir.world import WorldIR


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
