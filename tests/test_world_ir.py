from world_ir.coordinates import Frame
from world_ir.entity import Entity
from world_ir.world import WorldIR


def test_default_world_is_empty_and_versioned():
    world = WorldIR(id="w1")
    assert world.version == 1
    assert world.coordinate_system == Frame.WORLD
    assert len(world.entities) == 0


def test_opaque_sections_roundtrip_untouched():
    world = WorldIR(id="w1", environment={"weather": "clear"}, physics={"gravity": -9.81})
    restored = WorldIR.from_dict(world.to_dict())
    assert restored.environment == {"weather": "clear"}
    assert restored.physics == {"gravity": -9.81}


def test_entities_participate_in_roundtrip():
    world = WorldIR(id="w1")
    world.entities.add(Entity(id="e1", type="building"))
    restored = WorldIR.from_dict(world.to_dict())
    assert "e1" in restored.entities
    assert restored.entities.get("e1").type == "building"
