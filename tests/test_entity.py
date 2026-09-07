import pytest

from provenance import Provenance
from world_ir.entity import (
    DanglingRelationshipError,
    DuplicateEntityError,
    Entity,
    EntityRegistry,
    Relationship,
    UnknownEntityError,
)


def test_add_and_get():
    reg = EntityRegistry()
    reg.add(Entity(id="wall_01", type="wall", provenance=Provenance.OBSERVED))
    assert reg.get("wall_01").type == "wall"
    assert len(reg) == 1


def test_duplicate_id_raises():
    reg = EntityRegistry()
    reg.add(Entity(id="wall_01", type="wall"))
    with pytest.raises(DuplicateEntityError):
        reg.add(Entity(id="wall_01", type="wall"))


def test_unknown_entity_raises():
    reg = EntityRegistry()
    with pytest.raises(UnknownEntityError):
        reg.get("ghost")


def test_dangling_relationship_rejected_on_add():
    reg = EntityRegistry()
    entity = Entity(
        id="window_01",
        type="window",
        relationships=[Relationship(kind="attached_to", target_id="building_01")],
    )
    with pytest.raises(DanglingRelationshipError):
        reg.add(entity)


def test_self_relationship_allowed():
    reg = EntityRegistry()
    entity = Entity(
        id="loop_01",
        type="thing",
        relationships=[Relationship(kind="self_ref", target_id="loop_01")],
    )
    reg.add(entity)  # should not raise


def test_relationship_resolves_once_target_exists():
    reg = EntityRegistry()
    reg.add(Entity(id="building_01", type="building"))
    reg.add(Entity(
        id="window_01",
        type="window",
        relationships=[Relationship(kind="attached_to", target_id="building_01")],
    ))
    assert reg.get("window_01").relationships[0].target_id == "building_01"


def test_remove_blocked_while_referenced():
    reg = EntityRegistry()
    reg.add(Entity(id="building_01", type="building"))
    reg.add(Entity(
        id="window_01",
        type="window",
        relationships=[Relationship(kind="attached_to", target_id="building_01")],
    ))
    with pytest.raises(DanglingRelationshipError):
        reg.remove("building_01")


def test_by_type():
    reg = EntityRegistry()
    reg.add(Entity(id="w1", type="window"))
    reg.add(Entity(id="w2", type="window"))
    reg.add(Entity(id="d1", type="door"))
    assert {e.id for e in reg.by_type("window")} == {"w1", "w2"}


def test_roundtrip_to_from_list():
    reg = EntityRegistry()
    reg.add(Entity(id="building_01", type="building", provenance=Provenance.OBSERVED))
    reg.add(Entity(
        id="window_01",
        type="window",
        relationships=[Relationship(kind="attached_to", target_id="building_01")],
        semantic_labels=["glass", "operable"],
    ))
    restored = EntityRegistry.from_list(reg.to_list())
    assert len(restored) == 2
    assert restored.get("window_01").semantic_labels == ["glass", "operable"]
    assert restored.get("window_01").relationships[0].target_id == "building_01"
