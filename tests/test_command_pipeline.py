"""Tests for the command pipeline (spec sec 7): typed command -> validation
-> permission -> WorldAPI apply -> event -> provenance -> new version.
"""

import pytest

from engine.commands import (
    AllowAllPolicy,
    CommandNotFoundError,
    CommandValidationError,
    CreateEntityCommand,
    DeleteEntityCommand,
    PermissionDeniedError,
    SetEntityTransformCommand,
    WorldCommandProcessor,
)
from events.bus import EventBus
from events.types import ENTITY_CREATED_EVENT, ENTITY_DELETED_EVENT, ENTITY_TRANSFORM_SET_EVENT
from provenance import Provenance
from world_ir import Entity, WorldIR


def _processor():
    world = WorldIR(id="w1")
    bus = EventBus(seed=1)
    return WorldCommandProcessor(world, bus), world, bus


# ---- create ----

def test_create_entity_adds_entity_with_generated_provenance():
    processor, world, bus = _processor()
    result = processor.execute(CreateEntityCommand(entity_id="e1", entity_type="structure", name="Wall"))

    assert "e1" in world.entities
    entity = world.entities["e1"]
    assert entity.provenance == Provenance.GENERATED
    assert not entity.is_canonical() if hasattr(entity, "is_canonical") else True
    assert world.version == 2  # started at 1, bumped once
    assert result.world_version == 2

    events = bus.events_of_type(ENTITY_CREATED_EVENT)
    assert len(events) == 1
    assert events[0].source_refs == ("e1",)
    assert events[0].event_id == result.event_id


def test_create_entity_rejects_duplicate_id():
    processor, world, _ = _processor()
    processor.execute(CreateEntityCommand(entity_id="e1", entity_type="structure"))
    with pytest.raises(CommandValidationError):
        processor.execute(CreateEntityCommand(entity_id="e1", entity_type="structure"))


def test_create_entity_rejects_unknown_entity_type():
    processor, world, _ = _processor()
    with pytest.raises(CommandValidationError):
        processor.execute(CreateEntityCommand(entity_id="e1", entity_type="not-a-real-type"))
    assert world.version == 1  # rejected before any mutation -- version unchanged


# ---- set transform ----

def test_set_transform_updates_existing_entity():
    processor, world, bus = _processor()
    world.entities["e1"] = Entity(id="e1", name="Wall")

    processor.execute(SetEntityTransformCommand(entity_id="e1", position=(1.0, 2.0, 3.0)))

    assert world.entities["e1"].transform == {"position": {"x": 1.0, "y": 2.0, "z": 3.0}}
    assert world.version == 2
    assert len(bus.events_of_type(ENTITY_TRANSFORM_SET_EVENT)) == 1


def test_set_transform_rejects_unknown_entity():
    processor, world, _ = _processor()
    with pytest.raises(CommandNotFoundError):
        processor.execute(SetEntityTransformCommand(entity_id="nope", position=(0, 0, 0)))


def test_set_transform_does_not_change_entity_provenance():
    """Editing where an entity is placed isn't the same claim as editing
    what it's made of -- moving a RECONSTRUCTED entity must not silently
    make it look GENERATED or vice versa."""
    processor, world, _ = _processor()
    world.entities["e1"] = Entity(id="e1", provenance=Provenance.RECONSTRUCTED, confidence=0.9)
    processor.execute(SetEntityTransformCommand(entity_id="e1", position=(1.0, 0.0, 0.0)))
    assert world.entities["e1"].provenance == Provenance.RECONSTRUCTED


# ---- delete ----

def test_delete_entity_removes_it():
    processor, world, bus = _processor()
    world.entities["e1"] = Entity(id="e1")

    processor.execute(DeleteEntityCommand(entity_id="e1"))

    assert "e1" not in world.entities
    assert len(bus.events_of_type(ENTITY_DELETED_EVENT)) == 1


def test_delete_entity_rejects_unknown_entity():
    processor, world, _ = _processor()
    with pytest.raises(CommandNotFoundError):
        processor.execute(DeleteEntityCommand(entity_id="nope"))


# ---- permission stage ----

class DenyAllPolicy:
    def check(self, command, world):
        raise PermissionDeniedError("no actor is allowed to do anything in this test")


def test_permission_denied_blocks_execution_before_any_mutation():
    world = WorldIR(id="w1")
    bus = EventBus(seed=1)
    processor = WorldCommandProcessor(world, bus, permission_policy=DenyAllPolicy())

    with pytest.raises(PermissionDeniedError):
        processor.execute(CreateEntityCommand(entity_id="e1", entity_type="structure"))

    assert world.entities == {}
    assert world.version == 1
    assert bus.events == []


def test_permission_check_runs_after_validation():
    """A validation failure (unknown entity) must surface as that error,
    not be masked by a permission check that never got a chance to run
    on a nonexistent target -- validation is stage 1, permission is stage 2."""
    world = WorldIR(id="w1")
    bus = EventBus(seed=1)
    processor = WorldCommandProcessor(world, bus, permission_policy=DenyAllPolicy())

    with pytest.raises(CommandNotFoundError):
        processor.execute(SetEntityTransformCommand(entity_id="nope", position=(0, 0, 0)))


def test_allow_all_policy_permits_every_command():
    policy = AllowAllPolicy()
    world = WorldIR(id="w1")
    assert policy.check(CreateEntityCommand(entity_id="e1", entity_type="structure"), world) is None
