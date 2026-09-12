"""WorldCommandProcessor: the "WorldAPI" stage of the command pipeline
(spec sec 7): TYPED COMMAND -> VALIDATION -> PERMISSION CHECK -> WORLD API
-> STATE CHANGE -> EVENT -> PROVENANCE -> NEW VERSION.

This is the only code path in the repo allowed to mutate a WorldIR on
behalf of a command. Every mutation goes through validate() then
permission.check() before touching `world`, and every successful
mutation emits a real Event and bumps world.version -- so "the AI edited
something" is always visible in both the event log and the world's
version history, never a silent write.

Branching (spec: NEW VERSION/BRANCH) is intentionally not implemented
here beyond the version-counter bump WorldIR already has: a real branch
needs copy-on-write world state, which is Build Order steps 25+
(temporal branching/counterfactuals) -- a separate, larger, undone
piece. Claiming to branch here without that would be exactly the fake
scaffolding the project conventions forbid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from events.bus import EventBus
from events.types import ENTITY_CREATED_EVENT, ENTITY_DELETED_EVENT, ENTITY_TRANSFORM_SET_EVENT
from provenance import Provenance
from world_ir import Entity, EntityType, WorldIR

from .commands import Command, CreateEntityCommand, DeleteEntityCommand, SetEntityTransformCommand
from .permissions import AllowAllPolicy, PermissionPolicy


class CommandValidationError(ValueError):
    pass


class CommandNotFoundError(ValueError):
    pass


@dataclass(frozen=True)
class CommandResult:
    command: Command
    event_id: str
    world_version: int


class WorldCommandProcessor:
    """One processor per WorldIR + EventBus pair. `tick`/`timestamp` are
    caller-supplied (default 0) rather than read from a running clock --
    a Studio edit session isn't necessarily inside a simulation loop, and
    inventing a wall-clock default here would repeat the exact
    nondeterminism bug WorldIR.created_at already documents avoiding.
    """

    def __init__(
        self,
        world: WorldIR,
        event_bus: EventBus,
        permission_policy: Optional[PermissionPolicy] = None,
    ):
        self.world = world
        self.event_bus = event_bus
        self.permission_policy = permission_policy or AllowAllPolicy()

    def execute(self, command: Command, *, tick: int = 0, timestamp: float = 0.0) -> CommandResult:
        self._validate(command)
        self.permission_policy.check(command, self.world)  # raises PermissionDeniedError

        if isinstance(command, CreateEntityCommand):
            event_type, source_refs = self._apply_create(command)
        elif isinstance(command, SetEntityTransformCommand):
            event_type, source_refs = self._apply_set_transform(command)
        elif isinstance(command, DeleteEntityCommand):
            event_type, source_refs = self._apply_delete(command)
        else:
            raise CommandValidationError(f"unknown command type: {type(command).__name__}")

        self.world.version += 1
        event = self.event_bus.emit(
            event_type, timestamp=timestamp, tick=tick,
            actor_id=command.actor_id, source_refs=tuple(source_refs),
            parameters={"world_version": self.world.version},
        )
        return CommandResult(command=command, event_id=event.event_id, world_version=self.world.version)

    # ---- validation (parsing already happened -- the caller constructed
    # a typed Command; this stage checks it against current world state) ----

    def _validate(self, command: Command) -> None:
        if isinstance(command, CreateEntityCommand):
            if command.entity_id in self.world.entities:
                raise CommandValidationError(f"entity '{command.entity_id}' already exists")
            try:
                EntityType(command.entity_type)
            except ValueError:
                raise CommandValidationError(f"unknown entity_type: {command.entity_type!r}")
        elif isinstance(command, SetEntityTransformCommand):
            if command.entity_id not in self.world.entities:
                raise CommandNotFoundError(f"no such entity: {command.entity_id!r}")
            if len(command.position) != 3:
                raise CommandValidationError("position must be an (x, y, z) triple")
        elif isinstance(command, DeleteEntityCommand):
            if command.entity_id not in self.world.entities:
                raise CommandNotFoundError(f"no such entity: {command.entity_id!r}")
        else:
            raise CommandValidationError(f"unknown command type: {type(command).__name__}")

    # ---- state change ----

    def _apply_create(self, command: CreateEntityCommand):
        # GENERATED, not OBSERVED/RECONSTRUCTED: this entity exists because
        # a command created it, not because it was captured from reality
        # (spec sec 1.1 -- generated content must never silently become
        # canonical truth; Provenanced.is_canonical() already excludes it).
        entity = Entity(
            id=command.entity_id,
            name=command.name,
            type=EntityType(command.entity_type),
            provenance=Provenance.GENERATED,
            confidence=1.0,
        )
        self.world.entities[command.entity_id] = entity
        return ENTITY_CREATED_EVENT, [command.entity_id]

    def _apply_set_transform(self, command: SetEntityTransformCommand):
        entity = self.world.entities[command.entity_id]
        x, y, z = command.position
        entity.transform = {"position": {"x": x, "y": y, "z": z}}
        return ENTITY_TRANSFORM_SET_EVENT, [command.entity_id]

    def _apply_delete(self, command: DeleteEntityCommand):
        del self.world.entities[command.entity_id]
        return ENTITY_DELETED_EVENT, [command.entity_id]
