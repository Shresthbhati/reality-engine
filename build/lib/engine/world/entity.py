"""Entity registry: lifecycle, identity, and relationship management.

Invariants:
  - Entity IDs are globally unique within a world
  - Entity cannot be deleted twice
  - Entity cannot be created with duplicate ID
  - All observers notified atomically on create/destroy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
from enum import Enum


class EntityState(str, Enum):
    """Entity lifecycle state."""
    ALIVE = "ALIVE"
    DESTROYED = "DESTROYED"


@dataclass
class EntityMetadata:
    """Metadata for an entity."""
    id: str
    state: EntityState = EntityState.ALIVE
    tags: set[str] = field(default_factory=set)
    created_at_tick: int = 0
    destroyed_at_tick: Optional[int] = None


class EntityRegistry:
    """Manages entity lifecycle: create, destroy, query, observe.

    Thread-unsafe; designed for single-threaded simulation.
    """

    def __init__(self):
        self._entities: dict[str, EntityMetadata] = {}
        self._create_observers: list[Callable[[str, int], None]] = []
        self._destroy_observers: list[Callable[[str, int], None]] = []

    def create_entity(self, entity_id: str, tick: int) -> EntityMetadata:
        """Create a new entity with the given ID.

        Args:
            entity_id: Unique identifier
            tick: Tick at which entity was created

        Returns:
            EntityMetadata for the created entity

        Raises:
            ValueError: If entity ID already exists
        """
        if entity_id in self._entities:
            raise ValueError(f"Entity '{entity_id}' already exists")

        metadata = EntityMetadata(id=entity_id, created_at_tick=tick)
        self._entities[entity_id] = metadata

        # Notify all observers atomically
        for observer in self._create_observers:
            observer(entity_id, tick)

        return metadata

    def destroy_entity(self, entity_id: str, tick: int) -> None:
        """Destroy an entity.

        Args:
            entity_id: Entity to destroy
            tick: Tick at which entity was destroyed

        Raises:
            ValueError: If entity does not exist or already destroyed
        """
        if entity_id not in self._entities:
            raise ValueError(f"Entity '{entity_id}' does not exist")

        metadata = self._entities[entity_id]
        if metadata.state == EntityState.DESTROYED:
            raise ValueError(f"Entity '{entity_id}' already destroyed at tick {metadata.destroyed_at_tick}")

        metadata.state = EntityState.DESTROYED
        metadata.destroyed_at_tick = tick

        # Notify all observers atomically
        for observer in self._destroy_observers:
            observer(entity_id, tick)

    def get_entity(self, entity_id: str) -> Optional[EntityMetadata]:
        """Get metadata for an entity.

        Returns None if entity does not exist.
        """
        return self._entities.get(entity_id)

    def exists(self, entity_id: str) -> bool:
        """Check if entity exists and is alive."""
        metadata = self._entities.get(entity_id)
        return metadata is not None and metadata.state == EntityState.ALIVE

    def all_entities(self) -> list[EntityMetadata]:
        """Get all entities (including destroyed)."""
        return list(self._entities.values())

    def alive_entities(self) -> list[EntityMetadata]:
        """Get all alive entities."""
        return [m for m in self._entities.values() if m.state == EntityState.ALIVE]

    def tag_entity(self, entity_id: str, tag: str) -> None:
        """Add a tag to an entity."""
        if entity_id not in self._entities:
            raise ValueError(f"Entity '{entity_id}' does not exist")
        self._entities[entity_id].tags.add(tag)

    def untag_entity(self, entity_id: str, tag: str) -> None:
        """Remove a tag from an entity."""
        if entity_id not in self._entities:
            raise ValueError(f"Entity '{entity_id}' does not exist")
        self._entities[entity_id].tags.discard(tag)

    def entities_with_tag(self, tag: str) -> list[EntityMetadata]:
        """Get all entities with a given tag."""
        return [m for m in self._entities.values() if tag in m.tags and m.state == EntityState.ALIVE]

    def observe_create(self, observer: Callable[[str, int], None]) -> None:
        """Register an observer for entity creation.

        Observer is called with (entity_id, tick) atomically.
        """
        self._create_observers.append(observer)

    def observe_destroy(self, observer: Callable[[str, int], None]) -> None:
        """Register an observer for entity destruction.

        Observer is called with (entity_id, tick) atomically.
        """
        self._destroy_observers.append(observer)

    def clear(self) -> None:
        """Clear all entities and observers (for testing)."""
        self._entities.clear()
        self._create_observers.clear()
        self._destroy_observers.clear()
