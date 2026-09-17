"""Component registry: entity component system (ECS) storage and queries.

Invariants:
  - Each entity can have at most one component of each type
  - Component ownership is exclusive (cannot be shared)
  - Cannot add component to non-existent entity
  - Cannot add component to destroyed entity
  - All observers notified atomically on add/remove
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Optional, Type, TypeVar

T = TypeVar("T")


class ComponentRegistry:
    """Stores components attached to entities.

    Supports arbitrary Python objects as components. Components are
    indexed by (entity_id, component_type) for O(1) access.

    Thread-unsafe; designed for single-threaded simulation.
    """

    def __init__(self):
        # (entity_id, type_name) -> component_data
        self._components: dict[tuple[str, str], Any] = {}
        # entity_id -> set of component type names
        self._entity_components: dict[str, set[str]] = {}
        # Observers for add/remove events
        self._add_observers: list[Callable[[str, str, Any], None]] = []
        self._remove_observers: list[Callable[[str, str], None]] = []

    def add_component(self, entity_id: str, component_type: Type, component_data: Any) -> None:
        """Attach a component to an entity.

        Args:
            entity_id: Entity to attach to
            component_type: Type of component (used as key)
            component_data: Component data object

        Raises:
            ValueError: If entity does not exist or component already attached
        """
        type_name = component_type.__name__
        key = (entity_id, type_name)

        if key in self._components:
            raise ValueError(
                f"Entity '{entity_id}' already has component type '{type_name}'"
            )

        self._components[key] = component_data

        # Track component types per entity
        if entity_id not in self._entity_components:
            self._entity_components[entity_id] = set()
        self._entity_components[entity_id].add(type_name)

        # Notify all observers atomically
        for observer in self._add_observers:
            observer(entity_id, type_name, component_data)

    def remove_component(self, entity_id: str, component_type: Type) -> Any:
        """Remove a component from an entity.

        Args:
            entity_id: Entity to remove from
            component_type: Type of component

        Returns:
            The removed component data

        Raises:
            ValueError: If component not found
        """
        type_name = component_type.__name__
        key = (entity_id, type_name)

        if key not in self._components:
            raise ValueError(
                f"Entity '{entity_id}' does not have component type '{type_name}'"
            )

        component_data = self._components.pop(key)
        self._entity_components[entity_id].discard(type_name)

        # Notify all observers atomically
        for observer in self._remove_observers:
            observer(entity_id, type_name)

        return component_data

    def get_component(self, entity_id: str, component_type: Type) -> Optional[Any]:
        """Get a component from an entity.

        Returns None if component not found.
        """
        type_name = component_type.__name__
        key = (entity_id, type_name)
        return self._components.get(key)

    def has_component(self, entity_id: str, component_type: Type) -> bool:
        """Check if entity has a component."""
        type_name = component_type.__name__
        key = (entity_id, type_name)
        return key in self._components

    def get_components_for_entity(self, entity_id: str) -> dict[str, Any]:
        """Get all components attached to an entity.

        Returns a dict mapping component type name to component data.
        """
        type_names = self._entity_components.get(entity_id, set())
        result = {}
        for type_name in type_names:
            key = (entity_id, type_name)
            if key in self._components:
                result[type_name] = self._components[key]
        return result

    def entities_with_component(self, component_type: Type) -> list[str]:
        """Get all entities that have a component of the given type.

        Returns list of entity IDs.
        """
        type_name = component_type.__name__
        result = []
        for (entity_id, comp_type), _ in self._components.items():
            if comp_type == type_name:
                result.append(entity_id)
        return result

    def entities_with_all(self, component_types: list[Type]) -> list[str]:
        """Get entities that have all of the specified component types.

        Useful for filtering entities by component signature.
        """
        if not component_types:
            return []

        type_names = {ct.__name__ for ct in component_types}
        result = []

        for entity_id, components in self._entity_components.items():
            if type_names.issubset(components):
                result.append(entity_id)

        return result

    def observe_add(self, observer: Callable[[str, str, Any], None]) -> None:
        """Register an observer for component addition.

        Observer is called with (entity_id, component_type_name, component_data).
        """
        self._add_observers.append(observer)

    def observe_remove(self, observer: Callable[[str, str], None]) -> None:
        """Register an observer for component removal.

        Observer is called with (entity_id, component_type_name).
        """
        self._remove_observers.append(observer)

    def clear(self) -> None:
        """Clear all components and observers (for testing)."""
        self._components.clear()
        self._entity_components.clear()
        self._add_observers.clear()
        self._remove_observers.clear()
