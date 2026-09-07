"""Resource management: identity, lifecycle, reference counting.

Invariants:
  - Resource IDs are globally unique within a world
  - Resources are immutable after creation
  - Reference counting prevents premature cleanup
  - Cannot acquire reference to non-existent resource
  - Resource ownership is tracked for deterministic serialization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum


class ResourceType(str, Enum):
    """Standard resource types."""
    MESH = "MESH"
    MATERIAL = "MATERIAL"
    TEXTURE = "TEXTURE"
    SOUND = "SOUND"
    ANIMATION = "ANIMATION"
    CUSTOM = "CUSTOM"


@dataclass
class ResourceMetadata:
    """Metadata for a resource."""
    id: str
    type: ResourceType
    data: Any  # Immutable data object
    reference_count: int = 0
    created_at_tick: int = 0
    owner_entity: Optional[str] = None  # Entity that owns this resource


class ResourceManager:
    """Manages world resources: creation, acquisition, release, tracking.

    Resources are reference-counted. A resource is only available for
    cleanup when reference count reaches 0. Supports both owned (by entity)
    and unowned (global) resources.

    Thread-unsafe; designed for single-threaded simulation.
    """

    def __init__(self):
        self._resources: dict[str, ResourceMetadata] = {}
        self._acquire_observers: list[Callable[[str, int], None]] = []
        self._release_observers: list[Callable[[str, int], None]] = []

    def create_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
        data: Any,
        tick: int,
        owner_entity: Optional[str] = None,
    ) -> ResourceMetadata:
        """Create a new resource.

        Args:
            resource_id: Unique identifier
            resource_type: Type of resource
            data: Immutable data object
            tick: Tick at which resource was created
            owner_entity: Optional entity that owns this resource

        Returns:
            ResourceMetadata

        Raises:
            ValueError: If resource ID already exists
        """
        if resource_id in self._resources:
            raise ValueError(f"Resource '{resource_id}' already exists")

        metadata = ResourceMetadata(
            id=resource_id,
            type=resource_type,
            data=data,
            created_at_tick=tick,
            owner_entity=owner_entity,
        )
        self._resources[resource_id] = metadata
        return metadata

    def acquire_resource(self, resource_id: str, tick: int) -> ResourceMetadata:
        """Acquire a reference to a resource.

        Increments reference count. Resource remains available until all
        references are released.

        Args:
            resource_id: Resource to acquire
            tick: Tick at which resource was acquired

        Returns:
            ResourceMetadata

        Raises:
            ValueError: If resource does not exist
        """
        if resource_id not in self._resources:
            raise ValueError(f"Resource '{resource_id}' does not exist")

        metadata = self._resources[resource_id]
        metadata.reference_count += 1

        # Notify observers atomically
        for observer in self._acquire_observers:
            observer(resource_id, tick)

        return metadata

    def release_resource(self, resource_id: str, tick: int) -> int:
        """Release a reference to a resource.

        Decrements reference count. When count reaches 0, resource may be
        deleted.

        Args:
            resource_id: Resource to release
            tick: Tick at which resource was released

        Returns:
            Updated reference count

        Raises:
            ValueError: If resource does not exist or already has zero refs
        """
        if resource_id not in self._resources:
            raise ValueError(f"Resource '{resource_id}' does not exist")

        metadata = self._resources[resource_id]
        if metadata.reference_count <= 0:
            raise ValueError(f"Resource '{resource_id}' has no active references")

        metadata.reference_count -= 1

        # Notify observers atomically
        for observer in self._release_observers:
            observer(resource_id, tick)

        return metadata.reference_count

    def get_resource(self, resource_id: str) -> Optional[ResourceMetadata]:
        """Get metadata for a resource.

        Returns None if resource does not exist.
        """
        return self._resources.get(resource_id)

    def resources_by_type(self, resource_type: ResourceType) -> list[ResourceMetadata]:
        """Get all resources of a given type."""
        return [r for r in self._resources.values() if r.type == resource_type]

    def resources_by_owner(self, owner_entity: str) -> list[ResourceMetadata]:
        """Get all resources owned by an entity."""
        return [r for r in self._resources.values() if r.owner_entity == owner_entity]

    def global_resources(self) -> list[ResourceMetadata]:
        """Get all unowned (global) resources."""
        return [r for r in self._resources.values() if r.owner_entity is None]

    def unused_resources(self) -> list[ResourceMetadata]:
        """Get all resources with zero references (safe to delete)."""
        return [r for r in self._resources.values() if r.reference_count == 0]

    def delete_resource(self, resource_id: str) -> ResourceMetadata:
        """Delete a resource.

        Resource must have zero references.

        Args:
            resource_id: Resource to delete

        Returns:
            Deleted ResourceMetadata

        Raises:
            ValueError: If resource has active references
        """
        if resource_id not in self._resources:
            raise ValueError(f"Resource '{resource_id}' does not exist")

        metadata = self._resources[resource_id]
        if metadata.reference_count > 0:
            raise ValueError(
                f"Resource '{resource_id}' still has {metadata.reference_count} active references"
            )

        del self._resources[resource_id]
        return metadata

    def observe_acquire(self, observer: Callable[[str, int], None]) -> None:
        """Register an observer for resource acquisition."""
        self._acquire_observers.append(observer)

    def observe_release(self, observer: Callable[[str, int], None]) -> None:
        """Register an observer for resource release."""
        self._release_observers.append(observer)

    def clear(self) -> None:
        """Clear all resources and observers (for testing)."""
        self._resources.clear()
        self._acquire_observers.clear()
        self._release_observers.clear()
