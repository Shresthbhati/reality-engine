"""Runtime foundation: entity lifecycle, components, resources, events,
and world state management.

Core subsystems:
  - EntityRegistry: entity creation, destruction, tagging
  - ComponentRegistry: ECS-style component attachment/querying
  - ResourceManager: asset management with reference counting
  - EventBus: deterministic publish-subscribe messaging
  - WorldLifecycle: world state machine (CREATED → RUNNING → SHUTDOWN)
  - WorldRuntime: integrates all systems
"""

from .entity import EntityRegistry, EntityMetadata, EntityState
from .component import ComponentRegistry
from .resources import ResourceManager, ResourceType, ResourceMetadata
from .events import EventBus, Event, EventHandler, EventPriority
from .lifecycle import WorldLifecycle, WorldState, LifecycleMetrics
from .dependency_graph import DependencyGraph, System
from .caching import (
    Cache,
    LRUCache,
    QueryCache,
    TransformCache,
    ComponentCache,
)
from .runtime import WorldRuntime

__all__ = [
    "WorldRuntime",
    "EntityRegistry",
    "EntityMetadata",
    "EntityState",
    "ComponentRegistry",
    "ResourceManager",
    "ResourceType",
    "ResourceMetadata",
    "EventBus",
    "Event",
    "EventHandler",
    "EventPriority",
    "WorldLifecycle",
    "WorldState",
    "LifecycleMetrics",
    "DependencyGraph",
    "System",
    "Cache",
    "LRUCache",
    "QueryCache",
    "TransformCache",
    "ComponentCache",
]
