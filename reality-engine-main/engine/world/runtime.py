"""Runtime foundation: coordinates all subsystems.

WorldRuntime integrates:
  - Entity registry (lifecycle, identity)
  - Component registry (ECS storage)
  - Resource manager (asset management)
  - Event bus (deterministic messaging)
  - World lifecycle (state machine)
  - Coordinate frame resolution

Invariants:
  - Entity and component lifecycles are tightly coupled
  - Resources referenced by entities tracked automatically
  - All operations are deterministic
  - World state serialization is complete and lossless
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Type

from engine.core.logging import get_logger, Diagnostics
from world_ir import WorldIR, load_world, save_world
from world_ir.coordinates import CoordinateRegistry, Frame, Transform

from .entity import EntityRegistry, EntityMetadata
from .component import ComponentRegistry
from .resources import ResourceManager, ResourceType
from .events import EventBus, Event
from .lifecycle import WorldLifecycle, WorldState


def _iter_world_entities(world: WorldIR):
    """Yield Entity objects from either WorldIR shape this runtime supports.

    `world_ir.world_v1.WorldIR.entities` is a `dict[str, Entity]` (V1
    schema, entity.transform is an externalized dict). The legacy
    `world_ir.world.WorldIR.entities` is an `EntityRegistry` whose
    `__iter__` already yields Entity objects (real `Transform` objects).
    Iterating a dict directly yields its string keys, not Entity
    objects -- this is what previously crashed WorldRuntime.__init__ the
    moment a V1-schema world had any entities (KNOWN_LIMITATIONS.md).
    """
    entities = world.entities
    if isinstance(entities, dict):
        return entities.values()
    return entities


def _resolve_entity_transform(entity_id: str, transform: Any) -> Optional[Transform]:
    """Resolve an entity's transform field to a real `Transform`, regardless
    of WorldIR variant.

    Legacy `world_ir.world.WorldIR` entities carry a real `Transform`
    object already. V1-schema (`world_ir.world_v1.WorldIR`) entities carry
    an "externalized transform" dict -- per that field's own docstring,
    this is `Transform.to_dict()` output, not an arbitrary shape, so it
    round-trips through `Transform.from_dict()` like any other serialized
    Transform. A dict present but malformed (missing required keys, bad
    Frame value) is a real data problem and must fail loudly rather than
    silently resolve to "no transform" -- that was the previous no-op
    behavior this replaces.
    """
    if transform is None:
        return None
    if isinstance(transform, Transform):
        return transform
    if isinstance(transform, dict):
        try:
            return Transform.from_dict(transform)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"entity '{entity_id}' has a malformed transform dict: {exc}"
            ) from exc
    raise ValueError(
        f"entity '{entity_id}' has an unsupported transform type: {type(transform).__name__}"
    )


class WorldRuntime:
    """In-memory runtime over a WorldIR: entity queries, component storage,
    resources, events, and coordinate frame resolution.

    Serves as the central hub for all world operations. Provides:
      - Entity lifecycle management
      - Component attachment/querying (ECS)
      - Resource acquisition/release
      - Event publishing/subscription
      - Coordinate frame transforms
      - World serialization

    Thread-unsafe; designed for single-threaded simulation.
    """

    def __init__(self, world: WorldIR):
        self.world = world
        self._entity_registry = EntityRegistry()
        self._component_registry = ComponentRegistry()
        self._resource_manager = ResourceManager()
        self._event_bus = EventBus()
        self._lifecycle = WorldLifecycle()
        self._coordinates = CoordinateRegistry()
        self._logger = get_logger("engine.world.runtime")

        # Populate entity registry from WorldIR
        for entity in _iter_world_entities(world):
            self._entity_registry.create_entity(entity.id, tick=0)
            resolved_transform = _resolve_entity_transform(entity.id, entity.transform)
            if resolved_transform is not None:
                self._coordinates.register(resolved_transform)

    # ========== Entity Management ==========

    def create_entity(self, entity_id: str) -> EntityMetadata:
        """Create a new entity."""
        tick = self._lifecycle.metrics.ticks_simulated
        return self._entity_registry.create_entity(entity_id, tick)

    def destroy_entity(self, entity_id: str) -> None:
        """Destroy an entity and all its components."""
        tick = self._lifecycle.metrics.ticks_simulated
        self._entity_registry.destroy_entity(entity_id, tick)

        # Remove all components
        components = self._component_registry.get_components_for_entity(entity_id)
        for comp_type_name in list(components.keys()):
            # Create a dummy type for removal
            class DummyType:
                __name__ = comp_type_name
            self._component_registry.remove_component(entity_id, DummyType)

    def entity_exists(self, entity_id: str) -> bool:
        """Check if entity exists and is alive."""
        return self._entity_registry.exists(entity_id)

    def all_entities(self) -> list[EntityMetadata]:
        """Get all entities (including destroyed)."""
        return self._entity_registry.all_entities()

    def alive_entities(self) -> list[EntityMetadata]:
        """Get all alive entities."""
        return self._entity_registry.alive_entities()

    def get_entity_from_world(self, entity_id: str):
        """Get entity from WorldIR (for compatibility with loaded worlds)."""
        for entity in _iter_world_entities(self.world):
            if entity.id == entity_id:
                return entity
        return None

    def tag_entity(self, entity_id: str, tag: str) -> None:
        """Tag an entity for grouping."""
        self._entity_registry.tag_entity(entity_id, tag)

    def entities_with_tag(self, tag: str) -> list[EntityMetadata]:
        """Get all entities with a tag."""
        return self._entity_registry.entities_with_tag(tag)

    # ========== Component Management ==========

    def add_component(self, entity_id: str, component_type: Type, component_data: Any) -> None:
        """Attach a component to an entity."""
        if not self._entity_registry.exists(entity_id):
            raise ValueError(f"Entity '{entity_id}' does not exist")
        self._component_registry.add_component(entity_id, component_type, component_data)

    def remove_component(self, entity_id: str, component_type: Type) -> Any:
        """Remove a component from an entity."""
        if not self._entity_registry.exists(entity_id):
            raise ValueError(f"Entity '{entity_id}' does not exist")
        return self._component_registry.remove_component(entity_id, component_type)

    def get_component(self, entity_id: str, component_type: Type) -> Optional[Any]:
        """Get a component from an entity."""
        return self._component_registry.get_component(entity_id, component_type)

    def has_component(self, entity_id: str, component_type: Type) -> bool:
        """Check if entity has a component."""
        return self._component_registry.has_component(entity_id, component_type)

    def entities_with_component(self, component_type: Type) -> list[str]:
        """Get all entities with a component type."""
        return self._component_registry.entities_with_component(component_type)

    def entities_with_all(self, component_types: list[Type]) -> list[str]:
        """Get entities with all specified components."""
        return self._component_registry.entities_with_all(component_types)

    # ========== Resource Management ==========

    def create_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
        data: Any,
        owner_entity: Optional[str] = None,
    ) -> None:
        """Create a new resource."""
        tick = self._lifecycle.metrics.ticks_simulated
        self._resource_manager.create_resource(
            resource_id, resource_type, data, tick, owner_entity
        )

    def acquire_resource(self, resource_id: str) -> None:
        """Acquire a reference to a resource."""
        tick = self._lifecycle.metrics.ticks_simulated
        self._resource_manager.acquire_resource(resource_id, tick)

    def release_resource(self, resource_id: str) -> int:
        """Release a reference to a resource."""
        tick = self._lifecycle.metrics.ticks_simulated
        return self._resource_manager.release_resource(resource_id, tick)

    def get_resource(self, resource_id: str) -> Optional[Any]:
        """Get resource data."""
        metadata = self._resource_manager.get_resource(resource_id)
        return metadata.data if metadata else None

    # ========== Event Bus ==========

    def subscribe(self, event_type: str, handler) -> Any:
        """Subscribe to an event type."""
        return self._event_bus.subscribe(event_type, handler)

    def publish_event(self, event_type: str, data: Optional[dict] = None) -> None:
        """Publish an event."""
        tick = self._lifecycle.metrics.ticks_simulated
        timestamp = self._lifecycle.metrics.sim_time_s
        self._event_bus.publish(event_type, tick, timestamp, data)

    # ========== Lifecycle ==========

    @property
    def state(self) -> WorldState:
        """Get current world state."""
        return self._lifecycle.state

    def initialize(self) -> None:
        """Initialize the world."""
        self._lifecycle.initialize()
        self._logger.info("World initialized")

    def run(self) -> None:
        """Start running the world."""
        self._lifecycle.run()
        self._logger.info("World running")

    def step(self, dt: float) -> None:
        """Step the world forward by dt seconds."""
        if not self._lifecycle.can_step():
            raise ValueError(f"Cannot step world in state {self._lifecycle.state.value}")

        self._lifecycle.step(dt)

    def shutdown(self) -> None:
        """Shut down the world."""
        self._lifecycle.shutdown()
        self._logger.info(f"World shutdown. Ticks: {self._lifecycle.metrics.ticks_simulated}")

    # ========== Coordinate Frames ==========

    def resolve_transform(self, source: Frame, target: Frame) -> Optional[Transform]:
        """Get transform from source frame to target frame."""
        return self._coordinates.get(source, target)

    def resolve_point(
        self, point: tuple[float, float, float], source: Frame, target: Frame
    ) -> tuple[float, float, float]:
        """Transform a point from source frame to target frame."""
        transform = self._coordinates.get(source, target)
        if transform is None:
            raise ValueError(f"No known transform path from {source} to {target}")
        return transform.apply(point)

    # ========== Serialization ==========

    @staticmethod
    def load(path: str | Path) -> WorldRuntime:
        """Load a world from disk."""
        world = load_world(path)
        return WorldRuntime(world)

    def save(self, path: str | Path) -> Path:
        """Save the world to disk."""
        return save_world(self.world, path)

    # ========== Diagnostics ==========

    def get_diagnostics(self) -> Diagnostics:
        """Get runtime diagnostics."""
        metrics = self._lifecycle.metrics
        return Diagnostics(
            nan_count=0,
            inf_count=0,
            solver_iterations=0,
            energy_kinetic=0.0,
            contacts_resolved=0,
            timing_us=int(metrics.real_time_ms * 1000),
        )

    def log_state(self) -> None:
        """Log current world state."""
        metrics = self._lifecycle.metrics
        self._logger.info(
            "World state",
            context={
                "state": metrics.state.value,
                "ticks": metrics.ticks_simulated,
                "sim_time_s": metrics.sim_time_s,
                "real_time_ms": metrics.real_time_ms,
                "entities": len(self.alive_entities()),
            },
        )
