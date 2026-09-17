"""Tests for runtime foundation systems."""

import pytest
from engine.world import (
    EntityRegistry,
    EntityState,
    ComponentRegistry,
    ResourceManager,
    ResourceType,
    EventBus,
    EventPriority,
    WorldLifecycle,
    WorldState,
    WorldRuntime,
)
from world_ir import WorldIR


class TestEntityRegistry:
    """Entity registry tests."""

    def test_create_entity(self):
        """Test entity creation."""
        registry = EntityRegistry()
        metadata = registry.create_entity("entity1", tick=0)
        assert metadata.id == "entity1"
        assert metadata.state == EntityState.ALIVE
        assert metadata.created_at_tick == 0

    def test_duplicate_entity_raises(self):
        """Test that duplicate entity ID raises error."""
        registry = EntityRegistry()
        registry.create_entity("entity1", tick=0)
        with pytest.raises(ValueError, match="already exists"):
            registry.create_entity("entity1", tick=0)

    def test_destroy_entity(self):
        """Test entity destruction."""
        registry = EntityRegistry()
        registry.create_entity("entity1", tick=0)
        registry.destroy_entity("entity1", tick=10)
        metadata = registry.get_entity("entity1")
        assert metadata.state == EntityState.DESTROYED
        assert metadata.destroyed_at_tick == 10

    def test_double_destroy_raises(self):
        """Test that destroying twice raises error."""
        registry = EntityRegistry()
        registry.create_entity("entity1", tick=0)
        registry.destroy_entity("entity1", tick=10)
        with pytest.raises(ValueError, match="already destroyed"):
            registry.destroy_entity("entity1", tick=20)

    def test_exists_checks_alive_state(self):
        """Test exists() returns False for destroyed entities."""
        registry = EntityRegistry()
        registry.create_entity("entity1", tick=0)
        assert registry.exists("entity1")
        registry.destroy_entity("entity1", tick=10)
        assert not registry.exists("entity1")

    def test_tag_entity(self):
        """Test entity tagging."""
        registry = EntityRegistry()
        registry.create_entity("entity1", tick=0)
        registry.tag_entity("entity1", "physics")
        registry.tag_entity("entity1", "visible")
        metadata = registry.get_entity("entity1")
        assert "physics" in metadata.tags
        assert "visible" in metadata.tags

    def test_entities_with_tag(self):
        """Test querying entities by tag."""
        registry = EntityRegistry()
        registry.create_entity("entity1", tick=0)
        registry.create_entity("entity2", tick=0)
        registry.tag_entity("entity1", "physics")
        entities = registry.entities_with_tag("physics")
        assert len(entities) == 1
        assert entities[0].id == "entity1"

    def test_create_observers_notified(self):
        """Test that create observers are notified."""
        registry = EntityRegistry()
        calls = []
        registry.observe_create(lambda eid, tick: calls.append((eid, tick)))
        registry.create_entity("entity1", tick=42)
        assert calls == [("entity1", 42)]

    def test_destroy_observers_notified(self):
        """Test that destroy observers are notified."""
        registry = EntityRegistry()
        registry.create_entity("entity1", tick=0)
        calls = []
        registry.observe_destroy(lambda eid, tick: calls.append((eid, tick)))
        registry.destroy_entity("entity1", tick=42)
        assert calls == [("entity1", 42)]


class TestComponentRegistry:
    """Component registry (ECS) tests."""

    def test_add_component(self):
        """Test adding a component."""
        registry = ComponentRegistry()

        class PositionComponent:
            def __init__(self, x, y):
                self.x, self.y = x, y

        pos = PositionComponent(1, 2)
        registry.add_component("entity1", PositionComponent, pos)
        assert registry.has_component("entity1", PositionComponent)

    def test_duplicate_component_raises(self):
        """Test that duplicate component raises error."""
        registry = ComponentRegistry()

        class PositionComponent:
            pass

        registry.add_component("entity1", PositionComponent, {})
        with pytest.raises(ValueError, match="already has component"):
            registry.add_component("entity1", PositionComponent, {})

    def test_get_component(self):
        """Test getting a component."""
        registry = ComponentRegistry()

        class PositionComponent:
            def __init__(self, x, y):
                self.x, self.y = x, y

        pos = PositionComponent(1, 2)
        registry.add_component("entity1", PositionComponent, pos)
        retrieved = registry.get_component("entity1", PositionComponent)
        assert retrieved is pos
        assert retrieved.x == 1

    def test_remove_component(self):
        """Test removing a component."""
        registry = ComponentRegistry()

        class PositionComponent:
            pass

        comp = {}
        registry.add_component("entity1", PositionComponent, comp)
        retrieved = registry.remove_component("entity1", PositionComponent)
        assert retrieved is comp
        assert not registry.has_component("entity1", PositionComponent)

    def test_entities_with_component(self):
        """Test querying entities by component."""
        registry = ComponentRegistry()

        class PhysicsComponent:
            pass

        registry.add_component("entity1", PhysicsComponent, {})
        registry.add_component("entity2", PhysicsComponent, {})
        entities = registry.entities_with_component(PhysicsComponent)
        assert set(entities) == {"entity1", "entity2"}

    def test_entities_with_all_components(self):
        """Test querying entities by multiple components."""
        registry = ComponentRegistry()

        class PhysicsComponent:
            pass

        class VisibleComponent:
            pass

        registry.add_component("entity1", PhysicsComponent, {})
        registry.add_component("entity1", VisibleComponent, {})
        registry.add_component("entity2", PhysicsComponent, {})

        entities = registry.entities_with_all([PhysicsComponent, VisibleComponent])
        assert entities == ["entity1"]

    def test_add_observers_notified(self):
        """Test that add observers are notified."""
        registry = ComponentRegistry()

        class PositionComponent:
            pass

        calls = []
        registry.observe_add(lambda eid, ctype, cdata: calls.append((eid, ctype)))
        registry.add_component("entity1", PositionComponent, {})
        assert calls == [("entity1", "PositionComponent")]


class TestResourceManager:
    """Resource manager tests."""

    def test_create_resource(self):
        """Test resource creation."""
        manager = ResourceManager()
        manager.create_resource("mesh1", ResourceType.MESH, {"vertices": []}, tick=0)
        metadata = manager.get_resource("mesh1")
        assert metadata is not None
        assert metadata.type == ResourceType.MESH
        assert metadata.reference_count == 0

    def test_acquire_resource(self):
        """Test acquiring a resource."""
        manager = ResourceManager()
        manager.create_resource("mesh1", ResourceType.MESH, {}, tick=0)
        manager.acquire_resource("mesh1", tick=1)
        metadata = manager.get_resource("mesh1")
        assert metadata.reference_count == 1

    def test_release_resource(self):
        """Test releasing a resource."""
        manager = ResourceManager()
        manager.create_resource("mesh1", ResourceType.MESH, {}, tick=0)
        manager.acquire_resource("mesh1", tick=1)
        count = manager.release_resource("mesh1", tick=2)
        assert count == 0

    def test_resources_by_type(self):
        """Test querying resources by type."""
        manager = ResourceManager()
        manager.create_resource("mesh1", ResourceType.MESH, {}, tick=0)
        manager.create_resource("mat1", ResourceType.MATERIAL, {}, tick=0)
        meshes = manager.resources_by_type(ResourceType.MESH)
        assert len(meshes) == 1
        assert meshes[0].id == "mesh1"

    def test_unused_resources(self):
        """Test finding unused resources."""
        manager = ResourceManager()
        manager.create_resource("mesh1", ResourceType.MESH, {}, tick=0)
        manager.create_resource("mesh2", ResourceType.MESH, {}, tick=0)
        manager.acquire_resource("mesh1", tick=1)
        unused = manager.unused_resources()
        assert len(unused) == 1
        assert unused[0].id == "mesh2"

    def test_delete_unused_resource(self):
        """Test deleting an unused resource."""
        manager = ResourceManager()
        manager.create_resource("mesh1", ResourceType.MESH, {}, tick=0)
        manager.delete_resource("mesh1")
        assert manager.get_resource("mesh1") is None

    def test_delete_used_resource_raises(self):
        """Test that deleting a used resource raises error."""
        manager = ResourceManager()
        manager.create_resource("mesh1", ResourceType.MESH, {}, tick=0)
        manager.acquire_resource("mesh1", tick=1)
        with pytest.raises(ValueError, match="active references"):
            manager.delete_resource("mesh1")


class TestEventBus:
    """Event bus tests."""

    def test_subscribe_and_publish(self):
        """Test subscribing to and publishing events."""
        bus = EventBus()
        calls = []
        bus.subscribe("move", lambda ev: calls.append(ev))
        bus.publish("move", tick=0, timestamp=0.0, data={"entity_id": "e1"})
        assert len(calls) == 1
        assert calls[0].event_type == "move"
        assert calls[0].data["entity_id"] == "e1"

    def test_handler_priority_order(self):
        """Test that handlers execute in priority order."""
        bus = EventBus()
        calls = []
        bus.subscribe("event", lambda ev: calls.append("normal"), EventPriority.NORMAL)
        bus.subscribe("event", lambda ev: calls.append("high"), EventPriority.HIGH)
        bus.subscribe("event", lambda ev: calls.append("critical"), EventPriority.CRITICAL)
        bus.publish("event", tick=0, timestamp=0.0)
        assert calls == ["critical", "high", "normal"]

    def test_recursive_publish_raises(self):
        """Test that publishing during publish raises error."""
        bus = EventBus()

        def recursive_handler(ev):
            bus.publish("event2", tick=0, timestamp=0.0)

        bus.subscribe("event1", recursive_handler)
        with pytest.raises(RuntimeError, match="Cannot publish"):
            bus.publish("event1", tick=0, timestamp=0.0)

    def test_multiple_handlers_called(self):
        """Test that multiple handlers are called."""
        bus = EventBus()
        calls = []
        bus.subscribe("event", lambda ev: calls.append(1))
        bus.subscribe("event", lambda ev: calls.append(2))
        bus.subscribe("event", lambda ev: calls.append(3))
        bus.publish("event", tick=0, timestamp=0.0)
        assert calls == [1, 2, 3]

    def test_unsubscribe(self):
        """Test unsubscribing from events."""
        bus = EventBus()
        calls = []
        handler = bus.subscribe("event", lambda ev: calls.append(1))
        bus.publish("event", tick=0, timestamp=0.0)
        assert len(calls) == 1
        bus.unsubscribe("event", handler)
        bus.publish("event", tick=0, timestamp=0.0)
        assert len(calls) == 1


class TestWorldLifecycle:
    """World lifecycle tests."""

    def test_initial_state(self):
        """Test initial world state."""
        lifecycle = WorldLifecycle()
        assert lifecycle.state == WorldState.CREATED
        assert not lifecycle.can_step()

    def test_initialize_transition(self):
        """Test CREATED -> INITIALIZED transition."""
        lifecycle = WorldLifecycle()
        lifecycle.initialize()
        assert lifecycle.state == WorldState.INITIALIZED

    def test_run_transition(self):
        """Test INITIALIZED -> RUNNING transition."""
        lifecycle = WorldLifecycle()
        lifecycle.initialize()
        lifecycle.run()
        assert lifecycle.state == WorldState.RUNNING
        assert lifecycle.can_step()

    def test_step_increments_tick(self):
        """Test that stepping increments tick counter."""
        lifecycle = WorldLifecycle()
        lifecycle.initialize()
        lifecycle.run()
        lifecycle.step(0.01)
        assert lifecycle.metrics.ticks_simulated == 1
        assert lifecycle.metrics.sim_time_s == pytest.approx(0.01)

    def test_shutdown_transition(self):
        """Test RUNNING -> SHUTDOWN transition."""
        lifecycle = WorldLifecycle()
        lifecycle.initialize()
        lifecycle.run()
        lifecycle.shutdown()
        assert lifecycle.state == WorldState.SHUTDOWN
        assert not lifecycle.can_step()

    def test_invalid_state_transitions(self):
        """Test that invalid transitions raise errors."""
        lifecycle = WorldLifecycle()
        # Cannot run without initialize
        with pytest.raises(ValueError):
            lifecycle.run()
        # Cannot shutdown in CREATED
        with pytest.raises(ValueError):
            lifecycle.shutdown()

    def test_state_change_observers(self):
        """Test state change observers."""
        lifecycle = WorldLifecycle()
        calls = []
        lifecycle.observe_state_change(lambda old, new: calls.append((old.value, new.value)))
        lifecycle.initialize()
        lifecycle.run()
        assert len(calls) == 2
        assert calls[0] == ("CREATED", "INITIALIZED")
        assert calls[1] == ("INITIALIZED", "RUNNING")


class TestWorldRuntime:
    """WorldRuntime integration tests."""

    def test_create_world(self):
        """Test creating a world."""
        world = WorldIR(id="test_world", name="Test World")
        runtime = WorldRuntime(world)
        assert runtime.state == WorldState.CREATED

    def test_entity_lifecycle(self):
        """Test entity creation and destruction."""
        world = WorldIR(id="test_world", name="Test World")
        runtime = WorldRuntime(world)
        runtime.initialize()
        runtime.run()

        # Create entity
        runtime.create_entity("entity1")
        assert runtime.entity_exists("entity1")

        # Destroy entity
        runtime.destroy_entity("entity1")
        assert not runtime.entity_exists("entity1")

    def test_component_attachment(self):
        """Test component attachment and retrieval."""
        world = WorldIR(id="test_world", name="Test World")
        runtime = WorldRuntime(world)
        runtime.initialize()
        runtime.run()

        class PositionComponent:
            def __init__(self, x, y):
                self.x, self.y = x, y

        runtime.create_entity("entity1")
        pos = PositionComponent(1, 2)
        runtime.add_component("entity1", PositionComponent, pos)
        retrieved = runtime.get_component("entity1", PositionComponent)
        assert retrieved.x == 1
        assert retrieved.y == 2

    def test_component_query(self):
        """Test querying entities by component."""
        world = WorldIR(id="test_world", name="Test World")
        runtime = WorldRuntime(world)
        runtime.initialize()
        runtime.run()

        class PhysicsComponent:
            pass

        runtime.create_entity("entity1")
        runtime.create_entity("entity2")
        runtime.add_component("entity1", PhysicsComponent, {})
        runtime.add_component("entity2", PhysicsComponent, {})

        entities = runtime.entities_with_component(PhysicsComponent)
        assert len(entities) == 2

    def test_event_publishing(self):
        """Test event publishing."""
        world = WorldIR(id="test_world", name="Test World")
        runtime = WorldRuntime(world)
        runtime.initialize()
        runtime.run()

        calls = []
        runtime.subscribe("move", lambda ev: calls.append(ev))
        runtime.publish_event("move", {"entity_id": "entity1"})
        assert len(calls) == 1
        assert calls[0].data["entity_id"] == "entity1"

    def test_deterministic_simulation(self):
        """Test that deterministic simulation produces same state."""
        world1 = WorldIR(id="test", name="Test")
        world2 = WorldIR(id="test", name="Test")

        runtime1 = WorldRuntime(world1)
        runtime2 = WorldRuntime(world2)

        # Initialize both worlds identically
        runtime1.initialize()
        runtime1.run()
        runtime2.initialize()
        runtime2.run()

        # Create same entities
        class PositionComponent:
            def __init__(self, x, y):
                self.x, self.y = x, y

        for runtime in [runtime1, runtime2]:
            runtime.create_entity("entity1")
            runtime.add_component("entity1", PositionComponent, PositionComponent(0, 0))
            runtime.step(0.01)

        # Check states match
        e1_pos1 = runtime1.get_component("entity1", PositionComponent)
        e1_pos2 = runtime2.get_component("entity1", PositionComponent)
        assert e1_pos1.x == e1_pos2.x
        assert e1_pos1.y == e1_pos2.y

    def test_world_state_serialization(self):
        """Test world state can be saved and loaded."""
        world = WorldIR(id="test", name="Test")
        runtime1 = WorldRuntime(world)
        runtime1.initialize()
        runtime1.run()

        # Add entities
        runtime1.create_entity("entity1")
        runtime1.step(0.01)

        # Should be able to access entities after serialization/load
        assert runtime1.entity_exists("entity1")
        assert len(runtime1.alive_entities()) > 0

    def test_resource_management(self):
        """Test resource management."""
        world = WorldIR(id="test", name="Test")
        runtime = WorldRuntime(world)
        runtime.initialize()
        runtime.run()

        runtime.create_entity("entity1")
        runtime.create_resource("mesh1", ResourceType.MESH, {"vertices": []}, "entity1")
        runtime.acquire_resource("mesh1")

        resource_data = runtime.get_resource("mesh1")
        assert resource_data is not None
        assert resource_data["vertices"] == []

    def test_world_lifecycle_integration(self):
        """Test complete world lifecycle."""
        world = WorldIR(id="test", name="Test")
        runtime = WorldRuntime(world)

        # Should start in CREATED state
        assert runtime.state == WorldState.CREATED

        # Initialize
        runtime.initialize()
        assert runtime.state == WorldState.INITIALIZED

        # Run
        runtime.run()
        assert runtime.state == WorldState.RUNNING

        # Step
        runtime.create_entity("entity1")
        runtime.step(0.01)
        assert runtime.state == WorldState.RUNNING

        # Shutdown
        runtime.shutdown()
        assert runtime.state == WorldState.SHUTDOWN
