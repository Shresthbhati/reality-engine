"""Physics <-> event bus integration (spec sec 85 PHYSICS EVENT BUS,
V11 sec 930 EVENT SYSTEM). Wired at the PhysicsWorld level so the
IPhysicsBackend.step(world, dt) interface signature (sec 7.2) doesn't
need to change to support it.
"""

import pytest

from engine.physics.backend import PhysicsWorldConfig, SimpleRigidBodyBackend, StaticPlane
from engine.physics.collision.shapes import Box, Plane, Sphere
from engine.physics.materials import CANONICAL_MATERIALS
from engine.physics.math3 import Vec3
from engine.physics.rigid.body import RigidBody
from events import EventBus
from events.types import CONTACT_EVENT, IMPACT_EVENT


def ground_plane() -> StaticPlane:
    return StaticPlane(id="ground", plane=Plane(Vec3(0, 1, 0), 0.0), material=CANONICAL_MATERIALS["concrete"])


def test_no_events_emitted_without_a_bus_attached():
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0)))
    world.add_plane(ground_plane())
    world.add_body(RigidBody(id="ball", shape=Sphere(0.5), material=CANONICAL_MATERIALS["rubber"], mass=1.0, position=Vec3(0, 2, 0)))
    for _ in range(300):
        backend.step(world, 0.01)
    assert world.event_bus is None  # nothing to assert on the bus -- there isn't one


def test_impact_event_fires_on_first_contact_with_ground():
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0)))
    world.event_bus = EventBus(seed=1)
    world.add_plane(ground_plane())
    world.add_body(RigidBody(id="ball", shape=Sphere(0.5), material=CANONICAL_MATERIALS["rubber"], mass=1.0, position=Vec3(0, 3.0, 0)))

    for _ in range(150):
        backend.step(world, 0.01)

    impacts = world.event_bus.events_of_type(IMPACT_EVENT)
    assert len(impacts) >= 1
    first = impacts[0]
    assert first.target_refs == ("ball",)
    assert first.source_refs == ("ground",)
    assert first.parameters["approach_speed"] > 2.0


def test_gentle_placement_fires_contact_not_impact():
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0)))
    world.event_bus = EventBus(seed=1)
    world.add_plane(ground_plane())
    # Placed already resting on the ground -- negligible approach speed.
    world.add_body(RigidBody(id="box", shape=Box(Vec3(0.5, 0.5, 0.5)), material=CANONICAL_MATERIALS["wood"], mass=1.0, position=Vec3(0, 0.5, 0)))

    backend.step(world, 0.01)

    assert len(world.event_bus.events_of_type(IMPACT_EVENT)) == 0
    assert len(world.event_bus.events_of_type(CONTACT_EVENT)) >= 1


def test_ongoing_resting_contact_does_not_spam_events():
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0)))
    world.event_bus = EventBus(seed=1)
    world.add_plane(ground_plane())
    world.add_body(RigidBody(id="box", shape=Box(Vec3(0.5, 0.5, 0.5)), material=CANONICAL_MATERIALS["wood"], mass=1.0, position=Vec3(0, 0.5, 0)))

    for _ in range(200):
        backend.step(world, 0.01)

    # One contact begins, then holds -- not one event per tick for 200 ticks.
    total = len(world.event_bus.events_of_type(CONTACT_EVENT)) + len(world.event_bus.events_of_type(IMPACT_EVENT))
    assert total < 10


def test_body_body_contact_uses_body_ids_in_refs():
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, 0, 0)))  # no gravity, isolate body-body contact
    world.event_bus = EventBus(seed=1)
    world.add_body(RigidBody(id="left", shape=Sphere(1.0), material=CANONICAL_MATERIALS["steel"], mass=1.0, position=Vec3(-3, 0, 0), linear_velocity=Vec3(5, 0, 0)))
    world.add_body(RigidBody(id="right", shape=Sphere(1.0), material=CANONICAL_MATERIALS["steel"], mass=1.0, position=Vec3(3, 0, 0)))

    for _ in range(100):
        backend.step(world, 0.01)

    events = world.event_bus.events_of_type(IMPACT_EVENT) + world.event_bus.events_of_type(CONTACT_EVENT)
    assert len(events) >= 1
    assert events[0].source_refs == ("left",)
    assert events[0].target_refs == ("right",)


def test_event_ordering_is_deterministic_across_runs():
    def build_and_collect():
        backend = SimpleRigidBodyBackend()
        world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0)))
        world.event_bus = EventBus(seed=99)
        world.add_plane(ground_plane())
        world.add_body(RigidBody(id="ball", shape=Sphere(0.5), material=CANONICAL_MATERIALS["rubber"], mass=1.0, position=Vec3(0, 3.0, 0)))
        for _ in range(150):
            backend.step(world, 0.01)
        return world.event_bus.to_list()

    assert build_and_collect() == build_and_collect()
