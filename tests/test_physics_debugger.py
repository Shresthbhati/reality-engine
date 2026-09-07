"""Tests for the Physics Debugger debug-draw facade."""

import pytest

from engine.physics.backend import PhysicsWorldConfig, SimpleRigidBodyBackend, StaticPlane
from engine.physics.collision.shapes import Box, Plane, Sphere
from engine.physics.debug.debugger import PhysicsDebugger
from engine.physics.materials import CANONICAL_MATERIALS
from engine.physics.rigid.body import RigidBody, SleepState
from engine.physics.math3 import Vec3


def ground_plane() -> StaticPlane:
    return StaticPlane(
        id="ground",
        plane=Plane(normal=Vec3(0, 1, 0), distance=0.0),
        material=CANONICAL_MATERIALS["concrete"],
    )


def falling_cube_world(n_ticks=1, dt=0.01):
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0), seed=1))
    world.add_plane(ground_plane())
    cube = RigidBody(
        id="cube", shape=Box(Vec3(0.5, 0.5, 0.5)), material=CANONICAL_MATERIALS["wood"],
        mass=1.0, position=Vec3(0, 5.0, 0),
    )
    world.add_body(cube)
    for _ in range(n_ticks):
        backend.step(world, dt)
    return world


def resting_cube_world():
    return falling_cube_world(n_ticks=500, dt=0.01)


class TestBodiesDebug:
    def test_body_record_fields(self):
        debugger = PhysicsDebugger()
        world = falling_cube_world()
        records = debugger.bodies_debug(world)

        assert len(records) == 1
        record = records[0]
        assert record["id"] == "cube"
        assert record["shape"] == {"kind": "box", "half_extents": {"x": 0.5, "y": 0.5, "z": 0.5}}
        assert "position" in record
        assert "orientation" in record
        assert record["sleep_state"] == SleepState.AWAKE.value
        assert "linear_velocity" in record
        assert "angular_velocity" in record
        assert record["kinetic_energy"] >= 0.0

    def test_bodies_sorted_by_id(self):
        debugger = PhysicsDebugger()
        backend = SimpleRigidBodyBackend()
        world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0), seed=1))
        world.add_body(RigidBody(id="zebra", shape=Sphere(0.5), material=CANONICAL_MATERIALS["wood"], mass=1.0, position=Vec3(0, 10, 0)))
        world.add_body(RigidBody(id="apple", shape=Sphere(0.5), material=CANONICAL_MATERIALS["wood"], mass=1.0, position=Vec3(0, 20, 0)))

        records = debugger.bodies_debug(world)
        assert [r["id"] for r in records] == ["apple", "zebra"]

    def test_settled_body_is_sleeping_or_slow(self):
        debugger = PhysicsDebugger()
        world = resting_cube_world()
        record = debugger.bodies_debug(world)[0]
        # Settled body should have very low kinetic energy regardless of exact sleep threshold
        assert record["kinetic_energy"] < 0.5


class TestForcesDebug:
    def test_falling_body_has_no_force_record(self):
        """Gravity is applied and consumed within step(); accumulators are
        cleared after integration, so a body with no *externally held*
        force between steps produces no record -- this asserts that
        contract rather than assuming gravity leaves a visible residue."""
        debugger = PhysicsDebugger()
        world = falling_cube_world()
        records = debugger.forces_debug(world)
        assert records == []

    def test_manually_applied_force_appears(self):
        debugger = PhysicsDebugger()
        backend = SimpleRigidBodyBackend()
        world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, 0, 0), seed=1))
        body = RigidBody(id="pushed", shape=Sphere(0.5), material=CANONICAL_MATERIALS["wood"], mass=1.0, position=Vec3(0, 10, 0))
        world.add_body(body)
        body.apply_force(Vec3(5.0, 0.0, 0.0))

        records = debugger.forces_debug(world)
        assert len(records) == 1
        assert records[0]["id"] == "pushed"
        assert records[0]["force"]["x"] == 5.0


class TestContactsDebug:
    def test_no_contacts_when_falling(self):
        debugger = PhysicsDebugger()
        world = falling_cube_world()
        assert debugger.contacts_debug(world) == []

    def test_contact_recorded_against_plane_when_resting(self):
        debugger = PhysicsDebugger()
        world = resting_cube_world()
        contacts = debugger.contacts_debug(world)

        assert len(contacts) >= 1
        contact = contacts[0]
        assert contact["body_b"] == "cube"
        assert contact["plane_id"] == "ground"
        assert contact["body_a"] is None
        assert contact["penetration"] >= 0.0
        assert "normal" in contact


class TestNumericsDebug:
    def test_numerics_debug_passthrough(self):
        debugger = PhysicsDebugger()
        world = falling_cube_world()
        report = debugger.numerics_debug(world.last_numerics)
        assert report["nan_count"] == 0
        assert report["inf_count"] == 0


class TestSummary:
    def test_summary_counts(self):
        debugger = PhysicsDebugger()
        world = falling_cube_world()
        summary = debugger.summary(world)

        assert summary["body_count"] == 1
        assert summary["contact_count"] == 0
        assert summary["by_sleep_state"][SleepState.AWAKE.value] == 1
        assert summary["total_kinetic_energy"] > 0.0

    def test_summary_on_empty_world(self):
        debugger = PhysicsDebugger()
        backend = SimpleRigidBodyBackend()
        world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0), seed=1))
        summary = debugger.summary(world)

        assert summary["body_count"] == 0
        assert summary["contact_count"] == 0
        assert summary["total_kinetic_energy"] == 0.0
