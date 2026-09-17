"""Golden-scene style integration tests (spec sec 91 PHYSICS BENCHMARK
SUITE names these RB_001 falling cube / RB_002 stacked boxes / etc).
These aren't full RealityPhysicsBench yet -- there's no solver to
benchmark against a stored baseline until more than one backend exists
-- but they exercise the same scenes with the same pass/fail shape
(expected ranges, determinism, no NaN) called for in sec 92-93.
"""

import pytest

from engine.physics.backend import PhysicsWorldConfig, SimpleRigidBodyBackend, StaticPlane
from engine.physics.collision.shapes import Box, Plane, Sphere
from engine.physics.materials import CANONICAL_MATERIALS
from engine.physics.rigid.body import RigidBody, SleepState
from engine.physics.math3 import Vec3


def ground_plane(material_name="concrete") -> StaticPlane:
    return StaticPlane(
        id="ground",
        plane=Plane(normal=Vec3(0, 1, 0), distance=0.0),
        material=CANONICAL_MATERIALS[material_name],
    )


def run(world, backend, n_ticks, dt=0.01):
    for _ in range(n_ticks):
        backend.step(world, dt)
    return world


def test_rb001_falling_cube_settles_on_ground():
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0), seed=1))
    world.add_plane(ground_plane())
    cube = RigidBody(
        id="cube", shape=Box(Vec3(0.5, 0.5, 0.5)), material=CANONICAL_MATERIALS["wood"],
        mass=1.0, position=Vec3(0, 5.0, 0),
    )
    world.add_body(cube)

    run(world, backend, n_ticks=500, dt=0.01)  # 5 simulated seconds

    settled = world.get_body("cube")
    assert settled.position.y == pytest.approx(0.5, abs=0.05)
    assert abs(settled.linear_velocity.y) < 0.5
    assert world.last_numerics.nan_count == 0
    assert world.last_numerics.inf_count == 0


def test_rb002_stacked_boxes_remain_standing():
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0), seed=1))
    world.add_plane(ground_plane())

    bottom = RigidBody(
        id="bottom", shape=Box(Vec3(0.5, 0.5, 0.5)), material=CANONICAL_MATERIALS["wood"],
        mass=1.0, position=Vec3(0, 0.55, 0),  # small drop to settle onto the ground first
    )
    top = RigidBody(
        id="top", shape=Box(Vec3(0.5, 0.5, 0.5)), material=CANONICAL_MATERIALS["wood"],
        mass=1.0, position=Vec3(0, 1.6, 0),
    )
    world.add_body(bottom)
    world.add_body(top)

    run(world, backend, n_ticks=800, dt=0.01)  # 8 simulated seconds

    b = world.get_body("bottom")
    t = world.get_body("top")
    assert b.position.y == pytest.approx(0.5, abs=0.1)
    assert t.position.y == pytest.approx(1.5, abs=0.15)
    assert t.position.y > b.position.y  # stack order preserved, didn't collapse or swap
    assert abs(b.position.x) < 0.2 and abs(b.position.z) < 0.2  # didn't slide off
    assert abs(t.position.x) < 0.2 and abs(t.position.z) < 0.2
    assert world.last_numerics.nan_count == 0


def test_bouncy_ball_reverses_velocity_on_impact():
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0), seed=1))
    world.add_plane(ground_plane())
    ball = RigidBody(
        id="ball", shape=Sphere(0.5), material=CANONICAL_MATERIALS["rubber"],
        mass=1.0, position=Vec3(0, 3.0, 0), linear_damping=0.0,
    )
    world.add_body(ball)

    max_height_after_first_bounce = 0.0
    was_falling = False
    bounced = False
    for _ in range(300):
        backend.step(world, 0.01)
        b = world.get_body("ball")
        if b.linear_velocity.y < 0:
            was_falling = True
        if was_falling and b.linear_velocity.y > 0:
            bounced = True
        if bounced:
            max_height_after_first_bounce = max(max_height_after_first_bounce, b.position.y)

    assert bounced, "rubber ball should bounce off the ground at least once"
    assert max_height_after_first_bounce > 0.6  # rose back up meaningfully, not just jitter


def test_determinism_same_seed_same_final_state():
    def build_and_run():
        backend = SimpleRigidBodyBackend()
        world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0), seed=7))
        world.add_plane(ground_plane())
        world.add_body(RigidBody(
            id="cube", shape=Box(Vec3(0.5, 0.5, 0.5)), material=CANONICAL_MATERIALS["wood"],
            mass=1.0, position=Vec3(0.1, 4.0, -0.2),
        ))
        run(world, backend, n_ticks=200, dt=0.01)
        return backend.serialize(world)

    assert build_and_run() == build_and_run()


def test_serialize_deserialize_roundtrip_continues_identically():
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0), seed=3))
    world.add_plane(ground_plane())
    world.add_body(RigidBody(
        id="cube", shape=Box(Vec3(0.5, 0.5, 0.5)), material=CANONICAL_MATERIALS["wood"],
        mass=1.0, position=Vec3(0, 4.0, 0),
    ))
    run(world, backend, n_ticks=50, dt=0.01)

    snapshot = backend.serialize(world)
    restored = backend.deserialize(snapshot)

    run(world, backend, n_ticks=100, dt=0.01)
    run(restored, backend, n_ticks=100, dt=0.01)

    assert backend.serialize(world) == backend.serialize(restored)


def test_static_body_never_falls_through_itself():
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0), seed=1))
    world.add_plane(ground_plane())
    static_box = RigidBody(
        id="wall", shape=Box(Vec3(1, 1, 1)), material=CANONICAL_MATERIALS["concrete"],
        mass=0.0, position=Vec3(0, 1.0, 0),
    )
    world.add_body(static_box)
    run(world, backend, n_ticks=100, dt=0.01)
    assert world.get_body("wall").position == Vec3(0, 1.0, 0)
    assert world.get_body("wall").sleep_state == SleepState.STATIC


def test_mismatched_shape_pair_raises_not_implemented():
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig())
    world.add_body(RigidBody(id="s", shape=Sphere(1.0), material=CANONICAL_MATERIALS["steel"], mass=1.0))
    world.add_body(RigidBody(
        id="b", shape=Box(Vec3(1, 1, 1)), material=CANONICAL_MATERIALS["steel"], mass=1.0,
        position=Vec3(0.5, 0, 0),
    ))
    with pytest.raises(NotImplementedError):
        backend.step(world, 0.01)


def test_raycast_hits_nearest_body():
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig())
    world.add_body(RigidBody(id="near", shape=Sphere(1.0), material=CANONICAL_MATERIALS["steel"], mass=1.0, position=Vec3(5, 0, 0)))
    world.add_body(RigidBody(id="far", shape=Sphere(1.0), material=CANONICAL_MATERIALS["steel"], mass=1.0, position=Vec3(10, 0, 0)))

    hit = backend.query_raycast(world, origin=Vec3(0, 0, 0), direction=Vec3(1, 0, 0), max_distance=100)
    assert hit is not None
    assert hit.body_id == "near"
    assert hit.distance == pytest.approx(4.0)
