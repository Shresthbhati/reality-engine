"""End-to-end test: real reconstruction -> plane promotion -> physics
compilation -> a real physics integration step.

Proves the compiled bodies are not just structurally valid but actually
runnable by the existing physics engine (engine/physics/rigid/integrator.py)
-- the concrete demonstration that "isolated physics demos" and the real
reconstructed world are now connected, per the simulation campaign's
opening complaint.
"""

from __future__ import annotations

from evidence.promote_planes import promote_plane_to_entity
from perception.geometry.orientation import classify_planes
from perception.geometry.planes import detect_planes
from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult
from world_ir import WorldIR

from engine.compiler.physics_compiler import build_stepped_physics_world, compile_physics_world
from engine.physics.backend.interface import PhysicsWorldConfig
from engine.physics.backend.simple_backend import SimpleRigidBodyBackend
from engine.physics.math3 import Vec3
from engine.physics.rigid.integrator import integrate

_UP = (0.0, 1.0, 0.0)
_CAMS = [(2.0, 1.25, 1.5)]


def _point(x: float, y: float, z: float, counter=[0]) -> ReconstructedPoint:
    counter[0] += 1
    return ReconstructedPoint(position=(x, y, z), track_id=f"pt-{counter[0]:05d}", source_evidence_ids=["ev-1"])


def _room_result() -> ReconstructionResult:
    counter = [0]
    points = []
    for i in range(12):
        for j in range(12):
            points.append(_point(i * 0.25, 0.0, j * 0.25, counter))
            points.append(_point(i * 0.25, 2.5, j * 0.25, counter))
    for i in range(10):
        for j in range(10):
            points.append(_point(0.0, i * 0.25, j * 0.25, counter))
            points.append(_point(4.0, i * 0.25, j * 0.25, counter))
    for i in range(30):
        points.append(_point(2.0 + (i % 5) * 0.7, 1.25 + (i % 3) * 0.4, 1.5 + (i % 7) * 0.3, counter))
    return ReconstructionResult(points=points, camera_poses=[], registration_status="success")


def _promoted_world() -> WorldIR:
    result = _room_result()
    det = detect_planes(result, seed=42)
    oriented = classify_planes(det.planes, _CAMS, up=_UP)
    world = WorldIR()
    for n, o in enumerate(oriented):
        if o.role in ("wall", "floor", "ceiling"):
            promote_plane_to_entity(o, result, world, f"struct-{n}")
    return world


def test_promoted_structural_planes_compile_to_static_bodies():
    world = _promoted_world()
    diagnostics = compile_physics_world(world)

    assert len(diagnostics.compiled) >= 3
    assert len(diagnostics.skipped) == 0
    for result in diagnostics.compiled:
        assert result.body.is_static  # wall/floor/ceiling are all in _STATIC_ENTITY_TYPES


def test_static_body_does_not_move_under_gravity():
    world = _promoted_world()
    diagnostics = compile_physics_world(world)
    bodies = diagnostics.bodies()
    floor_body = next(b for b in bodies.values() if b.is_static)
    start_position = floor_body.position

    for _ in range(60):  # 1 second at 60Hz
        integrate(floor_body, gravity=Vec3(0.0, -9.81, 0.0), dt=1.0 / 60.0)

    assert floor_body.position.to_dict() == start_position.to_dict()


def test_dynamic_debris_above_the_room_falls_under_gravity():
    world = _promoted_world()
    from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3

    debris_geom = Geometry(
        id="geom-debris", type=GeometryType.BOX,
        bounds_min=Vector3(1.5, 2.0, 1.0), bounds_max=Vector3(2.0, 2.5, 1.5),
    )
    debris_entity = Entity(id="ent-debris", type=EntityType.DEBRIS, geometry_ids=[debris_geom.id], confidence=0.7)
    world.geometries[debris_geom.id] = debris_geom
    world.entities[debris_entity.id] = debris_entity

    diagnostics = compile_physics_world(world)
    bodies = diagnostics.bodies()
    debris_body = bodies["ent-debris"]
    assert not debris_body.is_static
    start_y = debris_body.position.y

    for _ in range(30):  # half a second
        integrate(debris_body, gravity=Vec3(0.0, -9.81, 0.0), dt=1.0 / 60.0)

    assert debris_body.position.y < start_y  # fell under gravity
    assert debris_body.linear_velocity.y < 0.0


def test_debris_collides_with_reconstructed_floor_via_real_backend():
    """The literal next simulation-campaign step: compiled bodies must
    collide with EACH OTHER through the real backend
    (engine/physics/backend/simple_backend.py), not just integrate
    independently. A debris box dropped above the reconstructed floor
    must come to rest on it, not tunnel through to an arbitrary negative
    y -- proof the reconstructed room and the physics engine are now one
    connected system."""
    from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3

    world = _promoted_world()
    debris_geom = Geometry(
        id="geom-debris", type=GeometryType.BOX,
        bounds_min=Vector3(1.5, 0.5, 1.0), bounds_max=Vector3(2.0, 1.0, 1.5),
    )
    debris_entity = Entity(id="ent-debris", type=EntityType.DEBRIS, geometry_ids=[debris_geom.id], confidence=0.7)
    world.geometries[debris_geom.id] = debris_geom
    world.entities[debris_entity.id] = debris_entity

    diagnostics = compile_physics_world(world)
    physics_world = build_stepped_physics_world(diagnostics, PhysicsWorldConfig(gravity=Vec3(0.0, -9.81, 0.0)))
    backend = SimpleRigidBodyBackend()

    debris_body = physics_world.get_body("phys-ent-debris")
    floor_body = next(
        b for eid, b in physics_world.bodies.items()
        if world.entities[eid.removeprefix("phys-")].type == EntityType.FLOOR
    )
    floor_top_y = floor_body.position.y + floor_body.shape.half_extents.y

    for _ in range(300):  # 5 seconds at 60Hz -- enough to fall ~0.25m and settle
        backend.step(physics_world, dt=1.0 / 60.0)

    debris_bottom_y = debris_body.position.y - debris_body.shape.half_extents.y
    # Resting on the floor (within the solver's contact tolerance), not
    # tunneled through it or still falling.
    assert debris_bottom_y >= floor_top_y - 0.05
    assert abs(debris_body.linear_velocity.y) < 0.5
