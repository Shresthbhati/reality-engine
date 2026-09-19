"""SimpleRigidBodyBackend: the one concrete IPhysicsBackend this pass
ships. P1 fidelity (spec sec 1.4 gameplay physics): semi-implicit Euler,
sphere/box/plane primitives, single-pass sequential-impulse contacts,
naive O(n^2) broadphase. Deterministic given the same config/seed and
input sequence (spec sec 90) -- ordering is always id-sorted, never
dict-iteration order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from engine.core.logging import Diagnostics, get_logger
from engine.physics.collision.broadphase import body_pairs
from engine.physics.collision.narrowphase import box_vs_box, box_vs_plane, sphere_vs_plane, sphere_vs_sphere
from engine.physics.collision.raycast import ray_vs_box, ray_vs_plane, ray_vs_sphere
from engine.physics.collision.shapes import Box, Plane, Sphere, shape_from_dict
from engine.physics.constraints.contact_solver import resolve_contact
from engine.physics.diagnostics.numerics import NumericsReport, check_world
from engine.physics.materials import PhysicsMaterial
from engine.physics.rigid.body import RigidBody
from engine.physics.rigid.integrator import integrate
from engine.physics.math3 import Vec3
from events import EventBus
from events.types import CONTACT_EVENT, IMPACT_EVENT

from .interface import IPhysicsBackend, PhysicsWorldConfig, RaycastHit

_NARROWPHASE_DISPATCH = {
    (Sphere, Sphere): sphere_vs_sphere,
    (Box, Box): box_vs_box,
}

# Approach speed (m/s, along the contact normal) above which a new
# contact is reported as an ImpactEvent instead of a plain ContactEvent
# -- spec sec 85 lists both as distinct event types.
IMPACT_SPEED_THRESHOLD = 2.0


class DuplicateBodyError(ValueError):
    pass


class UnknownBodyError(ValueError):
    pass


@dataclass(frozen=True)
class StaticPlane:
    id: str
    plane: Plane
    material: PhysicsMaterial


@dataclass(frozen=True)
class ContactRecord:
    """A resolved contact, kept for the step's query_contacts() result."""

    body_a_id: Optional[str]  # None means the static plane side
    body_b_id: str
    plane_id: Optional[str]
    normal: Vec3
    penetration: float


def _combine_restitution(a: PhysicsMaterial, b: PhysicsMaterial) -> float:
    return max(a.restitution, b.restitution)


def _combine_friction(a: PhysicsMaterial, b: PhysicsMaterial) -> float:
    return (a.friction_dynamic * b.friction_dynamic) ** 0.5


class PhysicsWorld:
    def __init__(self, config: PhysicsWorldConfig):
        self.config = config
        self.bodies: dict[str, RigidBody] = {}
        self.planes: dict[str, StaticPlane] = {}
        self.tick: int = 0
        self.time: float = 0.0
        self.last_numerics: Optional[NumericsReport] = None
        self.last_contacts: list[ContactRecord] = []
        self.event_bus: Optional[EventBus] = None
        self._last_energy: Optional[float] = None
        self._active_contact_keys: set[tuple] = set()

    def add_body(self, body: RigidBody) -> None:
        if body.id in self.bodies:
            raise DuplicateBodyError(f"body '{body.id}' already exists")
        self.bodies[body.id] = body

    def add_plane(self, static_plane: StaticPlane) -> None:
        if static_plane.id in self.planes:
            raise DuplicateBodyError(f"plane '{static_plane.id}' already exists")
        self.planes[static_plane.id] = static_plane

    def get_body(self, body_id: str) -> RigidBody:
        try:
            return self.bodies[body_id]
        except KeyError:
            raise UnknownBodyError(f"no body '{body_id}'") from None


class SimpleRigidBodyBackend(IPhysicsBackend):
    def create_world(self, config: PhysicsWorldConfig) -> PhysicsWorld:
        return PhysicsWorld(config)

    def step(self, world: PhysicsWorld, dt: float) -> None:
        for body in world.bodies.values():
            integrate(body, world.config.gravity, dt)

        next_tick = world.tick + 1
        next_time = world.time + dt
        contacts: list[ContactRecord] = []
        new_contact_keys: set[tuple] = set()

        positions = {bid: b.position for bid, b in world.bodies.items()}
        shapes = {bid: b.shape for bid, b in world.bodies.items() if isinstance(b.shape, (Sphere, Box))}
        for id_a, id_b in body_pairs(list(shapes.keys()), positions, shapes):
            body_a = world.bodies[id_a]
            body_b = world.bodies[id_b]
            handler = _NARROWPHASE_DISPATCH.get((type(body_a.shape), type(body_b.shape)))
            if handler is None:
                raise NotImplementedError(
                    f"no narrowphase handler for {type(body_a.shape).__name__} vs "
                    f"{type(body_b.shape).__name__} (bodies '{id_a}', '{id_b}')"
                )
            geom = handler(body_a.position, body_a.shape, body_b.position, body_b.shape)
            if geom is None:
                continue
            key = ("bb", id_a, id_b)
            new_contact_keys.add(key)
            approach_speed = max(0.0, -(body_b.linear_velocity - body_a.linear_velocity).dot(geom.normal))
            restitution = _combine_restitution(body_a.material, body_b.material)
            friction = _combine_friction(body_a.material, body_b.material)
            resolve_contact(body_a, body_b, geom.normal, geom.penetration, restitution, friction)
            contacts.append(ContactRecord(id_a, id_b, None, geom.normal, geom.penetration))
            if key not in world._active_contact_keys:
                self._emit_contact_event(world, next_tick, next_time, [id_a], [id_b], geom, approach_speed)

        for body_id in sorted(world.bodies.keys()):
            body = world.bodies[body_id]
            if body.is_static:
                continue
            for plane_id in sorted(world.planes.keys()):
                static_plane = world.planes[plane_id]
                if isinstance(body.shape, Sphere):
                    geom = sphere_vs_plane(body.position, body.shape, static_plane.plane)
                elif isinstance(body.shape, Box):
                    geom = box_vs_plane(body.position, body.shape, static_plane.plane)
                else:
                    raise NotImplementedError(
                        f"no plane narrowphase handler for {type(body.shape).__name__}"
                    )
                if geom is None:
                    continue
                key = ("bp", body_id, plane_id)
                new_contact_keys.add(key)
                approach_speed = max(0.0, -body.linear_velocity.dot(geom.normal))
                restitution = _combine_restitution(body.material, static_plane.material)
                friction = _combine_friction(body.material, static_plane.material)
                resolve_contact(None, body, geom.normal, geom.penetration, restitution, friction)
                contacts.append(ContactRecord(None, body_id, plane_id, geom.normal, geom.penetration))
                if key not in world._active_contact_keys:
                    self._emit_contact_event(world, next_tick, next_time, [plane_id], [body_id], geom, approach_speed)

        world._active_contact_keys = new_contact_keys
        world.last_contacts = contacts
        report = check_world(list(world.bodies.values()), world._last_energy)
        world.last_numerics = report
        world._last_energy = report.total_kinetic_energy
        world.tick = next_tick
        world.time = next_time

        # Emit diagnostics to logger
        logger = get_logger("engine.physics.backend")
        diag = Diagnostics(
            nan_count=report.nan_count,
            inf_count=report.inf_count,
            solver_iterations=len(contacts),
            energy_kinetic=report.total_kinetic_energy,
            contacts_resolved=len(contacts),
            timing_us=0,  # Timing not measured in this pass
        )
        diag.log_to(logger, world.tick)

    @staticmethod
    def _emit_contact_event(world: PhysicsWorld, tick: int, timestamp: float, source_refs, target_refs, geom, approach_speed: float) -> None:
        if world.event_bus is None:
            return
        event_type = IMPACT_EVENT if approach_speed >= IMPACT_SPEED_THRESHOLD else CONTACT_EVENT
        world.event_bus.emit(
            event_type,
            timestamp=timestamp,
            tick=tick,
            source_refs=tuple(source_refs),
            target_refs=tuple(target_refs),
            parameters={
                "normal": geom.normal.to_dict(),
                "penetration": geom.penetration,
                "approach_speed": approach_speed,
            },
        )

    def query_contacts(self, world: PhysicsWorld) -> list[ContactRecord]:
        return list(world.last_contacts)

    def query_raycast(
        self, world: PhysicsWorld, origin: Vec3, direction: Vec3, max_distance: float
    ) -> Optional[RaycastHit]:
        direction = direction.normalized()
        best: Optional[RaycastHit] = None

        for body_id in sorted(world.bodies.keys()):
            body = world.bodies[body_id]
            if isinstance(body.shape, Sphere):
                t = ray_vs_sphere(origin, direction, body.position, body.shape)
            elif isinstance(body.shape, Box):
                t = ray_vs_box(origin, direction, body.position, body.shape)
            else:
                continue
            if t is None or t > max_distance:
                continue
            if best is None or t < best.distance:
                point = origin + direction * t
                normal = (point - body.position).normalized()
                best = RaycastHit(body_id=body_id, point=point, normal=normal, distance=t)

        for plane_id in sorted(world.planes.keys()):
            static_plane = world.planes[plane_id]
            t = ray_vs_plane(origin, direction, static_plane.plane)
            if t is None or t > max_distance:
                continue
            if best is None or t < best.distance:
                point = origin + direction * t
                best = RaycastHit(body_id=plane_id, point=point, normal=static_plane.plane.normal, distance=t)

        return best

    def serialize(self, world: PhysicsWorld) -> dict:
        return {
            "format_version": 1,
            "config": {"gravity": world.config.gravity.to_dict(), "seed": world.config.seed},
            "tick": world.tick,
            "time": world.time,
            "bodies": [b.to_dict() for b in world.bodies.values()],
            "planes": [
                {"id": p.id, "plane": p.plane.to_dict(), "material": p.material.to_dict()}
                for p in world.planes.values()
            ],
            # Preserved so a restored world doesn't re-fire ContactEvent/
            # ImpactEvent as "new" for contacts that were already ongoing
            # at serialization time.
            "active_contact_keys": [list(k) for k in world._active_contact_keys],
        }

    def deserialize(self, data: dict) -> PhysicsWorld:
        if data.get("format_version") != 1:
            raise ValueError(f"unsupported physics world format_version {data.get('format_version')!r}")
        config = PhysicsWorldConfig(
            gravity=Vec3.from_dict(data["config"]["gravity"]),
            seed=data["config"]["seed"],
        )
        world = self.create_world(config)
        world.tick = data["tick"]
        world.time = data.get("time", 0.0)
        for body_data in data["bodies"]:
            world.add_body(RigidBody.from_dict(body_data))
        for plane_data in data["planes"]:
            world.add_plane(StaticPlane(
                id=plane_data["id"],
                plane=shape_from_dict(plane_data["plane"]),
                material=PhysicsMaterial.from_dict(plane_data["material"]),
            ))
        world._active_contact_keys = {tuple(k) for k in data.get("active_contact_keys", [])}
        return world
