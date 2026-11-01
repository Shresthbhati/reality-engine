"""Deterministic recorder + replayer for a physics simulation run.

Scope (see README.md): record a body-state timeline plus the full
event log for a `PhysicsWorld` run, then re-run the exact same
tick/dt/seed sequence against a fresh world seeded from the same
initial bodies and prove the two runs match. This is the "replay"
half of spec sec 90 determinism -- it does not implement branching,
scrubbing, or any UI; it is the recording+verification core only.

Provenance: each ContactEvent/ImpactEvent emitted by
`SimpleRigidBodyBackend._emit_contact_event` (engine/physics/backend/
simple_backend.py) carries the colliding bodies'/planes' ids in
`source_refs`/`target_refs` -- for a body-vs-body contact both are the
two body ids; for a body-vs-plane contact `source_refs` is the plane id
and `target_refs` is the body id. Recording the full event log via
`EventBus.to_list()` therefore ties every recorded contact back to the
WorldIR-derived body ids that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from events import EventBus
from engine.physics.backend.interface import PhysicsWorldConfig
from engine.physics.backend.simple_backend import PhysicsWorld, SimpleRigidBodyBackend
from engine.physics.math3 import Quat, Vec3
from engine.physics.rigid.body import RigidBody

# Same float-comparison convention already used elsewhere in this
# backend (see engine/physics/collision/narrowphase.py:31 and
# engine/physics/materials/material.py:31, both 1e-9) -- there's no
# shared tolerance constant in the repo to import, so this mirrors
# their inline value rather than inventing a new one.
_FLOAT_TOL = 1e-9


class ReplayDivergenceError(RuntimeError):
    """Raised by replay_and_verify() with the exact tick/body/field that
    first diverged -- never silently returns False with no diagnosis."""


def _vec3_to_tuple(v: Vec3) -> Tuple[float, float, float]:
    return (v.x, v.y, v.z)


def _quat_to_tuple(q: Quat) -> Tuple[float, float, float, float]:
    return (q.w, q.x, q.y, q.z)


def _snapshot_body(body: RigidBody) -> dict:
    return {
        "position": _vec3_to_tuple(body.position),
        "orientation": _quat_to_tuple(body.orientation),
        "linear_velocity": _vec3_to_tuple(body.linear_velocity),
        "angular_velocity": _vec3_to_tuple(body.angular_velocity),
    }


def _copy_body(body: RigidBody) -> RigidBody:
    """A fresh RigidBody with the same field values -- not the same
    instance -- so replay never mutates the caller's originals."""
    return RigidBody(
        id=body.id,
        shape=body.shape,
        material=body.material,
        mass=body.mass,
        position=Vec3(body.position.x, body.position.y, body.position.z),
        orientation=Quat(body.orientation.w, body.orientation.x, body.orientation.y, body.orientation.z),
        linear_velocity=Vec3(body.linear_velocity.x, body.linear_velocity.y, body.linear_velocity.z),
        angular_velocity=Vec3(body.angular_velocity.x, body.angular_velocity.y, body.angular_velocity.z),
        linear_damping=body.linear_damping,
        angular_damping=body.angular_damping,
        gravity_scale=body.gravity_scale,
    )


@dataclass(frozen=True)
class RecordedFrame:
    tick: int
    bodies: Dict[str, dict]

    def to_dict(self) -> dict:
        return {"tick": self.tick, "bodies": self.bodies}

    @staticmethod
    def from_dict(data: dict) -> "RecordedFrame":
        return RecordedFrame(tick=data["tick"], bodies=data["bodies"])


@dataclass(frozen=True)
class PhysicsRecording:
    seed: int
    dt: float
    frames: Tuple[RecordedFrame, ...]
    events: Tuple[dict, ...]

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "dt": self.dt,
            "frames": [f.to_dict() for f in self.frames],
            "events": list(self.events),
        }

    @staticmethod
    def from_dict(data: dict) -> "PhysicsRecording":
        return PhysicsRecording(
            seed=data["seed"],
            dt=data["dt"],
            frames=tuple(RecordedFrame.from_dict(f) for f in data["frames"]),
            events=tuple(data["events"]),
        )


def record_simulation(
    world: PhysicsWorld, backend: SimpleRigidBodyBackend, steps: int, dt: float, seed: int = 0
) -> PhysicsRecording:
    """Attach a fresh EventBus to `world` and record `steps` calls to
    `backend.step(world, dt)`.

    `world` must not already have an event_bus -- if it does, this
    raises rather than silently mixing a recording with pre-existing
    events, so a PhysicsRecording's event log always corresponds
    exactly to the steps it claims to cover.
    """
    if world.event_bus is not None:
        raise ValueError(
            "record_simulation requires a world with no event_bus set yet "
            "(got one already attached) -- pass a fresh PhysicsWorld so the "
            "recording's event log isn't mixed with pre-existing events"
        )
    world.event_bus = EventBus(seed=seed)

    def _snapshot() -> RecordedFrame:
        return RecordedFrame(
            tick=world.tick,
            bodies={bid: _snapshot_body(b) for bid, b in sorted(world.bodies.items())},
        )

    frames = [_snapshot()]
    for _ in range(steps):
        backend.step(world, dt)
        frames.append(_snapshot())

    return PhysicsRecording(
        seed=seed,
        dt=dt,
        frames=tuple(frames),
        events=tuple(world.event_bus.to_list()),
    )


def _floats_close(a: float, b: float) -> bool:
    return abs(a - b) <= _FLOAT_TOL * max(1.0, abs(a), abs(b))


def _tuples_close(a: tuple, b: tuple) -> bool:
    return len(a) == len(b) and all(_floats_close(x, y) for x, y in zip(a, b))


def replay_and_verify(
    recording: PhysicsRecording,
    initial_bodies: Dict[str, RigidBody],
    config: PhysicsWorldConfig,
    backend: Optional[SimpleRigidBodyBackend] = None,
) -> bool:
    """Re-run the recorded seed/dt/step sequence against a fresh world
    seeded from `initial_bodies` (deep-copied -- the caller's dict and
    bodies are never mutated) and `config`, and verify the resulting
    frames + event log exactly match `recording`.

    Returns True only on an exact match. Otherwise raises
    ReplayDivergenceError naming the first tick/body/field (or event
    index) that diverged -- never returns False.
    """
    backend = backend or SimpleRigidBodyBackend()
    world = backend.create_world(config)
    for body_id, body in sorted(initial_bodies.items()):
        world.add_body(_copy_body(body))

    steps = len(recording.frames) - 1
    replayed = record_simulation(world, backend, steps=steps, dt=recording.dt, seed=recording.seed)

    for frame_idx, (expected, actual) in enumerate(zip(recording.frames, replayed.frames)):
        if expected.tick != actual.tick:
            raise ReplayDivergenceError(
                f"tick mismatch at frame {frame_idx}: recorded tick={expected.tick}, replayed tick={actual.tick}"
            )
        if expected.bodies.keys() != actual.bodies.keys():
            raise ReplayDivergenceError(
                f"tick {expected.tick}: body id set differs -- "
                f"recorded={sorted(expected.bodies)}, replayed={sorted(actual.bodies)}"
            )
        for body_id in sorted(expected.bodies):
            exp_fields = expected.bodies[body_id]
            act_fields = actual.bodies[body_id]
            for field_name in exp_fields:
                exp_val = exp_fields[field_name]
                act_val = act_fields[field_name]
                if not _tuples_close(tuple(exp_val), tuple(act_val)):
                    raise ReplayDivergenceError(
                        f"tick {expected.tick}, body '{body_id}', field '{field_name}' diverged: "
                        f"recorded={exp_val}, replayed={act_val}"
                    )

    if len(recording.events) != len(replayed.events):
        raise ReplayDivergenceError(
            f"event count diverged: recorded {len(recording.events)} events, "
            f"replayed produced {len(replayed.events)}"
        )
    for idx, (exp_evt, act_evt) in enumerate(zip(recording.events, replayed.events)):
        for key in ("type", "tick", "source_refs", "target_refs"):
            if exp_evt.get(key) != act_evt.get(key):
                raise ReplayDivergenceError(
                    f"event {idx} field '{key}' diverged: recorded={exp_evt.get(key)!r}, "
                    f"replayed={act_evt.get(key)!r}"
                )

    return True
