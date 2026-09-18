"""The generic world-simulation loop (P17-01: WorldIR -> simulation ->
WorldStateDelta -> WorldIR branch/version).

This closes the loop the PhysicsCompiler bridge opened: the physics
stack already compiles a reconstructed world into real rigid bodies
(engine/compiler/physics_compiler.py) and steps them deterministically
(engine/physics/backend/simple_backend.py); what was missing is the
path BACK -- simulated end states becoming a WorldStateDelta that
applies to a versioned copy of the world.

Honesty rules (same discipline as the compile bridge):

  - Only MEASURED simulation results are recorded: per-entity position/
    orientation changes read back from the physics world after N fixed
    steps. Nothing is fabricated; a body that did not move is recorded
    as explicitly UNCHANGED, not omitted and not given invented motion.
  - Entity identity is preserved end to end: physics bodies are named
    "phys-<entity_id>" by the compiler; this module strips that prefix
    on the way back so deltas speak WorldIR entity ids.
  - Provenance is SIMULATED on the delta and stamped into the applied
    world's metadata -- simulated state never masquerades as observed.
  - Application NEVER mutates the source world: it returns a shallow
    versioned copy (version + 1). Missing entities raise rather than
    being silently created (no unobserved entities).
  - Deterministic: fixed dt, fixed step count, the backend's own
    id-sorted ordering; same inputs -> byte-identical deltas.

Reproducibility contract: a delta carries its exact simulation
parameters (dt, tick) so a consumer can re-run the same simulation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from provenance import Provenance
from world_ir.statement_state import StatementState
from world_ir.world_v1 import WorldIR

from engine.compiler.physics_compiler import (
    build_stepped_physics_world,
    compile_physics_world,
)

#: The physics compiler prefixes body ids with this; the loop strips it
#: so deltas reference WorldIR entity ids.
_BODY_ID_PREFIX = "phys-"


@dataclass(frozen=True)
class EntityChange:
    """Measured state change of one entity across a simulation run.

    position_delta/orientation_delta are None when that quantity was
    not simulated for the entity (e.g. a static body), never zero
    placeholders for "we did not look".
    """

    entity_id: str
    position_delta: Optional[Tuple[float, float, float]]
    orientation_delta: Optional[Tuple[float, float, float, float]]
    final_position: Tuple[float, float, float]
    final_orientation: Tuple[float, float, float, float]
    was_static: bool

    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "position_delta": (None if self.position_delta is None
                               else list(self.position_delta)),
            "orientation_delta": (None if self.orientation_delta is None
                                  else list(self.orientation_delta)),
            "final_position": list(self.final_position),
            "final_orientation": list(self.final_orientation),
            "was_static": self.was_static,
        }

    @staticmethod
    def from_dict(data: dict) -> "EntityChange":
        return EntityChange(
            entity_id=data["entity_id"],
            position_delta=(None if data.get("position_delta") is None
                            else tuple(data["position_delta"])),
            orientation_delta=(None if data.get("orientation_delta") is None
                               else tuple(data["orientation_delta"])),
            final_position=tuple(data["final_position"]),
            final_orientation=tuple(data["final_orientation"]),
            was_static=data.get("was_static", False),
        )


@dataclass(frozen=True)
class WorldStateDelta:
    """The measured difference between a world and its simulated end
    state, with the parameters needed to reproduce the run."""

    entity_changes: Dict[str, EntityChange] = field(default_factory=dict)
    unchanged_entity_ids: Tuple[str, ...] = ()
    tick: int = 0
    dt: float = 0.01
    #: Epistemic class of this delta: SIMULATED (world_ir/statement_state
    #: .py). The Provenance axis stays GENERATED -- simulation output is
    #: generated data whose origin is the simulator; SIMULATED vs
    #: PROCEDURAL etc. is the statement-state distinction. Recorded on
    #: BOTH axes so downstream can filter without guessing.
    provenance: Provenance = Provenance.GENERATED
    statement_state: StatementState = StatementState.SIMULATED

    def to_dict(self) -> dict:
        return {
            "entity_changes": {
                k: v.to_dict() for k, v in sorted(self.entity_changes.items())
            },
            "unchanged_entity_ids": list(self.unchanged_entity_ids),
            "tick": self.tick,
            "dt": self.dt,
            "provenance": self.provenance.value,
            "statement_state": self.statement_state.value,
        }

    @staticmethod
    def from_dict(data: dict) -> "WorldStateDelta":
        return WorldStateDelta(
            entity_changes={
                k: EntityChange.from_dict(v)
                for k, v in data.get("entity_changes", {}).items()
            },
            unchanged_entity_ids=tuple(data.get("unchanged_entity_ids", ())),
            tick=data.get("tick", 0),
            dt=data.get("dt", 0.01),
            provenance=Provenance(
                data.get("provenance", Provenance.GENERATED.value)),
            statement_state=StatementState(
                data.get("statement_state", StatementState.SIMULATED.value)),
        )


@dataclass(frozen=True)
class SimulationRun:
    """One deterministic simulation: the final state snapshot and the
    delta back to the source world."""

    delta: WorldStateDelta
    final_state: "object"  # world_ir.world_v1.SimulationState


def _final_state_snapshot(world, run) -> "SimulationState":
    from world_ir.world_v1 import SimulationState

    body_states = {}
    for bid in sorted(world.bodies):
        body = world.bodies[bid]
        entity_id = (bid[len(_BODY_ID_PREFIX):]
                     if bid.startswith(_BODY_ID_PREFIX) else bid)
        body_states[entity_id] = {
            "position": [body.position.x, body.position.y, body.position.z],
            "velocity": [body.linear_velocity.x, body.linear_velocity.y,
                         body.linear_velocity.z],
            "rotation": [body.orientation.w, body.orientation.x,
                         body.orientation.y, body.orientation.z],
            "angular_velocity": [body.angular_velocity.x,
                                 body.angular_velocity.y,
                                 body.angular_velocity.z],
        }
    return SimulationState(
        tick=run.tick,
        timestamp=run.time,
        dt=world.config.dt if hasattr(world.config, "dt") else 0.01,
        body_states=body_states,
        contact_count=len(run.last_contacts),
        provenance=Provenance.GENERATED,
    )


def simulate_and_diff(world: WorldIR, steps: int, dt: float,
                      config=None) -> SimulationRun:
    """Compile `world` to physics, simulate `steps` fixed steps of
    `dt`, and diff the end state against the world's compiled state.

    Deterministic: the backend's ordering is id-sorted and the step
    count/dt are fixed. `config` optionally overrides the
    PhysicsWorldConfig (gravity etc.) for scenarios that need it.
    """
    from engine.physics.backend.interface import PhysicsWorldConfig

    from engine.physics.backend.simple_backend import SimpleRigidBodyBackend

    diagnostics = compile_physics_world(world)
    physics = build_stepped_physics_world(
        diagnostics, config or PhysicsWorldConfig())
    compiled_ids = {r.entity_id: r.body for r in diagnostics.compiled}

    # The diagnostics hold references to the SAME body objects the
    # world steps, so the compiled state must be snapshotted BEFORE
    # stepping -- afterwards body.position IS the end state.
    compiled_state = {
        eid: ((b.position.x, b.position.y, b.position.z),
              (b.orientation.w, b.orientation.x, b.orientation.y,
               b.orientation.z),
              b.is_static)
        for eid, b in compiled_ids.items()
    }

    backend = SimpleRigidBodyBackend()
    for _ in range(steps):
        backend.step(physics, dt)

    snapshot = _final_state_snapshot(physics, physics)

    changes: Dict[str, EntityChange] = {}
    unchanged: list = []
    for entity_id in sorted(compiled_ids):
        compiled_pos, compiled_rot, static = compiled_state[entity_id]
        end = snapshot.body_states[entity_id]
        end_pos = tuple(end["position"])
        end_rot = tuple(end["rotation"])

        if static:
            unchanged.append(entity_id)
            continue

        pos_delta = (end_pos[0] - compiled_pos[0],
                     end_pos[1] - compiled_pos[1],
                     end_pos[2] - compiled_pos[2])
        rot_delta = (end_rot[0] - compiled_rot[0],
                     end_rot[1] - compiled_rot[1],
                     end_rot[2] - compiled_rot[2],
                     end_rot[3] - compiled_rot[3])
        moved = (pos_delta != (0.0, 0.0, 0.0)
                 or rot_delta != (0.0, 0.0, 0.0, 0.0))
        if not moved:
            # Measured result: the body genuinely did not move (e.g.
            # sleeping/resting). Recorded as unchanged -- an honest
            # measured fact, distinct from a skipped static body.
            unchanged.append(entity_id)
            continue
        changes[entity_id] = EntityChange(
            entity_id=entity_id,
            position_delta=pos_delta,
            orientation_delta=rot_delta,
            final_position=end_pos,
            final_orientation=end_rot,
            was_static=False,
        )

    delta = WorldStateDelta(
        entity_changes=changes,
        unchanged_entity_ids=tuple(unchanged),
        tick=steps,
        dt=dt,
        provenance=Provenance.GENERATED,
        statement_state=StatementState.SIMULATED,
    )
    return SimulationRun(delta=delta, final_state=snapshot)


def apply_delta_to_world(world: WorldIR, delta: WorldStateDelta) -> WorldIR:
    """Apply a WorldStateDelta to `world`, returning a versioned copy
    (version + 1). NEVER mutates `world`.

    - Entity transforms are updated to the simulated final positions
      (positions measured by the simulation, not extrapolated).
    - The delta's provenance is stamped into the world's metadata so
      simulated state remains distinguishable from observed state.
    - Entities referenced by the delta but absent from the world raise
      KeyError -- the delta never creates unobserved entities.
    """
    missing = [eid for eid in sorted(delta.entity_changes)
               if eid not in world.entities]
    if missing:
        raise KeyError(
            f"delta references entities not in world: {missing}")

    new_world = copy.copy(world)
    new_world.entities = dict(world.entities)
    new_world.metadata = dict(world.metadata)
    new_world.version = world.version + 1

    for entity_id, change in delta.entity_changes.items():
        entity = world.entities[entity_id]
        base = dict(entity.transform) if entity.transform else {}
        # The MEASURED absolute end state: the physics compiler places
        # bodies at the geometry AABB center (geometry space == world
        # space for this loop), so the body's final position IS the
        # entity's simulated world position. Writing base + delta would
        # lose that origin for entities compiled without a transform.
        new_transform = dict(base)
        new_transform["position"] = {
            "x": change.final_position[0],
            "y": change.final_position[1],
            "z": change.final_position[2],
        }
        new_entity = copy.copy(entity)
        new_entity.transform = new_transform
        new_world.entities[entity_id] = new_entity

    new_world.metadata["last_delta_statement_state"] = (
        delta.statement_state.value)
    new_world.metadata["last_delta_provenance"] = delta.provenance.value
    new_world.metadata["last_delta_tick"] = delta.tick
    return new_world
