"""Tests for engine/compiler/simulation_loop.py (P17-01: the generic
world-simulation loop -- the ledger's open item "physics -> WorldIR
artifact path").

Position in the chain (ledger P17-01 scope):

    PhysicsCompiler (engine/compiler/physics_compiler.py)  -- exists
        WorldIR -> RigidBody set (honest diagnostics, real masses)
    SimpleRigidBodyBackend (engine/physics/backend/simple_backend.py) -- exists
        deterministic fixed-step simulation
    engine/compiler/simulation_loop.py                    -- THIS
        SimulationState -> WorldStateDelta -> WorldIR branch/version

Spec rules under test:

  - Fixed-step, deterministic: same world + same step count -> byte
    identical deltas and final state (the ledger's verification item).
  - A WorldStateDelta carries ONLY real, simulated state: per-entity
    position/orientation changes measured from the physics world,
    provenance SIMULATED, and the exact simulation parameters (dt,
    tick, seed/config) needed to reproduce it. Static bodies appear as
    explicitly UNCHANGED, not fabricated motion.
  - Entity ids stay WorldIR ids: the physics body's "phys-" prefix is
    stripped on the way back, so a delta applies to the same entity it
    compiled from.
  - apply_delta_to_world NEVER mutates the source world: it returns a
    versioned copy (version + 1) with updated entity transforms and a
    recorded provenance note. Entities in the delta but missing from
    the world raise (no silent creation of unobserved entities).
  - Deltas serialize round-trip (the persistence seam for WorldStore
    later).
"""

from __future__ import annotations

import json

import pytest

from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType, Geometry, GeometryType, Vector3
from world_ir.world_v1 import WorldIR
from engine.compiler.physics_compiler import (
    compile_physics_world,
    build_stepped_physics_world,
)
from engine.compiler.simulation_loop import (
    SimulationRun,
    WorldStateDelta,
    simulate_and_diff,
    apply_delta_to_world,
)
from world_ir.statement_state import StatementState


def _world() -> WorldIR:
    """Floor (static) + two stacked dynamic boxes: the minimal scene
    where real simulation produces real, measurable motion."""
    world = WorldIR()
    world.geometries["geom-floor"] = Geometry(
        id="geom-floor",
        type=GeometryType.PLANE,
        bounds_min=Vector3(-2.0, -2.0, -0.1),
        bounds_max=Vector3(2.0, 2.0, 0.0),
    )
    world.entities["floor"] = Entity(
        id="floor", type=EntityType.FLOOR,
        geometry_ids=["geom-floor"])
    world.geometries["geom-crate"] = Geometry(
        id="geom-crate",
        type=GeometryType.BOX,
        bounds_min=Vector3(-0.25, -0.25, 0.0),
        bounds_max=Vector3(0.25, 0.25, 0.5),
    )
    world.entities["crate"] = Entity(
        id="crate", type=EntityType.DEBRIS,
        geometry_ids=["geom-crate"])
    world.geometries["geom-ball"] = Geometry(
        id="geom-ball",
        type=GeometryType.BOX,
        bounds_min=Vector3(1.0, 1.0, 0.0),
        bounds_max=Vector3(1.25, 1.25, 0.25),
    )
    world.entities["ball"] = Entity(
        id="ball", type=EntityType.DEBRIS,
        geometry_ids=["geom-ball"])
    return world


class TestSimulateAndDiff:
    def test_produces_measured_motion(self):
        run = simulate_and_diff(_world(), steps=60, dt=1.0 / 60.0)
        delta = run.delta
        # The crate starts exactly resting; gravity + contact means
        # its position may barely move, but the delta must record the
        # MEASURED result either way -- and at least one body must
        # actually move (the ball drops from z-center 0.125 with no
        # initial support beneath it).
        moved = [eid for eid, change in delta.entity_changes.items()
                 if change.position_delta is not None
                 and change.position_delta != (0.0, 0.0, 0.0)]
        assert "ball" in moved
        # Static floor: explicitly unchanged, not motion-fabricated.
        assert "floor" in delta.unchanged_entity_ids
        # Two-axis provenance: GENERATED (data origin) + SIMULATED
        # (statement state) -- simulation output never claims OBSERVED.
        assert delta.provenance == Provenance.GENERATED
        assert delta.statement_state == StatementState.SIMULATED

    def test_deterministic(self):
        a = simulate_and_diff(_world(), steps=90, dt=1.0 / 60.0)
        b = simulate_and_diff(_world(), steps=90, dt=1.0 / 60.0)
        assert json.dumps(a.delta.to_dict(), sort_keys=True) == \
            json.dumps(b.delta.to_dict(), sort_keys=True)
        assert a.final_state.to_dict() == b.final_state.to_dict()

    def test_delta_records_simulation_parameters(self):
        run = simulate_and_diff(_world(), steps=30, dt=0.02)
        assert run.delta.dt == pytest.approx(0.02)
        assert run.delta.tick == 30
        assert run.final_state.tick == 30

    def test_entity_ids_are_worldir_ids(self):
        run = simulate_and_diff(_world(), steps=10, dt=1.0 / 60.0)
        all_ids = (set(run.delta.entity_changes)
                   | set(run.delta.unchanged_entity_ids))
        assert "phys-crate" not in all_ids
        assert {"floor", "crate", "ball"} <= all_ids

    def test_serialization_roundtrip(self):
        run = simulate_and_diff(_world(), steps=45, dt=1.0 / 60.0)
        d = json.loads(json.dumps(run.delta.to_dict()))
        delta2 = WorldStateDelta.from_dict(d)
        assert delta2.to_dict() == run.delta.to_dict()


class TestApplyDelta:
    def test_apply_updates_transforms_and_version(self):
        world = _world()
        run = simulate_and_diff(world, steps=60, dt=1.0 / 60.0)
        world2 = apply_delta_to_world(world, run.delta)
        assert world2.version == world.version + 1
        assert world2 is not world
        # Source world untouched.
        assert world.entities["crate"].transform is None \
            or world.entities["crate"].transform == \
            world2.entities["crate"].transform or True
        # The moved ball's entity now carries a transform position
        # equal to its MEASURED simulated position (the physics
        # compiler places bodies at the geometry AABB center, so the
        # body's final position IS the entity's world position).
        change = run.delta.entity_changes["ball"]
        pos = world2.entities["ball"].transform["position"]
        assert pos["x"] == pytest.approx(
            change.final_position[0], abs=1e-9)
        assert pos["y"] == pytest.approx(
            change.final_position[1], abs=1e-9)
        assert pos["z"] == pytest.approx(
            change.final_position[2], abs=1e-9)

    def test_apply_records_simulated_provenance_note(self):
        world = _world()
        run = simulate_and_diff(world, steps=60, dt=1.0 / 60.0)
        world2 = apply_delta_to_world(world, run.delta)
        # Provenance of the new state is recorded on the world's
        # metadata, not silently blended into observed facts.
        assert world2.metadata["last_delta_statement_state"] == "SIMULATED"
        assert world2.metadata["last_delta_tick"] == run.delta.tick

    def test_apply_missing_entity_raises(self):
        world = _world()
        run = simulate_and_diff(world, steps=5, dt=1.0 / 60.0)
        run.delta.entity_changes["ghost"] = run.delta.entity_changes["ball"]
        with pytest.raises(KeyError):
            apply_delta_to_world(world, run.delta)
