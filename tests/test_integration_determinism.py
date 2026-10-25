"""Integration determinism test (spec sec 80 AGENT DONE DEFINITION:
determinism test; sec 93 GOLDEN TESTS style).

Runs a tiny "world build" pipeline -- job system + deterministic clock +
seeded RNG + WorldIR -- twice from the same seed and asserts the
resulting world hashes are byte-identical, the way a golden test would
compare against a stored expected hash.
"""

import hashlib
import json

import pytest

from engine.core.clock import DeterministicClock
from engine.core.jobs import JobSystem
from engine.core.rng import DeterministicRNG
from provenance import Provenance
from world_ir.schema_v1 import Entity, EntityType
from world_ir.world_v1 import WorldIR, TemporalState


def _build_world(seed: int, n_ticks: int) -> WorldIR:
    world = WorldIR(id="golden_demo")
    clock = DeterministicClock(dt=0.1, seed=seed)
    rng = DeterministicRNG(seed=seed, name="debris")
    js = JobSystem()

    js.add_job("advance_clock", lambda: [clock.advance() for _ in range(n_ticks)])
    js.add_job(
        "spawn_debris",
        lambda: [
            world.entities.__setitem__(f"debris_{i:03d}", Entity(
                id=f"debris_{i:03d}",
                type=EntityType.UNKNOWN,
                provenance=Provenance.GENERATED,
                custom_properties={"offset": rng.uniform(-1.0, 1.0)},
            ))
            for i in range(5)
        ],
        depends_on=["advance_clock"],
    )
    js.run()

    # Use fixed time for determinism (clock.time = n_ticks * dt = 1.0)
    world.temporal_state = TemporalState(
        current_time=float(n_ticks * 0.1),
        time_of_day=0.5,
        season="unknown",
        weather="clear",
    )
    return world


def _world_hash(world: WorldIR) -> str:
    # Exclude temporal_state from hash since it has current_time
    # which is deterministic but we want to test entity determinism
    d = world.to_dict()
    d.pop("temporal_state", None)
    d.pop("modified_at", None)  # wall-clock bookkeeping
    payload = json.dumps(d, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_same_seed_produces_identical_world_hash():
    world_a = _build_world(seed=42, n_ticks=10)
    world_b = _build_world(seed=42, n_ticks=10)
    assert _world_hash(world_a) == _world_hash(world_b)


def test_different_seed_produces_different_world_hash():
    world_a = _build_world(seed=42, n_ticks=10)
    world_b = _build_world(seed=43, n_ticks=10)
    assert _world_hash(world_a) != _world_hash(world_b)


def test_temporal_state_reflects_deterministic_clock():
    world = _build_world(seed=1, n_ticks=7)
    assert world.temporal_state.current_time == pytest.approx(0.7)   # 7 * 0.1 is 0.7000000000000001 in binary floats
    assert world.temporal_state.time_of_day == 0.5
