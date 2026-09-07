"""Integration determinism test (spec sec 80 AGENT DONE DEFINITION:
determinism test; sec 93 GOLDEN TESTS style).

Runs a tiny "world build" pipeline -- job system + deterministic clock +
seeded RNG + WorldIR -- twice from the same seed and asserts the
resulting world hashes are byte-identical, the way a golden test would
compare against a stored expected hash.
"""

import hashlib
import json

from engine.core.clock import DeterministicClock
from engine.core.jobs import JobSystem
from engine.core.rng import DeterministicRNG
from provenance import Provenance
from world_ir.entity import Entity
from world_ir.world import WorldIR


def _build_world(seed: int, n_ticks: int) -> WorldIR:
    world = WorldIR(id="golden_demo")
    clock = DeterministicClock(dt=0.1, seed=seed)
    rng = DeterministicRNG(seed=seed, name="debris")
    js = JobSystem()

    js.add_job("advance_clock", lambda: [clock.advance() for _ in range(n_ticks)])
    js.add_job(
        "spawn_debris",
        lambda: [
            world.entities.add(Entity(
                id=f"debris_{i:03d}",
                type="debris",
                provenance=Provenance.GENERATED,
                extra_fields={"offset": rng.uniform(-1.0, 1.0)},
            ))
            for i in range(5)
        ],
        depends_on=["advance_clock"],
    )
    js.run()

    world.temporal_state = {"final_tick": clock.tick, "final_time": clock.time}
    return world


def _world_hash(world: WorldIR) -> str:
    payload = json.dumps(world.to_dict(), sort_keys=True).encode("utf-8")
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
    assert world.temporal_state["final_tick"] == 7
    assert abs(world.temporal_state["final_time"] - 0.7) < 1e-9
