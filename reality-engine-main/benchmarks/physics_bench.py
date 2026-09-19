"""Real benchmarks against SimpleRigidBodyBackend and the command pipeline."""

from __future__ import annotations

from benchmarks.harness import BenchmarkResult, run_benchmark
from engine.commands.commands import CreateEntityCommand
from engine.commands.processor import WorldCommandProcessor
from engine.physics.backend import PhysicsWorldConfig, SimpleRigidBodyBackend
from engine.physics.collision.shapes import Box
from engine.physics.materials import CANONICAL_MATERIALS
from engine.physics.math3 import Vec3
from engine.physics.rigid.body import RigidBody
from events import EventBus
from world_ir import WorldIR


def _build_body_world(body_count: int):
    backend = SimpleRigidBodyBackend()
    world = backend.create_world(PhysicsWorldConfig(gravity=Vec3(0, -9.81, 0), seed=1))
    for i in range(body_count):
        world.add_body(RigidBody(
            id=f"body-{i}",
            shape=Box(Vec3(0.5, 0.5, 0.5)),
            material=CANONICAL_MATERIALS["wood"],
            mass=1.0,
            position=Vec3(float(i % 10), 5.0 + i, float(i // 10)),
        ))
    return backend, world


def bench_rigid_body_step(body_count: int, n_ticks: int = 100, iterations: int = 10) -> BenchmarkResult:
    """Times stepping a fixed-size rigid body world for n_ticks, iterations times.
    Each iteration builds a fresh world so tick count and op count stay identical
    across iterations (deterministic operation count).
    """
    def run_once():
        backend, world = _build_body_world(body_count)
        for _ in range(n_ticks):
            backend.step(world, 0.01)

    return run_benchmark(
        name=f"rigid_body_step_n{body_count}",
        iterations=iterations,
        fn=run_once,
        metadata={"body_count": body_count, "ticks_per_iteration": n_ticks},
    )


def bench_rigid_body_step_10() -> BenchmarkResult:
    return bench_rigid_body_step(body_count=10)


def bench_rigid_body_step_100() -> BenchmarkResult:
    return bench_rigid_body_step(body_count=100)


def bench_command_pipeline_throughput(command_count: int = 1000, iterations: int = 5) -> BenchmarkResult:
    """Times executing `command_count` CreateEntityCommands back to back,
    `iterations` times (fresh WorldIR/EventBus/processor each time).
    """
    def run_once():
        world = WorldIR()
        bus = EventBus()
        processor = WorldCommandProcessor(world, bus)
        for i in range(command_count):
            processor.execute(CreateEntityCommand(entity_id=f"ent-{i}", entity_type="debris"))

    return run_benchmark(
        name=f"command_pipeline_create_n{command_count}",
        iterations=iterations,
        fn=run_once,
        metadata={"command_count": command_count},
    )


BENCHMARKS = [
    bench_rigid_body_step_10,
    bench_rigid_body_step_100,
    bench_command_pipeline_throughput,
]
