"""Deterministic simulation clock (spec sec 90 DETERMINISM, sec 83 CORE
INTERFACES: SimulationContext.{time, dt, tick, seed}).

Fixed timestep, monotonic tick counter, stable across CPU/platform since
it does no floating-point accumulation of dt (time = tick * dt exactly).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationContext:
    """Passed to every ISolver.step() call (spec sec 83)."""

    time: float
    dt: float
    tick: int
    seed: int


class DeterministicClock:
    """Fixed-timestep clock. Advancing N ticks from the same seed and dt
    always yields the same sequence of SimulationContext values.
    """

    def __init__(self, dt: float, seed: int = 0):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = dt
        self.seed = seed
        self.tick = 0

    @property
    def time(self) -> float:
        # tick * dt, never dt accumulated in a loop -- avoids float drift.
        return self.tick * self.dt

    def context(self) -> SimulationContext:
        return SimulationContext(time=self.time, dt=self.dt, tick=self.tick, seed=self.seed)

    def advance(self) -> SimulationContext:
        self.tick += 1
        return self.context()

    def reset(self) -> None:
        self.tick = 0

    def run(self, n_ticks: int):
        """Yield SimulationContext for n_ticks starting from tick 0/time 0."""
        self.reset()
        yield self.context()
        for _ in range(n_ticks):
            yield self.advance()
