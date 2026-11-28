"""World lifecycle management: initialization, running, shutdown states.

Invariants:
  - World transitions through states in order: CREATED -> INITIALIZED -> RUNNING -> SHUTDOWN
  - Cannot transition backwards
  - Cannot step world unless in RUNNING state
  - All systems notified on state transitions
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional
from dataclasses import dataclass
import time


class WorldState(str, Enum):
    """World lifecycle state."""
    CREATED = "CREATED"        # Initial state
    INITIALIZED = "INITIALIZED"  # Ready to run
    RUNNING = "RUNNING"        # Simulation active
    SHUTDOWN = "SHUTDOWN"      # Terminated


@dataclass
class LifecycleMetrics:
    """Metrics about world lifecycle."""
    state: WorldState
    ticks_simulated: int = 0
    real_time_ms: float = 0.0  # Real time elapsed
    sim_time_s: float = 0.0    # Simulation time elapsed


class WorldLifecycle:
    """Manages world state transitions and lifecycle events.

    Ensures proper initialization, running, and shutdown sequences.
    Prevents invalid state transitions. Notifies observers of changes.

    Thread-unsafe; designed for single-threaded simulation.
    """

    def __init__(self):
        self._state: WorldState = WorldState.CREATED
        self._metrics = LifecycleMetrics(state=WorldState.CREATED)
        self._state_change_observers: list[Callable[[WorldState, WorldState], None]] = []
        self._start_time_ms: Optional[float] = None

    @property
    def state(self) -> WorldState:
        """Get current world state."""
        return self._state

    @property
    def metrics(self) -> LifecycleMetrics:
        """Get lifecycle metrics."""
        return self._metrics

    def initialize(self) -> None:
        """Initialize the world.

        Transition from CREATED to INITIALIZED.

        Raises:
            ValueError: If not in CREATED state
        """
        if self._state != WorldState.CREATED:
            raise ValueError(f"Cannot initialize world in state {self._state.value}")

        old_state = self._state
        self._state = WorldState.INITIALIZED
        self._metrics.state = self._state

        for observer in self._state_change_observers:
            observer(old_state, self._state)

    def run(self) -> None:
        """Start running the world.

        Transition from INITIALIZED to RUNNING.

        Raises:
            ValueError: If not in INITIALIZED state
        """
        if self._state != WorldState.INITIALIZED:
            raise ValueError(f"Cannot run world in state {self._state.value}")

        old_state = self._state
        self._state = WorldState.RUNNING
        self._metrics.state = self._state
        self._start_time_ms = time.time() * 1000  # Convert to ms

        for observer in self._state_change_observers:
            observer(old_state, self._state)

    def step(self, dt: float) -> None:
        """Record a simulation step.

        Updates metrics. Must be in RUNNING state.

        Args:
            dt: Delta time for this step (seconds)

        Raises:
            ValueError: If not in RUNNING state
        """
        if self._state != WorldState.RUNNING:
            raise ValueError(f"Cannot step world in state {self._state.value}")

        self._metrics.ticks_simulated += 1
        self._metrics.sim_time_s += dt

        if self._start_time_ms is not None:
            elapsed_ms = time.time() * 1000 - self._start_time_ms
            self._metrics.real_time_ms = elapsed_ms

    def shutdown(self) -> None:
        """Shut down the world.

        Transition from RUNNING (or INITIALIZED) to SHUTDOWN.

        Raises:
            ValueError: If already shutdown or in CREATED state
        """
        if self._state == WorldState.CREATED:
            raise ValueError("Cannot shutdown world in CREATED state")
        if self._state == WorldState.SHUTDOWN:
            raise ValueError("World already shut down")

        old_state = self._state
        self._state = WorldState.SHUTDOWN
        self._metrics.state = self._state

        for observer in self._state_change_observers:
            observer(old_state, self._state)

    def observe_state_change(self, observer: Callable[[WorldState, WorldState], None]) -> None:
        """Register an observer for state changes.

        Observer is called with (old_state, new_state).
        """
        self._state_change_observers.append(observer)

    def can_step(self) -> bool:
        """Check if world can be stepped."""
        return self._state == WorldState.RUNNING

    def clear(self) -> None:
        """Reset to initial state (for testing)."""
        self._state = WorldState.CREATED
        self._metrics = LifecycleMetrics(state=WorldState.CREATED)
        self._state_change_observers.clear()
        self._start_time_ms = None
