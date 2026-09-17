"""Destruction event data (spec §12).

FractureEvent is the shared record type produced by fracture/glass/debris
solvers. It has no common abstract backend interface -- glass.py and
debris.py never implemented the previous IDestructionBackend ABC, and
nothing in the codebase dispatches over it polymorphically, so it was
pure unused ceremony. If a real second destruction backend needs a
shared contract later, reintroduce the interface then.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.physics.math3 import Vec3


@dataclass
class FractureEvent:
    """An object has fractured."""
    entity_id: str
    impact_point: Vec3
    impact_velocity: float
    impact_normal: Vec3
    fragment_count: int
    energy_released_j: float
    tick: int
    timestamp: float
