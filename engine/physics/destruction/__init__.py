"""Destruction systems: fracture, debris, damage.

Tier 5 subsystems implementing §12-14 (basic fracture, glass physics, debris).
"""

from .interface import FractureEvent
from .fracture import BasicFractureSolver
from .glass import GlassPhysicsSolver, GlassPane, GlassTemper
from .debris import DebrisManager, DebrisConfig, DebrisFragment, DebrisPool

__all__ = [
    "FractureEvent",
    "BasicFractureSolver",
    "GlassPhysicsSolver",
    "GlassPane",
    "GlassTemper",
    "DebrisManager",
    "DebrisConfig",
    "DebrisFragment",
    "DebrisPool",
]
