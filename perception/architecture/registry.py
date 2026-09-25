"""Architectural class registry (directive sections 4 + 32): the
EXTENSIBLE table that maps semantic component classes to WorldIR entity
types, the parametric fit kind that produces their geometry, per-class
geometric acceptance thresholds, and the directive phase they belong
to.

Why a registry: new component classes (cornices, balustrades,
minarets, statues, ...) must be addable WITHOUT redesigning the
pipeline. Every consumer (fitting, acceptance, promotion, benchmark
reporting) reads thresholds and mappings from here, so adding a class
is adding a row -- not editing consumers.

Honesty rules carried over from the rest of perception/:
  - Thresholds are stored facts about what this class accepts, not
    per-callsite magic numbers.
  - A class whose thresholds are unknown declares them as None --
    unknown stays unknown; the components layer refuses rather than
    guessing.
  - Re-registering a name is refused (silent overrides would let one
    import change another module's semantics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from enum import Enum


class FitKind(str, Enum):
    """Which geometric fit produces this class's component geometry."""

    PLANE = "plane"          # wall/floor/ceiling/roof facets
    CYLINDER = "cylinder"    # columns, pillars, minarets, masts
    SPHERE = "sphere"        # domes (hemispherical and ellipsoidal approx)
    CIRCLE = "circle"        # arches (cross-section ring + span)
    POINT_CLOUD = "point_cloud"  # classes whose geometry is the raw segment


@dataclass(frozen=True)
class ArchClass:
    """One architectural component class's contract."""

    name: str
    worldir_type: str
    fit_kind: FitKind
    #: Directive phase (1-4) this class is scheduled in. Data, not
    #: folklore: the benchmark reports progress per phase from this.
    phase: int
    #: --- Per-class geometric acceptance thresholds (all optional:
    #: None means "no gate of this kind" -- never a fabricated zero).
    #: Cylinder classes: fitted axis must be this close to vertical.
    min_axis_up_dot: Optional[float] = None
    #: Circle/arch classes: measured angular span window.
    min_angular_span_rad: Optional[float] = None
    max_angular_span_rad: Optional[float] = None
    #: Sphere classes: fraction of points required above the fitted
    #: center plane (dome-like, not full-bubble).
    min_upper_fraction: Optional[float] = None
    #: Minimum measured extent (meters) for extent-gated classes.
    min_extent_m: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "worldir_type": self.worldir_type,
            "fit_kind": self.fit_kind.value,
            "phase": self.phase,
            "min_axis_up_dot": self.min_axis_up_dot,
            "min_angular_span_rad": self.min_angular_span_rad,
            "max_angular_span_rad": self.max_angular_span_rad,
            "min_upper_fraction": self.min_upper_fraction,
            "min_extent_m": self.min_extent_m,
        }


def _default_registry() -> "ArchitectureRegistry":
    reg = ArchitectureRegistry()
    # ---- Phase 1 (directive section 32) ----
    reg.register(ArchClass("wall", "wall", FitKind.PLANE, 1))
    reg.register(ArchClass("floor", "floor", FitKind.PLANE, 1))
    reg.register(ArchClass("ceiling", "ceiling", FitKind.PLANE, 1))
    reg.register(ArchClass("roof", "roof", FitKind.PLANE, 1))
    reg.register(ArchClass(
        "column", "column", FitKind.CYLINDER, 1, min_axis_up_dot=0.9,
    ))
    reg.register(ArchClass("door", "door", FitKind.PLANE, 1))
    reg.register(ArchClass("window", "window", FitKind.PLANE, 1))
    reg.register(ArchClass(
        "arch", "structure", FitKind.CIRCLE, 1,
        min_angular_span_rad=0.9 * 3.141592653589793,
        max_angular_span_rad=2.05 * 3.141592653589793,
    ))
    reg.register(ArchClass(
        "dome", "structure", FitKind.SPHERE, 1, min_upper_fraction=0.45,
    ))
    reg.register(ArchClass(
        "stairs", "stairs", FitKind.POINT_CLOUD, 1, min_extent_m=0.5,
    ))
    reg.register(ArchClass(
        "corridor", "corridor", FitKind.POINT_CLOUD, 1,
        min_extent_m=2.0,
    ))
    # ---- Phase 2 ----
    reg.register(ArchClass("cornice", "beam", FitKind.PLANE, 2))
    reg.register(ArchClass("balustrade", "wall", FitKind.POINT_CLOUD, 2))
    reg.register(ArchClass("pediment", "roof", FitKind.PLANE, 2))
    reg.register(ArchClass(
        "tower", "structure", FitKind.CYLINDER, 2, min_axis_up_dot=0.85,
    ))
    reg.register(ArchClass("spire", "structure", FitKind.POINT_CLOUD, 2))
    # ---- Phase 3 ----
    reg.register(ArchClass("statue", "debris", FitKind.POINT_CLOUD, 3))
    # ---- Building/site container classes (used by the graph) ----
    reg.register(ArchClass("building", "building", FitKind.POINT_CLOUD, 1))
    reg.register(ArchClass("facade", "wall", FitKind.POINT_CLOUD, 1))
    return reg


class ArchitectureRegistry:
    """Name -> ArchClass table with no silent overrides and no guessed
    lookups."""

    def __init__(self) -> None:
        self._classes: Dict[str, ArchClass] = {}

    def register(self, cls: ArchClass) -> None:
        if cls.name in self._classes:
            raise ValueError(
                f"architectural class {cls.name!r} is already registered -- "
                f"re-registration would silently change semantics; use a "
                f"new name or an explicit replacement API"
            )
        self._classes[cls.name] = cls

    def has(self, name: str) -> bool:
        return name in self._classes

    def get(self, name: str) -> ArchClass:
        if name not in self._classes:
            raise KeyError(
                f"unknown architectural class {name!r} -- registered: "
                f"{sorted(self._classes)}"
            )
        return self._classes[name]

    def names(self) -> tuple:
        return tuple(sorted(self._classes))

    def by_phase(self, phase: int) -> tuple:
        return tuple(
            self._classes[n] for n in sorted(self._classes)
            if self._classes[n].phase == phase
        )


_DEFAULT: Optional[ArchitectureRegistry] = None


def get_default_registry() -> ArchitectureRegistry:
    """The shared default registry (built-ins only; external modules
    register into their own instances or go through an explicit
    replacement API -- never silently mutate this one)."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = _default_registry()
    return _DEFAULT
