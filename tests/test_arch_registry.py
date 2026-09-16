"""Tests for the architectural class registry (directive sections 4,
32): an EXTENSIBLE registry mapping semantic classes to WorldIR entity
types and per-class geometric acceptance thresholds, so new component
classes (cornices, balustrades, statues, ...) can be added WITHOUT
redesigning the pipeline.

Rules under test:
  - Built-in Phase-1 classes are registered with their WorldIR type,
    required fit kind, and acceptance predicate parameters.
  - `register()` adds a new class at runtime; re-registering the same
    name is an error (silent overrides would let one import change
    another module's semantics).
  - Registry lookups never guess: an unknown class name raises.
  - Every class declares its directive phase (1-4) so phasing is data,
    not folklore.
  - Acceptance thresholds are evaluated through the registry, so the
    per-class honesty gates live in one place.
"""

from __future__ import annotations

import pytest

from perception.architecture.registry import (
    ArchClass,
    ArchitectureRegistry,
    FitKind,
    get_default_registry,
)


class TestBuiltinClasses:
    def test_phase1_classes_present(self):
        reg = get_default_registry()
        for name in ("wall", "floor", "ceiling", "roof", "column",
                     "door", "window", "arch", "dome", "stairs"):
            assert reg.has(name), f"missing Phase-1 class {name!r}"

    def test_column_maps_to_worldir_column(self):
        reg = get_default_registry()
        cls = reg.get("column")
        assert cls.worldir_type == "column"
        assert cls.fit_kind is FitKind.CYLINDER

    def test_dome_and_arch_kinds(self):
        reg = get_default_registry()
        assert reg.get("dome").fit_kind is FitKind.SPHERE
        assert reg.get("arch").fit_kind is FitKind.CIRCLE
        # Wall comes from the plane classifier, not a parametric fit.
        assert reg.get("wall").fit_kind is FitKind.PLANE

    def test_phase_declared(self):
        reg = get_default_registry()
        assert reg.get("column").phase == 1
        assert reg.get("cornice").phase == 2
        assert reg.get("statue").phase == 3

    def test_unknown_class_raises(self):
        reg = get_default_registry()
        with pytest.raises(KeyError):
            reg.get("flying_buttress")

    def test_no_silent_override(self):
        reg = get_default_registry()
        with pytest.raises(ValueError):
            reg.register(ArchClass(
                name="column", worldir_type="column",
                fit_kind=FitKind.CYLINDER, phase=1,
            ))


class TestExtensibility:
    def test_new_class_registered_and_resolvable(self):
        reg = get_default_registry()
        fresh = ArchitectureRegistry()
        fresh.register(ArchClass(
            name="minaret", worldir_type="structure",
            fit_kind=FitKind.CYLINDER, phase=2,
            min_axis_up_dot=0.9,
        ))
        cls = fresh.get("minaret")
        assert cls.phase == 2
        assert cls.min_axis_up_dot == pytest.approx(0.9)
        # The default registry is untouched (no cross-import leakage).
        assert not get_default_registry().has("minaret")

    def test_accepts_thresholds_defaulted_not_fabricated(self):
        reg = get_default_registry()
        cls = reg.get("arch")
        # Registry stores thresholds; the components layer evaluates
        # them. An arch needs its measured angular span gate.
        assert cls.min_angular_span_rad is not None
        assert cls.max_angular_span_rad is not None
        assert cls.min_angular_span_rad < cls.max_angular_span_rad
