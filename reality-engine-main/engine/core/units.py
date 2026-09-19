"""Canonical unit system (spec sec 88 UNITS).

No subsystem may silently mix units. Every physical quantity that flows
between subsystems must be tagged with its canonical unit so a mismatch
raises immediately instead of producing a silently wrong simulation.
"""

from __future__ import annotations

from enum import Enum


class Unit(str, Enum):
    """Canonical SI units. All values crossing a solver boundary are in
    exactly these units — no subsystem converts implicitly."""

    METER = "meter"
    KILOGRAM = "kilogram"
    SECOND = "second"
    METER_PER_SECOND = "meter/second"
    METER_PER_SECOND2 = "meter/second^2"
    NEWTON = "newton"
    PASCAL = "pascal"
    JOULE = "joule"
    KELVIN = "kelvin"
    DIMENSIONLESS = "dimensionless"


class UnitMismatchError(ValueError):
    """Raised when a value tagged with one unit is combined with another."""


class Quantity:
    """A scalar value paired with its canonical unit.

    Arithmetic between Quantities of different units raises
    UnitMismatchError instead of silently producing a wrong number —
    this is the enforcement point for the "no mixed units" rule.
    """

    __slots__ = ("value", "unit")

    def __init__(self, value: float, unit: Unit):
        self.value = float(value)
        self.unit = unit

    def _check(self, other: "Quantity") -> None:
        if not isinstance(other, Quantity):
            raise UnitMismatchError(
                f"cannot combine Quantity({self.unit}) with non-Quantity {other!r}"
            )
        if other.unit != self.unit:
            raise UnitMismatchError(f"unit mismatch: {self.unit} vs {other.unit}")

    def __add__(self, other: "Quantity") -> "Quantity":
        self._check(other)
        return Quantity(self.value + other.value, self.unit)

    def __sub__(self, other: "Quantity") -> "Quantity":
        self._check(other)
        return Quantity(self.value - other.value, self.unit)

    def __mul__(self, scalar: float) -> "Quantity":
        if isinstance(scalar, Quantity):
            raise UnitMismatchError(
                "Quantity * Quantity is not defined; derive a new unit explicitly"
            )
        return Quantity(self.value * scalar, self.unit)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> "Quantity":
        if isinstance(scalar, Quantity):
            raise UnitMismatchError(
                "Quantity / Quantity is not defined; derive a new unit explicitly"
            )
        return Quantity(self.value / scalar, self.unit)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Quantity)
            and self.unit == other.unit
            and self.value == other.value
        )

    def __repr__(self) -> str:
        return f"Quantity({self.value!r}, {self.unit.value})"

    def to_dict(self) -> dict:
        return {"value": self.value, "unit": self.unit.value}

    @staticmethod
    def from_dict(data: dict) -> "Quantity":
        return Quantity(data["value"], Unit(data["unit"]))
