import pytest

from engine.core.units import Quantity, Unit, UnitMismatchError


def test_same_unit_arithmetic():
    a = Quantity(3.0, Unit.METER)
    b = Quantity(4.0, Unit.METER)
    assert (a + b) == Quantity(7.0, Unit.METER)
    assert (a - b) == Quantity(-1.0, Unit.METER)


def test_scalar_multiplication():
    a = Quantity(3.0, Unit.KILOGRAM)
    assert (a * 2) == Quantity(6.0, Unit.KILOGRAM)
    assert (2 * a) == Quantity(6.0, Unit.KILOGRAM)


def test_mismatched_units_raise():
    length = Quantity(1.0, Unit.METER)
    mass = Quantity(1.0, Unit.KILOGRAM)
    with pytest.raises(UnitMismatchError):
        length + mass


def test_quantity_quantity_multiplication_raises():
    a = Quantity(1.0, Unit.METER)
    b = Quantity(1.0, Unit.METER)
    with pytest.raises(UnitMismatchError):
        a * b


def test_roundtrip_dict():
    q = Quantity(9.81, Unit.METER_PER_SECOND2)
    assert Quantity.from_dict(q.to_dict()) == q
