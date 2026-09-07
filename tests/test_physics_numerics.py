import pytest

from engine.physics.collision.shapes import Sphere
from engine.physics.diagnostics.numerics import SimulationDivergedError, check_world
from engine.physics.materials import CANONICAL_MATERIALS
from engine.physics.rigid.body import RigidBody
from engine.physics.math3 import Vec3


def make_body(**overrides) -> RigidBody:
    kwargs = dict(id="b", shape=Sphere(1.0), material=CANONICAL_MATERIALS["steel"], mass=1.0)
    kwargs.update(overrides)
    return RigidBody(**kwargs)


def test_healthy_state_reports_zero_counts():
    body = make_body(linear_velocity=Vec3(1, 2, 3))
    report = check_world([body], previous_energy=None)
    assert report.nan_count == 0
    assert report.inf_count == 0
    assert report.body_count == 1


def test_nan_velocity_raises():
    body = make_body(linear_velocity=Vec3(float("nan"), 0, 0))
    with pytest.raises(SimulationDivergedError):
        check_world([body], previous_energy=None)


def test_inf_velocity_raises():
    body = make_body(linear_velocity=Vec3(float("inf"), 0, 0))
    with pytest.raises(SimulationDivergedError):
        check_world([body], previous_energy=None)


def test_explosive_growth_above_absolute_floor_raises():
    body = make_body(linear_velocity=Vec3(1000, 0, 0))  # huge KE this tick
    with pytest.raises(SimulationDivergedError):
        check_world([body], previous_energy=1e-6)


def test_growth_below_absolute_floor_is_not_flagged():
    body = make_body(linear_velocity=Vec3(0.01, 0, 0))  # tiny KE, still "explosive" in ratio
    report = check_world([body], previous_energy=1e-12)
    assert report is not None


def test_max_velocity_tracks_largest_body():
    slow = make_body(id="slow", linear_velocity=Vec3(1, 0, 0))
    fast = make_body(id="fast", linear_velocity=Vec3(0, 5, 0))
    report = check_world([slow, fast], previous_energy=None)
    assert report.max_velocity == pytest.approx(5.0)
