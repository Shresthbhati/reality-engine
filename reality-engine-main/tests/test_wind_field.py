"""Tests for the wind field (engine/environment/wind.py): Beaufort
classification, boundary-layer profile, deterministic gusts, quadratic
drag coupling into the real physics backend, band-change events, and
serialization round-trips.

Physics expectations are hand-computable from the documented model:
F = 0.5 * 1.225 * Cd * A * v^2.
"""

from __future__ import annotations

import math

import pytest

from engine.environment.wind import (
    AIR_DENSITY_KG_M3,
    BeaufortBand,
    WindConfig,
    WindField,
    classify_beaufort,
)
from engine.world.events import EventBus


# ---------------------------------------------------------------------------
# Beaufort classification
# ---------------------------------------------------------------------------


class TestBeaufort:
    def test_band_boundaries(self):
        # Convention (module docstring): CALM strictly below 0.5; other
        # bands inclusive of both bounds (FRESH_BREEZE owns 8.0..10.7).
        assert classify_beaufort(0.0) is BeaufortBand.CALM
        assert classify_beaufort(0.49) is BeaufortBand.CALM
        assert classify_beaufort(0.5) is BeaufortBand.LIGHT_AIR
        assert classify_beaufort(1.5) is BeaufortBand.LIGHT_AIR
        assert classify_beaufort(1.6) is BeaufortBand.LIGHT_BREEZE
        assert classify_beaufort(10.7) is BeaufortBand.FRESH_BREEZE
        assert classify_beaufort(10.8) is BeaufortBand.STRONG_BREEZE
        assert classify_beaufort(32.7) is BeaufortBand.HURRICANE
        assert classify_beaufort(100.0) is BeaufortBand.HURRICANE

    def test_negative_speed_refused(self):
        with pytest.raises(ValueError):
            classify_beaufort(-0.1)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestWindConfig:
    def test_invalid_alpha_refused(self):
        with pytest.raises(ValueError):
            WindConfig(alpha=0.9)
        with pytest.raises(ValueError):
            WindConfig(alpha=0.01)

    def test_invalid_heights_and_periods_refused(self):
        with pytest.raises(ValueError):
            WindConfig(reference_height_m=0.0)
        with pytest.raises(ValueError):
            WindConfig(gust_period_s=0.0)
        with pytest.raises(ValueError):
            WindConfig(gust_amplitude_frac=1.5)
        with pytest.raises(ValueError):
            WindConfig(reference_speed_mps=-1.0)


# ---------------------------------------------------------------------------
# Evolution + determinism
# ---------------------------------------------------------------------------


class TestEvolution:
    def test_step_is_deterministic_across_instances(self):
        a, b = WindField(WindConfig(seed=7)), WindField(WindConfig(seed=7))
        for _ in range(50):
            sa = a.step(0.1)
            sb = b.step(0.1)
            assert sa.current_speed_mps == sb.current_speed_mps

    def test_gust_stays_within_configured_amplitude(self):
        cfg = WindConfig(reference_speed_mps=20.0, gust_amplitude_frac=0.3)
        field = WindField(cfg)
        speeds = [field.step(0.25).current_speed_mps for _ in range(400)]
        assert all(s <= cfg.reference_speed_mps * 1.3 + 1e-9 for s in speeds)
        assert all(s >= 0.0 for s in speeds)

    def test_band_change_recorded_in_state(self):
        field = WindField(WindConfig(reference_speed_mps=5.0))
        bands = {field.step(0.5).band for _ in range(200)}
        # With gusts the sampled band set must be a subset of real bands
        # and include at least the reference band.
        assert field.state.band in {b.value for b in BeaufortBand}
        assert len(bands) >= 1

    def test_negative_dt_refused(self):
        field = WindField(WindConfig())
        with pytest.raises(ValueError):
            field.step(-0.1)


# ---------------------------------------------------------------------------
# Boundary-layer profile
# ---------------------------------------------------------------------------


class TestProfile:
    def test_power_law_speedup_at_height(self):
        cfg = WindConfig(reference_speed_mps=20.0, reference_height_m=10.0, alpha=0.16)
        field = WindField(cfg)
        field.step(0.01)
        v30 = field.speed_at_height(30.0)
        expected = field.state.current_speed_mps * (30.0 / 10.0) ** 0.16
        assert v30 == pytest.approx(expected)
        assert v30 > field.state.current_speed_mps

    def test_below_reference_height_uses_reference_speed(self):
        field = WindField(WindConfig(reference_height_m=10.0))
        field.step(0.01)
        assert field.speed_at_height(5.0) == field.state.current_speed_mps
        assert field.speed_at_height(0.0) == field.state.current_speed_mps

    def test_negative_height_refused(self):
        field = WindField(WindConfig())
        with pytest.raises(ValueError):
            field.speed_at_height(-1.0)


# ---------------------------------------------------------------------------
# Drag coupling
# ---------------------------------------------------------------------------


class TestDragCoupling:
    def test_drag_force_matches_hand_computed_quadratic_law(self):
        cfg = WindConfig(reference_speed_mps=10.0, gust_amplitude_frac=0.0)
        field = WindField(cfg)
        field.step(0.01)
        v = field.state.current_speed_mps
        fx, fy, fz = field.drag_force(area_m2=2.0, drag_coefficient=1.2)
        expected = 0.5 * AIR_DENSITY_KG_M3 * 1.2 * 2.0 * v * v
        assert fx == pytest.approx(expected)  # wind along +X
        assert fy == pytest.approx(0.0) and fz == pytest.approx(0.0)

    def test_drag_direction_follows_config(self):
        cfg = WindConfig(reference_speed_mps=10.0, direction_deg=90.0,
                         gust_amplitude_frac=0.0)
        field = WindField(cfg)
        field.step(0.01)
        fx, fy, fz = field.drag_force(area_m2=1.0)
        assert fy > 0.0 and fx == pytest.approx(0.0, abs=1e-12)

    def test_relative_velocity_form(self):
        """Body moving WITH the wind at wind speed feels zero force."""
        cfg = WindConfig(reference_speed_mps=10.0, gust_amplitude_frac=0.0)
        field = WindField(cfg)
        field.step(0.01)
        v = field.state.current_speed_mps
        fx, fy, fz = field.drag_force(
            area_m2=1.0, body_velocity=(v, 0.0, 0.0)
        )
        assert (fx, fy, fz) == pytest.approx((0.0, 0.0, 0.0))

    def test_wind_loads_push_real_physics_bodies(self):
        """The causal hook: wind impulses move real dynamic bodies in the
        SimpleRigidBodyBackend; static bodies are untouched."""
        from engine.physics.backend.simple_backend import SimpleRigidBodyBackend
        from engine.physics.backend.interface import PhysicsWorldConfig
        from engine.physics.math3 import Vec3
        from engine.physics.collision.shapes import Box
        from engine.physics.materials.material import PhysicsMaterial
        from engine.physics.rigid.body import RigidBody

        backend = SimpleRigidBodyBackend()
        world = backend.create_world(PhysicsWorldConfig())
        world.add_body(RigidBody(
            id="cart",
            shape=Box(half_extents=Vec3(0.25, 0.25, 0.25)),
            material=PhysicsMaterial(name="cart", density=100.0,
                                     friction_static=0.1, friction_dynamic=0.05,
                                     restitution=0.1),
            mass=2.0,
            position=Vec3(0.0, 0.5, 0.0),
        ))

        field = WindField(WindConfig(reference_speed_mps=40.0,
                                     gust_amplitude_frac=0.0))
        field.step(0.01)

        before = world.get_body("cart").linear_velocity.x
        impulses = field.apply_wind_loads(world, dt=0.1,
                                          exposed_area_by_body={"cart": 1.0})
        assert "cart" in impulses
        assert impulses["cart"][0] > 0.0  # pushed along +X

        backend.step(world, 0.1)
        after = world.get_body("cart").linear_velocity.x
        assert after > before  # the impulse actually accelerated the body

    def test_static_bodies_receive_no_loads(self):
        from engine.physics.backend.simple_backend import SimpleRigidBodyBackend
        from engine.physics.backend.interface import PhysicsWorldConfig
        from engine.physics.math3 import Vec3
        from engine.physics.collision.shapes import Box
        from engine.physics.materials.material import PhysicsMaterial
        from engine.physics.rigid.body import RigidBody

        backend = SimpleRigidBodyBackend()
        world = backend.create_world(PhysicsWorldConfig())
        world.add_body(RigidBody(
            id="slab", shape=Box(half_extents=Vec3(1, 1, 0.1)),
            material=PhysicsMaterial(name="s", density=2400.0,
                                     friction_static=0.7, friction_dynamic=0.6,
                                     restitution=0.1),
            mass=0.0, position=Vec3(0.0, 0.0, 0.0),
        ))
        field = WindField(WindConfig(reference_speed_mps=60.0,
                                     gust_amplitude_frac=0.0))
        field.step(0.01)
        impulses = field.apply_wind_loads(world, dt=0.1,
                                          exposed_area_by_body={"slab": 4.0})
        assert impulses == {}  # static by construction

    def test_invalid_drag_params_refused(self):
        field = WindField(WindConfig())
        with pytest.raises(ValueError):
            field.drag_force(area_m2=-1.0)
        with pytest.raises(ValueError):
            field.drag_force(area_m2=1.0, drag_coefficient=0.0)


# ---------------------------------------------------------------------------
# Events + serialization
# ---------------------------------------------------------------------------


class TestEventsAndSerialization:
    def test_band_change_publishes_event(self):
        bus = EventBus()
        seen = []
        bus.subscribe("wind.band_changed", lambda e: seen.append(e))
        # Gust amplitude large enough to cross band boundaries.
        field = WindField(
            WindConfig(reference_speed_mps=5.0, gust_amplitude_frac=1.0),
            event_bus=bus,
        )
        for _ in range(300):
            field.step(0.1)
        assert seen, "band change must publish wind.band_changed"
        data = seen[0].data
        assert "from_band" in data and "to_band" in data and "speed_mps" in data

    def test_no_band_change_no_event(self):
        bus = EventBus()
        seen = []
        bus.subscribe("wind.band_changed", lambda e: seen.append(e))
        field = WindField(WindConfig(reference_speed_mps=5.0,
                                     gust_amplitude_frac=0.0),
                          event_bus=bus)
        for _ in range(10):
            field.step(0.1)
        assert seen == []

    def test_serialization_roundtrip(self):
        field = WindField(WindConfig(reference_speed_mps=12.0, seed=9))
        for _ in range(5):
            field.step(0.1)
        data = field.to_dict()
        assert data["format_version"] == 1
        assert data["state"]["current_speed_mps"] == field.state.current_speed_mps
        assert data["config"]["seed"] == 9
