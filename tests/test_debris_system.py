"""Tests for debris system."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.physics

try:
    from engine.physics.destruction import DebrisManager, DebrisConfig, DebrisFragment
    from engine.physics.math3 import Vec3
except ImportError:
    pytest.skip("engine.physics module not available - requires reality-engine-child", allow_module_level=True)


class TestDebrisFragment:
    """Debris fragment lifecycle tests."""

    def test_create_fragment(self):
        """Test creating a debris fragment."""
        fragment = DebrisFragment(
            "frag_1",
            Vec3(0, 0, 1),
            Vec3(1, 0, 0),
            size_m=0.1,
            mass_kg=0.5,
            lifetime_s=10.0,
            created_at_tick=0,
            sharpness=0.8,
        )

        assert fragment.fragment_id == "frag_1"
        assert fragment.position == Vec3(0, 0, 1)
        assert fragment.velocity == Vec3(1, 0, 0)
        assert fragment.age_s == 0.0
        assert not fragment.is_sleeping
        assert not fragment.is_expired()

    def test_fragment_age_increases(self):
        """Test fragment aging."""
        fragment = DebrisFragment(
            "frag_1",
            Vec3(0, 0, 1),
            Vec3(0, 0, 0),
            size_m=0.1,
            mass_kg=0.5,
            lifetime_s=10.0,
            created_at_tick=0,
        )

        fragment.step(1.0, 9.81)
        assert fragment.age_s == 1.0

        fragment.step(2.0, 9.81)
        assert fragment.age_s == 3.0

    def test_fragment_expiry(self):
        """Test fragment lifetime expiry."""
        fragment = DebrisFragment(
            "frag_1",
            Vec3(0, 0, 1),
            Vec3(0, 0, 0),
            size_m=0.1,
            mass_kg=0.5,
            lifetime_s=1.0,
            created_at_tick=0,
        )

        fragment.step(0.5, 9.81)
        assert not fragment.is_expired()

        fragment.step(0.6, 9.81)
        assert fragment.is_expired()

    def test_fragment_gravity_application(self):
        """Test gravity is applied to velocity."""
        fragment = DebrisFragment(
            "frag_1",
            Vec3(0, 10, 0),
            Vec3(0, 0, 0),
            size_m=0.1,
            mass_kg=1.0,
            lifetime_s=10.0,
            created_at_tick=0,
        )

        fragment.step(1.0, 9.81)
        assert fragment.velocity.y == pytest.approx(-9.81)

    def test_fragment_air_resistance(self):
        """Test air resistance slows velocity."""
        fragment = DebrisFragment(
            "frag_1",
            Vec3(0, 0, 0),
            Vec3(10, 0, 0),
            size_m=0.1,
            mass_kg=1.0,
            lifetime_s=10.0,
            created_at_tick=0,
        )

        fragment.step(1.0, 0.0)  # No gravity
        # Velocity should decrease due to air resistance
        assert fragment.velocity.x < 10.0


class TestDebrisManager:
    """Debris manager pool and lifecycle tests."""

    def test_create_pool(self):
        """Test creating a debris pool."""
        pool = DebrisManager(initial_capacity=10)
        assert pool.stats().total_capacity == 10

    def test_acquire_fragment(self):
        """Test acquiring a fragment from pool."""
        pool = DebrisManager(initial_capacity=10)
        fragment = pool.acquire(
            "frag_1",
            Vec3(0, 0, 1),
            Vec3(1, 0, 0),
            size_m=0.1,
            mass_kg=0.5,
            lifetime_s=10.0,
        )

        assert fragment.fragment_id == "frag_1"
        assert pool.stats().active_count == 1

    def test_release_fragment(self):
        """Test releasing fragment back to pool."""
        pool = DebrisManager(initial_capacity=10)
        fragment = pool.acquire("frag_1", Vec3(0, 0, 1), Vec3(1, 0, 0), 0.1, 0.5, 10.0)
        pool.release("frag_1")

        assert pool.stats().active_count == 0

    def test_release_nonexistent(self):
        """Test releasing non-existent fragment."""
        pool = DebrisManager(initial_capacity=10)
        pool.release("nonexistent")  # Should not raise

    def test_pool_capacity_expansion(self):
        """Test pool expands when capacity exceeded."""
        pool = DebrisManager(initial_capacity=2)
        f1 = pool.acquire("f1", Vec3(0, 0, 0), Vec3(0, 0, 0), 0.1, 1.0, 10.0)
        f2 = pool.acquire("f2", Vec3(0, 0, 0), Vec3(0, 0, 0), 0.1, 1.0, 10.0)
        f3 = pool.acquire("f3", Vec3(0, 0, 0), Vec3(0, 0, 0), 0.1, 1.0, 10.0)

        # Pool should expand
        assert pool.stats().total_capacity >= 3

    def test_fragment_id_uniqueness(self):
        """Test fragment IDs must be unique in pool."""
        pool = DebrisManager(initial_capacity=10)
        pool.acquire("frag_1", Vec3(0, 0, 1), Vec3(1, 0, 0), 0.1, 0.5, 10.0)

        with pytest.raises(ValueError):
            pool.acquire("frag_1", Vec3(0, 0, 1), Vec3(1, 0, 0), 0.1, 0.5, 10.0)

    def test_pool_stats(self):
        """Test pool statistics."""
        pool = DebrisManager(initial_capacity=5)
        pool.acquire("f1", Vec3(0, 0, 0), Vec3(0, 0, 0), 0.1, 1.0, 10.0)
        pool.acquire("f2", Vec3(0, 0, 0), Vec3(0, 0, 0), 0.1, 1.0, 10.0)

        stats = pool.stats()
        assert stats.total_capacity == 5
        assert stats.active_count == 2
        assert stats.inactive_count == 3


class TestDebrisConfig:
    """Debris configuration tests."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DebrisConfig()
        assert config.gravity == 9.81
        assert config.air_resistance > 0.0
        assert config.max_lifetime_s > 0.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = DebrisConfig(gravity=3.71, air_resistance=0.02, max_lifetime_s=5.0)
        assert config.gravity == 3.71
        assert config.air_resistance == 0.02
        assert config.max_lifetime_s == 5.0


class TestDebrisManagerStep:
    """Test DebrisManager step integration."""

    def test_step_updates_all_fragments(self):
        """Test step advances all active fragments."""
        pool = DebrisManager(initial_capacity=10)
        pool.acquire("f1", Vec3(0, 10, 0), Vec3(0, 0, 0), 0.1, 1.0, 10.0)

        pool.step(1.0)
        pool.step(1.0)

        stats = pool.stats()
        assert stats.active_count == 1

    def test_step_removes_expired(self):
        """Test expired fragments are removed from pool."""
        pool = DebrisManager(initial_capacity=10)
        pool.acquire("f1", Vec3(0, 0, 0), Vec3(0, 0, 0), 0.1, 1.0, 0.5)

        # Step past lifetime
        pool.step(1.0)

        assert pool.stats().active_count == 0

    def test_step_applies_gravity(self):
        """Test gravity applied to all fragments."""
        pool = DebrisManager(initial_capacity=10)
        pool.acquire("f1", Vec3(0, 10, 0), Vec3(0, 0, 0), 0.1, 1.0, 10.0)

        pool.step(1.0)

        fragment = pool.get_fragment("f1")
        assert fragment.velocity.y < 0  # Gravity pulled down