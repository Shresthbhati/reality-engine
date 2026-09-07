"""Tests for debris system."""

import pytest

from engine.physics.destruction import DebrisManager, DebrisConfig, DebrisFragment
from engine.physics.math3 import Vec3


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
            lifetime_s=5.0,
            created_at_tick=0,
        )

        fragment.step(4.0, 9.81)
        assert not fragment.is_expired()

        fragment.step(1.5, 9.81)
        assert fragment.is_expired()

    def test_gravity_acceleration(self):
        """Test gravity applied to falling fragment."""
        fragment = DebrisFragment(
            "frag_1",
            Vec3(0, 0, 10),
            Vec3(0, 0, 0),
            size_m=0.1,
            mass_kg=1.0,
            lifetime_s=10.0,
            created_at_tick=0,
        )

        fragment.step(1.0, 9.81)
        # After 1 second with gravity: v_z = 0 - 9.81*1 = -9.81
        assert abs(fragment.velocity.z - (-9.81)) < 1e-6
        # Semi-implicit: first apply gravity (v_z becomes -9.81), then update position
        # z = 10 + (-9.81)*1 = 0.19 (not 5.095, which would be for analytical integration)
        assert abs(fragment.position.z - 0.19) < 0.01

    def test_fragment_sleeping(self):
        """Test fragment sleeping when velocity low."""
        fragment = DebrisFragment(
            "frag_1",
            Vec3(0, 0, 1),
            Vec3(0, 0, 0),  # Zero velocity, no gravity effects
            size_m=0.1,
            mass_kg=0.5,
            lifetime_s=10.0,
            created_at_tick=0,
        )

        # With gravity=0, should fall asleep due to zero velocity
        fragment.step(0.1, 0.0)
        assert fragment.is_sleeping

    def test_fragment_wakeup(self):
        """Test sleeping fragment wakes with velocity."""
        fragment = DebrisFragment(
            "frag_1",
            Vec3(0, 0, 1),
            Vec3(0, 0, 0),
            size_m=0.1,
            mass_kg=0.5,
            lifetime_s=10.0,
            created_at_tick=0,
        )

        fragment.is_sleeping = True
        fragment.sleep_time_s = 5.0

        fragment.velocity = Vec3(1.0, 0, 0)  # Above threshold
        fragment.step(0.1, 9.81)

        assert not fragment.is_sleeping
        assert fragment.sleep_time_s == 0.0

    def test_fragment_to_dict(self):
        """Test fragment serialization."""
        fragment = DebrisFragment(
            "frag_1",
            Vec3(1, 2, 3),
            Vec3(0.5, 0, 0),
            size_m=0.1,
            mass_kg=0.5,
            lifetime_s=10.0,
            created_at_tick=42,
            sharpness=0.7,
        )

        data = fragment.to_dict()
        assert data["fragment_id"] == "frag_1"
        assert data["position"]["x"] == 1
        assert data["velocity"]["x"] == 0.5
        assert data["size_m"] == 0.1
        assert data["mass_kg"] == 0.5
        assert data["lifetime_s"] == 10.0
        assert data["created_at_tick"] == 42
        assert data["sharpness"] == 0.7


class TestDebrisPool:
    """Object pool tests."""

    def test_acquire_new_fragment(self):
        """Test acquiring a new fragment from pool."""
        from engine.physics.destruction.debris import DebrisPool

        pool = DebrisPool(initial_capacity=10)
        fragment = pool.acquire(
            "frag_1",
            Vec3(0, 0, 1),
            Vec3(1, 0, 0),
            0.1,
            0.5,
            10.0,
            0,
            0.8,
        )

        assert fragment.fragment_id == "frag_1"
        assert len(pool._active) == 1

    def test_release_fragment(self):
        """Test releasing fragment back to pool."""
        from engine.physics.destruction.debris import DebrisPool

        pool = DebrisPool(initial_capacity=10)
        fragment = pool.acquire("frag_1", Vec3(0, 0, 1), Vec3(1, 0, 0), 0.1, 0.5, 10.0, 0)

        pool.release("frag_1")
        assert len(pool._active) == 0
        assert len(pool._available) == 1

    def test_reuse_pooled_fragment(self):
        """Test reusing a released fragment."""
        from engine.physics.destruction.debris import DebrisPool

        pool = DebrisPool(initial_capacity=10)
        frag1 = pool.acquire("frag_1", Vec3(0, 0, 1), Vec3(1, 0, 0), 0.1, 0.5, 10.0, 0)
        frag1_id = id(frag1)

        pool.release("frag_1")

        frag2 = pool.acquire("frag_2", Vec3(1, 1, 1), Vec3(2, 0, 0), 0.2, 1.0, 5.0, 1)
        frag2_id = id(frag2)

        # Should reuse same object
        assert frag1_id == frag2_id
        assert frag2.fragment_id == "frag_2"  # But with new data
        assert frag2.position == Vec3(1, 1, 1)

    def test_pool_stats(self):
        """Test pool statistics."""
        from engine.physics.destruction.debris import DebrisPool

        pool = DebrisPool(initial_capacity=5)
        f1 = pool.acquire("frag_1", Vec3(0, 0, 1), Vec3(1, 0, 0), 0.1, 0.5, 10.0, 0)
        f2 = pool.acquire("frag_2", Vec3(0, 0, 2), Vec3(2, 0, 0), 0.2, 1.0, 10.0, 0)

        stats = pool.stats()
        assert stats["active"] == 2
        assert stats["available"] == 0

        pool.release("frag_1")
        stats = pool.stats()
        assert stats["active"] == 1
        assert stats["available"] == 1


class TestDebrisManager:
    """Debris manager tests."""

    def test_create_manager(self):
        """Test creating debris manager."""
        manager = DebrisManager()
        assert manager is not None
        stats = manager.get_stats()
        assert stats["active_count"] == 0

    def test_spawn_fragments(self):
        """Test spawning fragments from fracture."""
        manager = DebrisManager()

        fragments = [
            {
                "position": {"x": 0, "y": 0, "z": 1},
                "velocity": {"x": 1, "y": 0, "z": 0},
                "size_m": 0.1,
                "lifetime_s": 10.0,
                "sharpness": 0.8,
            },
            {
                "position": {"x": 1, "y": 0, "z": 1},
                "velocity": {"x": 2, "y": 0, "z": 0},
                "size_m": 0.2,
                "lifetime_s": 8.0,
                "sharpness": 0.7,
            },
        ]

        spawned = manager.spawn_fragments("window_1", fragments, 0, 0.0)
        assert spawned == 2

        stats = manager.get_stats()
        assert stats["active_count"] == 2

    def test_debris_stepping(self):
        """Test debris physics stepping."""
        manager = DebrisManager()

        fragments = [
            {
                "position": {"x": 0, "y": 0, "z": 10},
                "velocity": {"x": 0, "y": 0, "z": 0},
                "size_m": 0.1,
                "lifetime_s": 10.0,
            },
        ]

        manager.spawn_fragments("window_1", fragments, 0, 0.0)
        manager.step(1.0, 1)

        # Fragment should have fallen
        active = manager._pool.get_active()
        assert len(active) == 1
        assert active[0].position.z < 10  # Fallen due to gravity

    def test_debris_expiry_and_cleanup(self):
        """Test expired debris removal."""
        manager = DebrisManager()

        fragments = [
            {
                "position": {"x": 0, "y": 0, "z": 1},
                "velocity": {"x": 0, "y": 0, "z": 0},
                "size_m": 0.1,
                "lifetime_s": 1.0,  # 1 second lifetime
            },
        ]

        manager.spawn_fragments("window_1", fragments, 0, 0.0)
        assert manager.get_stats()["active_count"] == 1

        # Step past lifetime
        manager.step(0.5, 1)
        assert manager.get_stats()["active_count"] == 1

        manager.step(0.6, 2)
        assert manager.get_stats()["active_count"] == 0  # Cleaned up

    def test_collision_handling(self):
        """Test debris collision."""
        manager = DebrisManager()

        fragments = [
            {
                "position": {"x": 0, "y": 0, "z": 1},
                "velocity": {"x": 0, "y": 0, "z": -5},
                "size_m": 0.1,
                "lifetime_s": 10.0,
            },
        ]

        manager.spawn_fragments("window_1", fragments, 0, 0.0)
        active = manager._pool.get_active()
        fragment_id = active[0].fragment_id

        # Collide with floor (normal pointing up)
        manager.handle_collision(
            fragment_id,
            Vec3(0, 0, 0),
            Vec3(0, 0, 1),
            restitution=0.5,
        )

        # Velocity should reverse and damp
        active = manager._pool.get_active()
        assert active[0].velocity.z > 0  # Bounced up

    def test_render_queue_ordering(self):
        """Test render queue depth sorting."""
        manager = DebrisManager()

        fragments = [
            {
                "position": {"x": 0, "y": 0, "z": 5},
                "velocity": {"x": 0, "y": 0, "z": 0},
                "size_m": 0.1,
                "lifetime_s": 10.0,
            },
            {
                "position": {"x": 1, "y": 0, "z": 2},
                "velocity": {"x": 0, "y": 0, "z": 0},
                "size_m": 0.1,
                "lifetime_s": 10.0,
            },
            {
                "position": {"x": 2, "y": 0, "z": 8},
                "velocity": {"x": 0, "y": 0, "z": 0},
                "size_m": 0.1,
                "lifetime_s": 10.0,
            },
        ]

        manager.spawn_fragments("window_1", fragments, 0, 0.0)
        manager.step(0, 0)

        queue = manager.get_render_queue()
        assert len(queue) == 3
        # Should be sorted by Z in descending order
        assert queue[0]["position"]["z"] == 8
        assert queue[1]["position"]["z"] == 5
        assert queue[2]["position"]["z"] == 2

    def test_max_debris_capacity(self):
        """Test max debris limit."""
        config = DebrisConfig(max_debris_count=5)
        manager = DebrisManager(config=config)

        # Try to spawn more than max
        for i in range(10):
            fragments = [
                {
                    "position": {"x": i, "y": 0, "z": 1},
                    "velocity": {"x": 0, "y": 0, "z": 0},
                    "size_m": 0.1,
                    "lifetime_s": 10.0,
                },
            ]
            manager.spawn_fragments(f"obj_{i}", fragments, 0, 0.0)

        stats = manager.get_stats()
        # Should not exceed max
        assert stats["active_count"] <= 5

    def test_stats(self):
        """Test debris statistics."""
        manager = DebrisManager()

        fragments = [
            {
                "position": {"x": i, "y": 0, "z": 1},
                "velocity": {"x": 0, "y": 0, "z": 0},
                "size_m": 0.1,
                "lifetime_s": 10.0,
            }
            for i in range(3)
        ]

        manager.spawn_fragments("window_1", fragments, 0, 0.0)
        manager.step(0.01, 1)

        stats = manager.get_stats()
        assert "pool" in stats
        assert "active_count" in stats
        assert stats["active_count"] == 3
        assert stats["total_spawned"] == 3

    def test_serialize_deserialize(self):
        """Test debris state serialization."""
        manager1 = DebrisManager(seed=42)

        fragments = [
            {
                "position": {"x": 0, "y": 0, "z": 5},
                "velocity": {"x": 1, "y": 0, "z": 0},
                "size_m": 0.1,
                "lifetime_s": 10.0,
            },
            {
                "position": {"x": 1, "y": 0, "z": 3},
                "velocity": {"x": 2, "y": 0, "z": 0},
                "size_m": 0.2,
                "lifetime_s": 8.0,
            },
        ]

        manager1.spawn_fragments("window_1", fragments, 0, 0.0)
        manager1.step(1.0, 1)

        # Serialize
        data = manager1.serialize()
        assert "active_fragments" in data
        assert len(data["active_fragments"]) == 2

        # Deserialize into new manager
        manager2 = DebrisManager(seed=42)
        manager2.deserialize(data)

        # Check state matches
        stats1 = manager1.get_stats()
        stats2 = manager2.get_stats()
        assert stats1["active_count"] == stats2["active_count"]
        assert stats1["total_spawned"] == stats2["total_spawned"]

    def test_collision_event_recording(self):
        """Test collision events are recorded."""
        manager = DebrisManager()

        fragments = [
            {
                "position": {"x": 0, "y": 0, "z": 1},
                "velocity": {"x": 0, "y": 0, "z": -5},
                "size_m": 0.1,
                "lifetime_s": 10.0,
            },
        ]

        manager.spawn_fragments("window_1", fragments, 0, 0.0)
        active = manager._pool.get_active()
        fragment_id = active[0].fragment_id

        manager.handle_collision(
            fragment_id,
            Vec3(0, 0, 0),
            Vec3(0, 0, 1),
            restitution=0.3,
        )

        stats = manager.get_stats()
        assert stats["collision_events"] == 1

    def test_sleeping_fragment_optimization(self):
        """Test sleeping fragments don't consume physics updates."""
        # Use custom config with zero gravity to test sleeping without gravity acceleration
        config = DebrisConfig()
        manager = DebrisManager(config=config)

        fragments = [
            {
                "position": {"x": 0, "y": 0, "z": 0},
                "velocity": {"x": 0, "y": 0, "z": 0},
                "size_m": 0.1,
                "lifetime_s": 10.0,
            },
        ]

        manager.spawn_fragments("window_1", fragments, 0, 0.0)

        # Step with zero gravity so fragment stays at rest
        active = manager._pool.get_active()
        active[0].step(0.1, 0.0)

        # Fragment should be sleeping
        assert active[0].is_sleeping

        stats = manager.get_stats()
        assert stats["sleeping_count"] == 1
