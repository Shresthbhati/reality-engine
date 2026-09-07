"""Tests for destruction (fracture) system."""

import pytest
import math

from engine.physics.destruction import BasicFractureSolver, FractureEvent
from engine.physics.math3 import Vec3


class TestBasicFractureSolver:
    """Basic fracture solver tests."""

    def test_creation(self):
        """Test creating a fracture solver."""
        solver = BasicFractureSolver(seed=42)
        assert solver is not None

    def test_can_fracture_glass(self):
        """Test checking if glass can fracture."""
        solver = BasicFractureSolver()
        assert solver.can_fracture("glass_pane", mass=2.0, material_name="glass")

    def test_cannot_fracture_unknown_material(self):
        """Test that unknown materials cannot fracture."""
        solver = BasicFractureSolver()
        assert not solver.can_fracture("entity1", mass=1.0, material_name="rubber")

    def test_cannot_fracture_twice(self):
        """Test that already-fractured entity cannot fracture again."""
        solver = BasicFractureSolver()
        impact_point = Vec3(0, 0, 0)
        impact_normal = Vec3(0, 1, 0)

        # First fracture
        event1 = solver.fracture(
            "entity1",
            impact_point,
            impact_velocity=10.0,
            impact_normal=impact_normal,
            tick=0,
            timestamp=0.0,
            material_name="glass",
            mass_kg=1.0,
        )
        assert event1 is not None

        # Second fracture should fail
        event2 = solver.fracture(
            "entity1",
            impact_point,
            impact_velocity=10.0,
            impact_normal=impact_normal,
            tick=1,
            timestamp=0.01,
            material_name="glass",
            mass_kg=1.0,
        )
        assert event2 is None

    def test_low_velocity_no_fracture(self):
        """Test that low-velocity impact doesn't fracture."""
        solver = BasicFractureSolver()
        event = solver.fracture(
            "entity1",
            Vec3(0, 0, 0),
            impact_velocity=1.0,  # Below minimum
            impact_normal=Vec3(0, 1, 0),
            tick=0,
            timestamp=0.0,
            material_name="glass",
            mass_kg=1.0,
        )
        assert event is None

    def test_high_velocity_fracture(self):
        """Test that high-velocity impact does fracture."""
        solver = BasicFractureSolver()
        event = solver.fracture(
            "entity1",
            Vec3(0, 0, 0),
            impact_velocity=5.0,  # Well above minimum
            impact_normal=Vec3(0, 1, 0),
            tick=0,
            timestamp=0.0,
            material_name="glass",
            mass_kg=1.0,
        )
        assert event is not None
        assert event.entity_id == "entity1"
        assert event.fragment_count > 0

    def test_fracture_event_properties(self):
        """Test that fracture event has correct properties."""
        solver = BasicFractureSolver()
        impact_point = Vec3(1, 2, 3)
        impact_velocity = 5.0
        impact_normal = Vec3(0, 1, 0)

        event = solver.fracture(
            "entity1",
            impact_point,
            impact_velocity,
            impact_normal,
            tick=42,
            timestamp=1.23,
            material_name="glass",
            mass_kg=2.0,
        )

        assert event is not None
        assert event.entity_id == "entity1"
        assert event.impact_point == impact_point
        assert event.impact_velocity == impact_velocity
        assert event.impact_normal == impact_normal
        assert event.tick == 42
        assert event.timestamp == 1.23
        assert event.fragment_count >= 3  # Glass min fragments

    def test_fragment_count_within_range(self):
        """Test that fragment count is within material limits."""
        solver = BasicFractureSolver(seed=42)
        min_frags = 3
        max_frags = 12

        for _ in range(10):
            event = solver.fracture(
                f"entity_{_}",
                Vec3(0, 0, 0),
                impact_velocity=5.0,
                impact_normal=Vec3(0, 1, 0),
                tick=0,
                timestamp=0.0,
                material_name="glass",
                mass_kg=1.0,
            )
            assert event is not None
            assert min_frags <= event.fragment_count <= max_frags

    def test_material_glass(self):
        """Test glass fracture properties."""
        solver = BasicFractureSolver()
        event = solver.fracture(
            "glass1",
            Vec3(0, 0, 0),
            impact_velocity=3.0,  # Just above glass minimum
            impact_normal=Vec3(0, 1, 0),
            tick=0,
            timestamp=0.0,
            material_name="glass",
            mass_kg=1.0,
        )
        assert event is not None
        assert 3 <= event.fragment_count <= 12

    def test_material_concrete(self):
        """Test concrete fracture properties."""
        solver = BasicFractureSolver()
        event = solver.fracture(
            "concrete1",
            Vec3(0, 0, 0),
            impact_velocity=6.0,  # Just above concrete minimum
            impact_normal=Vec3(0, 1, 0),
            tick=0,
            timestamp=0.0,
            material_name="concrete",
            mass_kg=100.0,  # Concrete blocks are heavy
        )
        assert event is not None
        assert 1 <= event.fragment_count <= 6

    def test_generate_fragments(self):
        """Test fragment generation."""
        solver = BasicFractureSolver(seed=42)
        event = solver.fracture(
            "entity1",
            Vec3(0, 0, 0),
            impact_velocity=5.0,
            impact_normal=Vec3(0, 1, 0),
            tick=0,
            timestamp=0.0,
            material_name="glass",
            mass_kg=1.0,
        )
        assert event is not None

        geometry_data = {
            "bounds_min": Vec3(-1, -1, -1),
            "bounds_max": Vec3(1, 1, 1),
        }

        fragments = solver.generate_fragments("entity1", event, geometry_data)
        assert len(fragments) == event.fragment_count

        # Check each fragment
        for i, frag in enumerate(fragments):
            assert frag["id"] == f"entity1_frag_{i}"
            assert frag["parent_id"] == "entity1"
            assert "position" in frag
            assert "velocity" in frag
            assert frag["size_m"] > 0
            assert frag["volume_m3"] > 0
            assert frag["lifetime_s"] > 0

    def test_fragment_positions_near_impact(self):
        """Test that fragments are positioned near impact point."""
        solver = BasicFractureSolver(seed=42)
        impact_point = Vec3(5, 10, 15)
        event = solver.fracture(
            "entity1",
            impact_point,
            impact_velocity=5.0,
            impact_normal=Vec3(0, 1, 0),
            tick=0,
            timestamp=0.0,
            material_name="glass",
            mass_kg=1.0,
        )
        assert event is not None

        geometry_data = {
            "bounds_min": Vec3(-1, -1, -1),
            "bounds_max": Vec3(1, 1, 1),
        }

        fragments = solver.generate_fragments("entity1", event, geometry_data)

        # Fragments should be near impact point
        for frag in fragments:
            pos = Vec3.from_dict(frag["position"])
            distance = (pos - impact_point).length()
            # Fragment size is roughly 0.5-1.0m, so distance should be under 3m
            assert distance < 3.0

    def test_deterministic_with_seed(self):
        """Test that same seed produces same results."""
        impact_point = Vec3(0, 0, 0)
        impact_normal = Vec3(0, 1, 0)

        # Run 1
        solver1 = BasicFractureSolver(seed=123)
        event1 = solver1.fracture(
            "entity1",
            impact_point,
            impact_velocity=5.0,
            impact_normal=impact_normal,
            tick=0,
            timestamp=0.0,
            material_name="glass",
            mass_kg=1.0,
        )
        frags1 = solver1.generate_fragments("entity1", event1, {
            "bounds_min": Vec3(-1, -1, -1),
            "bounds_max": Vec3(1, 1, 1),
        })

        # Run 2 (same seed)
        solver2 = BasicFractureSolver(seed=123)
        event2 = solver2.fracture(
            "entity1",
            impact_point,
            impact_velocity=5.0,
            impact_normal=impact_normal,
            tick=0,
            timestamp=0.0,
            material_name="glass",
            mass_kg=1.0,
        )
        frags2 = solver2.generate_fragments("entity1", event2, {
            "bounds_min": Vec3(-1, -1, -1),
            "bounds_max": Vec3(1, 1, 1),
        })

        # Should be identical
        assert event1.fragment_count == event2.fragment_count
        assert len(frags1) == len(frags2)

        for f1, f2 in zip(frags1, frags2):
            assert f1["size_m"] == f2["size_m"]
            pos1 = Vec3.from_dict(f1["position"])
            pos2 = Vec3.from_dict(f2["position"])
            assert (pos1 - pos2).length() < 1e-6  # Near-identical

    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        impact_point = Vec3(0, 0, 0)
        impact_normal = Vec3(0, 1, 0)

        solver1 = BasicFractureSolver(seed=123)
        event1 = solver1.fracture(
            "entity1",
            impact_point,
            impact_velocity=5.0,
            impact_normal=impact_normal,
            tick=0,
            timestamp=0.0,
            material_name="glass",
            mass_kg=1.0,
        )
        frags1 = solver1.generate_fragments("entity1", event1, {
            "bounds_min": Vec3(-1, -1, -1),
            "bounds_max": Vec3(1, 1, 1),
        })

        solver2 = BasicFractureSolver(seed=456)  # Different seed
        event2 = solver2.fracture(
            "entity1",
            impact_point,
            impact_velocity=5.0,
            impact_normal=impact_normal,
            tick=0,
            timestamp=0.0,
            material_name="glass",
            mass_kg=1.0,
        )
        frags2 = solver2.generate_fragments("entity1", event2, {
            "bounds_min": Vec3(-1, -1, -1),
            "bounds_max": Vec3(1, 1, 1),
        })

        # At least some difference (highly likely with different seeds)
        differences = 0
        for f1, f2 in zip(frags1, frags2):
            pos1 = Vec3.from_dict(f1["position"])
            pos2 = Vec3.from_dict(f2["position"])
            if (pos1 - pos2).length() > 1e-6:
                differences += 1

        # With different seeds, expect at least some difference
        assert differences > 0

    def test_serialize_deserialize(self):
        """Test serialization and deserialization."""
        solver1 = BasicFractureSolver(seed=42)

        # Fracture an entity
        solver1.fracture(
            "entity1",
            Vec3(0, 0, 0),
            impact_velocity=5.0,
            impact_normal=Vec3(0, 1, 0),
            tick=0,
            timestamp=0.0,
            material_name="glass",
            mass_kg=1.0,
        )

        # Serialize
        data = solver1.serialize()
        assert "fractured_entities" in data
        assert "entity1" in data["fractured_entities"]

        # Deserialize into new solver
        solver2 = BasicFractureSolver()
        solver2.deserialize(data)

        # Should not be able to fracture the same entity
        event = solver2.fracture(
            "entity1",
            Vec3(0, 0, 0),
            impact_velocity=5.0,
            impact_normal=Vec3(0, 1, 0),
            tick=1,
            timestamp=0.01,
            material_name="glass",
            mass_kg=1.0,
        )
        assert event is None

    def test_energy_calculation(self):
        """Test that energy is calculated correctly."""
        solver = BasicFractureSolver()
        mass_kg = 2.0
        velocity_m_s = 3.0

        event = solver.fracture(
            "entity1",
            Vec3(0, 0, 0),
            impact_velocity=velocity_m_s,
            impact_normal=Vec3(0, 1, 0),
            tick=0,
            timestamp=0.0,
            material_name="glass",
            mass_kg=mass_kg,
        )

        if event is not None:
            # KE = 0.5 * m * v^2
            expected_ke = 0.5 * mass_kg * velocity_m_s ** 2
            assert abs(event.energy_released_j - expected_ke) < 1e-6

    def test_zero_mass_cannot_fracture(self):
        """Test that zero-mass entities cannot fracture."""
        solver = BasicFractureSolver()
        assert not solver.can_fracture("entity1", mass=0.0, material_name="glass")

    def test_negative_mass_cannot_fracture(self):
        """Test that negative-mass entities cannot fracture."""
        solver = BasicFractureSolver()
        assert not solver.can_fracture("entity1", mass=-1.0, material_name="glass")
