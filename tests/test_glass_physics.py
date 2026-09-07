"""Tests for glass physics system."""

import pytest
import math

from engine.physics.destruction import GlassPhysicsSolver, GlassPane, GlassTemper
from engine.physics.math3 import Vec3


class TestGlassPane:
    """Glass pane geometry tests."""

    def test_create_pane(self):
        """Test creating a glass pane."""
        pane = GlassPane("window1", width_m=1.0, height_m=2.0, thickness_m=4.0)
        assert pane.pane_id == "window1"
        assert pane.width_m == 1.0
        assert pane.height_m == 2.0
        assert abs(pane.thickness_m - 0.004) < 1e-6  # 4mm = 0.004m

    def test_pane_surface_area(self):
        """Test pane surface area calculation."""
        pane = GlassPane("window1", width_m=2.0, height_m=3.0, thickness_m=4.0)
        assert pane.surface_area_m2() == 6.0

    def test_pane_volume(self):
        """Test pane volume calculation."""
        pane = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0)
        expected_volume = 1.0 * 1.0 * 0.004
        assert abs(pane.volume_m3() - expected_volume) < 1e-6

    def test_add_frame_attachment(self):
        """Test adding frame attachment points."""
        pane = GlassPane("window1", width_m=1.0, height_m=2.0, thickness_m=4.0)
        corner1 = Vec3(0, 0, 0)
        corner2 = Vec3(1, 2, 0)
        pane.add_frame_attachment(corner1)
        pane.add_frame_attachment(corner2)
        assert len(pane.frame_attachment_points) == 2

    def test_pane_temper_types(self):
        """Test different glass temper types."""
        for temper in [GlassTemper.ANNEALED, GlassTemper.TEMPERED, GlassTemper.LAMINATED]:
            pane = GlassPane("window1", width_m=1.0, height_m=2.0, thickness_m=4.0, temper=temper)
            assert pane.temper == temper


class TestGlassPhysicsSolver:
    """Glass physics solver tests."""

    def test_creation(self):
        """Test creating a glass solver."""
        solver = GlassPhysicsSolver(seed=42)
        assert solver is not None

    def test_register_pane(self):
        """Test registering a glass pane."""
        solver = GlassPhysicsSolver()
        pane = GlassPane("window1", width_m=1.0, height_m=2.0, thickness_m=4.0)
        solver.register_pane(pane)
        assert "window1" in solver._panes

    def test_fracture_annealed_glass(self):
        """Test fracturing annealed glass."""
        solver = GlassPhysicsSolver()
        pane = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0, temper=GlassTemper.ANNEALED)
        solver.register_pane(pane)

        event = solver.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=3.0,  # Above annealed minimum (2.0)
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )

        assert event is not None
        assert event.entity_id == "window1"
        assert 6 <= event.fragment_count <= 20  # Annealed range

    def test_fracture_tempered_glass(self):
        """Test fracturing tempered glass (more resistant)."""
        solver = GlassPhysicsSolver()
        pane = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0, temper=GlassTemper.TEMPERED)
        solver.register_pane(pane)

        # Low velocity shouldn't fracture tempered
        event1 = solver.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=3.0,  # Below tempered minimum (5.0)
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )
        assert event1 is None

        # High velocity should fracture tempered
        event2 = solver.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=6.0,  # Above tempered minimum
            impact_normal=Vec3(0, 0, 1),
            tick=1,
            timestamp=0.01,
        )
        assert event2 is not None
        assert 50 <= event2.fragment_count <= 200  # Tempered creates many granules

    def test_fracture_laminated_glass(self):
        """Test fracturing laminated glass (most resistant)."""
        solver = GlassPhysicsSolver()
        pane = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0, temper=GlassTemper.LAMINATED)
        solver.register_pane(pane)

        # Very high velocity needed for laminated
        event = solver.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=9.0,
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )

        assert event is not None
        assert 3 <= event.fragment_count <= 8  # Laminated creates fewer, larger pieces

    def test_cannot_fracture_twice(self):
        """Test that a pane cannot fracture twice."""
        solver = GlassPhysicsSolver()
        pane = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0)
        solver.register_pane(pane)

        # First fracture
        event1 = solver.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=5.0,
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )
        assert event1 is not None

        # Second fracture should fail
        event2 = solver.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=5.0,
            impact_normal=Vec3(0, 0, 1),
            tick=1,
            timestamp=0.01,
        )
        assert event2 is None

    def test_generate_radial_shards(self):
        """Test radial shard generation for annealed glass."""
        solver = GlassPhysicsSolver(seed=42)
        pane = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0, temper=GlassTemper.ANNEALED)
        solver.register_pane(pane)

        event = solver.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=3.0,
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )
        assert event is not None

        shards = solver.generate_shards(pane, Vec3(0.5, 0.5, 0), event)
        assert len(shards) == event.fragment_count

        for shard in shards:
            assert "position" in shard
            assert "velocity" in shard
            assert shard["size_m"] > 0
            assert shard["sharpness"] > 0.8  # Annealed = sharp

    def test_generate_spiderweb_shards(self):
        """Test spider-web shard generation for tempered glass."""
        solver = GlassPhysicsSolver(seed=42)
        pane = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0, temper=GlassTemper.TEMPERED)
        solver.register_pane(pane)

        event = solver.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=6.0,
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )
        assert event is not None

        shards = solver.generate_shards(pane, Vec3(0.5, 0.5, 0), event)
        assert len(shards) == event.fragment_count

        for shard in shards:
            assert shard["size_m"] < 0.2  # Granules are small
            assert shard["sharpness"] < 0.5  # Granules are dull
            assert shard["mesh_type"] == "granule"

    def test_generate_laminated_shards(self):
        """Test laminated shard generation."""
        solver = GlassPhysicsSolver(seed=42)
        pane = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0, temper=GlassTemper.LAMINATED)
        solver.register_pane(pane)

        event = solver.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=9.0,
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )
        assert event is not None

        shards = solver.generate_shards(pane, Vec3(0.5, 0.5, 0), event)
        assert len(shards) == event.fragment_count

        for shard in shards:
            assert shard["held_by_interlayer"]
            assert shard["mesh_type"] == "chunk"

    def test_shard_positions_near_impact(self):
        """Test that shards are positioned near impact point."""
        solver = GlassPhysicsSolver(seed=42)
        impact_point = Vec3(0.5, 0.5, 0)
        pane = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0)
        solver.register_pane(pane)

        event = solver.fracture_pane(
            "window1",
            impact_point,
            impact_velocity=3.0,
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )
        assert event is not None

        shards = solver.generate_shards(pane, impact_point, event)

        for shard in shards:
            pos = Vec3.from_dict(shard["position"])
            distance = (pos - impact_point).length()
            # Shards should be within reasonable distance from impact point
            # Radial shards can scatter quite far - allow up to 2x pane size
            assert distance <= max(pane.width_m, pane.height_m) * 2.0

    def test_deterministic_with_seed(self):
        """Test that same seed produces same shard patterns."""
        pane = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0)

        # Run 1
        solver1 = GlassPhysicsSolver(seed=123)
        solver1.register_pane(pane)
        event1 = solver1.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=3.0,
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )
        shards1 = solver1.generate_shards(pane, Vec3(0.5, 0.5, 0), event1)

        # Run 2
        pane2 = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0)
        solver2 = GlassPhysicsSolver(seed=123)
        solver2.register_pane(pane2)
        event2 = solver2.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=3.0,
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )
        shards2 = solver2.generate_shards(pane2, Vec3(0.5, 0.5, 0), event2)

        # Should have same structure
        assert len(shards1) == len(shards2)
        assert event1.fragment_count == event2.fragment_count

        # Shards should be at same positions
        for s1, s2 in zip(shards1, shards2):
            pos1 = Vec3.from_dict(s1["position"])
            pos2 = Vec3.from_dict(s2["position"])
            assert (pos1 - pos2).length() < 1e-6

    def test_serialize_deserialize(self):
        """Test serialization and deserialization."""
        solver1 = GlassPhysicsSolver(seed=42)
        pane = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0)
        solver1.register_pane(pane)

        # Fracture pane
        solver1.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=5.0,
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )

        # Serialize
        data = solver1.serialize()
        assert "pane_states" in data
        assert "window1" in data["pane_states"]
        assert data["pane_states"]["window1"]["is_fractured"]

        # Deserialize into new solver
        solver2 = GlassPhysicsSolver()
        pane2 = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0)
        solver2.register_pane(pane2)
        solver2.deserialize(data)

        # Should not be able to fracture again
        event = solver2.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=5.0,
            impact_normal=Vec3(0, 0, 1),
            tick=1,
            timestamp=0.01,
        )
        assert event is None

    def test_energy_based_fracture_decision(self):
        """Test that energy-based fracture decision is correct."""
        solver = GlassPhysicsSolver()
        pane = GlassPane("window1", width_m=2.0, height_m=2.0, thickness_m=4.0, temper=GlassTemper.ANNEALED)
        solver.register_pane(pane)

        # Small pane, low velocity - shouldn't fracture
        event1 = solver.fracture_pane(
            "window1",
            Vec3(1, 1, 0),
            impact_velocity=1.0,  # Below threshold
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )
        assert event1 is None

    def test_unknown_pane(self):
        """Test that fracturing unknown pane returns None."""
        solver = GlassPhysicsSolver()

        event = solver.fracture_pane(
            "unknown_window",
            Vec3(0, 0, 0),
            impact_velocity=5.0,
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )
        assert event is None

    def test_multiple_panes(self):
        """Test managing multiple panes."""
        solver = GlassPhysicsSolver()

        pane1 = GlassPane("window1", width_m=1.0, height_m=1.0, thickness_m=4.0, temper=GlassTemper.ANNEALED)
        pane2 = GlassPane("window2", width_m=2.0, height_m=2.0, thickness_m=6.0, temper=GlassTemper.TEMPERED)

        solver.register_pane(pane1)
        solver.register_pane(pane2)

        # Fracture pane 1
        event1 = solver.fracture_pane(
            "window1",
            Vec3(0.5, 0.5, 0),
            impact_velocity=3.0,
            impact_normal=Vec3(0, 0, 1),
            tick=0,
            timestamp=0.0,
        )
        assert event1 is not None

        # Pane 2 should still be fractureable
        event2 = solver.fracture_pane(
            "window2",
            Vec3(1, 1, 0),
            impact_velocity=6.0,
            impact_normal=Vec3(0, 0, 1),
            tick=1,
            timestamp=0.01,
        )
        assert event2 is not None
