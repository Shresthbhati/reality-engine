"""Tests for water body simulation: WaterBody and WaterState."""

import pytest

from engine.environment.water import WaterConfig, WaterBody, WaterState


class TestWaterBody:
    """Tests for WaterBody class."""

    def test_body_creation_with_defaults(self):
        """Test creating a water body with default depth."""
        body = WaterBody(body_id="lake1", surface_area_m2=10.0)
        assert body.body_id == "lake1"
        assert body.surface_area_m2 == 10.0
        assert body.depth_m == 0.0

    def test_body_creation_with_depth(self):
        """Test creating a water body with specified depth."""
        body = WaterBody(body_id="pond1", surface_area_m2=5.5, depth_m=2.3)
        assert body.body_id == "pond1"
        assert body.surface_area_m2 == 5.5
        assert body.depth_m == 2.3

    def test_volume_computation(self):
        """Test that volume is computed correctly as surface_area * depth."""
        body = WaterBody(body_id="lake1", surface_area_m2=10.0, depth_m=2.0)
        assert body.volume_m3 == 20.0

    def test_volume_with_zero_depth(self):
        """Test volume when depth is zero."""
        body = WaterBody(body_id="empty", surface_area_m2=100.0, depth_m=0.0)
        assert body.volume_m3 == 0.0

    def test_rejects_zero_surface_area(self):
        """Test that zero surface area raises ValueError."""
        with pytest.raises(ValueError, match="surface_area_m2 must be > 0"):
            WaterBody(body_id="invalid", surface_area_m2=0.0)

    def test_rejects_negative_surface_area(self):
        """Test that negative surface area raises ValueError."""
        with pytest.raises(ValueError, match="surface_area_m2 must be > 0"):
            WaterBody(body_id="invalid", surface_area_m2=-5.0)


class TestWaterState:
    """Tests for WaterState class."""

    def test_state_creation(self):
        """Test creating a WaterState with config."""
        config = WaterConfig()
        state = WaterState(config)
        assert state.config == config
        assert state.format_version == 1

    def test_custom_config(self):
        """Test WaterState with custom config values."""
        config = WaterConfig(water_density_kg_m3=1025.0, gravity_m_s2=9.8)
        state = WaterState(config)
        assert state.config.water_density_kg_m3 == 1025.0
        assert state.config.gravity_m_s2 == 9.8

    def test_register_body(self):
        """Test registering a water body."""
        config = WaterConfig()
        state = WaterState(config)
        body = WaterBody(body_id="lake1", surface_area_m2=10.0, depth_m=2.0)
        state.register_body(body)
        assert state.get_body("lake1") == body

    def test_duplicate_registration_raises(self):
        """Test that registering duplicate body_id raises ValueError."""
        config = WaterConfig()
        state = WaterState(config)
        body1 = WaterBody(body_id="lake1", surface_area_m2=10.0)
        body2 = WaterBody(body_id="lake1", surface_area_m2=5.0)

        state.register_body(body1)
        with pytest.raises(
            ValueError, match="Water body 'lake1' is already registered"
        ):
            state.register_body(body2)

    def test_get_unknown_body_raises(self):
        """Test that getting an unknown body raises ValueError."""
        config = WaterConfig()
        state = WaterState(config)
        with pytest.raises(ValueError, match="Unknown water body: 'unknown'"):
            state.get_body("unknown")

    def test_buoyancy_force_worked_example(self):
        """Test buoyancy formula against worked example."""
        # Worked example: 1000.0 * 9.81 * 1.5 = 14715.0 N
        config = WaterConfig(water_density_kg_m3=1000.0, gravity_m_s2=9.81)
        state = WaterState(config)
        body = WaterBody(
            body_id="lake1", surface_area_m2=10.0, depth_m=2.0
        )  # volume = 20.0 m³
        state.register_body(body)

        force = state.buoyancy_force_n("lake1", submerged_volume_m3=1.5)
        assert force == 14715.0

    def test_buoyancy_force_zero_volume(self):
        """Test buoyancy with zero submerged volume."""
        config = WaterConfig()
        state = WaterState(config)
        body = WaterBody(body_id="lake1", surface_area_m2=10.0, depth_m=2.0)
        state.register_body(body)

        force = state.buoyancy_force_n("lake1", submerged_volume_m3=0.0)
        assert force == 0.0

    def test_buoyancy_force_negative_volume_raises(self):
        """Test that negative submerged volume raises ValueError."""
        config = WaterConfig()
        state = WaterState(config)
        body = WaterBody(body_id="lake1", surface_area_m2=10.0, depth_m=2.0)
        state.register_body(body)

        with pytest.raises(
            ValueError, match="submerged_volume_m3 must be >= 0"
        ):
            state.buoyancy_force_n("lake1", submerged_volume_m3=-1.0)

    def test_buoyancy_force_exceeds_body_volume_raises(self):
        """Test that submersing more than body volume raises ValueError."""
        config = WaterConfig()
        state = WaterState(config)
        body = WaterBody(
            body_id="lake1", surface_area_m2=10.0, depth_m=2.0
        )  # volume = 20.0 m³
        state.register_body(body)

        with pytest.raises(
            ValueError, match="submerged_volume_m3.*exceeds.*body volume"
        ):
            state.buoyancy_force_n("lake1", submerged_volume_m3=20.1)

    def test_buoyancy_force_at_body_volume_limit(self):
        """Test buoyancy at the exact body volume limit."""
        config = WaterConfig(water_density_kg_m3=1000.0, gravity_m_s2=10.0)
        state = WaterState(config)
        body = WaterBody(
            body_id="lake1", surface_area_m2=5.0, depth_m=4.0
        )  # volume = 20.0 m³
        state.register_body(body)

        # Should succeed at exact volume limit
        force = state.buoyancy_force_n("lake1", submerged_volume_m3=20.0)
        assert force == 1000.0 * 10.0 * 20.0

    def test_serialize_empty_state(self):
        """Test serializing an empty water state."""
        config = WaterConfig()
        state = WaterState(config)
        data = state.serialize()

        assert data["format_version"] == 1
        assert data["bodies"] == {}

    def test_serialize_with_bodies(self):
        """Test serializing water state with registered bodies."""
        config = WaterConfig()
        state = WaterState(config)

        body1 = WaterBody(body_id="lake1", surface_area_m2=10.0, depth_m=2.0)
        body2 = WaterBody(body_id="pond1", surface_area_m2=5.0, depth_m=1.5)

        state.register_body(body1)
        state.register_body(body2)

        data = state.serialize()

        assert data["format_version"] == 1
        assert "lake1" in data["bodies"]
        assert "pond1" in data["bodies"]
        assert data["bodies"]["lake1"]["surface_area_m2"] == 10.0
        assert data["bodies"]["lake1"]["depth_m"] == 2.0
        assert data["bodies"]["pond1"]["surface_area_m2"] == 5.0
        assert data["bodies"]["pond1"]["depth_m"] == 1.5

    def test_deserialize_empty_state(self):
        """Test deserializing an empty state."""
        config = WaterConfig()
        state = WaterState(config)

        data = {"format_version": 1, "bodies": {}}
        state.deserialize(data)

        with pytest.raises(ValueError, match="Unknown water body"):
            state.get_body("lake1")

    def test_deserialize_with_bodies(self):
        """Test deserializing state with bodies."""
        config = WaterConfig()
        state = WaterState(config)

        data = {
            "format_version": 1,
            "bodies": {
                "lake1": {"surface_area_m2": 10.0, "depth_m": 2.0},
                "pond1": {"surface_area_m2": 5.0, "depth_m": 1.5},
            },
        }

        state.deserialize(data)

        lake = state.get_body("lake1")
        assert lake.body_id == "lake1"
        assert lake.surface_area_m2 == 10.0
        assert lake.depth_m == 2.0
        assert lake.volume_m3 == 20.0

        pond = state.get_body("pond1")
        assert pond.body_id == "pond1"
        assert pond.surface_area_m2 == 5.0
        assert pond.depth_m == 1.5
        assert pond.volume_m3 == 7.5

    def test_serialize_deserialize_roundtrip(self):
        """Test that serialize->deserialize is a perfect roundtrip."""
        config = WaterConfig(
            water_density_kg_m3=1050.0, gravity_m_s2=9.82, seed=123
        )
        state = WaterState(config)

        # Register some bodies
        body1 = WaterBody(body_id="lake1", surface_area_m2=100.0, depth_m=5.0)
        body2 = WaterBody(body_id="ocean1", surface_area_m2=1000.0, depth_m=100.0)
        state.register_body(body1)
        state.register_body(body2)

        # Serialize
        data = state.serialize()

        # Deserialize into a new state
        state2 = WaterState(config)
        state2.deserialize(data)

        # Verify bodies are identical
        lake = state2.get_body("lake1")
        assert lake.surface_area_m2 == 100.0
        assert lake.depth_m == 5.0
        assert lake.volume_m3 == 500.0

        ocean = state2.get_body("ocean1")
        assert ocean.surface_area_m2 == 1000.0
        assert ocean.depth_m == 100.0
        assert ocean.volume_m3 == 100000.0

        # Verify buoyancy still works the same
        force1 = state.buoyancy_force_n("lake1", submerged_volume_m3=10.0)
        force2 = state2.buoyancy_force_n("lake1", submerged_volume_m3=10.0)
        assert force1 == force2

    def test_deserialize_format_version_mismatch(self):
        """Test that format_version mismatch raises ValueError."""
        config = WaterConfig()
        state = WaterState(config)

        data = {"format_version": 2, "bodies": {}}
        with pytest.raises(ValueError, match="Format version mismatch"):
            state.deserialize(data)

    def test_deserialize_replaces_existing_bodies(self):
        """Test that deserialize replaces existing bodies."""
        config = WaterConfig()
        state = WaterState(config)

        # Register initial body
        body1 = WaterBody(body_id="lake1", surface_area_m2=10.0, depth_m=2.0)
        state.register_body(body1)

        # Verify it's there
        assert state.get_body("lake1").surface_area_m2 == 10.0

        # Deserialize different data
        data = {
            "format_version": 1,
            "bodies": {
                "pond1": {"surface_area_m2": 5.0, "depth_m": 1.0},
            },
        }
        state.deserialize(data)

        # Old body should be gone
        with pytest.raises(ValueError, match="Unknown water body"):
            state.get_body("lake1")

        # New body should be there
        assert state.get_body("pond1").surface_area_m2 == 5.0
