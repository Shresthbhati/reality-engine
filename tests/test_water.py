"""Tests for water body simulation: WaterBody and WaterState."""

import pytest

from engine.environment.water import WaterConfig, WaterBody, WaterState
from engine.world.events import EventBus


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

    def test_serialize_deserialize_roundtrip_preserves_max_depth_m(self):
        """max_depth_m must survive a serialize->deserialize roundtrip
        (regression: it was previously dropped, silently losing the
        overflow threshold and breaking get_diagnostics() post-restore)."""
        config = WaterConfig()
        state = WaterState(config)

        body = WaterBody(
            body_id="pond1",
            surface_area_m2=10.0,
            depth_m=0.3,
            max_depth_m=0.2,
        )
        state.register_body(body)

        data = state.serialize()

        state2 = WaterState(config)
        state2.deserialize(data)

        restored = state2.get_body("pond1")
        assert restored.max_depth_m == 0.2

        # Overflow status is re-derivable from depth_m + max_depth_m alone;
        # no separate "was_overflowing" state is persisted (that flag is
        # in-memory only, scoped to a single step() call), so diagnostics
        # correctly report the restored body as overflowing.
        diagnostics = state2.get_diagnostics()
        assert diagnostics["overflowing_body_count"] == 1

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

    # Task 2: Flow between adjacent bodies and drainage tests

    def test_body_with_drainage(self):
        """Test creating a water body with drainage rate."""
        body = WaterBody(
            body_id="draining_lake",
            surface_area_m2=10.0,
            depth_m=2.0,
            drainage_rate_m3_s=0.5,
        )
        assert body.body_id == "draining_lake"
        assert body.drainage_rate_m3_s == 0.5
        assert body.depth_m == 2.0

    def test_body_rejects_negative_drainage(self):
        """Test that negative drainage rate raises ValueError."""
        with pytest.raises(ValueError, match="drainage_rate_m3_s must be >= 0"):
            WaterBody(
                body_id="invalid",
                surface_area_m2=10.0,
                drainage_rate_m3_s=-1.0,
            )

    def test_connect_bodies_bidirectional(self):
        """Test that connect_bodies creates bidirectional adjacency."""
        config = WaterConfig()
        state = WaterState(config)

        body_a = WaterBody(body_id="lake_a", surface_area_m2=10.0, depth_m=2.0)
        body_b = WaterBody(body_id="lake_b", surface_area_m2=5.0, depth_m=1.0)

        state.register_body(body_a)
        state.register_body(body_b)

        state.connect_bodies("lake_a", "lake_b")

        # Verify bidirectional adjacency
        assert "lake_b" in state._adjacency["lake_a"]
        assert "lake_a" in state._adjacency["lake_b"]

    def test_connect_unknown_body_a_raises(self):
        """Test that connecting unknown body_a raises ValueError."""
        config = WaterConfig()
        state = WaterState(config)

        body_b = WaterBody(body_id="lake_b", surface_area_m2=5.0)
        state.register_body(body_b)

        with pytest.raises(ValueError, match="Unknown water body: 'unknown_a'"):
            state.connect_bodies("unknown_a", "lake_b")

    def test_connect_unknown_body_b_raises(self):
        """Test that connecting unknown body_b raises ValueError."""
        config = WaterConfig()
        state = WaterState(config)

        body_a = WaterBody(body_id="lake_a", surface_area_m2=10.0)
        state.register_body(body_a)

        with pytest.raises(ValueError, match="Unknown water body: 'unknown_b'"):
            state.connect_bodies("lake_a", "unknown_b")

    def test_connect_duplicate_raises(self):
        """Test that connecting same pair twice raises ValueError."""
        config = WaterConfig()
        state = WaterState(config)

        body_a = WaterBody(body_id="lake_a", surface_area_m2=10.0)
        body_b = WaterBody(body_id="lake_b", surface_area_m2=5.0)

        state.register_body(body_a)
        state.register_body(body_b)

        state.connect_bodies("lake_a", "lake_b")

        # Try to connect again
        with pytest.raises(
            ValueError,
            match="Water bodies 'lake_a' and 'lake_b' are already connected",
        ):
            state.connect_bodies("lake_a", "lake_b")

    def test_flow_between_two_bodies_worked_example(self):
        """Test flow with hand-computed expected depths.

        Worked example:
        - area_a = 10 m², depth_a = 3.0 m => V_a = 30 m³
        - area_b = 5 m², depth_b = 1.0 m => V_b = 5 m³
        - flow_rate_coefficient = 0.5, dt = 1.0 s
        - Rate-limited flow = 0.5 * (3.0 - 1.0) * 1.0 = 1.0 m³
        - F_equalizing = 10 * 5 * (3.0 - 1.0) / (10 + 5) = 100 * 2 / 15 = 13.33... m³
        - Actual transfer = min(1.0, 13.33...) = 1.0 m³
        - New V_a = 30 - 1 = 29 m³ => depth_a' = 29 / 10 = 2.9 m
        - New V_b = 5 + 1 = 6 m³ => depth_b' = 6 / 5 = 1.2 m
        """
        config = WaterConfig(flow_rate_coefficient=0.5)
        state = WaterState(config)

        body_a = WaterBody(
            body_id="lake_a", surface_area_m2=10.0, depth_m=3.0
        )
        body_b = WaterBody(body_id="lake_b", surface_area_m2=5.0, depth_m=1.0)

        state.register_body(body_a)
        state.register_body(body_b)
        state.connect_bodies("lake_a", "lake_b")

        # Execute one step
        state.step(dt=1.0, tick=1)

        # Verify depths
        assert abs(state.get_body("lake_a").depth_m - 2.9) < 1e-9
        assert abs(state.get_body("lake_b").depth_m - 1.2) < 1e-9

    def test_flow_overshoot_clamped_at_equalization(self):
        """Test that flow never overshoots equalization in one step.

        Set up a case where rate-limited flow would exceed equalizing flow:
        - area_a = 10 m², depth_a = 2.0 m => V_a = 20 m³
        - area_b = 1 m², depth_b = 0.0 m => V_b = 0 m³
        - flow_rate_coefficient = 10.0 (very high), dt = 1.0 s
        - Rate-limited flow = 10.0 * (2.0 - 0.0) * 1.0 = 20.0 m³
        - F_equalizing = 10 * 1 * (2.0 - 0.0) / (10 + 1) = 20 / 11 ≈ 1.818... m³
        - Actual transfer = min(20.0, 1.818...) = 1.818... m³
        - New depth_a' = (20 - 1.818...) / 10 ≈ 1.818... m
        - New depth_b' = (0 + 1.818...) / 1 ≈ 1.818... m
        - Both should be equal (or very close)
        """
        config = WaterConfig(flow_rate_coefficient=10.0)
        state = WaterState(config)

        body_a = WaterBody(body_id="lake_a", surface_area_m2=10.0, depth_m=2.0)
        body_b = WaterBody(body_id="lake_b", surface_area_m2=1.0, depth_m=0.0)

        state.register_body(body_a)
        state.register_body(body_b)
        state.connect_bodies("lake_a", "lake_b")

        # Execute one step
        state.step(dt=1.0, tick=1)

        # Both depths should be equal (within floating point precision)
        depth_a = state.get_body("lake_a").depth_m
        depth_b = state.get_body("lake_b").depth_m

        assert abs(depth_a - depth_b) < 1e-9

    def test_flow_multi_neighbor_uses_start_of_step_snapshot(self):
        """A body with two neighbors must compute both transfers from its
        depth at the START of the step, not from a value already mutated by
        processing the first pair.

        Star topology: A connected to both B and C.
        - A: area=10, depth=3.0
        - B: area=5,  depth=1.0
        - C: area=2,  depth=0.5
        - flow_rate_coefficient = 0.1, dt = 1.0 s (rate-limited flow binds
          for both pairs, since equalizing flow is much larger)

        Pair A-B (using A's snapshot depth 3.0, not any mutated value):
        - depth_diff = 2.0
        - rate_limited = 0.1 * 2.0 * 1.0 = 0.2
        - equalizing = 10*5*2.0/15 = 6.666... => transfer = 0.2 (A -> B)

        Pair A-C (using A's snapshot depth 3.0):
        - depth_diff = 2.5
        - rate_limited = 0.1 * 2.5 * 1.0 = 0.25
        - equalizing = 10*2*2.5/12 = 4.1666... => transfer = 0.25 (A -> C)

        Net delta for A = -0.2 - 0.25 = -0.45 m³
        - depth_A' = 3.0 + (-0.45 / 10) = 2.955 m
        - depth_B' = 1.0 + (0.2 / 5) = 1.04 m
        - depth_C' = 0.5 + (0.25 / 2) = 0.625 m

        A buggy implementation that mutates A's depth after the A-B pair
        (e.g. to 2.96) would use that mutated depth for the A-C pair instead
        of the snapshot value, producing different (wrong) numbers.
        """
        config = WaterConfig(flow_rate_coefficient=0.1)
        state = WaterState(config)

        body_a = WaterBody(body_id="a", surface_area_m2=10.0, depth_m=3.0)
        body_b = WaterBody(body_id="b", surface_area_m2=5.0, depth_m=1.0)
        body_c = WaterBody(body_id="c", surface_area_m2=2.0, depth_m=0.5)

        state.register_body(body_a)
        state.register_body(body_b)
        state.register_body(body_c)
        state.connect_bodies("a", "b")
        state.connect_bodies("a", "c")

        state.step(dt=1.0, tick=1)

        assert abs(state.get_body("a").depth_m - 2.955) < 1e-9
        assert abs(state.get_body("b").depth_m - 1.04) < 1e-9
        assert abs(state.get_body("c").depth_m - 0.625) < 1e-9

    def test_drainage_reduces_volume(self):
        """Test that drainage reduces water depth correctly."""
        config = WaterConfig()
        state = WaterState(config)

        # Create body with drainage: 0.5 m³/s
        # After dt=2.0 s, should drain 1.0 m³
        # depth_m = (10.0 * 2.0 - 1.0) / 10.0 = 19.0 / 10.0 = 1.9 m
        body = WaterBody(
            body_id="draining_lake",
            surface_area_m2=10.0,
            depth_m=2.0,
            drainage_rate_m3_s=0.5,
        )
        state.register_body(body)

        state.step(dt=2.0, tick=1)

        expected_depth = 1.9
        assert abs(state.get_body("draining_lake").depth_m - expected_depth) < 1e-9

    def test_drainage_floors_at_zero(self):
        """Test that drainage never goes below zero depth."""
        config = WaterConfig()
        state = WaterState(config)

        # Create body that would drain completely if not floored
        # volume = 2.0 m³, drainage = 2.0 m³/s, dt = 2.0 s => would drain 4.0 m³
        body = WaterBody(
            body_id="small_pond",
            surface_area_m2=1.0,
            depth_m=2.0,
            drainage_rate_m3_s=2.0,
        )
        state.register_body(body)

        state.step(dt=2.0, tick=1)

        assert state.get_body("small_pond").depth_m == 0.0

    def test_serialize_with_drainage_and_adjacency(self):
        """Test serialization includes drainage rates and adjacency."""
        config = WaterConfig(flow_rate_coefficient=0.5)
        state = WaterState(config)

        body_a = WaterBody(
            body_id="lake_a",
            surface_area_m2=10.0,
            depth_m=2.0,
            drainage_rate_m3_s=0.1,
        )
        body_b = WaterBody(
            body_id="lake_b",
            surface_area_m2=5.0,
            depth_m=1.0,
            drainage_rate_m3_s=0.05,
        )

        state.register_body(body_a)
        state.register_body(body_b)
        state.connect_bodies("lake_a", "lake_b")

        data = state.serialize()

        # Verify drainage rates are serialized
        assert data["bodies"]["lake_a"]["drainage_rate_m3_s"] == 0.1
        assert data["bodies"]["lake_b"]["drainage_rate_m3_s"] == 0.05

        # Verify adjacency is serialized (as a pair)
        assert ["lake_a", "lake_b"] in data["adjacency"] or ["lake_b", "lake_a"] in data["adjacency"]

    def test_deserialize_with_drainage_and_adjacency(self):
        """Test deserialization restores drainage rates and adjacency."""
        config = WaterConfig()
        state = WaterState(config)

        data = {
            "format_version": 1,
            "bodies": {
                "lake_a": {
                    "surface_area_m2": 10.0,
                    "depth_m": 2.0,
                    "drainage_rate_m3_s": 0.1,
                },
                "lake_b": {
                    "surface_area_m2": 5.0,
                    "depth_m": 1.0,
                    "drainage_rate_m3_s": 0.05,
                },
            },
            "adjacency": [["lake_a", "lake_b"]],
        }

        state.deserialize(data)

        # Verify drainage rates are restored
        assert state.get_body("lake_a").drainage_rate_m3_s == 0.1
        assert state.get_body("lake_b").drainage_rate_m3_s == 0.05

        # Verify adjacency is restored
        assert "lake_b" in state._adjacency["lake_a"]
        assert "lake_a" in state._adjacency["lake_b"]

    def test_serialize_deserialize_roundtrip_with_flow(self):
        """Test complete roundtrip with adjacency and drainage."""
        config = WaterConfig(flow_rate_coefficient=0.75)
        state = WaterState(config)

        # Create bodies with drainage and connect them
        body_a = WaterBody(
            body_id="lake_a",
            surface_area_m2=10.0,
            depth_m=3.0,
            drainage_rate_m3_s=0.2,
        )
        body_b = WaterBody(
            body_id="lake_b",
            surface_area_m2=5.0,
            depth_m=1.5,
            drainage_rate_m3_s=0.1,
        )

        state.register_body(body_a)
        state.register_body(body_b)
        state.connect_bodies("lake_a", "lake_b")

        # Run one step of simulation
        state.step(dt=1.0, tick=1)

        # Serialize
        data = state.serialize()

        # Deserialize into new state
        state2 = WaterState(config)
        state2.deserialize(data)

        # Verify all properties are preserved
        lake_a = state2.get_body("lake_a")
        lake_b = state2.get_body("lake_b")

        assert lake_a.surface_area_m2 == 10.0
        assert lake_a.drainage_rate_m3_s == 0.2
        assert lake_b.surface_area_m2 == 5.0
        assert lake_b.drainage_rate_m3_s == 0.1

        # Depths should be preserved as well
        assert abs(lake_a.depth_m - state.get_body("lake_a").depth_m) < 1e-9
        assert abs(lake_b.depth_m - state.get_body("lake_b").depth_m) < 1e-9

        # Adjacency should be preserved
        assert "lake_b" in state2._adjacency["lake_a"]
        assert "lake_a" in state2._adjacency["lake_b"]


class TestWaterEvents:
    """Tests for overflow event publishing via WaterState.step()."""

    def _make_overflow_scenario(self, event_bus=None, max_depth_m=0.05):
        """Two connected bodies: a deep source and a shallow, max-depth-limited sink."""
        config = WaterConfig(flow_rate_coefficient=0.5)
        state = WaterState(config, event_bus=event_bus)
        deep = WaterBody(body_id="deep", surface_area_m2=100.0, depth_m=10.0)
        shallow = WaterBody(
            body_id="shallow",
            surface_area_m2=1.0,
            depth_m=0.0,
            max_depth_m=max_depth_m,
        )
        state.register_body(deep)
        state.register_body(shallow)
        state.connect_bodies("deep", "shallow")
        return state

    def test_event_published_once_on_crossing(self):
        """Event fires exactly once on the step where depth crosses above max_depth_m."""
        events = []
        bus = EventBus()
        bus.subscribe("water.body_overflowed", lambda e: events.append(e))
        state = self._make_overflow_scenario(event_bus=bus)

        state.step(dt=0.1, tick=1)
        assert state.get_body("shallow").depth_m > 0.05
        assert len(events) == 1
        assert events[0].data["body_id"] == "shallow"
        assert events[0].data["max_depth_m"] == 0.05
        assert events[0].data["depth_m"] > 0.05

    def test_no_repeated_event_while_still_over_threshold(self):
        """Staying above max_depth_m on subsequent steps does not refire the event."""
        events = []
        bus = EventBus()
        bus.subscribe("water.body_overflowed", lambda e: events.append(e))
        state = self._make_overflow_scenario(event_bus=bus)

        state.step(dt=0.1, tick=1)
        assert len(events) == 1

        state.step(dt=0.1, tick=2)
        state.step(dt=0.1, tick=3)
        assert state.get_body("shallow").depth_m > 0.05
        assert len(events) == 1

    def test_no_event_when_max_depth_is_none(self):
        """No max_depth_m set means no overflow detection, even if depth grows large."""
        events = []
        bus = EventBus()
        bus.subscribe("water.body_overflowed", lambda e: events.append(e))
        state = self._make_overflow_scenario(event_bus=bus, max_depth_m=None)

        state.step(dt=0.1, tick=1)
        assert len(events) == 0

    def test_no_crash_when_event_bus_is_none(self):
        """event_bus=None (the default) must not crash on an overflow crossing."""
        state = self._make_overflow_scenario(event_bus=None)

        state.step(dt=0.1, tick=1)  # must not raise
        assert state.get_body("shallow").depth_m > 0.05
