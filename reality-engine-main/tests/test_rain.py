"""Tests for rain intensity model and surface accumulation."""

import pytest

from engine.environment import RainIntensity, RainConfig, RainState, RainSurface
from engine.world.events import EventBus


class TestRainIntensity:
    """Rain intensity enum tests."""

    def test_enum_values(self):
        """Test that all rain intensity bands are defined."""
        assert RainIntensity.NONE.value == "none"
        assert RainIntensity.LIGHT.value == "light"
        assert RainIntensity.MODERATE.value == "moderate"
        assert RainIntensity.HEAVY.value == "heavy"
        assert RainIntensity.EXTREME.value == "extreme"

    def test_enum_count(self):
        """Test that exactly 5 intensity bands exist."""
        assert len(list(RainIntensity)) == 5


class TestRainConfig:
    """Rain configuration tests."""

    def test_default_config(self):
        """Test default RainConfig values."""
        config = RainConfig()
        assert config.max_intensity_mm_h == 50.0
        assert config.visibility_reduction_per_mm_h == 0.015
        assert config.seed == 42

    def test_custom_config(self):
        """Test custom RainConfig values."""
        config = RainConfig(
            max_intensity_mm_h=100.0,
            visibility_reduction_per_mm_h=0.02,
            seed=123,
        )
        assert config.max_intensity_mm_h == 100.0
        assert config.visibility_reduction_per_mm_h == 0.02
        assert config.seed == 123

    def test_config_immutable(self):
        """Test that RainConfig is frozen (immutable)."""
        config = RainConfig()
        with pytest.raises(AttributeError):
            config.max_intensity_mm_h = 75.0


class TestRainState:
    """Rain state manager tests."""

    def test_initialization(self):
        """Test RainState initialization."""
        config = RainConfig()
        rain = RainState(config)

        assert rain.intensity_mm_h == 0.0
        assert rain.intensity_band() == RainIntensity.NONE

    def test_set_intensity_basic(self):
        """Test setting rainfall intensity."""
        config = RainConfig()
        rain = RainState(config)

        rain.set_intensity(10.0, tick=1, timestamp=0.1)
        assert rain.intensity_mm_h == 10.0

    def test_set_intensity_clamping_above_max(self):
        """Test that intensity is clamped to max_intensity_mm_h."""
        config = RainConfig(max_intensity_mm_h=50.0)
        rain = RainState(config)

        rain.set_intensity(100.0, tick=1, timestamp=0.1)
        assert rain.intensity_mm_h == 50.0

    def test_set_intensity_clamping_below_max(self):
        """Test that intensity below max is not clamped."""
        config = RainConfig(max_intensity_mm_h=50.0)
        rain = RainState(config)

        rain.set_intensity(30.0, tick=1, timestamp=0.1)
        assert rain.intensity_mm_h == 30.0

    def test_set_intensity_negative_raises(self):
        """Test that negative intensity raises ValueError."""
        config = RainConfig()
        rain = RainState(config)

        with pytest.raises(ValueError, match="Negative intensity is invalid"):
            rain.set_intensity(-5.0, tick=1, timestamp=0.1)

    def test_set_intensity_zero(self):
        """Test setting intensity to zero."""
        config = RainConfig()
        rain = RainState(config)

        rain.set_intensity(10.0, tick=1, timestamp=0.1)
        rain.set_intensity(0.0, tick=2, timestamp=0.2)
        assert rain.intensity_mm_h == 0.0

    def test_intensity_band_none(self):
        """Test NONE band (0 mm/h)."""
        config = RainConfig()
        rain = RainState(config)

        rain.set_intensity(0.0, tick=1, timestamp=0.1)
        assert rain.intensity_band() == RainIntensity.NONE

    def test_intensity_band_light(self):
        """Test LIGHT band (0-2.5 mm/h)."""
        config = RainConfig()
        rain = RainState(config)

        # Just above 0
        rain.set_intensity(0.1, tick=1, timestamp=0.1)
        assert rain.intensity_band() == RainIntensity.LIGHT

        # At 2.5 boundary (should be MODERATE, not LIGHT)
        rain.set_intensity(2.5, tick=2, timestamp=0.2)
        assert rain.intensity_band() == RainIntensity.MODERATE

        # Just below 2.5 (should be LIGHT)
        rain.set_intensity(2.49, tick=3, timestamp=0.3)
        assert rain.intensity_band() == RainIntensity.LIGHT

    def test_intensity_band_moderate(self):
        """Test MODERATE band (2.5-7.6 mm/h)."""
        config = RainConfig()
        rain = RainState(config)

        # At 2.5 boundary
        rain.set_intensity(2.5, tick=1, timestamp=0.1)
        assert rain.intensity_band() == RainIntensity.MODERATE

        # Middle of band
        rain.set_intensity(5.0, tick=2, timestamp=0.2)
        assert rain.intensity_band() == RainIntensity.MODERATE

        # Just below 7.6 boundary
        rain.set_intensity(7.59, tick=3, timestamp=0.3)
        assert rain.intensity_band() == RainIntensity.MODERATE

    def test_intensity_band_heavy(self):
        """Test HEAVY band (7.6-50 mm/h)."""
        config = RainConfig()
        rain = RainState(config)

        # At 7.6 boundary
        rain.set_intensity(7.6, tick=1, timestamp=0.1)
        assert rain.intensity_band() == RainIntensity.HEAVY

        # Middle of band
        rain.set_intensity(25.0, tick=2, timestamp=0.2)
        assert rain.intensity_band() == RainIntensity.HEAVY

        # Just below 50.0 boundary
        rain.set_intensity(49.99, tick=3, timestamp=0.3)
        assert rain.intensity_band() == RainIntensity.HEAVY

    def test_intensity_band_extreme(self):
        """Test EXTREME band (>50 mm/h)."""
        config = RainConfig()
        rain = RainState(config)

        # At 50.0 boundary (should be EXTREME, not HEAVY)
        rain.set_intensity(50.0, tick=1, timestamp=0.1)
        assert rain.intensity_band() == RainIntensity.EXTREME

        # Well above 50.0
        rain.set_intensity(100.0, tick=2, timestamp=0.2)
        rain.set_intensity(100.0, tick=3, timestamp=0.3)  # Clamped to 50.0
        # 50.0 clamped, which is at EXTREME boundary
        assert rain.intensity_band() == RainIntensity.EXTREME

    def test_visibility_factor_no_rain(self):
        """Test visibility factor at 0 mm/h (no rain)."""
        config = RainConfig()
        rain = RainState(config)

        rain.set_intensity(0.0, tick=1, timestamp=0.1)
        assert rain.visibility_factor() == 1.0

    def test_visibility_factor_light_rain(self):
        """Test visibility factor during light rain."""
        config = RainConfig(visibility_reduction_per_mm_h=0.015)
        rain = RainState(config)

        # 1.0 mm/h: reduction = 1.0 * 0.015 = 0.015
        rain.set_intensity(1.0, tick=1, timestamp=0.1)
        expected = 1.0 - min(0.9, 0.015)
        assert abs(rain.visibility_factor() - expected) < 1e-9
        assert abs(rain.visibility_factor() - 0.985) < 1e-9

    def test_visibility_factor_moderate_rain(self):
        """Test visibility factor during moderate rain."""
        config = RainConfig(visibility_reduction_per_mm_h=0.015)
        rain = RainState(config)

        # 10.0 mm/h: reduction = 10.0 * 0.015 = 0.15
        rain.set_intensity(10.0, tick=1, timestamp=0.1)
        expected = 1.0 - min(0.9, 0.15)
        assert abs(rain.visibility_factor() - expected) < 1e-9
        assert abs(rain.visibility_factor() - 0.85) < 1e-9

    def test_visibility_factor_heavy_rain(self):
        """Test visibility factor during heavy rain."""
        config = RainConfig(visibility_reduction_per_mm_h=0.015)
        rain = RainState(config)

        # 50.0 mm/h: reduction = 50.0 * 0.015 = 0.75
        rain.set_intensity(50.0, tick=1, timestamp=0.1)
        expected = 1.0 - min(0.9, 0.75)
        assert abs(rain.visibility_factor() - expected) < 1e-9
        assert abs(rain.visibility_factor() - 0.25) < 1e-9

    def test_visibility_factor_extreme_rain(self):
        """Test visibility factor during extreme rain (clamped to floor)."""
        config = RainConfig(visibility_reduction_per_mm_h=0.015)
        rain = RainState(config)

        # Very high intensity: reduction would be > 0.9, clamped to 0.9
        rain.set_intensity(100.0, tick=1, timestamp=0.1)
        rain.set_intensity(100.0, tick=2, timestamp=0.2)  # Clamped to max 50.0
        # 50.0 * 0.015 = 0.75, min(0.9, 0.75) = 0.75
        expected = 1.0 - 0.75
        assert abs(rain.visibility_factor() - expected) < 1e-9

    def test_visibility_factor_monotonic(self):
        """Test that visibility factor decreases monotonically with intensity."""
        config = RainConfig(visibility_reduction_per_mm_h=0.015)
        rain = RainState(config)

        intensities = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]
        factors = []

        for intensity in intensities:
            rain.set_intensity(intensity, tick=1, timestamp=0.1)
            factors.append(rain.visibility_factor())

        # Check that each visibility factor is >= the next (decreasing)
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1], \
                f"Visibility not monotonic: {factors[i]} should be >= {factors[i + 1]}"

    def test_visibility_factor_floored_at_0_1(self):
        """Test that visibility factor is floored at 0.1 when reduction clamps to 0.9."""
        # Use high reduction coefficient so max intensity hits the 0.9 clamp
        # max_intensity_mm_h=50.0, visibility_reduction_per_mm_h=0.02
        # => reduction = 50.0 * 0.02 = 1.0, clamped to 0.9
        # => visibility_factor = 1.0 - 0.9 = 0.1
        config = RainConfig(max_intensity_mm_h=50.0, visibility_reduction_per_mm_h=0.02)
        rain = RainState(config)

        # Set to max intensity; reduction should clamp to 0.9, floor to 0.1
        rain.set_intensity(50.0, tick=1, timestamp=0.1)
        assert pytest.approx(rain.visibility_factor(), rel=1e-9) == 0.1

        # Even higher intensity (clamped to max): still 0.1
        rain.set_intensity(1000.0, tick=2, timestamp=0.2)
        assert pytest.approx(rain.visibility_factor(), rel=1e-9) == 0.1

    def test_intensity_m_s_conversion(self):
        """Test conversion from mm/h to m/s."""
        config = RainConfig()
        rain = RainState(config)

        # 1 mm/h should convert to 1/3600000 m/s
        rain.set_intensity(1.0, tick=1, timestamp=0.1)
        expected_m_s = 1.0 / 3600000.0
        assert abs(rain.intensity_m_s - expected_m_s) < 1e-15

        # 3600000 mm/h should convert to 1 m/s
        rain.set_intensity(3600000.0, tick=2, timestamp=0.2)
        rain.set_intensity(50.0, tick=3, timestamp=0.3)  # Will be clamped to 50.0
        clamped_m_s = 50.0 / 3600000.0
        assert abs(rain.intensity_m_s - clamped_m_s) < 1e-15

    def test_serialize_to_dict(self):
        """Test serialization to dictionary."""
        config = RainConfig()
        rain = RainState(config)

        rain.set_intensity(15.0, tick=5, timestamp=0.5)

        data = rain.to_dict()
        assert data["format_version"] == 1
        assert data["intensity_mm_h"] == 15.0
        assert data["intensity_band"] == "heavy"
        assert data["last_tick"] == 5
        assert data["last_timestamp"] == 0.5

    def test_serialize_alias(self):
        """Test that serialize() is an alias for to_dict()."""
        config = RainConfig()
        rain = RainState(config)

        rain.set_intensity(15.0, tick=5, timestamp=0.5)

        data1 = rain.to_dict()
        data2 = rain.serialize()
        assert data1 == data2

    def test_deserialize_instance_method(self):
        """Test that deserialize() instance method restores state in place."""
        config = RainConfig()
        original = RainState(config)

        original.set_intensity(25.0, tick=10, timestamp=1.0)
        data = original.serialize()

        # Use instance method deserialize
        restored = RainState(config)
        restored.deserialize(data)

        assert restored.intensity_mm_h == 25.0
        assert restored.intensity_band() == RainIntensity.HEAVY
        assert restored._last_tick == 10
        assert restored._last_timestamp == 1.0

    def test_deserialize_unsupported_version(self):
        """Test that deserialize raises ValueError on unsupported format_version."""
        config = RainConfig()
        rain = RainState(config)

        bad_data = {
            "format_version": 2,  # Unsupported version
            "intensity_mm_h": 10.0,
        }

        with pytest.raises(ValueError, match="Unsupported rain state format_version"):
            rain.deserialize(bad_data)

    def test_round_trip_serialization(self):
        """Test full serialize/deserialize round-trip."""
        config = RainConfig(max_intensity_mm_h=75.0, visibility_reduction_per_mm_h=0.02)
        original = RainState(config)

        # Set various states
        test_cases = [
            (0.0, RainIntensity.NONE),
            (1.5, RainIntensity.LIGHT),
            (5.0, RainIntensity.MODERATE),
            (25.0, RainIntensity.HEAVY),
            (75.0, RainIntensity.EXTREME),
        ]

        for intensity, expected_band in test_cases:
            original.set_intensity(intensity, tick=1, timestamp=0.1)
            assert original.intensity_band() == expected_band

            # Serialize
            data = original.serialize()

            # Deserialize using instance method
            restored = RainState(config)
            restored.deserialize(data)

            # Verify all state
            assert restored.intensity_mm_h == intensity
            assert restored.intensity_band() == expected_band
            assert restored._last_tick == 1
            assert restored._last_timestamp == 0.1
            assert abs(restored.visibility_factor() - original.visibility_factor()) < 1e-9

    def test_band_change_detection(self):
        """Test that band changes are tracked (extension point for Task 2)."""
        config = RainConfig()
        rain = RainState(config)

        # Start with no rain
        assert rain._current_band == RainIntensity.NONE

        # Transition to light
        rain.set_intensity(1.0, tick=1, timestamp=0.1)
        assert rain._current_band == RainIntensity.LIGHT
        assert rain._previous_band == RainIntensity.NONE

        # Transition to moderate
        rain.set_intensity(5.0, tick=2, timestamp=0.2)
        assert rain._current_band == RainIntensity.MODERATE
        assert rain._previous_band == RainIntensity.LIGHT

        # Stay in same band
        rain.set_intensity(6.0, tick=3, timestamp=0.3)
        assert rain._current_band == RainIntensity.MODERATE
        assert rain._previous_band == RainIntensity.MODERATE

    def test_intensity_property_read_only(self):
        """Test that intensity_mm_h and intensity_m_s are read-only properties."""
        config = RainConfig()
        rain = RainState(config)

        rain.set_intensity(10.0, tick=1, timestamp=0.1)

        # These should raise AttributeError when trying to set
        with pytest.raises(AttributeError):
            rain.intensity_mm_h = 20.0

        with pytest.raises(AttributeError):
            rain.intensity_m_s = 0.1

    def test_config_with_different_seeds(self):
        """Test that different seeds create independent RNG streams."""
        config1 = RainConfig(seed=42)
        config2 = RainConfig(seed=123)

        rain1 = RainState(config1)
        rain2 = RainState(config2)

        # Both start at same state
        assert rain1.intensity_mm_h == rain2.intensity_mm_h == 0.0
        assert rain1.intensity_band() == rain2.intensity_band() == RainIntensity.NONE

        # After setting same intensity, should still be same
        rain1.set_intensity(20.0, tick=1, timestamp=0.1)
        rain2.set_intensity(20.0, tick=1, timestamp=0.1)

        assert rain1.intensity_mm_h == rain2.intensity_mm_h == 20.0
        assert rain1.visibility_factor() == rain2.visibility_factor()


class TestRainSurface:
    """Rain surface accumulation tests."""

    def test_rain_surface_initialization(self):
        """Test RainSurface initialization with defaults."""
        surface = RainSurface("test_surface", area_m2=100.0)

        assert surface.surface_id == "test_surface"
        assert surface.area_m2 == 100.0
        assert surface.absorption_coefficient == 0.3
        assert surface.accumulated_depth_m == 0.0

    def test_rain_surface_custom_absorption(self):
        """Test RainSurface with custom absorption coefficient."""
        surface = RainSurface(
            "roof",
            area_m2=50.0,
            absorption_coefficient=0.1,
        )

        assert surface.absorption_coefficient == 0.1
        assert surface.accumulated_depth_m == 0.0

    def test_rain_surface_absorption_zero(self):
        """Test RainSurface with zero absorption (all rain accumulates)."""
        surface = RainSurface("impermeable", area_m2=25.0, absorption_coefficient=0.0)
        assert surface.absorption_coefficient == 0.0

    def test_rain_surface_absorption_one(self):
        """Test RainSurface with full absorption (no accumulation)."""
        surface = RainSurface("soil", area_m2=75.0, absorption_coefficient=1.0)
        assert surface.absorption_coefficient == 1.0

    def test_rain_surface_absorption_out_of_range_high(self):
        """Test that absorption > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="absorption_coefficient must be in"):
            RainSurface("bad", area_m2=10.0, absorption_coefficient=1.1)

    def test_rain_surface_absorption_out_of_range_low(self):
        """Test that absorption < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="absorption_coefficient must be in"):
            RainSurface("bad", area_m2=10.0, absorption_coefficient=-0.1)

    def test_register_surface_basic(self):
        """Test registering a surface."""
        config = RainConfig()
        rain = RainState(config)
        surface = RainSurface("roof", area_m2=100.0, absorption_coefficient=0.2)

        rain.register_surface(surface)

        # Should not raise
        assert rain.get_accumulation("roof") == 0.0

    def test_register_surface_duplicate_raises(self):
        """Test that registering duplicate surface_id raises ValueError."""
        config = RainConfig()
        rain = RainState(config)
        surface1 = RainSurface("roof", area_m2=100.0)
        surface2 = RainSurface("roof", area_m2=200.0)

        rain.register_surface(surface1)

        with pytest.raises(ValueError, match="Surface already registered"):
            rain.register_surface(surface2)

    def test_get_accumulation_unknown_surface(self):
        """Test that get_accumulation raises ValueError for unknown surface."""
        config = RainConfig()
        rain = RainState(config)

        with pytest.raises(ValueError, match="Unknown surface"):
            rain.get_accumulation("nonexistent")

    def test_step_no_surfaces_no_error(self):
        """Test that step() with no registered surfaces doesn't error."""
        config = RainConfig()
        rain = RainState(config)
        rain.set_intensity(10.0, tick=1, timestamp=0.1)

        # Should not raise
        rain.step(dt=1.0, tick=1)

    def test_step_zero_intensity_no_accumulation(self):
        """Test that step() with zero intensity doesn't accumulate."""
        config = RainConfig()
        rain = RainState(config)
        surface = RainSurface("test", area_m2=100.0, absorption_coefficient=0.3)
        rain.register_surface(surface)

        rain.set_intensity(0.0, tick=1, timestamp=0.1)
        rain.step(dt=1.0, tick=1)

        assert rain.get_accumulation("test") == 0.0

    def test_step_single_surface_accumulation_exact(self):
        """Test accumulation with exact hand-computed values.

        Worked example from brief:
        - intensity_mm_h = 10.0 => intensity_m_s = 10.0 / 3600000 = 2.7777...e-6
        - absorption_coefficient = 0.3
        - dt = 1.0 second
        - accumulated_depth_m += 2.7777...e-6 * 1.0 * 0.7 = 1.9444...e-6
        """
        config = RainConfig()
        rain = RainState(config)
        surface = RainSurface("test", area_m2=1.0, absorption_coefficient=0.3)
        rain.register_surface(surface)

        rain.set_intensity(10.0, tick=1, timestamp=0.1)
        rain.step(dt=1.0, tick=1)

        # Expected: intensity_m_s * dt * (1 - absorption)
        intensity_m_s = 10.0 / 3600000.0
        expected_depth = intensity_m_s * 1.0 * 0.7
        assert pytest.approx(rain.get_accumulation("test"), rel=1e-9) == expected_depth

    def test_step_accumulation_over_hour(self):
        """Test accumulation over one hour (3600 seconds).

        Worked example:
        - 10 mm/h rain, 30% absorbed => 70% of 10mm = 7mm per hour
        - intensity_m_s = 10.0 / 3600000
        - accumulated over 3600s = 2.7777...e-6 * 3600 * 0.7 = 0.007 meters (7mm)
        """
        config = RainConfig()
        rain = RainState(config)
        surface = RainSurface("test", area_m2=1.0, absorption_coefficient=0.3)
        rain.register_surface(surface)

        rain.set_intensity(10.0, tick=1, timestamp=0.1)
        rain.step(dt=3600.0, tick=1)

        # Expected: 10 mm/h * (1 - 0.3) = 7 mm = 0.007 meters
        expected_depth = 0.007  # 7mm in meters
        assert pytest.approx(rain.get_accumulation("test"), rel=1e-3) == expected_depth

    def test_step_multiple_steps_cumulative(self):
        """Test that accumulation is cumulative across multiple steps."""
        config = RainConfig()
        rain = RainState(config)
        surface = RainSurface("test", area_m2=1.0, absorption_coefficient=0.3)
        rain.register_surface(surface)

        rain.set_intensity(10.0, tick=1, timestamp=0.1)

        # Two 1800-second steps (total 1 hour)
        rain.step(dt=1800.0, tick=1)
        depth_after_half_hour = rain.get_accumulation("test")

        rain.step(dt=1800.0, tick=2)
        depth_after_hour = rain.get_accumulation("test")

        # After 1 hour total: 10 mm/h * 0.7 = 7 mm
        expected_hour = 0.007
        assert pytest.approx(depth_after_hour, rel=1e-3) == expected_hour
        # After half hour: ~3.5 mm
        assert pytest.approx(depth_after_half_hour, rel=1e-3) == expected_hour / 2

    def test_step_multiple_surfaces_independent(self):
        """Test that multiple surfaces accumulate independently."""
        config = RainConfig()
        rain = RainState(config)

        # Surface 1: low absorption
        surface1 = RainSurface("roof", area_m2=100.0, absorption_coefficient=0.1)
        # Surface 2: high absorption
        surface2 = RainSurface("soil", area_m2=100.0, absorption_coefficient=0.8)

        rain.register_surface(surface1)
        rain.register_surface(surface2)

        rain.set_intensity(10.0, tick=1, timestamp=0.1)
        rain.step(dt=3600.0, tick=1)

        # Surface 1 (0.1 absorption): 10 * (1 - 0.1) = 9 mm = 0.009 m
        depth1 = rain.get_accumulation("roof")
        assert pytest.approx(depth1, rel=1e-3) == 0.009

        # Surface 2 (0.8 absorption): 10 * (1 - 0.8) = 2 mm = 0.002 m
        depth2 = rain.get_accumulation("soil")
        assert pytest.approx(depth2, rel=1e-3) == 0.002

        # Verify independence
        assert depth1 > depth2

    def test_step_with_changing_intensity(self):
        """Test that intensity changes are reflected in accumulation."""
        config = RainConfig()
        rain = RainState(config)
        surface = RainSurface("test", area_m2=1.0, absorption_coefficient=0.3)
        rain.register_surface(surface)

        # Step 1: 10 mm/h for 1800 seconds
        rain.set_intensity(10.0, tick=1, timestamp=0.1)
        rain.step(dt=1800.0, tick=1)
        depth1 = rain.get_accumulation("test")

        # Step 2: 20 mm/h for 1800 seconds (double intensity)
        rain.set_intensity(20.0, tick=2, timestamp=0.2)
        rain.step(dt=1800.0, tick=2)
        depth2 = rain.get_accumulation("test")

        # depth2 should be about 1.5x depth1 (half hour at 10 mm/h, half hour at 20 mm/h)
        # 10 * 0.7 * 0.5 + 20 * 0.7 * 0.5 = 3.5 + 7 = 10.5 mm = 0.0105 m
        expected = (10.0 * 0.7 * 1800.0 + 20.0 * 0.7 * 1800.0) / 3600000.0
        assert pytest.approx(depth2, rel=1e-3) == expected

    def test_serialize_with_surfaces(self):
        """Test serialization includes registered surfaces."""
        config = RainConfig()
        rain = RainState(config)
        surface = RainSurface("roof", area_m2=100.0, absorption_coefficient=0.2)
        rain.register_surface(surface)

        rain.set_intensity(10.0, tick=5, timestamp=0.5)
        rain.step(dt=100.0, tick=5)

        data = rain.serialize()

        assert "surfaces" in data
        assert "roof" in data["surfaces"]
        assert data["surfaces"]["roof"]["area_m2"] == 100.0
        assert data["surfaces"]["roof"]["absorption_coefficient"] == 0.2
        assert data["surfaces"]["roof"]["accumulated_depth_m"] > 0

    def test_deserialize_restores_surfaces(self):
        """Test deserialization restores surface state."""
        config = RainConfig()
        original = RainState(config)
        surface = RainSurface("roof", area_m2=100.0, absorption_coefficient=0.2)
        original.register_surface(surface)

        original.set_intensity(10.0, tick=1, timestamp=0.1)
        original.step(dt=3600.0, tick=1)

        data = original.serialize()

        # Restore to new state
        restored = RainState(config)
        restored.deserialize(data)

        # Verify surface was restored
        assert restored.get_accumulation("roof") > 0
        assert pytest.approx(
            restored.get_accumulation("roof"),
            rel=1e-6,
        ) == original.get_accumulation("roof")

    def test_serialize_deserialize_round_trip_surfaces(self):
        """Test full round-trip serialization with multiple surfaces."""
        config = RainConfig()
        original = RainState(config)

        # Register multiple surfaces
        surface1 = RainSurface("roof", area_m2=100.0, absorption_coefficient=0.1)
        surface2 = RainSurface("soil", area_m2=200.0, absorption_coefficient=0.9)
        original.register_surface(surface1)
        original.register_surface(surface2)

        original.set_intensity(15.0, tick=10, timestamp=1.0)
        original.step(dt=7200.0, tick=10)  # 2 hours

        # Serialize
        data = original.serialize()

        # Deserialize
        restored = RainState(config)
        restored.deserialize(data)

        # Verify all state
        assert restored.intensity_mm_h == 15.0
        assert restored.intensity_band() == RainIntensity.HEAVY
        assert restored._last_tick == 10
        assert restored._last_timestamp == 1.0

        # Verify surfaces
        depth1_original = original.get_accumulation("roof")
        depth1_restored = restored.get_accumulation("roof")
        assert pytest.approx(depth1_restored, rel=1e-6) == depth1_original

        depth2_original = original.get_accumulation("soil")
        depth2_restored = restored.get_accumulation("soil")
        assert pytest.approx(depth2_restored, rel=1e-6) == depth2_original

    def test_deserialize_empty_surfaces_dict(self):
        """Test deserialization with empty surfaces dict."""
        config = RainConfig()
        rain = RainState(config)

        rain.set_intensity(10.0, tick=1, timestamp=0.1)
        data = rain.serialize()

        # Ensure surfaces dict is empty
        assert data["surfaces"] == {}

        # Deserialize should not raise
        restored = RainState(config)
        restored.deserialize(data)

        # Verify intensity is restored
        assert restored.intensity_mm_h == 10.0


class TestRainEvents:
    """Rain band-change event publishing tests."""

    def test_event_published_on_band_change(self):
        """Test that a rain.intensity_changed event fires when the band changes."""
        config = RainConfig()
        bus = EventBus()
        received = []
        bus.subscribe("rain.intensity_changed", lambda event: received.append(event))

        rain = RainState(config, event_bus=bus)
        rain.set_intensity(5.0, tick=1, timestamp=0.1)  # NONE -> MODERATE

        assert len(received) == 1
        event = received[0]
        assert event.event_type == "rain.intensity_changed"
        assert event.tick == 1
        assert event.timestamp == 0.1
        assert event.data == {
            "from_band": "none",
            "to_band": "moderate",
            "intensity_mm_h": 5.0,
        }

    def test_no_event_on_same_band_change(self):
        """Test that no event fires when intensity changes but band stays the same."""
        config = RainConfig()
        bus = EventBus()
        received = []
        bus.subscribe("rain.intensity_changed", lambda event: received.append(event))

        rain = RainState(config, event_bus=bus)
        rain.set_intensity(1.0, tick=1, timestamp=0.1)  # NONE -> LIGHT (1 event)
        received.clear()

        rain.set_intensity(1.5, tick=2, timestamp=0.2)  # LIGHT -> LIGHT (no event)

        assert rain.intensity_band() == RainIntensity.LIGHT
        assert received == []

    def test_no_event_bus_does_not_crash(self):
        """Test that band changes with event_bus=None do not crash or attempt publish."""
        config = RainConfig()
        rain = RainState(config)  # default event_bus=None

        # Should not raise, even across multiple band changes
        rain.set_intensity(1.0, tick=1, timestamp=0.1)   # NONE -> LIGHT
        rain.set_intensity(5.0, tick=2, timestamp=0.2)   # LIGHT -> MODERATE
        rain.set_intensity(0.0, tick=3, timestamp=0.3)   # MODERATE -> NONE

        assert rain.intensity_band() == RainIntensity.NONE


class TestRainDiagnostics:
    """Rain diagnostics tests."""

    def test_get_diagnostics_no_surfaces(self):
        """Test diagnostics with no registered surfaces."""
        config = RainConfig()
        rain = RainState(config)
        rain.set_intensity(3.0, tick=1, timestamp=0.1)

        diag = rain.get_diagnostics()
        assert diag["intensity_mm_h"] == 3.0
        assert diag["band"] == "moderate"
        assert diag["visibility_factor"] == rain.visibility_factor()
        assert diag["surface_count"] == 0
        assert diag["total_accumulated_volume_m3"] == 0.0

    def test_get_diagnostics_with_surfaces(self):
        """Test diagnostics sums depth*area across registered surfaces."""
        config = RainConfig()
        rain = RainState(config)
        rain.register_surface(RainSurface("roof", area_m2=10.0, absorption_coefficient=0.0))
        rain.register_surface(RainSurface("ground", area_m2=20.0, absorption_coefficient=0.0))

        rain.set_intensity(36.0, tick=1, timestamp=0.1)  # 36 mm/h = 1e-5 m/s
        rain.step(dt=100.0, tick=1)

        diag = rain.get_diagnostics()
        assert diag["surface_count"] == 2
        expected_volume = sum(
            rain.get_accumulation(sid) * area
            for sid, area in (("roof", 10.0), ("ground", 20.0))
        )
        assert pytest.approx(diag["total_accumulated_volume_m3"], rel=1e-9) == expected_volume
        assert diag["total_accumulated_volume_m3"] > 0.0
