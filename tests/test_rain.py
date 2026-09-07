"""Tests for rain intensity model."""

import pytest

from engine.environment import RainIntensity, RainConfig, RainState


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
        """Test that visibility factor never goes below 0.1."""
        config = RainConfig(visibility_reduction_per_mm_h=0.015)
        rain = RainState(config)

        # Test at max intensity
        rain.set_intensity(50.0, tick=1, timestamp=0.1)
        assert rain.visibility_factor() >= 0.1

        # Test at extreme levels (clamped to max)
        rain.set_intensity(1000.0, tick=2, timestamp=0.2)
        assert rain.visibility_factor() >= 0.1

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

    def test_deserialize_from_dict(self):
        """Test deserialization from dictionary."""
        config = RainConfig()
        original = RainState(config)

        original.set_intensity(25.0, tick=10, timestamp=1.0)
        data = original.to_dict()

        # Create new state from serialized data
        restored = RainState.from_dict(config, data)
        assert restored.intensity_mm_h == 25.0
        assert restored.intensity_band() == RainIntensity.HEAVY
        assert restored._last_tick == 10
        assert restored._last_timestamp == 1.0

    def test_deserialize_alias(self):
        """Test that deserialize() is an alias for from_dict()."""
        config = RainConfig()
        original = RainState(config)

        original.set_intensity(25.0, tick=10, timestamp=1.0)
        data = original.serialize()

        # Test both methods
        restored1 = RainState.from_dict(config, data)
        restored2 = RainState.deserialize(config, data)

        assert restored1.intensity_mm_h == restored2.intensity_mm_h
        assert restored1.intensity_band() == restored2.intensity_band()

    def test_deserialize_unsupported_version(self):
        """Test that deserialize raises ValueError on unsupported format_version."""
        config = RainConfig()

        bad_data = {
            "format_version": 2,  # Unsupported version
            "intensity_mm_h": 10.0,
        }

        with pytest.raises(ValueError, match="Unsupported rain state format_version"):
            RainState.deserialize(config, bad_data)

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

            # Deserialize
            restored = RainState.deserialize(config, data)

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
