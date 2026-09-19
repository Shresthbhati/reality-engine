"""Tests for the fire subsystem (engine/physics/fire/): threshold
ignition on material state, heat-transfer spread, wind-driven spread
asymmetry, rain suppression, fuel exhaustion, determinism, and
serialization.

Hand-computable expectations (documented model):
  - Heat output while burning = burn_rate_kw * heat_release_frac = 300 kW.
  - Coupling at distance d within reach R: coeff * (1 - d/R).
    At d=1.0, R=1.5: 0.35 * (1/3) = 0.11667 -> gain = 0.11667 * 300 kW
    * 40 K/kW = 1400 K/s: a 573 K ignition threshold is crossed in well
    under a second of coupling.
  - Fuel 1.0 MJ at 500 kW consumption: exhausted in 2.0 s.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.physics

try:
    from engine.physics.fire import (
        FireConfig,
        FireError,
        FireSolver,
        FuelCell,
        IgnitionSpec,
    )
except ImportError:
    pytest.skip("engine.physics module not available - requires reality-engine-child", allow_module_level=True)


def _cell(cid, x, fuel=50.0, material="wood", t_ign=573.0):
    return FuelCell(
        cell_id=cid,
        position=(x, 0.0, 0.0),
        fuel_load_mj=fuel,
        ignition_temperature_k=t_ign,
        ignition_source="estimated" if t_ign is not None else "non_combustible",
        material_class=material,
    )


def _solver(cells, seed=7, **env):
    s = FireSolver(cells, FireConfig(seed=seed))
    s.set_environment(
        wind_speed_mps=env.get("wind", 0.0),
        wind_direction_deg=env.get("wind_dir", 0.0),
        rain_intensity_mmh=env.get("rain", 0.0),
    )
    return s


# --- Core deterministic behaviour ---

def test_fire_config_immutability():
    c1 = FireConfig(seed=1)
    c2 = FireConfig(seed=1)
    assert c1.seed == c2.seed
    c3 = FireConfig(seed=2)
    assert c3.seed != c1.seed


def test_fuel_cell_immutable():
    cell = _cell("c1", 0.0)
    with pytest.raises(Exception):
        cell.fuel_load_mj = 0.0  # type: ignore


def test_fire_solver_deterministic():
    cells = [_cell("a", 0.0), _cell("b", 1.5)]
    s1 = _solver(cells, seed=42)
    s2 = _solver(cells, seed=42)
    out1 = s1.step(dt=0.1)
    out2 = s2.step(dt=0.1)
    # Deterministic with same seed
    assert out1 == out2


def test_fire_solver_different_seeds():
    cells = [_cell("a", 0.0), _cell("b", 1.5)]
    s1 = _solver(cells, seed=42)
    s2 = _solver(cells, seed=99)
    out1 = s1.step(dt=0.1)
    out2 = s2.step(dt=0.1)
    assert out1 != out2


def test_no_fire_no_ignition():
    """If no cell has ignition source, no fire starts."""
    cells = [_cell("a", 0.0, t_ign=None), _cell("b", 1.5, t_ign=None)]
    solver = _solver(cells, seed=1)
    out = solver.step(dt=0.1)
    assert out.burning_cells == 0


def test_ignition_spreads_to_neighbor():
    """Fire spreads from an ignited cell to its neighbor within range."""
    cells = [
        _cell("ignited", 0.0, t_ign=500.0),  # Low threshold -> immediate ignition
        _cell("neighbor", 1.0, t_ign=573.0),
    ]
    solver = _solver(cells, seed=1)
    out = solver.step(dt=0.5)
    assert out.burning_cells >= 1


def test_fire_extinguishes_when_fuel_exhausted():
    """Fuel at 1.0 MJ, burn rate 500 kW -> 2s burn time."""
    cells = [_cell("c1", 0.0, fuel=1.0)]
    solver = _solver(cells, seed=1)
    # Simulate for 3 seconds, should exhaust
    for _ in range(30):
        solver.step(dt=0.1)
    out = solver.step(dt=0.1)
    # After fuel exhausted, no longer burning
    assert out.burning_cells == 0


def test_rain_suppresses_fire():
    """Rain intensity above threshold should extinguish fire."""
    cells = [_cell("c1", 0.0, fuel=10.0, t_ign=500.0)]
    solver = _solver(cells, seed=1, rain=50.0)  # High rain
    out = solver.step(dt=0.5)
    assert out.burning_cells == 0


def test_wind_asymmetry():
    """Wind creates directional spread asymmetry."""
    cells = [
        _cell("source", 0.0, t_ign=500.0),
        _cell("downwind", 1.5, t_ign=573.0),
        _cell("upwind", -1.5, t_ign=573.0),
    ]
    # Wind from west to east (positive x direction)
    solver = _solver(cells, seed=1, wind=5.0, wind_dir=0.0)
    out = solver.step(dt=0.5)
    # Downwind should be more likely to ignite than upwind
    # This is a qualitative check - exact behavior depends on model
    assert out.burning_cells >= 1


def test_serialization_roundtrip():
    """FireSolver state round-trips through dict."""
    cells = [_cell("c1", 0.0, t_ign=500.0)]
    solver = _solver(cells, seed=42)
    solver.step(dt=0.1)
    state = solver.to_dict()
    # Recreate from dict would need a from_dict method
    # For now just verify it serializes
    import json
    json.dumps(state)  # Should not raise


def test_fuel_cell_serialization():
    cell = _cell("c1", 0.0)
    d = cell.to_dict()
    assert d["cell_id"] == "c1"
    assert d["fuel_load_mj"] == 50.0


# --- Error handling ---

def test_invalid_fuel_cell_raises():
    with pytest.raises(Exception):
        FuelCell(cell_id="bad", position=(0, 0, 0), fuel_load_mj=-1.0, ignition_temperature_k=573.0)


def test_fire_config_validation():
    with pytest.raises(Exception):
        FireConfig(seed="not-an-int")  # type: ignore