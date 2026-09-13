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

from engine.physics.fire import (
    FireConfig,
    FireError,
    FireSolver,
    FuelCell,
    IgnitionSpec,
)


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


def _run(solver, seconds, dt=0.1):
    events = []
    for _ in range(int(seconds / dt)):
        events.extend(solver.step(dt))
    return events


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_cells_refused(self):
        with pytest.raises(FireError, match="at least one fuel cell"):
            FireSolver([])

    def test_duplicate_ids_refused(self):
        with pytest.raises(FireError, match="duplicate"):
            FireSolver([_cell("a", 0.0), _cell("a", 1.0)])

    def test_negative_fuel_refused(self):
        with pytest.raises(FireError, match="negative"):
            FuelCell("a", (0, 0, 0), fuel_load_mj=-1.0,
                     ignition_temperature_k=573.0,
                     ignition_source="estimated", material_class="wood")

    def test_nonpositive_ignition_temperature_refused(self):
        with pytest.raises(FireError, match="positive"):
            FuelCell("a", (0, 0, 0), fuel_load_mj=1.0,
                     ignition_temperature_k=0.0,
                     ignition_source="estimated", material_class="wood")

    def test_max_cells_cap(self):
        cfg = FireConfig(max_cells=2)
        with pytest.raises(FireError, match="max_cells"):
            FireSolver([_cell("a", 0), _cell("b", 1), _cell("c", 2)], cfg)

    def test_invalid_config_refused(self):
        with pytest.raises(FireError):
            FireConfig(burn_rate_kw=0.0)
        with pytest.raises(FireError):
            FireConfig(heat_transfer_coeff=1.5)
        with pytest.raises(FireError):
            FireConfig(flame_temperature_k=200.0)


# ---------------------------------------------------------------------------
# Ignition (threshold honesty)
# ---------------------------------------------------------------------------


class TestIgnition:
    def test_spark_ignites_combustible_cell(self):
        s = _solver([_cell("a", 0.0)])
        assert s.ignite(IgnitionSpec("a", "spark", 0.0)) is True
        assert s.burning_cells() == ["a"]
        assert s.state["a"].ignited_at_s == 0.0

    def test_non_combustible_refuses_ignition(self):
        s = _solver([_cell("wall", 0.0, material="concrete", t_ign=None)])
        assert s.ignite(IgnitionSpec("wall", "explosion", 0.0)) is False
        assert s.burning_cells() == []

    def test_double_ignition_refused(self):
        s = _solver([_cell("a", 0.0)])
        assert s.ignite(IgnitionSpec("a", "spark", 0.0)) is True
        assert s.ignite(IgnitionSpec("a", "spark", 1.0)) is False

    def test_unknown_cell_raises(self):
        s = _solver([_cell("a", 0.0)])
        with pytest.raises(FireError, match="unknown fuel cell"):
            s.ignite(IgnitionSpec("ghost", "spark", 0.0))


# ---------------------------------------------------------------------------
# Spread (heat transfer, not timers)
# ---------------------------------------------------------------------------


class TestSpread:
    def test_spread_chain_ignites_neighbors_in_order(self):
        s = _solver([_cell("a", 0.0), _cell("b", 1.0), _cell("c", 2.0)])
        s.ignite(IgnitionSpec("a", "spark", 0.0))
        events = _run(s, 30)
        ignitions = [e for e in events if e["type"] == "IgnitionEvent"
                     and e["source_refs"] == ("heat_transfer",)]
        assert [e["target_refs"][0] for e in ignitions] == ["b", "c"]
        # Causal citations: neighbor ignition names heat_transfer as cause.
        assert all(e["parameters"]["temperature_at_ignition_k"] >= 573.0
                   for e in ignitions)

    def test_non_combustible_never_ignites_from_neighbors(self):
        s = _solver([_cell("a", 0.0),
                     _cell("wall", 1.0, material="concrete", t_ign=None)])
        s.ignite(IgnitionSpec("a", "spark", 0.0))
        events = _run(s, 60)
        assert not any(
            e["type"] == "IgnitionEvent" and e["target_refs"] == ("wall",)
            for e in events
        )

    def test_wind_accelerates_downwind_spread(self):
        # b upwind (x=-1), c downwind (x=+1); wind blows toward +x.
        def run(wind):
            s = _solver(
                [_cell("src", 0.0, fuel=400.0), _cell("up", -1.0),
                 _cell("down", 1.0)],
                wind=15.0, wind_dir=0.0 if wind == "down" else 180.0,
            )
            s.ignite(IgnitionSpec("src", "spark", 0.0))
            events = _run(s, 60)
            times = {
                e["target_refs"][0]: e["time_s"]
                for e in events
                if e["type"] == "IgnitionEvent"
                and e["source_refs"] == ("heat_transfer",)
            }
            return times

        down = run("down")   # wind toward +x
        up = run("up")       # wind toward -x
        assert down["down"] < up["down"], (
            "downwind neighbor must ignite sooner when wind blows at it"
        )
        assert down["up"] > down["down"], (
            "upwind neighbor must ignite later than downwind in the same run"
        )


# ---------------------------------------------------------------------------
# Suppression (rain + fuel exhaustion)
# ---------------------------------------------------------------------------


class TestSuppression:
    def test_heavy_rain_prevents_spread(self):
        s = _solver([_cell("a", 0.0), _cell("b", 1.0)], rain=50.0)
        s.ignite(IgnitionSpec("a", "spark", 0.0))
        events = _run(s, 30)
        assert "b" not in s.burning_cells()
        assert not any(
            e["type"] == "IgnitionEvent" and e["target_refs"] == ("b",)
            for e in events
        )

    def test_fuel_exhaustion_extinguishes(self):
        s = _solver([_cell("a", 0.0, fuel=1.0)])  # 1 MJ at 500 kW = 2.0 s
        s.ignite(IgnitionSpec("a", "spark", 0.0))
        events = _run(s, 5)
        ext = [e for e in events if e["type"] == "ExtinguishedEvent"]
        assert len(ext) == 1
        assert ext[0]["parameters"]["reason"] == "fuel_exhausted"
        assert ext[0]["time_s"] == pytest.approx(2.0, abs=0.11)
        assert s.state["a"].fuel_remaining_mj == pytest.approx(0.0, abs=1e-9)
        assert s.burning_cells() == []

    def test_burn_duration_matches_hand_computation(self):
        # 50 MJ at 500 kW -> 100 s of burning.
        s = _solver([_cell("a", 0.0, fuel=50.0)])
        s.ignite(IgnitionSpec("a", "spark", 0.0))
        events = _run(s, 120, dt=0.5)
        ext = [e for e in events if e["type"] == "ExtinguishedEvent"]
        assert ext[0]["time_s"] == pytest.approx(100.0, abs=1.0)


# ---------------------------------------------------------------------------
# Determinism + serialization
# ---------------------------------------------------------------------------


class TestDeterminismAndSerialization:
    def test_identical_seeds_identical_evolution(self):
        def run():
            s = _solver([_cell("a", 0.0), _cell("b", 1.0)],
                        seed=11, wind=8.0, wind_dir=30.0)
            s.ignite(IgnitionSpec("a", "spark", 0.0))
            _run(s, 20)
            return s.to_dict()

        assert run() == run()

    def test_to_dict_roundtrip(self):
        s = _solver([_cell("a", 0.0), _cell("b", 1.0)])
        s.ignite(IgnitionSpec("a", "spark", 0.0))
        _run(s, 3)
        data = s.to_dict()
        assert data["format_version"] == 1
        restored = {cid: type(s.state[cid]).from_dict(d)
                    for cid, d in data["state"].items()}
        assert restored["a"].burning == s.state["a"].burning
        assert restored["a"].temperature_k == pytest.approx(
            s.state["a"].temperature_k, abs=1e-3
        )

    def test_negative_dt_refused(self):
        s = _solver([_cell("a", 0.0)])
        with pytest.raises(FireError):
            s.step(-0.1)
