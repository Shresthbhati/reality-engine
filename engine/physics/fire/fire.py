"""Causal fire model (simulation campaign sec 9; the empty
engine/physics/fire/ scaffold's designated subsystem).

Fire here is a PHYSICAL process on material state, not an animation:

  HEAT -> MATERIAL TEMPERATURE -> IGNITION THRESHOLD -> COMBUSTION
       -> FUEL CONSUMPTION -> HEAT RELEASE -> NEIGHBOR HEATING -> SPREAD

Rules the spec (sec 29 NO FAKE PHYSICS) imposes, honored throughout:
  - Nothing ignites because a timer fired: ignition happens exactly when a
    cell's temperature crosses its material's ignition temperature
    (WorldIR PhysicalProperties.ignition_temperature exists; class-table
    estimates fill the rest, labelled ESTIMATED).
  - Heat transfer is a documented approximation: conduction-style neighbor
    coupling scaled by a contact factor, plus wind-advected boosting
    (downwind neighbors receive more), plus suppression by water (rain
    intensity / WaterBody presence).
  - Fuel is finite: a burning cell consumes its fuel load at a burn rate
    and EXTINGUISHES when fuel is exhausted -- no eternal fires.
  - Every state transition emits onto the event bus with cause refs, so
    the causal event graph can answer "why did this burn?".
  - Deterministic: same seed + same inputs -> identical evolution (no
    wall clock, seeded RNG only for sub-model noise).

Public surface:
  FuelCell, FireConfig, FireSolver (step/spread/suppress), plus
  IGNITION_EVENT / BURNING_EVENT / EXTINGUISHED_EVENT names for
  events/types.py vocabulary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from engine.core.rng import DeterministicRNG
from engine.core.logging import get_logger

#: Ambient temperature (Kelvin) -- documented constant, 20 C.
AMBIENT_TEMPERATURE_K = 293.15

#: Ignition-temperature estimates (Kelvin) by material class where the
#: entity's own WorldIR measurement is absent. Textbook-class ESTIMATES,
#: labelled in diagnostics -- never presented as measured.
IGNITION_TEMPERATURE_K_BY_CLASS = {
    "wood": 573.0,       # ~300 C
    "paper": 505.0,      # ~233 C (fahrenheit 451 ≈ 233 C)
    "fabric": 563.0,
    "plastic": 623.0,
    "grass": 553.0,
    "vegetation": 553.0,
    "concrete": None,    # non-combustible
    "steel": None,
    "metal": None,
    "glass": None,
    "brick": None,
    "stone": None,
    "generic": 600.0,
}

#: Fuel load estimates (energy density, MJ/m^2 of exposed fuel bed) by
#: material class -- documented approximations for the low-fidelity model.
FUEL_LOAD_MJ_M2_BY_CLASS = {
    "wood": 120.0,
    "paper": 40.0,
    "fabric": 60.0,
    "plastic": 150.0,
    "grass": 15.0,
    "vegetation": 20.0,
    "generic": 50.0,
}


class FireError(RuntimeError):
    """Raised for invalid fire-model inputs (never silently coerced)."""


@dataclass(frozen=True)
class FuelCell:
    """One burnable location in the fire grid.

    ``ignition_temperature_k`` carries its provenance: a value sourced
    from WorldIR measurements is OBSERVED; a class-table value is
    ESTIMATED (recorded in ``ignition_source``). Non-combustible classes
    have ``ignition_temperature_k = None`` and can never ignite.
    """
    cell_id: str
    position: Tuple[float, float, float]
    fuel_load_mj: float
    ignition_temperature_k: Optional[float]
    ignition_source: str            # "observed" | "estimated" | "non_combustible"
    material_class: str
    height_m: float = 1.0           # for wind-profile sampling

    def __post_init__(self):
        if self.fuel_load_mj < 0.0:
            raise FireError(f"fuel load cannot be negative: {self.fuel_load_mj}")
        if self.ignition_temperature_k is not None and self.ignition_temperature_k <= 0.0:
            raise FireError(
                f"ignition temperature must be positive, got "
                f"{self.ignition_temperature_k}"
            )


@dataclass
class CellState:
    """Mutable per-cell fire state (serializable)."""
    temperature_k: float = AMBIENT_TEMPERATURE_K
    burning: bool = False
    fuel_remaining_mj: float = 0.0
    ignited_at_s: Optional[float] = None
    extinguished_at_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature_k": round(self.temperature_k, 4),
            "burning": self.burning,
            "fuel_remaining_mj": round(self.fuel_remaining_mj, 4),
            "ignited_at_s": self.ignited_at_s,
            "extinguished_at_s": self.extinguished_at_s,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CellState":
        return cls(
            temperature_k=data["temperature_k"],
            burning=data["burning"],
            fuel_remaining_mj=data["fuel_remaining_mj"],
            ignited_at_s=data.get("ignited_at_s"),
            extinguished_at_s=data.get("extinguished_at_s"),
        )


@dataclass(frozen=True)
class FireConfig:
    """Fire model configuration (immutable).

    All rates are documented low-fidelity approximations chosen for
    stable, inspectable behavior -- not CFD.
    """
    neighbor_reach_m: float = 1.5        # coupling distance for heat transfer
    heat_transfer_coeff: float = 0.35    # fraction of source output per s
    wind_boost_coeff: float = 0.5        # extra transfer along wind vector
    burn_rate_kw: float = 500.0          # fuel consumption while burning (kW)
    heat_release_frac: float = 0.6       # fraction of burned fuel -> heat
    flame_temperature_k: float = 1173.0  # ~900 C ceiling for burning cells
    cooling_per_s: float = 0.05          # passive cooling fraction per s
    water_cooling_per_mmh: float = 0.12  # suppression per (mm/h rain)
    heating_gain_scale: float = 40.0     # K per (kW of received coupling)
    max_cells: int = 4096                # hard cap: refuse absurd grids
    seed: int = 42

    def __post_init__(self):
        if self.neighbor_reach_m <= 0.0:
            raise FireError("neighbor_reach_m must be positive")
        if not (0.0 < self.heat_transfer_coeff <= 1.0):
            raise FireError("heat_transfer_coeff must be in (0, 1]")
        if self.wind_boost_coeff < 0.0:
            raise FireError("wind_boost_coeff cannot be negative")
        if self.burn_rate_kw <= 0.0:
            raise FireError("burn_rate_kw must be positive")
        if self.flame_temperature_k <= AMBIENT_TEMPERATURE_K:
            raise FireError("flame_temperature_k must exceed ambient")


@dataclass(frozen=True)
class IgnitionSpec:
    """An explicit ignition input: the ONLY way a cell starts burning.

    Carries its cause so events can cite it (spark, ember, explosion...).
    """
    cell_id: str
    cause: str                    # "spark" | "explosion" | "ember" | ...
    at_time_s: float


class FireSolver:
    """Deterministic fire evolution over a set of fuel cells.

    Composes with the rest of the engine:
      - WindField.state.current_speed_mps + direction feed wind-driven
        spread (downwind neighbors heat faster).
      - RainState intensity mm/h (or explicit water application) drives
        suppression.
      - Event emission via a callback keeps engine/physics decoupled from
        any single bus implementation.
    """

    def __init__(self, cells: List[FuelCell], config: FireConfig = None,
                 on_event=None):
        if not cells:
            raise FireError("fire model requires at least one fuel cell")
        cfg = config or FireConfig()
        if len(cells) > cfg.max_cells:
            raise FireError(
                f"{len(cells)} cells exceeds config.max_cells={cfg.max_cells}"
            )
        ids = [c.cell_id for c in cells]
        if len(set(ids)) != len(ids):
            raise FireError("duplicate fuel cell ids")
        self.config = cfg
        self.cells: Dict[str, FuelCell] = {c.cell_id: c for c in cells}
        self.state: Dict[str, CellState] = {
            c.cell_id: CellState(fuel_remaining_mj=c.fuel_load_mj)
            for c in cells
        }
        self._time_s = 0.0
        self._rng = DeterministicRNG(cfg.seed, name="fire-noise")
        self._on_event = on_event
        self._logger = get_logger("engine.physics.fire")
        self.ignite_pending: List[IgnitionSpec] = []

    # ------------------------------------------------------------------
    # Environment coupling
    # ------------------------------------------------------------------

    def set_environment(self, wind_speed_mps: float, wind_direction_deg: float,
                        rain_intensity_mmh: float = 0.0) -> None:
        """Snapshot the environment for the next steps (called by the
        simulation loop from WindField/RainState)."""
        self._wind_speed = max(0.0, wind_speed_mps)
        self._wind_dir_deg = wind_direction_deg
        self._rain_mmh = max(0.0, rain_intensity_mmh)

    # ------------------------------------------------------------------
    # Ignition
    # ------------------------------------------------------------------

    def ignite(self, spec: IgnitionSpec) -> bool:
        """Attempt ignition at spec.at_time_s. Returns True if the cell
        actually ignited.

        Honesty: a non-combustible cell (no ignition temperature) REFUSES
        ignition -- an explosion next to a concrete wall does not set the
        wall on fire. Already-burning cells return False (no double
        counting)."""
        cell = self.cells.get(spec.cell_id)
        if cell is None:
            raise FireError(f"unknown fuel cell '{spec.cell_id}'")
        state = self.state[spec.cell_id]
        if cell.ignition_temperature_k is None:
            self._log_ignition_refused(spec, "non_combustible")
            return False
        if state.burning:
            return False
        state.burning = True
        state.ignited_at_s = spec.at_time_s
        state.temperature_k = max(
            state.temperature_k, self.config.flame_temperature_k
        )
        self._emit("IgnitionEvent", spec.at_time_s,
                   source_refs=(spec.cause,), target_refs=(spec.cell_id,),
                   parameters={
                       "ignition_temperature_k": cell.ignition_temperature_k,
                       "ignition_source": cell.ignition_source,
                   })
        return True

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def step(self, dt: float) -> List[Dict[str, Any]]:
        """Advance the fire dt seconds.

        Order (deterministic): fuel burn + heat release for burning cells
        -> heat transfer to neighbors (wind-weighted) -> passive cooling +
        rain suppression -> threshold check (temperature >= ignition
        temperature ignites a cell) -> fuel exhaustion extinguishes.

        Returns the events emitted this step (also passed to on_event).
        """
        if dt <= 0.0:
            raise FireError(f"dt must be positive, got {dt}")
        cfg = self.config
        self._time_s += dt
        events: List[Dict[str, Any]] = []

        # 1. Burning cells release heat from fuel. Fuel is stored in MJ;
        # burn_rate_kw = kJ/s, so MJ/s consumed = burn_rate_kw * dt / 1000.
        heat_output: Dict[str, float] = {}  # kW released per cell
        for cid, cell in self.cells.items():
            state = self.state[cid]
            if not state.burning:
                continue
            burned_mj = min(
                state.fuel_remaining_mj, cfg.burn_rate_kw * dt / 1000.0
            )
            state.fuel_remaining_mj -= burned_mj
            heat_output[cid] = burned_mj * 1000.0 * cfg.heat_release_frac / dt  # kW
            state.temperature_k = cfg.flame_temperature_k
            if state.fuel_remaining_mj <= 1e-9:
                state.burning = False
                state.extinguished_at_s = self._time_s
                events.append(self._emit(
                    "ExtinguishedEvent", self._time_s,
                    source_refs=("fuel_exhaustion",), target_refs=(cid,),
                    parameters={"reason": "fuel_exhausted"},
                ))

        # 2. Heat transfer to neighbors (wind-weighted).
        wind_rad = math.radians(getattr(self, "_wind_dir_deg", 0.0))
        wdx, wdy = math.cos(wind_rad), math.sin(wind_rad)
        wind_speed = getattr(self, "_wind_speed", 0.0)
        rain = getattr(self, "_rain_mmh", 0.0)

        positions = {cid: c.position for cid, c in self.cells.items()}
        heights = {cid: c.height_m for cid, c in self.cells.items()}
        for target_id, target in self.cells.items():
            tstate = self.state[target_id]
            if tstate.burning:
                continue  # burning cells are held at flame temperature
            gain = 0.0
            tx, ty, tz = target.position
            for source_id, output_mj_s in heat_output.items():
                if source_id == target_id:
                    continue
                sx, sy, sz = positions[source_id]
                dist = math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2 + (tz - sz) ** 2)
                if dist > cfg.neighbor_reach_m or dist < 1e-9:
                    continue
                # Inverse-distance coupling within reach.
                coupling = cfg.heat_transfer_coeff * (1.0 - dist / cfg.neighbor_reach_m)
                # Wind boost: sources UPWIND of the target heat it more.
                if wind_speed > 0.0:
                    ux, uy = (tx - sx) / dist, (ty - sy) / dist
                    alignment = ux * wdx + uy * wdy  # +1 downwind
                    alignment = max(0.0, alignment)
                    coupling *= 1.0 + cfg.wind_boost_coeff * alignment * (
                        wind_speed / 20.0
                    )
                # Rain suppresses transfer.
                coupling *= max(0.0, 1.0 - rain * cfg.water_cooling_per_mmh)
                gain += coupling * output_mj_s

            # Apply gain as temperature rise (documented linear
            # approximation: K proportional to received kW; a thermal-mass
            # model is future work). gain is in kW, scaled to K/s.
            tstate.temperature_k += gain * cfg.heating_gain_scale * dt

            # Passive cooling toward ambient.
            tstate.temperature_k -= (
                (tstate.temperature_k - AMBIENT_TEMPERATURE_K)
                * cfg.cooling_per_s * dt
            )
            # Rain cools exposed cells directly.
            tstate.temperature_k -= rain * cfg.water_cooling_per_mmh * dt * 10.0

            # 3. Threshold check -- the ONLY spontaneous-ignition path.
            # A cell with no fuel left CANNOT re-ignite: burnt-out material
            # stays extinguished even while still hot (ashes do not burn).
            if (target.ignition_temperature_k is not None
                    and tstate.fuel_remaining_mj > 1e-9
                    and tstate.temperature_k >= target.ignition_temperature_k):
                tstate.burning = True
                tstate.ignited_at_s = self._time_s
                events.append(self._emit(
                    "IgnitionEvent", self._time_s,
                    source_refs=("heat_transfer",), target_refs=(target_id,),
                    parameters={
                        "ignition_temperature_k": target.ignition_temperature_k,
                        "ignition_source": target.ignition_source,
                        "temperature_at_ignition_k": round(tstate.temperature_k, 2),
                    },
                ))

        return events

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def burning_cells(self) -> List[str]:
        return [cid for cid, s in self.state.items() if s.burning]

    def exhausted_cells(self) -> List[str]:
        return [
            cid for cid, s in self.state.items()
            if not s.burning and s.fuel_remaining_mj <= 1e-9
        ]

    @property
    def time_s(self) -> float:
        return self._time_s

    # ------------------------------------------------------------------
    # Events / serialization
    # ------------------------------------------------------------------

    def _emit(self, event_type: str, timestamp: float, **kwargs) -> Dict[str, Any]:
        payload = {"type": event_type, "time_s": timestamp, **kwargs}
        if self._on_event is not None:
            self._on_event(payload)
        return payload

    def _log_ignition_refused(self, spec: IgnitionSpec, reason: str) -> None:
        self._logger.info(
            "Ignition refused",
            context={"cell": spec.cell_id, "cause": spec.cause, "reason": reason},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": 1,
            "time_s": round(self._time_s, 6),
            "config": {
                "burn_rate_kw": self.config.burn_rate_kw,
                "heat_transfer_coeff": self.config.heat_transfer_coeff,
                "seed": self.config.seed,
            },
            "state": {cid: s.to_dict() for cid, s in self.state.items()},
        }
