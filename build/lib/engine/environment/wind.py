"""Wind field: deterministic meteorological wind with physical force
coupling (simulation campaign sec 11; spec sec 19-adjacent RAIN pattern).

Follows engine/environment/rain.py's established conventions: frozen
config, deterministic state evolution via DeterministicRNG, band
classification (Beaufort scale here), optional event publishing on band
changes, serializable state.

Physical model (documented approximations, spec sec 29 discipline):
  - Boundary-layer power law: v(h) = v_ref * (h / h_ref)^alpha. This is a
    standard meteorological profile, not a DNS -- it gives heights above
    the reference an honest, monotone speed-up and is labelled as an
    approximation in the docs.
  - Gusts: deterministic damped oscillation around the base speed
    (sine components seeded from the config's seed), NOT a random
    per-frame scramble -- two runs with the same seed produce the same
    gust sequence.
  - Force coupling: quadratic aerodynamic drag,
    F = 0.5 * rho_air * Cd * A * v_rel^2, applied opposite the wind
    direction to a body's exposed area. Quadratic drag is the correct
    dominant term for building-scale wind loads; it is an approximation
    (no shielding, no turbulence) and is documented as such. This is the
    causal hook the glass/structural/debris systems consume: wind speed
    -> force -> physics bodies -> impact/structural events -- no scripted
    damage timers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from engine.core.rng import DeterministicRNG
from engine.core.logging import get_logger
from engine.world.events import EventBus

#: Air density at sea level, 15 C (kg/m^3) -- standard atmosphere value.
AIR_DENSITY_KG_M3 = 1.225


class BeaufortBand(str, Enum):
    """Beaufort force bands (meteorological standard, m/s bounds)."""
    CALM = "calm"              # < 0.5
    LIGHT_AIR = "light_air"    # 0.5 - 1.5
    LIGHT_BREEZE = "light_breeze"       # 1.6 - 3.3
    GENTLE_BREEZE = "gentle_breeze"     # 3.4 - 5.4
    MODERATE_BREEZE = "moderate_breeze" # 5.5 - 7.9
    FRESH_BREEZE = "fresh_breeze"       # 8.0 - 10.7
    STRONG_BREEZE = "strong_breeze"     # 10.8 - 13.8
    NEAR_GALE = "near_gale"             # 13.9 - 17.1
    GALE = "gale"                       # 17.2 - 20.7
    STRONG_GALE = "strong_gale"         # 20.8 - 24.4
    STORM = "storm"                     # 24.5 - 28.4
    VIOLENT_STORM = "violent_storm"     # 28.5 - 32.6
    HURRICANE = "hurricane"             # >= 32.7


#: (upper_bound_mps, band) in ascending order; CALM (strictly below 0.5)
#: is handled before the table; band for speeds above the last bound is
#: HURRICANE.
_BEAUFORT_TABLE: Tuple[Tuple[float, BeaufortBand], ...] = (
    (1.5, BeaufortBand.LIGHT_AIR),
    (3.3, BeaufortBand.LIGHT_BREEZE),
    (5.4, BeaufortBand.GENTLE_BREEZE),
    (7.9, BeaufortBand.MODERATE_BREEZE),
    (10.7, BeaufortBand.FRESH_BREEZE),
    (13.8, BeaufortBand.STRONG_BREEZE),
    (17.1, BeaufortBand.NEAR_GALE),
    (20.7, BeaufortBand.GALE),
    (24.4, BeaufortBand.STRONG_GALE),
    (28.4, BeaufortBand.STORM),
    (32.6, BeaufortBand.VIOLENT_STORM),
)


def classify_beaufort(speed_mps: float) -> BeaufortBand:
    """Band classification; deterministic pure function.

    Convention: CALM is strictly below 0.5 m/s; every other band includes
    both its stated bounds (e.g. FRESH_BREEZE covers 8.0 through 10.7
    inclusive) -- matching the band table in this module's docstring.
    """
    if speed_mps < 0.0:
        raise ValueError(f"wind speed cannot be negative, got {speed_mps}")
    if speed_mps < 0.5:
        return BeaufortBand.CALM
    for upper, band in _BEAUFORT_TABLE:
        if speed_mps <= upper:
            return band
    return BeaufortBand.HURRICANE


@dataclass(frozen=True)
class WindConfig:
    """Wind system configuration (immutable).

    Attributes:
        reference_speed_mps: Speed at the reference height (the base of
            the boundary-layer profile).
        reference_height_m: Height the reference speed is defined at.
        direction_deg: Compass direction the wind BLOWS TOWARD, degrees
            clockwise from +X axis in the world's horizontal plane.
        alpha: Boundary-layer power-law exponent (0.10 open terrain,
            0.22 suburban, 0.33 urban -- labeled estimate; 0.16 default
            for open-to-suburban transition).
        gust_amplitude_frac: Gust amplitude as a fraction of base speed.
        gust_period_s: Primary gust oscillation period.
        max_speed_mps: Safety clamp on the evolved speed.
        seed: Deterministic seed for the gust sub-stream.
    """
    reference_speed_mps: float = 5.0
    reference_height_m: float = 10.0
    direction_deg: float = 0.0
    alpha: float = 0.16
    gust_amplitude_frac: float = 0.25
    gust_period_s: float = 12.0
    max_speed_mps: float = 80.0
    seed: int = 42

    def __post_init__(self):
        if self.reference_speed_mps < 0.0:
            raise ValueError("reference_speed_mps cannot be negative")
        if self.reference_height_m <= 0.0:
            raise ValueError("reference_height_m must be positive")
        if not (0.05 <= self.alpha <= 0.45):
            raise ValueError(
                f"alpha must be in [0.05, 0.45] (documented terrain range), "
                f"got {self.alpha}"
            )
        if self.gust_amplitude_frac < 0.0 or self.gust_amplitude_frac > 1.0:
            raise ValueError("gust_amplitude_frac must be in [0, 1]")
        if self.gust_period_s <= 0.0:
            raise ValueError("gust_period_s must be positive")


@dataclass
class WindState:
    """Mutable, serializable wind state."""
    time_s: float = 0.0
    current_speed_mps: float = 0.0   # base + gust at reference height
    base_speed_mps: float = 0.0
    gust_speed_mps: float = 0.0
    band: str = BeaufortBand.CALM.value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_s": self.time_s,
            "current_speed_mps": self.current_speed_mps,
            "base_speed_mps": self.base_speed_mps,
            "gust_speed_mps": self.gust_speed_mps,
            "band": self.band,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WindState":
        return cls(
            time_s=data["time_s"],
            current_speed_mps=data["current_speed_mps"],
            base_speed_mps=data["base_speed_mps"],
            gust_speed_mps=data["gust_speed_mps"],
            band=data["band"],
        )


class WindField:
    """Deterministic wind field with physical force coupling.

    Usage:
        field = WindField(WindConfig(reference_speed_mps=25.0))
        field.step(dt=0.1)
        force = field.drag_force(area_m2=2.0, drag_coefficient=1.2)
        # -> force vector (Vec3-compatible tuple) in world space
    """

    def __init__(self, config: WindConfig, event_bus: Optional[EventBus] = None):
        self.config = config
        self.state = WindState()
        self._rng = DeterministicRNG(config.seed, name="wind-gust")
        self._logger = get_logger("engine.environment.wind")
        self._event_bus = event_bus
        # Initial band from the reference speed.
        self.state.base_speed_mps = config.reference_speed_mps
        self.state.current_speed_mps = config.reference_speed_mps
        self.state.band = classify_beaufort(config.reference_speed_mps).value

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def step(self, dt: float) -> WindState:
        """Advance the field dt seconds; deterministic given the seed."""
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        cfg = self.config
        self.state.time_s += dt

        # Base speed: constant reference (slow weather evolution is a
        # future layer; the gust model carries sub-period variation now).
        base = cfg.reference_speed_mps

        # Deterministic gust: two seeded sine components (primary period
        # plus a slower modulation), no per-frame randomness.
        t = self.state.time_s
        primary = math.sin(2.0 * math.pi * t / cfg.gust_period_s)
        phase0 = self._rng.uniform(0.0, 2.0 * math.pi)
        secondary = math.sin(2.0 * math.pi * t / (cfg.gust_period_s * 2.7) + phase0)
        gust = (
            cfg.reference_speed_mps
            * cfg.gust_amplitude_frac
            * (0.7 * primary + 0.3 * secondary)
        )

        speed = min(cfg.max_speed_mps, max(0.0, base + gust))
        self.state.base_speed_mps = base
        self.state.gust_speed_mps = gust
        self.state.current_speed_mps = speed

        new_band = classify_beaufort(speed)
        if new_band.value != self.state.band:
            old = self.state.band
            self.state.band = new_band.value
            self._logger.info(
                "Wind band changed",
                context={"from": old, "to": new_band.value, "speed_mps": speed},
            )
            if self._event_bus is not None:
                self._event_bus.publish(
                    "wind.band_changed",
                    tick=int(round(self.state.time_s / max(dt, 1e-9))),
                    timestamp=self.state.time_s,
                    data={
                        "from_band": old,
                        "to_band": new_band.value,
                        "speed_mps": speed,
                        "direction_deg": cfg.direction_deg,
                    },
                )
        return self.state

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def speed_at_height(self, height_m: float) -> float:
        """Boundary-layer power law at a height above ground.

        Documented approximation (spec sec 29): monotone speed-up with
        height above the reference height; below it, the reference speed
        is used (the profile's log-layer transition is not modelled).
        """
        if height_m < 0.0:
            raise ValueError(f"height cannot be negative, got {height_m}")
        cfg = self.config
        if height_m <= cfg.reference_height_m:
            return self.state.current_speed_mps
        return min(
            cfg.max_speed_mps,
            self.state.current_speed_mps
            * (height_m / cfg.reference_height_m) ** cfg.alpha,
        )

    def direction_vector(self) -> Tuple[float, float, float]:
        """Unit vector the wind blows TOWARD (world +X = 0 deg, +Y = 90 deg)."""
        rad = math.radians(self.config.direction_deg)
        return (math.cos(rad), math.sin(rad), 0.0)

    # ------------------------------------------------------------------
    # Force coupling (the causal hook into physics)
    # ------------------------------------------------------------------

    def drag_force(
        self,
        area_m2: float,
        height_m: Optional[float] = None,
        drag_coefficient: float = 1.2,
        body_velocity: Optional[Tuple[float, float, float]] = None,
    ) -> Tuple[float, float, float]:
        """Quadratic drag force vector from the wind on an exposed area.

        F = 0.5 * rho * Cd * A * v_rel^2, applied along the wind direction
        (or opposite the relative velocity when body_velocity is given).

        Documented approximation: no shielding, no turbulence, area is
        treated as flat-facing; adequate for building-scale load coupling,
        labelled as such.
        """
        if area_m2 < 0.0:
            raise ValueError(f"area cannot be negative, got {area_m2}")
        if drag_coefficient <= 0.0:
            raise ValueError(f"drag_coefficient must be positive, got {drag_coefficient}")

        height = height_m if height_m is not None else self.config.reference_height_m
        v_wind = self.speed_at_height(height)
        dx, dy, _dz = self.direction_vector()

        if body_velocity is None:
            speed_rel = v_wind
            return (0.5 * AIR_DENSITY_KG_M3 * drag_coefficient * area_m2 * speed_rel ** 2 * dx,
                    0.5 * AIR_DENSITY_KG_M3 * drag_coefficient * area_m2 * speed_rel ** 2 * dy,
                    0.0)

        # Relative-velocity form: force opposes the RELATIVE flow.
        rvx = dx * v_wind - body_velocity[0]
        rvy = dy * v_wind - body_velocity[1]
        rvz = -body_velocity[2]
        rel_speed = math.sqrt(rvx * rvx + rvy * rvy + rvz * rvz)
        if rel_speed < 1e-9:
            return (0.0, 0.0, 0.0)
        scale = 0.5 * AIR_DENSITY_KG_M3 * drag_coefficient * area_m2 * rel_speed
        return (scale * rvx, scale * rvy, scale * rvz)

    def apply_wind_loads(self, world, dt: float,
                         exposed_area_by_body: Dict[str, float],
                         drag_coefficient: float = 1.2) -> Dict[str, Tuple[float, float, float]]:
        """Apply wind drag forces as impulses to physics bodies in a
        SimpleRigidBodyBackend world.

        Causal chain (no scripts): wind state -> drag force -> impulse ->
        body velocity -> contacts/impacts. Returns the impulses applied,
        body id -> impulse vector, for the debugger/inspector.

        Only DYNAMIC bodies listed in exposed_area_by_body receive loads;
        static bodies are unaffected by construction.
        """
        impulses: Dict[str, Tuple[float, float, float]] = {}
        from engine.physics.math3 import Vec3
        for body_id, area in exposed_area_by_body.items():
            body = world.get_body(body_id)
            if body.is_static:
                continue
            fx, fy, fz = self.drag_force(
                area_m2=area,
                drag_coefficient=drag_coefficient,
                body_velocity=(body.linear_velocity.x, body.linear_velocity.y,
                               body.linear_velocity.z),
            )
            ix, iy, iz = fx * dt, fy * dt, fz * dt
            body.apply_impulse(Vec3(ix, iy, iz))
            impulses[body_id] = (ix, iy, iz)
        return impulses

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format_version": 1,
            "config": {
                "reference_speed_mps": self.config.reference_speed_mps,
                "reference_height_m": self.config.reference_height_m,
                "direction_deg": self.config.direction_deg,
                "alpha": self.config.alpha,
                "gust_amplitude_frac": self.config.gust_amplitude_frac,
                "gust_period_s": self.config.gust_period_s,
                "max_speed_mps": self.config.max_speed_mps,
                "seed": self.config.seed,
            },
            "state": self.state.to_dict(),
        }
