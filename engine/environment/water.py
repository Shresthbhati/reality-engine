"""Water body simulation: depth, volume, buoyancy."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

from engine.core.logging import get_logger
from engine.world.events import EventBus


@dataclass(frozen=True)
class WaterConfig:
    """Configuration for water physics simulation."""

    water_density_kg_m3: float = 1000.0
    """Density of fresh water in kg/m³."""

    gravity_m_s2: float = 9.81
    """Gravitational acceleration in m/s²."""

    seed: int = 42
    """Random seed for deterministic water behavior."""

    flow_rate_coefficient: float = 0.5
    """P1 simplification: unitless coefficient governing rate of level equalization
    between adjacent bodies. Real-world hydraulic conductivity is much more complex
    (depends on substrate, temperature, head differential). This is a tunable knob
    that controls speed of flow. Default 0.5 gives moderate equalization.
    """


class WaterBody:
    """A single water body (pond, lake, etc.) with depth and volume."""

    def __init__(
        self,
        body_id: str,
        surface_area_m2: float,
        depth_m: float = 0.0,
        drainage_rate_m3_s: float = 0.0,
        max_depth_m: Optional[float] = None,
    ):
        """Initialize a water body.

        Args:
            body_id: Unique identifier for this water body.
            surface_area_m2: Surface area in square meters.
            depth_m: Current depth in meters (default 0.0).
            drainage_rate_m3_s: Volume drained per second (default 0.0).
            max_depth_m: Optional overflow threshold in meters (default None
                = no limit). Used by WaterState.step() to detect overflow.

        Raises:
            ValueError: If surface_area_m2 <= 0 or drainage_rate_m3_s < 0.
        """
        if surface_area_m2 <= 0:
            raise ValueError(
                f"surface_area_m2 must be > 0, got {surface_area_m2}"
            )
        if drainage_rate_m3_s < 0:
            raise ValueError(
                f"drainage_rate_m3_s must be >= 0, got {drainage_rate_m3_s}"
            )
        self.body_id = body_id
        self.surface_area_m2 = surface_area_m2
        self.depth_m = depth_m
        self.drainage_rate_m3_s = drainage_rate_m3_s
        self.max_depth_m = max_depth_m

    @property
    def volume_m3(self) -> float:
        """Compute volume as surface_area_m2 * depth_m."""
        return self.surface_area_m2 * self.depth_m


class WaterState:
    """Manages multiple water bodies and computes water physics."""

    def __init__(self, config: WaterConfig, event_bus: Optional[EventBus] = None):
        """Initialize water state with a config.

        Args:
            config: WaterConfig instance.
            event_bus: Optional EventBus to publish "water.body_overflowed"
                events to on overflow crossings. Default None preserves
                prior (Task 1/2) behavior of no event publishing.
        """
        self.config = config
        self._bodies: Dict[str, WaterBody] = {}
        self._adjacency: Dict[str, Set[str]] = {}
        self.format_version = 1
        self._logger = get_logger("engine.environment.water")
        self._event_bus = event_bus

    def register_body(self, body: WaterBody) -> None:
        """Register a water body.

        Args:
            body: WaterBody to register.

        Raises:
            ValueError: If a body with the same body_id is already registered.
        """
        if body.body_id in self._bodies:
            raise ValueError(
                f"Water body '{body.body_id}' is already registered"
            )
        self._bodies[body.body_id] = body
        self._logger.info(
            "Water body registered",
            context={
                "body_id": body.body_id,
                "surface_area_m2": body.surface_area_m2,
                "depth_m": body.depth_m,
            },
        )

    def get_body(self, body_id: str) -> WaterBody:
        """Retrieve a registered water body by ID.

        Args:
            body_id: ID of the water body.

        Returns:
            The WaterBody object.

        Raises:
            ValueError: If body_id is not registered.
        """
        if body_id not in self._bodies:
            raise ValueError(f"Unknown water body: '{body_id}'")
        return self._bodies[body_id]

    def buoyancy_force_n(
        self, body_id: str, submerged_volume_m3: float
    ) -> float:
        """Compute buoyant force on an object in water.

        Uses Archimedes' principle: F = ρ * g * V_submerged
        where ρ is water density, g is gravity, V_submerged is submerged volume.

        This is a P1 simplification: we assume the submerged volume is
        instantaneously available without flow dynamics or surface tension.
        A full model would include buoyancy over time as water level rises/falls.

        Args:
            body_id: ID of the water body containing the submerged object.
            submerged_volume_m3: Volume of the object submerged in cubic meters.

        Returns:
            Buoyant force in Newtons.

        Raises:
            ValueError: If submerged_volume_m3 < 0 or exceeds the body's volume.
        """
        if submerged_volume_m3 < 0:
            raise ValueError(
                f"submerged_volume_m3 must be >= 0, got {submerged_volume_m3}"
            )

        body = self.get_body(body_id)
        if submerged_volume_m3 > body.volume_m3:
            raise ValueError(
                f"submerged_volume_m3 ({submerged_volume_m3}) exceeds "
                f"body volume ({body.volume_m3}) for body '{body_id}'"
            )

        force = (
            self.config.water_density_kg_m3
            * self.config.gravity_m_s2
            * submerged_volume_m3
        )
        return force

    def connect_bodies(self, body_id_a: str, body_id_b: str) -> None:
        """Register a bidirectional adjacency between two water bodies.

        Allows flow equalization during step().

        Args:
            body_id_a: ID of first body.
            body_id_b: ID of second body.

        Raises:
            ValueError: If either body is unknown or the pair is already connected.
        """
        if body_id_a not in self._bodies:
            raise ValueError(f"Unknown water body: '{body_id_a}'")
        if body_id_b not in self._bodies:
            raise ValueError(f"Unknown water body: '{body_id_b}'")

        if body_id_a not in self._adjacency:
            self._adjacency[body_id_a] = set()
        if body_id_b not in self._adjacency:
            self._adjacency[body_id_b] = set()

        if body_id_b in self._adjacency[body_id_a]:
            raise ValueError(
                f"Water bodies '{body_id_a}' and '{body_id_b}' are already connected"
            )

        self._adjacency[body_id_a].add(body_id_b)
        self._adjacency[body_id_b].add(body_id_a)

        self._logger.info(
            "Water bodies connected",
            context={"body_id_a": body_id_a, "body_id_b": body_id_b},
        )

    def step(self, dt: float, tick: int) -> None:
        """Execute one simulation step: flow between adjacent bodies and drainage.

        For each adjacent pair of bodies where body A is deeper than body B,
        transfer volume from A to B. The transfer amount is the minimum of:
        - Rate-limited flow: flow_rate_coefficient * (depth_a - depth_b) * dt
        - Equalizing flow: the exact volume that makes depths equal,
          computed as (depth_a - depth_b) * area_a * area_b / (area_a + area_b)

        This ensures no single step overshoots equalization. Also applies per-body
        drainage: subtracts drainage_rate_m3_s * dt from each body's volume,
        floored at zero.

        Args:
            dt: Time step in seconds.
            tick: Simulation tick (for logging/debugging).
        """
        # Snapshot every body's depth at the START of the step. All pair
        # calculations below read only this snapshot (never live/mutated
        # state), so a body with multiple neighbors gets consistent,
        # order-independent flow amounts for this tick.
        depth_snapshot = {
            body_id: body.depth_m for body_id, body in self._bodies.items()
        }
        # Snapshot overflow status BEFORE this step's flow/drainage, so we
        # can detect a body newly crossing above max_depth_m this step
        # (fires once on the crossing, not on every subsequent step spent
        # above the threshold).
        was_overflowing = {
            body_id: (
                body.max_depth_m is not None
                and depth_snapshot[body_id] > body.max_depth_m
            )
            for body_id, body in self._bodies.items()
        }
        # Net volume delta (m3) accumulated per body across all pairs this
        # step, applied to live state only after every pair is computed.
        net_delta_m3: Dict[str, float] = {
            body_id: 0.0 for body_id in self._bodies
        }

        # First pass: compute all flows between adjacent bodies using the
        # snapshot only.
        processed_pairs = set()
        for body_id_a, neighbors in list(self._adjacency.items()):
            for body_id_b in neighbors:
                # Avoid processing the same pair twice
                pair_key = tuple(sorted([body_id_a, body_id_b]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                body_a = self._bodies[body_id_a]
                body_b = self._bodies[body_id_b]
                depth_a = depth_snapshot[body_id_a]
                depth_b = depth_snapshot[body_id_b]

                # Determine which body is deeper; only flow from deeper to shallower
                if depth_a > depth_b:
                    deeper_body, shallower_body = body_a, body_b
                    deeper_depth, shallower_depth = depth_a, depth_b
                    source_id, dest_id = body_id_a, body_id_b
                elif depth_b > depth_a:
                    deeper_body, shallower_body = body_b, body_a
                    deeper_depth, shallower_depth = depth_b, depth_a
                    source_id, dest_id = body_id_b, body_id_a
                else:
                    # Depths equal, no flow
                    continue

                # Compute the two candidate flows
                depth_diff = deeper_depth - shallower_depth
                rate_limited_flow = (
                    self.config.flow_rate_coefficient * depth_diff * dt
                )

                # Equalizing flow: volume that exactly balances the depths
                # Derived from: (V_a - F) / area_a = (V_b + F) / area_b
                # Solving: F = area_a * area_b * (depth_a - depth_b) / (area_a + area_b)
                equalizing_flow = (
                    deeper_body.surface_area_m2
                    * shallower_body.surface_area_m2
                    * depth_diff
                    / (
                        deeper_body.surface_area_m2
                        + shallower_body.surface_area_m2
                    )
                )

                # Apply the smaller flow to guarantee no overshoot
                transfer_volume = min(rate_limited_flow, equalizing_flow)

                # Accumulate net delta; do not mutate body state yet.
                net_delta_m3[source_id] -= transfer_volume
                net_delta_m3[dest_id] += transfer_volume

                self._logger.info(
                    "Water flow between bodies",
                    context={
                        "tick": tick,
                        "source": source_id,
                        "dest": dest_id,
                        "transfer_m3": transfer_volume,
                        "rate_limited_m3": rate_limited_flow,
                        "equalizing_m3": equalizing_flow,
                    },
                )

        # Second pass: apply the accumulated net delta from the snapshot to
        # each body's actual depth, floored at zero.
        for body_id, delta in net_delta_m3.items():
            body = self._bodies[body_id]
            body.depth_m = max(
                0.0,
                depth_snapshot[body_id] + delta / body.surface_area_m2,
            )

        # Third pass: apply drainage to all bodies
        for body in self._bodies.values():
            drainage_volume = body.drainage_rate_m3_s * dt
            if drainage_volume > 0:
                body.depth_m = max(
                    0.0, body.depth_m - drainage_volume / body.surface_area_m2
                )
                self._logger.info(
                    "Water drainage",
                    context={
                        "tick": tick,
                        "body_id": body.body_id,
                        "drainage_m3": drainage_volume,
                    },
                )

        # Fourth pass: detect newly-crossed overflow (depth now above
        # max_depth_m, wasn't before this step) and publish an event.
        for body_id, body in self._bodies.items():
            if body.max_depth_m is None:
                continue
            is_overflowing = body.depth_m > body.max_depth_m
            if is_overflowing and not was_overflowing[body_id]:
                self._logger.info(
                    "Water body overflowed",
                    context={
                        "tick": tick,
                        "body_id": body_id,
                        "depth_m": body.depth_m,
                        "max_depth_m": body.max_depth_m,
                    },
                )
                if self._event_bus is not None:
                    self._event_bus.publish(
                        event_type="water.body_overflowed",
                        tick=tick,
                        timestamp=float(tick),
                        data={
                            "body_id": body_id,
                            "depth_m": body.depth_m,
                            "max_depth_m": body.max_depth_m,
                        },
                    )

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get current water diagnostics for debugging/inspection.

        Returns:
            Dict with total volume across all bodies (m3), body count,
            connection count, and overflowing body count (bodies currently
            above their max_depth_m).
        """
        total_volume_m3 = sum(
            body.volume_m3 for body in self._bodies.values()
        )
        connection_count = sum(
            len(neighbors) for neighbors in self._adjacency.values()
        ) // 2
        overflowing_count = sum(
            1
            for body in self._bodies.values()
            if body.max_depth_m is not None and body.depth_m > body.max_depth_m
        )
        return {
            "total_volume_m3": total_volume_m3,
            "body_count": len(self._bodies),
            "connection_count": connection_count,
            "overflowing_body_count": overflowing_count,
        }

    def serialize(self) -> dict:
        """Serialize all registered water bodies, adjacency, and drainage rates.

        Returns:
            Dictionary with format_version, bodies, and adjacency data.
        """
        bodies_data = {}
        for body_id, body in self._bodies.items():
            bodies_data[body_id] = {
                "surface_area_m2": body.surface_area_m2,
                "depth_m": body.depth_m,
                "drainage_rate_m3_s": body.drainage_rate_m3_s,
            }

        # Serialize adjacency as list of sorted pairs (to avoid duplicates)
        adjacency_pairs = []
        processed = set()
        for body_id_a, neighbors in self._adjacency.items():
            for body_id_b in neighbors:
                pair_key = tuple(sorted([body_id_a, body_id_b]))
                if pair_key not in processed:
                    adjacency_pairs.append(list(pair_key))
                    processed.add(pair_key)

        return {
            "format_version": self.format_version,
            "bodies": bodies_data,
            "adjacency": adjacency_pairs,
        }

    def deserialize(self, data: dict) -> None:
        """Deserialize water bodies, adjacency, and drainage rates from serialized data.

        Replaces any existing registered bodies and adjacency.

        Args:
            data: Dictionary with format_version, bodies, and adjacency data.

        Raises:
            ValueError: If format_version does not match.
        """
        if data.get("format_version") != self.format_version:
            raise ValueError(
                f"Format version mismatch: expected {self.format_version}, "
                f"got {data.get('format_version')}"
            )

        self._bodies = {}
        self._adjacency = {}

        for body_id, body_data in data.get("bodies", {}).items():
            body = WaterBody(
                body_id=body_id,
                surface_area_m2=body_data["surface_area_m2"],
                depth_m=body_data["depth_m"],
                drainage_rate_m3_s=body_data.get("drainage_rate_m3_s", 0.0),
            )
            self.register_body(body)

        # Restore adjacency relationships
        for pair in data.get("adjacency", []):
            if len(pair) == 2:
                body_id_a, body_id_b = pair
                # Use internal adjacency dict directly to avoid duplicate validation
                if body_id_a not in self._adjacency:
                    self._adjacency[body_id_a] = set()
                if body_id_b not in self._adjacency:
                    self._adjacency[body_id_b] = set()
                self._adjacency[body_id_a].add(body_id_b)
                self._adjacency[body_id_b].add(body_id_a)
