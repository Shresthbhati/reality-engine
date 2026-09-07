"""Debris system: P2 destruction with fragment lifecycle and rendering.

Implements fragment lifecycle management with:
  - Object pooling for performance
  - Collision handling between debris
  - Rendering queue for batching
  - Age-based culling
  - Deterministic physics on fragments

Spec §14: DEBRIS SYSTEM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import math

from engine.core.rng import DeterministicRNG
from engine.core.logging import get_logger
from engine.physics.math3 import Vec3


@dataclass(frozen=True)
class DebrisConfig:
    """Debris system configuration."""
    max_debris_count: int = 1000
    debris_lifetime_s: float = 10.0
    collision_damping: float = 0.8
    gravity_m_s2: float = 9.81
    min_velocity_threshold_m_s: float = 0.01
    render_batch_size: int = 256


class DebrisFragment:
    """Individual debris fragment with lifecycle state."""

    def __init__(
        self,
        fragment_id: str,
        position: Vec3,
        velocity: Vec3,
        size_m: float,
        mass_kg: float,
        lifetime_s: float,
        created_at_tick: int,
        sharpness: float = 0.5,
    ):
        """Initialize debris fragment.

        Args:
            fragment_id: Unique fragment identifier
            position: Initial position (world space)
            velocity: Initial velocity (m/s)
            size_m: Fragment characteristic size
            mass_kg: Fragment mass (kg)
            lifetime_s: Time until automatic removal
            created_at_tick: Simulation tick when created
            sharpness: Edge sharpness (0.0 dull to 1.0 sharp)
        """
        self.fragment_id = fragment_id
        self.position = position
        self.velocity = velocity
        self.size_m = size_m
        self.mass_kg = mass_kg
        self.lifetime_s = lifetime_s
        self.created_at_tick = created_at_tick
        self.sharpness = sharpness
        self.age_s = 0.0
        self.is_sleeping = False
        self.sleep_time_s = 0.0

    def step(self, dt: float, gravity: float) -> None:
        """Step fragment physics forward.

        Args:
            dt: Timestep (seconds)
            gravity: Gravitational acceleration (m/s²)
        """
        self.age_s += dt

        # Wake if velocity exceeds threshold
        if self.is_sleeping and self.velocity.length() > 0.01:
            self.is_sleeping = False
            self.sleep_time_s = 0.0

        # Skip physics for sleeping fragments
        if self.is_sleeping:
            self.sleep_time_s += dt
            return

        # Apply gravity (create new Vec3)
        self.velocity = Vec3(
            self.velocity.x,
            self.velocity.y,
            self.velocity.z - gravity * dt,
        )

        # Update position
        self.position = self.position + self.velocity * dt

        # Sleep if velocity falls below threshold
        if self.velocity.length() < 0.01:
            self.is_sleeping = True

    def is_expired(self) -> bool:
        """Check if fragment has exceeded lifetime."""
        return self.age_s >= self.lifetime_s

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "fragment_id": self.fragment_id,
            "position": self.position.to_dict(),
            "velocity": self.velocity.to_dict(),
            "size_m": self.size_m,
            "mass_kg": self.mass_kg,
            "lifetime_s": self.lifetime_s,
            "created_at_tick": self.created_at_tick,
            "sharpness": self.sharpness,
            "age_s": self.age_s,
            "is_sleeping": self.is_sleeping,
        }


class DebrisPool:
    """Object pool for debris fragments."""

    def __init__(self, initial_capacity: int = 100):
        """Initialize debris pool.

        Args:
            initial_capacity: Initial pool size
        """
        self._available: List[DebrisFragment] = []
        self._active: Dict[str, DebrisFragment] = {}
        self._initial_capacity = initial_capacity

    def acquire(
        self,
        fragment_id: str,
        position: Vec3,
        velocity: Vec3,
        size_m: float,
        mass_kg: float,
        lifetime_s: float,
        created_at_tick: int,
        sharpness: float = 0.5,
    ) -> DebrisFragment:
        """Acquire a fragment from pool or create new.

        Args:
            fragment_id: Unique fragment identifier
            position: Initial position
            velocity: Initial velocity
            size_m: Fragment size
            mass_kg: Fragment mass
            lifetime_s: Fragment lifetime
            created_at_tick: Creation tick
            sharpness: Edge sharpness

        Returns:
            DebrisFragment ready for use
        """
        if self._available:
            fragment = self._available.pop()
            # Reinitialize
            fragment.fragment_id = fragment_id
            fragment.position = position
            fragment.velocity = velocity
            fragment.size_m = size_m
            fragment.mass_kg = mass_kg
            fragment.lifetime_s = lifetime_s
            fragment.created_at_tick = created_at_tick
            fragment.sharpness = sharpness
            fragment.age_s = 0.0
            fragment.is_sleeping = False
            fragment.sleep_time_s = 0.0
        else:
            fragment = DebrisFragment(
                fragment_id,
                position,
                velocity,
                size_m,
                mass_kg,
                lifetime_s,
                created_at_tick,
                sharpness,
            )

        self._active[fragment_id] = fragment
        return fragment

    def release(self, fragment_id: str) -> None:
        """Release fragment back to pool.

        Args:
            fragment_id: Fragment to release
        """
        if fragment_id in self._active:
            fragment = self._active.pop(fragment_id)
            self._available.append(fragment)

    def get_active(self) -> List[DebrisFragment]:
        """Get all active fragments."""
        return list(self._active.values())

    def clear(self) -> None:
        """Clear all fragments."""
        self._active.clear()
        self._available.clear()

    def stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        return {
            "active": len(self._active),
            "available": len(self._available),
            "total_capacity": len(self._active) + len(self._available),
        }


class DebrisManager:
    """Manages debris fragment lifecycle, physics, and rendering."""

    def __init__(self, config: Optional[DebrisConfig] = None, seed: int = 42):
        """Initialize debris manager.

        Args:
            config: Debris configuration
            seed: RNG seed for determinism
        """
        self.config = config or DebrisConfig()
        self._rng = DeterministicRNG(seed)
        self._logger = get_logger("engine.physics.destruction.debris")
        self._pool = DebrisPool(self.config.max_debris_count)
        self._render_queue: List[DebrisFragment] = []
        self._collision_events: List[Dict[str, Any]] = []
        self._total_spawned = 0

    def spawn_fragments(
        self,
        parent_id: str,
        fragments: List[Dict[str, Any]],
        tick: int,
        timestamp: float,
    ) -> int:
        """Spawn debris fragments from fracture/destruction event.

        Args:
            parent_id: ID of object that fractured
            fragments: List of fragment specifications from fracture system
            tick: Current simulation tick
            timestamp: Current simulation time

        Returns:
            Number of fragments actually spawned
        """
        spawned = 0
        for i, frag_spec in enumerate(fragments):
            if len(self._pool._active) >= self.config.max_debris_count:
                self._logger.warning(
                    "Debris pool full, dropping fragments",
                    context={"active": len(self._pool._active), "max": self.config.max_debris_count},
                )
                break

            fragment_id = f"{parent_id}_debris_{self._total_spawned}_{i}"
            position = Vec3.from_dict(frag_spec.get("position", {"x": 0, "y": 0, "z": 0}))
            velocity = Vec3.from_dict(frag_spec.get("velocity", {"x": 0, "y": 0, "z": 0}))
            size_m = frag_spec.get("size_m", 0.1)
            sharpness = frag_spec.get("sharpness", 0.5)
            lifetime_s = frag_spec.get("lifetime_s", self.config.debris_lifetime_s)

            # Compute mass from size (approximate cube)
            density = 2500.0  # kg/m³ (glass-like)
            volume = size_m ** 3
            mass_kg = max(0.01, volume * density)

            fragment = self._pool.acquire(
                fragment_id,
                position,
                velocity,
                size_m,
                mass_kg,
                lifetime_s,
                tick,
                sharpness,
            )

            self._total_spawned += 1
            spawned += 1

            self._logger.debug(
                "Debris fragment spawned",
                context={
                    "fragment_id": fragment_id,
                    "parent_id": parent_id,
                    "size_m": size_m,
                    "lifetime_s": lifetime_s,
                },
            )

        return spawned

    def step(self, dt: float, tick: int) -> None:
        """Step debris simulation forward.

        Args:
            dt: Timestep (seconds)
            tick: Simulation tick
        """
        active = self._pool.get_active()

        # Step all active fragments
        for fragment in active:
            fragment.step(dt, self.config.gravity_m_s2)

        # Remove expired fragments
        expired = [f for f in active if f.is_expired()]
        for fragment in expired:
            self._pool.release(fragment.fragment_id)

        # Build render queue (sorted by depth for proper blending)
        self._render_queue = sorted(
            self._pool.get_active(),
            key=lambda f: f.position.z,
            reverse=True,
        )

        if expired:
            self._logger.debug(
                "Debris cleaned up",
                context={
                    "expired": len(expired),
                    "remaining": len(self._pool.get_active()),
                },
            )

    def handle_collision(
        self,
        fragment_id: str,
        collision_point: Vec3,
        collision_normal: Vec3,
        restitution: float = 0.3,
    ) -> None:
        """Handle collision between fragment and environment.

        Args:
            fragment_id: Fragment involved in collision
            collision_point: Collision location
            collision_normal: Surface normal at collision
            restitution: Bounce factor (0.0-1.0)
        """
        active = self._pool.get_active()
        fragment = next((f for f in active if f.fragment_id == fragment_id), None)

        if not fragment:
            return

        # Reflect velocity around normal
        dot = fragment.velocity.x * collision_normal.x + \
              fragment.velocity.y * collision_normal.y + \
              fragment.velocity.z * collision_normal.z

        if dot < 0:  # Only if moving toward surface
            # Reflect: v' = v - (1 + r) * (v·n) * n
            reflected = Vec3(
                fragment.velocity.x - (1.0 + restitution) * dot * collision_normal.x,
                fragment.velocity.y - (1.0 + restitution) * dot * collision_normal.y,
                fragment.velocity.z - (1.0 + restitution) * dot * collision_normal.z,
            )

            # Apply damping
            fragment.velocity = reflected * self.config.collision_damping

            # Move fragment away from surface
            fragment.position = fragment.position + collision_normal * 0.01

            # Record collision event
            self._collision_events.append({
                "fragment_id": fragment_id,
                "collision_point": collision_point.to_dict(),
                "collision_normal": collision_normal.to_dict(),
                "velocity_after": fragment.velocity.to_dict(),
            })

            self._logger.debug(
                "Debris collision handled",
                context={
                    "fragment_id": fragment_id,
                    "restitution": restitution,
                    "velocity_mag": fragment.velocity.length(),
                },
            )

    def get_render_queue(self) -> List[Dict[str, Any]]:
        """Get fragments in rendering order.

        Returns:
            List of fragment data sorted for rendering
        """
        return [
            {
                "fragment_id": f.fragment_id,
                "position": f.position.to_dict(),
                "velocity": f.velocity.to_dict(),
                "size_m": f.size_m,
                "sharpness": f.sharpness,
                "age_s": f.age_s,
                "lifetime_s": f.lifetime_s,
                "progress": min(1.0, f.age_s / f.lifetime_s),
            }
            for f in self._render_queue
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get debris system statistics."""
        pool_stats = self._pool.stats()
        active = self._pool.get_active()

        return {
            "pool": pool_stats,
            "total_spawned": self._total_spawned,
            "active_count": len(active),
            "collision_events": len(self._collision_events),
            "sleeping_count": sum(1 for f in active if f.is_sleeping),
            "render_queue_size": len(self._render_queue),
        }

    def serialize(self) -> Dict[str, Any]:
        """Serialize debris manager state."""
        active = self._pool.get_active()
        return {
            "format_version": 1,
            "total_spawned": self._total_spawned,
            "active_fragments": [f.to_dict() for f in active],
            "collision_events": self._collision_events,
        }

    def deserialize(self, data: Dict[str, Any]) -> None:
        """Restore debris manager state."""
        if data.get("format_version") != 1:
            raise ValueError(f"Unsupported debris state version: {data.get('format_version')}")

        self._total_spawned = data.get("total_spawned", 0)
        self._collision_events = data.get("collision_events", [])

        # Restore fragments
        for frag_data in data.get("active_fragments", []):
            position = Vec3.from_dict(frag_data["position"])
            velocity = Vec3.from_dict(frag_data["velocity"])

            self._pool.acquire(
                frag_data["fragment_id"],
                position,
                velocity,
                frag_data["size_m"],
                frag_data["mass_kg"],
                frag_data["lifetime_s"],
                frag_data["created_at_tick"],
                frag_data["sharpness"],
            )
