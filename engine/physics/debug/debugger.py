"""Physics Debugger: read-only debug-draw records for rigid bodies,
contacts, forces, and numerics health (§36 debug visualization).

Same pattern as Viewport/Inspector: a headless engine has no GPU to draw
to, so "debug visualization" here means structured records a renderer
would consume (bounding shape + transform, contact normal/penetration,
force/velocity vectors, sleep-state, energy) -- not new physics state.
Reads directly off the real `PhysicsWorld`/`ContactRecord` the golden
tests already exercise (`engine/physics/backend/simple_backend.py`),
not a parallel data shape.
"""

from __future__ import annotations

from typing import Any, Dict, List

from engine.physics.backend.simple_backend import ContactRecord, PhysicsWorld
from engine.physics.diagnostics.numerics import NumericsReport
from engine.physics.rigid.body import RigidBody, SleepState


class PhysicsDebugger:
    """Stateless -- every method takes the physics state it describes and
    returns plain-dict debug records. No mutation, no stored snapshot."""

    def bodies_debug(self, world: PhysicsWorld) -> List[Dict[str, Any]]:
        return [
            {
                "id": b.id,
                "shape": b.shape.to_dict(),
                "position": b.position.to_dict(),
                "orientation": b.orientation.to_dict(),
                "sleep_state": b.sleep_state.value,
                "linear_velocity": b.linear_velocity.to_dict(),
                "angular_velocity": b.angular_velocity.to_dict(),
                "kinetic_energy": b.kinetic_energy(),
            }
            for _, b in sorted(world.bodies.items())
        ]

    def forces_debug(self, world: PhysicsWorld) -> List[Dict[str, Any]]:
        """Force/torque accumulators, for bodies where either is non-zero
        (drawing a zero-length vector is never useful for debugging)."""
        records = []
        for body_id, b in sorted(world.bodies.items()):
            force, torque = b.force, b.torque
            if force.length_sq() > 0.0 or torque.length_sq() > 0.0:
                records.append({"id": body_id, "force": force.to_dict(), "torque": torque.to_dict()})
        return records

    def contacts_debug(self, world: PhysicsWorld) -> List[Dict[str, Any]]:
        return [self._contact_record_to_dict(c) for c in world.last_contacts]

    @staticmethod
    def _contact_record_to_dict(c: ContactRecord) -> Dict[str, Any]:
        return {
            "body_a": c.body_a_id,  # None means the static-plane side
            "body_b": c.body_b_id,
            "plane_id": c.plane_id,
            "normal": c.normal.to_dict(),
            "penetration": c.penetration,
        }

    def numerics_debug(self, report: NumericsReport) -> Dict[str, Any]:
        return report.to_dict()

    def summary(self, world: PhysicsWorld) -> Dict[str, Any]:
        sleep_counts = {state.value: 0 for state in SleepState}
        total_ke = 0.0
        for b in world.bodies.values():
            sleep_counts[b.sleep_state.value] += 1
            total_ke += b.kinetic_energy()

        return {
            "body_count": len(world.bodies),
            "by_sleep_state": sleep_counts,
            "contact_count": len(world.last_contacts),
            "total_kinetic_energy": total_ke,
        }
