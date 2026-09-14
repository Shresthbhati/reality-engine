# Robotics Integration

Status: MISSING (2026-09-14) — intentionally deferred (ROADMAP P5+)

## Purpose

Make compiled worlds usable by robots: collision-relevant geometry,
navigable free space, object affordances, and re-localization of a
robot's sensors into the world frame.

## Current state

WorldIR entities carry geometry (meshes/points) and semantics; there
is no collision representation, no navigation model, no
re-localization interface. The pipeline's camera poses are the only
"agent" trajectories.

## Scope when implemented

1. **Collision geometry.** Simplified convex hulls / voxel collision
   grids derived from mesh artifacts, with LOD and correctness
   documented (agents collide with the hull, not the render mesh).
2. **Free space / navigation.** Occupancy grid from fused geometry +
   camera trajectory evidence; floors/walls from existing plane
   entities become navigation constraints.
3. **Re-localization.** Robot camera/depth stream → pose in the
   world frame (registration P1 methods re-used; landmark-based
   preferred).
4. **Task semantics.** Entities already carry identity/material;
   affordance layer (openable, climbable) is a small schema extension.

## Downstream coupling

- Physics (P7) shares collision representations.
- Uncertainty (P3): navigation margins use geometry uncertainty —
  a robot plans against confidence, not raw surfaces.

## Non-goals (v1)

- Running on-robot in real time (Studio/offline compilation first).
- ROS ecosystem integration until a consumer exists (avoid premature
  interface design; revisit with the first real consumer).

## Failure modes

- Mesh holes → over-conservative collision; document the tradeoff,
  keep the evidence (holes are visible in Studio).
- Re-localization failure → honest UNLOCALIZED state; robot does not
  silently operate in a guessed frame.

## Acceptance criteria

- Deterministic: room world → occupancy grid matches known free
  space in fixture; hulls contain mesh within tolerance.
- Re-localization fixture: synthetic query views → correct pose,
  report error.

## Priority

P5 (after WorldStore; depends on physics-adjacent geometry work).
