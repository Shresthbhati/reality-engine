# Advanced Physics

Status: FOUNDATION (rigid bodies, collision, friction, contact
solver, destruction/fire/wind/rain/water foundations exist) (2026-09-14)

## Purpose

Simulation fidelity: from the existing rigid-body/contact foundation
to the mechanics a real environment simulator needs — constrained
bodies, structural response, fluids, thermal effects — consuming
WorldIR geometry/materials with propagated uncertainty.

## Current state (foundation inventory)

Implemented/partial per the audit: rigid-body dynamics, collision,
friction, contact solver, destruction foundations, fire foundations,
wind, rain, water.

Missing/partial below.

## Workstreams

### CCD (continuous collision)
- Model: conservative advancement + swept-convex tests between steps.
- Invariant: no tunneling through static geometry at max velocity.
- Test: bullet-through-thin-wall fixture at increasing dt → no
  pass-through for any dt in range.

### Constraints / joints
- Model: sequential impulses first; XPBD for stability where
  stiffness varies.
- Types v1: distance, hinge, slider, fixed. Typed API, no hidden
  global solver state.
- Test: pendulum energy drift bounded; chain under gravity stable.

### Structural mechanics
- Model: reduced-order (position-based damage proxy) before FEM;
  FEM only with a real consumer demand.
- Couples to destruction foundations: damage → fracture thresholds.

### Fluids
- Model: SPH for interactive-scale water; FLIP/PIC if fidelity
  demands and hardware allows.
- Invariant: volume conservation tracked and reported as diagnostic.
- Test: dam-break fixture, measured volume drift < tolerance.

### Thermal coupling
- Model: heat equation on entity material properties; fire model
  consumes ignition/thermal state (couples to existing fire
  foundations).

### Fire
- Model: reaction + thermal + airflow coupling on existing
  foundations; honest scope — stylized-but-physical, not combustion
  chemistry.

### Physics → WorldIR
- Simulation results are derived artifacts (state, contact events,
  fractures) persisted with provenance and full lineage; never
  overwrite the compiled world.

## Global rules

- Determinism: fixed-step, reproducible runs; seeded randomness
  explicit.
- Materials come from perception (P2) with uncertainty (P3);
  defaults are recorded as class priors, never silent.
- Performance targets measured (§34), not assumed.

## Acceptance criteria per workstream

Each lands with: model doc, deterministic fixture, invariant checks,
diagnostics, and registry status transition evidence.

## Priority

P7 (after material perception P2 and uncertainty P3 make inputs
credible; CCD/constraints first — they gate everything else).
