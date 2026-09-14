# Uncertainty Propagation

Status: MISSING (2026-09-14)

## Purpose

Every derived quantity in the world (positions, dimensions, poses,
measurements, material properties) carries an explicit, propagating
uncertainty, so downstream consumers (physics, Studio, comparisons)
know what is trustworthy — and the system never presents an estimate
as a fact.

## Current state

Uncertainty is represented ad hoc: scale state carries a basis
(measurement/prior/unknown), COLMAP reports reprojection error,
detections carry scores, planes carry residual stats. There is no
systematic covariance model, no propagation rules, no unified
representation in WorldIR.

## Model

- **Per-quantity representation:** each uncertain scalar/vector gets
  `value + uncertainty + basis` where uncertainty is std, covariance,
  or a bounded interval; basis ∈ {measured, derived, estimated,
  prior, unknown} (CLAUDE.md §44 taxonomy).
- **Propagation rules (standard error propagation):**
  - sum/difference: variances add (covariance if correlated)
  - scale: relative uncertainty preserved through scaling
  - pose composition: Jacobian-based covariance propagation
  - rigid transform of points: rotate covariance blocks
- **Derived quantities record their inputs' uncertainties** — the
  provenance graph (see `PROVENANCE_GRAPH.md`) is the structure this
  rides on: an edge producer→consumer with the transform.

## WorldIR integration

Extend entity/measurements/geometry metadata schema with
`uncertainty` (typed, versioned) — additive change; old worlds
without it read as UNKNOWN everywhere (explicit, not zero).

## Failure modes

- Unknown inputs: propagate UNKNOWN, never assume zero uncertainty
  (that is the silent-lie failure).
- Correlations ignored in v1 → document as approximation; flag where
  it matters (bundle-adjusted poses are correlated).

## Acceptance criteria

- Deterministic tests: known-input covariance through scale/transform
  chains matches analytic results within tolerance.
- UNKNOWN input → UNKNOWN output asserted for every operator.

## Priority

P3 (rides on provenance; gates physics credibility).
