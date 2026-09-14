# Incremental Compilation

Status: MISSING (2026-09-14)

## Purpose

When a world already exists and new evidence arrives (more photos, a
video pass, a depth session), compile the update without redoing all
work — while guaranteeing the result equals a from-scratch recompile
given the same total inputs.

## Current state

Every `reality compile` run is from scratch. Correct, but O(full
session) each time; the WorldStore (P4) needs incremental commits to
be practical for a lived-in world.

## Model

Compilation is a pure function `f(existing_world, new_inputs,
existing_artifacts) → new_world`. Correctness invariant:

> incremental_result == full_recompile(all_inputs)  (within
> deterministic-compiler tolerance, ideally bit-identical)

The deterministic-compiler rule (same inputs → same hashes) makes
this testable: run both paths on the same fixture, diff hashes.

## Stage-wise incrementality (ordered by value)

1. **Artifact reuse (free).** Unchanged stage inputs → stage outputs
   already exist by content hash; skip recompute. Biggest win for
   artifact-heavy stages (depth, meshing).
2. **SfM extension.** New images against existing model: sequential
   registration into the existing reconstruction; new points only.
   COLMAP supports this (`model_converter` + `point_triangulator`).
3. **Fusion update.** New depth views merge into the fused cloud;
   per-image provenance makes per-view increments natural.
4. **Mesh refresh.** Re-mesh on cloud delta; only when cloud changed
   beyond a threshold (config, recorded).
5. **Identity/semantic update.** New detections associate into
   existing entities (P2 association); entity revisions append.

## Failure modes

- Drift: incremental path diverges from full recompile → the
  equivalence test catches it; on failure, fall back to full
  recompile (correct, slower) and record the incident.
- SfM extension failure for a new image → that image is UNVERIFIED
  evidence; never corrupts the existing model.

## Acceptance criteria

- Deterministic fixture: compile, add 2 images, incremental vs full
  recompile → identical artifact hashes for reused stages, equivalent
  world hashes.
- Reuse accounting: report lists which stages were reused vs
  recomputed (honesty in the report, always).

## Priority

P4 (with WorldStore).
