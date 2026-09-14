# Reality Studio

Status: PARTIAL (viewer app exists; Studio as editing/management
surface is MISSING) (2026-09-14)

## Purpose

The human-facing client of WorldIR/WorldStore: inspect compiled
worlds, review low-confidence evidence (identity merges, GPS gaps,
scale basis), correct them (which is new evidence with
`user_supplied` provenance), and export/deliver.

## Current state

`apps/viewer/` renders compiled worlds: point clouds, mesh artifacts
(PR #16), planes, objects as primitives, layer toggles, basic
framing. No editing, no review queue, no WorldStore integration
(WorldStore itself is P4), no diff view.

## Capabilities (ordered)

1. **Inspect (exists, extend).** Viewer over WorldStore revisions —
   open any revision, diff view between revisions, provenance
   inspection of any entity/artifact (P3 graph UI).
2. **Review queue.** Surfaces low-confidence associations, UNKNOWN
   scale/GPS, failed stages — each as an actionable card. Nothing is
   silently wrong; the queue makes uncertainty operable (P2/P3
   dependency).
3. **Correct.** Merge/split entities, fix measurements, supply
   anchors. Every correction persists as evidence with provenance
   `user_supplied` and yields a new world revision (P4 commit).
4. **Export/deliver.** glTF (exists), USD, Blender (exists as
   exporter) — from any revision.

## Rules

- Studio is a **client** of WorldIR/WorldStore, never a second world
  representation (Decision 018).
- Every mutation is a new revision; no destructive edits.
- Viewer fallbacks must stay honest: AABB/placeholder geometry is
  labeled as fallback in the UI (CLAUDE.md §42), and the mesh layer
  renders the real mesh when the artifact resolves.

## Failure modes

- WorldStore absent → Studio runs read-only against a compiled world
  directory; review queue works, commits disabled (explicit state).
- Large worlds → LOD strategy for meshes/clouds (3D-scene discipline;
  see `docs/future/large-world/LARGE_WORLD.md`).

## Acceptance criteria

- E2E: compile real capture → open in Studio → review queue shows
  the envelope-filter count and scale basis → apply a correction →
  new revision visible in diff view.
- No regression in existing viewer rendering (screenshot + layers).

## Priority

P6 (needs P3 provenance + P4 WorldStore to be more than a viewer).
