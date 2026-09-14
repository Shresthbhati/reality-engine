# Provenance Graph

Status: FOUNDATION (artifacts + per-artifact provenance records
exist) → target IMPLEMENTED as a queryable graph (2026-09-14)

## Purpose

Every fact in the world must be traceable to its origins: which
source files, which processing stages, which models, which parameters,
which other artifacts. Provenance is the backbone that uncertainty
propagation (P3), incremental compilation (P4), and honest status all
stand on.

## Current state

- ArtifactStore persists content-addressed artifacts (sha256) with
  metadata (producer, params) — implemented (PRs #14/#16).
- WorldIR records `data_uri`/`data_hash` for geometry/measurements.
- Compile reports record per-stage inputs/outputs/diagnostics.
- What does **not** exist: a linked graph (artifact→artifact edges),
  a query interface (`reality provenance <artifact_id>`), or
  cross-session lineage.

## Target model

- **Nodes:** artifacts (by hash), sources (by content hash), stages
  (producer+version+params), models (registry entry: name/version/
  checkpoint/checksum), captures.
- **Edges:** derived_from (artifact→artifact), produced_by
  (artifact→stage), consumed (stage→artifact), observes
  (entity→source evidence).
- **Invariants:** immutable nodes; edges only added; deleting a
  source tombstones, never rewrites history (CLAUDE.md §43).

## Interface

- `reality provenance <artifact-id>` — ancestors/descendants walk.
- `reality diff <world-a> <world-b>` — uses lineage to explain what
  changed and why (which new inputs produced which new artifacts).
- Studio inspection (P6) reads the same graph.

## Acceptance criteria

- Deterministic: compile a small world → the full ancestor chain of
  the mesh artifact walks back to source images with hashes at every
  hop.
- Tamper test: modify an artifact → lineage check fails loudly.

## Priority

P3 (graph edges and query on top of the existing per-artifact
records; needed before WorldStore incremental logic).
