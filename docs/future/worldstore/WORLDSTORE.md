# WorldStore

Status: MISSING (2026-09-14)

## Purpose

The persistent world database: a durable, versioned store of compiled
WorldIR worlds and their artifacts that supports incremental updates
(new capture → world revision), history (every prior revision
retrievable), and multi-session accumulation (today's room scan joins
yesterday's).

## Current state

Worlds exist as per-run JSON exports; ArtifactStore provides
content-addressed persistence underneath. Nothing links revisions,
nothing merges captures into a shared world, nothing versions the
world itself.

## Model

- **World = append-only sequence of revisions.** Each revision is a
  complete WorldIR snapshot whose artifacts live in the ArtifactStore;
  a revision references its parent and the input artifacts that
  produced it (lineage from the provenance graph, P3).
- **Identity:** world_id stable across revisions; revision_id =
  content hash of the compiled world.
- **Operations:** `open(world_id)`, `commit(new_inputs)` → new
  revision, `checkout(revision)`, `diff(rev_a, rev_b)`, `gc()`.

## Incremental compilation

A commit recompiles only what new inputs can change (new images →
SfM extension; new depth → fused cloud update; identity changes →
entity updates) — see `INCREMENTAL_COMPILATION.md`. v1 may honestly
recompile everything and record that it did; incrementality is an
optimization with a correctness invariant (identical output to full
recompile given identical inputs).

## Multi-session merge

Two sessions in the same space → cross-source registration (P1)
produces the transform, identity association (P2) decides which
entities are the same, then revisions merge. Merges are new
revisions, never in-place edits.

## Failure modes

- Corrupt revision → detected by hash chain; earlier revision served.
- Merge conflicts (same entity, contradictory measurements) → both
  kept with provenance + confidence; Studio surfaces the conflict
  (never silent overwrite).

## Acceptance criteria

- Deterministic: two sequential commits → two revisions, parent links
  intact, checkout of rev1 reproduces rev1 exactly.
- Crash during commit → store opens at last complete revision.

## Priority

P4 (depends on provenance graph; Studio depends on it).
