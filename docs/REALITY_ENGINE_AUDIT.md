# Reality Engine — Repository Audit

**Date:** 2026-09-12
**Method:** Direct inspection (`find`, `git log`, `pytest`), not documentation claims.
**Scope:** Compares actual repository state against the full "evidence-grounded
world compiler + perception engine + simulation platform + application"
vision, not just the WorldIR/physics core this repo started with.

## Headline honest status

The repository is a **real, tested, dependency-clean physics/WorldIR/evidence
core** (522 tests passing before this audit's own addition; 534 after adding
the scene graph query engine below) with **zero perception,
reconstruction-at-scale, scene-graph-as-a-module, studio-application, or
export capability beyond scaffolding**. The directories the full vision
calls for exist, but seven of them (`perception/`, `exporters/`, `apps/`,
`benchmarks/`, `gpu/`, `shaders/`, `plugins/`) contain **only a README.md
each** — no code, no tests, nothing importable. Calling any of them
"scaffolded" would be generous; they are directory placeholders.

## What is actually implemented (verified, not claimed)

| Area | Status | Evidence |
|---|---|---|
| WorldIR schema (entities, materials, geometry, relationships, temporal events, causal relations, branches/scenarios stubs) | FUNCTIONAL | `world_ir/schema_v1.py`, `world_ir/world_v1.py`, round-trip tested |
| Provenance model (OBSERVED/RECONSTRUCTED/ESTIMATED/INFERRED/GENERATED/UNKNOWN/CONFLICT) | FUNCTIONAL | `provenance/provenance.py`, used consistently across evidence/reconstruction/command modules |
| Physics (rigid body, collision LOD0/1, contacts, materials, events) | FUNCTIONAL for the fidelity level implemented | `engine/physics/*`, ~130+ tests |
| Destruction/glass/debris/replay/viewport/inspector/physics-debugger (Steps 12-18) | FUNCTIONAL at the level committed | per `docs/BUILD_LEDGER.md` |
| Rain/weather intensity+accumulation (Step 19) | FUNCTIONAL | per `docs/BUILD_LEDGER.md` |
| Evidence layer (Session/Dataset, append-only, merge with honest `not_performed` registration) | FUNCTIONAL/PARTIAL | `evidence/`, this session's prior audits |
| Real COLMAP reconstruction backend | PARTIAL — one real 4-image scene run end-to-end (3/4 registered, 228 points); no automated CI test invokes the real binary | `reconstruction/backend/colmap_backend.py` |
| Reconstruction validation wired into promotion (CONFLICT provenance on disagreement) | PARTIAL | `evidence/promote_reconstruction.py` |
| Studio foundation (Outliner, Selection, StudioSession, provenance panel) | PARTIAL | `engine/studio/` |
| Command pipeline (typed command -> validate -> permission -> apply -> event -> version) | PARTIAL — 3 command kinds only | `engine/commands/` |

## What does NOT exist (verified by direct inspection, not inferred from absence of a README claim)

- **Perception engine**: no camera calibration, feature extraction (beyond COLMAP's internal SIFT), depth estimation, segmentation, tracking, or 2D->3D instance lifting exists as Reality Engine code. `perception/{depth,detection,segmentation,tracking,materials,change_detection}/` are README-only.
- **Scene graph as a first-class module**: `Relationship`/`RelationshipKind` exist on `Entity` (spec-compliant edge data), and `Inspector` exposes a few entity-scoped queries (`get_relationships`, `find_supporting`), but there is no graph-level query engine (path queries, "what's inside room X", reverse-lookup by relationship kind across the whole world). This is the highest-leverage gap that's actually reachable with zero new dependencies — see "Next increment" below.
- **Automatic room/wall/floor reasoning, material inference from imagery, measurement-from-geometry, evidence fusion across competing observations**: none implemented. `evidence/promote.py` only turns a human-entered `MANUAL_MEASUREMENT` into a WorldIR Measurement; nothing infers geometry-derived measurements.
- **Ontology**: `EntityType` enum exists but is a flat 10-value list (`BUILDING, STRUCTURE, VEHICLE, TERRAIN, VEGETATION, WATER, NATURAL_HAZARD, DEBRIS, SENSOR, UNKNOWN`) — far short of the room/wall/floor/door/window/stairs/road granularity the vision calls for.
- **Exporters** (Blender/glTF/USD/Unreal): directory exists, README only, zero code.
- **Reality Studio as an application**: `engine/studio/` is a headless Python data-model layer (viewport math, selection state, command wiring) with no UI, no windowing, no rendering. There is no `apps/studio` code, only its README.
- **Gaussian splatting, depth models, SAM2, VGGT, DUSt3R, Pointcept, or any of the researched perception ecosystem**: not installed, not adapted, no adapter interfaces exist yet (`ICameraReconstructionBackend`-equivalent exists only for COLMAP as `IReconstructionBackend`; no `IDepthBackend`, `ISegmentationBackend`, etc.)
- **Technology registry / licensing tracking**: `docs/RECONSTRUCTION_BACKEND_DECISION.md` documents the one decision made (COLMAP, BSD) but there is no `docs/TECHNOLOGY_REGISTRY.md` and no license audit for any other candidate technology.
- **Benchmark suite, golden end-to-end scene (photos -> WorldIR -> Studio -> export), GPU/hardware detection, disaster coupling beyond rain**: none implemented.

## Architecture gaps that matter for future work

1. **No adapter interface layer for perception.** `IReconstructionBackend` is the only backend interface in the repo; a real perception engine needs the same pattern (`IDepthBackend`, `ISegmentationBackend`, `I3DPerceptionBackend`) *before* any specific model gets adapted in, so licensing/hardware/quality tradeoffs stay isolated. This should be designed the same way `IPhysicsBackend` and `IReconstructionBackend` already were.
2. **EntityType ontology is too flat** for room/wall/floor/door-level reasoning. Extending it is cheap; building the reasoning that would populate those types (geometric plane detection, room boundary inference) is not, and needs real algorithms this repo doesn't have yet.
3. **No scene-graph query layer.** The data (`Relationship`) is there; the query engine (item 22/59 of the full vision: answer "what's inside Room 4" without an LLM) is not. This is buildable today with zero new dependencies — implemented in this same session, see below.

## Licensing status (item 49)

No technology registry exists yet. The only external technology actually
integrated is COLMAP (BSD 3-Clause, confirmed via `docs/RECONSTRUCTION_BACKEND_DECISION.md`
and the installed binary's own license file). Every other candidate named in
the full vision (SAM2, VGGT, DUSt3R, MASt3R, Pointcept, gsplat, Nerfstudio,
Depth Anything, TripoSR) has **not been installed, evaluated, or licensed**
in this repository. Claiming otherwise would be exactly the "fake
completion" rule 41 forbids.

## Recommended real next increments, in dependency-safe order

1. **Scene graph query engine** (this session) — zero new dependencies, uses existing `Relationship` data, directly serves items 22/59.
2. **Perception adapter interfaces** (`IDepthBackend`, `ISegmentationBackend`) with zero concrete implementations yet — mirrors the existing `IPhysicsBackend`/`IReconstructionBackend` pattern, unblocks future model integration without committing to one now.
3. **Technology registry** (`docs/TECHNOLOGY_REGISTRY.md`) — a real research/licensing pass on 2-3 candidate depth/segmentation models before installing anything, per item 10.
4. Only after 2-3: install and adapt one real depth or segmentation backend, benchmarked, license-checked, isolated behind its interface.

Item 1 is implemented: `engine/scene_graph/` — `SceneGraph` answers
`contents_of`/`container_of`/`supporters_of`/`path_exists`/`query_by_kind`
purely by walking existing `Entity.relationships` data, no LLM, no new
dependency. 12 new tests.

Item 2 is implemented: `perception/depth/interface.py` (`IDepthBackend`,
`DepthMap`) and `perception/segmentation/interface.py`
(`ISegmentationBackend`, `SegmentedRegion`, `SegmentationResult`), both
mirroring `IReconstructionBackend`'s exact pattern. **Zero concrete
backends implement either interface** — no depth model, no SAM2, nothing
installed. This is SCAFFOLDED, explicitly distinguished from PARTIAL in
`CAPABILITY_MATRIX.md`. 10 new tests exercise the interface contract via
minimal fakes.

544/544 tests passing overall after both increments. Items 3-4 (technology
registry, then one real depth/segmentation backend installed and adapted)
remain real, substantial, undone work — each is its own multi-session
effort, not a checkbox.

## Repository consolidation (2026-09-12)

The rain (Step 19) and water (Step 20) simulation systems existed only on
two separate, unmerged git branches/worktrees (`step19-rain`,
`step20-water`), siblings that forked from the same point right after
Step 18 and never rejoined the main line where all the evidence/
reconstruction/studio/command-pipeline/perception work in this audit
happened. Both are now merged into this branch: 105 real tests (58 rain +
47 water) that existed but weren't visible from `main`/this branch until
now. Resolved two stale hand-maintained test-count doc conflicts (already
out of sync before the merge) and one real code conflict
(`engine/environment/__init__.py`, combined both modules' exports). No
test logic was touched. All previously-separate `reality-engine-*`
worktree directories have been removed -- one directory now contains
everything.

## Geometric reasoning slice (2026-09-12, follow-on session)

Implemented the audit's "highest-leverage gap actually reachable with
zero new dependencies" (gap 2 in the list above): the first code that
assigns WALL/FLOOR/CEILING automatically and derives measurements from
reconstructed geometry.

- `perception/geometry/planes.py` -- deterministic dependency-free RANSAC
  over a `ReconstructionResult`'s points (seeded via engine/core/rng).
  During development the tests caught two real algorithm bugs: a strict-
  majority noise rule made extracting any second plane impossible (each
  structure is a minority of the cloud -- replaced by an absolute
  min-inlier floor + refit re-check), and a single post-consensus refit
  could lock in a tilted compromise plane under slight contamination
  (replaced by iterative refit-and-recollect plus a final least-squares
  refit of the reported inliers, the standard RANSAC contract). Observed
  on a synthetic room (518 points, 30 noise): all four planes found with
  exact constants (d in {-4, -2.5, 0, 0}), identical structure under
  seeds 7/42/123/9999, honest unassigned-point reporting.
- `perception/geometry/orientation.py` -- camera-side floor/ceiling
  disambiguation, near-vertical wall classification, honest UNKNOWN for
  sloped planes and for `up=None` (refuses to guess). Raises when given
  no camera positions instead of inventing a camera side.
- `evidence/promote_planes.py` -- promotion into WorldIR as typed
  entities (INFERRED) + `GeometryType.PLANE` geometry (RECONSTRUCTED,
  real inlier bounds) + ESTIMATED extent measurement from real inlier
  span (precision from fit RMS) + wall thickness ONLY between paired
  opposite faces within 0.5 m with span overlap (tests caught a
  same-facing-normal pairing bug and a plane-gap sign bug here). The
  synthetic-room smoke run produced exactly floor+ceiling+2 walls with
  byte-identical round-trip serialization.
- `GeometryType.PLANE` added to the schema (additive, documented).
- 32 new tests (`tests/test_geometric_reasoning.py`), hand-computed
  expectations, determinism + noise-fraction + refusal + round-trip
  coverage. Full suite: 725 passing (excluding one other agent's
  in-flight test file, which had a collection error unrelated to this
  work at last observation).

Still undone in geometric reasoning (unchanged): ROOM inference from
connected plane structure, supports/support reasoning, evidence fusion
across competing plane fits, curvature/mesh reasoning, and validation
against real photos at scale (the synthetic room is a fixture, not a
benchmark).

## What this audit does NOT claim

This audit does not claim the full vision in the originating prompt (a
perception engine with COLMAP/VGGT/SAM2/depth-model adapters, automatic
room/wall/material inference, a Blender/USD/glTF/Unreal export system, a
windowed Studio application, golden end-to-end photo->WorldIR->export
scenes, a benchmark suite, or a technology registry) is implemented. None
of it is. That prompt describes a multi-month to multi-year systems-engineering
effort across computer vision, graphics, and application engineering. This
audit's job was to state the real gap honestly and implement one real,
dependency-safe increment against it, not to pretend the gap is closed.
