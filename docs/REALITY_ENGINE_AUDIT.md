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
- **Material inference from imagery**: none implemented. **Wall/floor/ceiling reasoning, measurement-from-geometry, evidence fusion, and room inference**: no longer in this list — see the dated updates below (`perception/geometry/`, `evidence/promote_planes.py` merged via PR #3; `reconstruction/fusion/fusion.py` and `evidence/promote_rooms.py` in this pass). Material inference remains undone.
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

## Dated update — 2026-09-12: evidence fusion core (this pass)

`reconstruction/fusion/fusion.py` implements the fusion core the audit
previously listed under "none implemented": `fuse_quantity()` combines
independent observations of one real quantity (e.g. LiDAR vs
photogrammetry distance, competing reconstructions' estimates) by
inverse-variance confidence weighting; detects conflicts at 5 combined
sigma (difference > 5·sqrt(σₐ²+σᵦ²), i.e. statistically real
disagreement under honest claimed uncertainties — a 5 cm delta with
centimeter-scale uncertainties is a conflict, the same delta with
decimeter-scale uncertainties is consistent); resolves conflicts
HONESTLY — the fused value stays the weighted mean of ALL sources
(never a winner-pick or silent discard), provenance becomes
`Provenance.CONFLICT`, confidence drops to the weakest contributor, and
every conflicting pair is preserved verbatim in the result alongside
the raw contributing observations and the observed pairwise spread
(`agreement_rms`). Passthrough keeps a single source's own provenance
and precision exactly. Unit discipline enforced: mixing units raises
`FusionError` rather than fusing. Bridges to WorldIR via
`fused_to_measurement()` (CONFLICT provenance survives the bridge).
20 tests including the spec's canonical LiDAR-3.17/photogrammetry-3.22
scenario with hand-computed sigma multiples, input-order determinism,
and error paths. Full suite: 761 passed, 1 skipped.

Still undone in fusion, labelled not hidden: outlier rejection / robust
estimation, temporal consistency across epochs, pose-level (non-scalar)
fusion, and wiring `fuse_quantity` into an automatic promotion path
(it is a deterministic core waiting for callers that hold two
independent observations — e.g. competing plane fits, which row T2
defers explicitly).

## Dated corrections — 2026-09-12: stale claims in earlier audit sections

Several claims in the older sections above predate merged work and are
no longer true (kept here because this file is read as ground truth):

- "EntityType is a flat 10-value list" — stale. Now 24 values
  (ROOM/WALL/FLOOR/CEILING/ROOF/DOOR/WINDOW/STAIRS/ROAD/... added in
  the ontology extension; wall/floor/ceiling get ASSIGNED automatically
  by the merged geometric-reasoning slice, row T2).
- "Exporters: directory exists, README only, zero code" — stale.
  `exporters/gltf/exporter.py` and `exporters/usd/exporter.py` are real
  (glTF 2.0 JSON and USD ASCII with structural tests), BOX-geometry
  scope only; no Blender/Unreal target.
- "No scene-graph query engine" — stale. `engine/scene_graph/graph.py`
  (row N) answers contents_of/supporters_of/path_exists queries over
  real Relationship edges; what's still missing is automatic geometric
  derivation of most relationships and room-level containment.
- "No docs/TECHNOLOGY_REGISTRY.md" — stale. It exists (row Q) with
  depth/segmentation candidates; no registered technology beyond COLMAP
  is installed. (MiDaS depth backend landed 2026-09-12 via another
  agent's commit — `perception/depth/` is no longer README-only.)
- "Measurement-from-geometry: nothing infers geometry-derived
  measurements" — stale for planes: extent + conditional wall thickness
  come from `evidence/promote_planes.py` (row T2). Room dimensions,
  area/volume, and object dimensions remain undone.

## Dated update — 2026-09-12: room inference (this pass)

`evidence/promote_rooms.py` closes the next geometric-reasoning loop:
ROOM entities are inferred from plane structure when walls rest on a
floor, share a top height, and their wall-intersection-floor LINES close
a boundary ring of corners the walls' inlier support reaches. Promotion
writes a ROOM Entity (INFERRED) with CONTAINS room→part and PART_OF
part→room edges (INFERRED, derivation metadata) and ESTIMATED floor-
area/extents/height measurements; `SceneGraph.contents_of()` then
answers "what does this room contain?" from WorldIR alone. Honesty
designs worth recording (all test-observed): a wall that PIERCES a
floor plane reads as non-contact (min-of-|distance| wrongly accepted
piercing slabs — real bug, fixed with a signed lowest-point test); the
ring walks DIRECTED angles with the floor's inlier centroid on each
line's left (folded-angle ordering falsely placed parallel walls
adjacent — real bug, fixed); a horizontal slab classifies as a second
floor and is rejected by the interior-support (walkable-floor) test
rather than spawning a duplicate room; height prefers the observed
ceiling plane (wall top rows are legitimately lost to the ceiling
plane — wall-top-only height measured 1.8 vs the true 2.0) and is
labelled a lower bound without one. 18 tests, full suite 798 passed /
2 skipped. Not built, labelled: non-convex (L-shaped) rooms (fail
closed, not silently), DOOR/WINDOW/ROOF assignment, shared-wall
multi-room ownership.

## Dated update — 2026-09-13: evidence packages (ingestion abstraction, this pass)

`evidence/packages.py` implements the structured evidence-system layer
the deep-implementation phase's sec-2 FIRST priority named — the layer
above the append-only Session that answers "what does the engine hold,
where did it come from, what happened to it, and can I trust it":

- `EvidenceSource` (device/platform/operator identity), `EvidenceAsset`
  (frozen, hash-addressed — sha256 is both the id component and the
  dedup key; `acquired_at` is caller-supplied, no wall clocks anywhere,
  the same determinism discipline as WorldIR's `created_at=0.0`),
  `EvidenceReference` (asset → WorldIR-observation binding with role),
  `ObservationSet` (named reference group feeding ONE downstream
  quantity), `EvidencePackage` (append-only store + observation sets),
  and `DeterministicPackageBuilder` (the single hashing/validation/
  dedup path — policy in the builder, passive storage below it).
- Deterministic identity, tested to the byte: asset ids are
  `ev-{seed}-{index}-{sha256[:16]}`, the package id is content-derived
  from the sorted asset hashes, and rebuilding from the same payloads
  reproduces the package identically (`to_dict()` equality).
- Corruption handling is real, not a boolean: minimum payload size plus
  magic-byte container signatures (JPEG/PNG/TIFF/EXIV/MPEG-TS/LAS/
  laz/E57/PLY/PCD). A GIF named `.jpg` raises `CorruptEvidenceError` at
  build time; extensionless sensor logs (GPS/IMU CSV, depth dumps) pass
  the size gate only.
- Duplicate detection by content hash: same bytes under a different
  filename is the same evidence; `DuplicateEvidenceError` (the existing
  session-layer vocabulary) names the existing asset id for
  de-dup-by-reference. On the package itself a duplicate hash returns
  the existing id; policy raises, storage de-duplicates.
- Processing history (`ProcessingRecord`, reused from `session.py` — one
  vocabulary, not two) is the ONLY mutation an asset supports, via
  frozen-replacement: history grows, evidence never changes.
- Two structured callers make this an ingestion entry point, not a
  standalone abstraction: `ObservationSet` feeds `fuse_quantity()`
  (tested with the canonical LiDAR 3.17 m / photogrammetry 3.22 m
  CONFLICT scenario, every claim's evidence chain resolvable to a real
  asset), and `to_evidence_items()` bridges assets into the
  reconstruction backends' `List[EvidenceItem]` contract, from which
  points carry `source_evidence_ids` into the world compiler end-to-end
  (tested).

27 tests; full suite **859 passed, 2 skipped** (baseline 832 + 27, no
regressions). Still undone, labelled: no disk/camera file importer
(payloads are caller-supplied bytes — the `source_uri` gap noted in row
A stands), no EXIF/GPS metadata decoding (fields exist, no decoder),
quality metrics are caller-measured, no payload blob storage (hashes +
`source_uri` only).

## Dated update — 2026-09-13: media preprocessing / disk importers (this pass)

`evidence/importers.py` + `evidence/frames.py` close the ingestion gap
the evidence-package pass explicitly flagged ("no disk/camera file
importer — payloads are caller-supplied bytes"):

- **Folder -> package, deterministically.** `import_folder` walks a
  capture folder in sorted path order and feeds
  `DeterministicPackageBuilder`; `acquired_at` comes only from EXIF
  DateTimeOriginal/DateTime (never file mtimes, never a wall clock), so
  re-importing an unchanged folder reproduces the package byte-for-byte
  (tested).
- **EXIF/GPS decode is real.** Camera make/model, exposure, F-number,
  ISO, DateTimeOriginal->epoch, and GPS lat/lon DMS->signed decimal
  degrees (N/S/E/W hemisphere flips, altitude with below-sea-level
  reference, WGS84-assumption note). Corrupt-EXIF payloads that pass
  container validation degrade to metadata-poor assets — unreadable
  metadata is not unreadable evidence.
- **Quality metrics measured, or honestly unmeasured.** Laplacian-
  variance blur, luma mean, clipped fraction over real decoded pixels;
  a failed decode yields `measured: 0.0` plus a note, never a fabricated
  score. Resolution always comes from container headers.
- **Two-layer duplicate handling.** Exact: content hash, skip-and-record
  (new builder `asset_id_for_payload()` pre-check). Near: 64-bit dhash
  over decoded luma at hamming <= 10, marked with a first-seen
  reference. Pixel decode required for near-dup; honestly skipped where
  unavailable.
- **Video never blindly processed.** The container imports unmodified
  as VIDEO evidence; grab-only enumeration feeds an extensible
  `IFrameSelectionStrategy` (`evidence/frames.py`); the deterministic
  default (`UniformTimeSamplingStrategy`, integer index math, first+last
  always included) picks the subset, and only selected frames are fully
  decoded and re-encoded as DERIVED frame assets linked back to the
  container via `source_video_asset_id`. Frame timestamps stay
  container-relative metadata — never claimed as acquisition times.
- **Zero-install honesty.** PIL/cv2/numpy (already present: Pillow
  12.3.0, OpenCV 5.0.0, numpy 2.5.2 — registry rows added) are probed
  at import, never hard-required; absence raises
  `ImporterCapabilityError` naming what to install. `pyproject.toml`
  still declares no core dependencies.
- **End-to-end chains tested.** Mixed folder -> validated package ->
  `to_evidence_items()` -> reconstruction -> compiled WorldIR with every
  3D point's `source_evidence_ids` resolvable to real ingested assets;
  folder -> `ObservationSet` -> `fuse_quantity()`.

35 tests; full suite **894 passed, 2 skipped** (859 baseline + 35, no
regressions). Still undone, labelled: no RAW/HEIC decode, no
package->Session registration bridge (Sessions and Packages remain two
stores), blur/exposure are quality signals not yet a quality gate,
no camera-facing capture UI.

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
