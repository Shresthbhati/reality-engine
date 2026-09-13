# Reality Engine — Current State

**Updated:** 2026-09-13 (WorldDiff + production audit session complete)
**Branch:** `claude/reality-engine-next-8ffa19` (worktree off `main`, which already has PR #4 evidence-fusion merged)
**Verified baseline before this session:** 890 passed, 1 skipped.
**Verified after this session:** 941 passed, 1 skipped (observed 54.3s) — +3 SDK external-consumer tests on top of the prior 938, zero regressions.

## Completed this session (2026-09-13, latest): public SDK (sdk/reality.py)

Platformization campaign, Phases 1/2/80. Before this, every capability
(world compilation, validation, diffing, physics compilation, export)
was reachable only by importing internal modules directly -- no stable
surface an external caller could depend on. `sdk/reality.py`: 5 thin
pass-through functions (`compile_world_from_reconstruction`, `validate`,
`diff`, `compile_physics`, `export`) calling the real engine underneath
-- no separate fake runtime, nothing reimplemented. `export()` dispatches
by format name to the three real exporters and raises a typed
`UnsupportedExportFormatError` for anything else (never a silent no-op).

`tests/test_sdk_external_consumer.py` (Phase 80 "External Consumer
Test"): a tiny test app that imports `sdk.reality` ONLY -- never
`evidence.*`/`engine.compiler.*`/`exporters.*` directly -- and runs the
full ingest → compile → validate → physics-compile → export (all 3
formats) → diff loop through the SDK surface alone, proving the public
boundary is real. 3 tests total (full flow, typed error on unknown
export format, typed error on empty evidence).

## Completed this session (2026-09-13, latest): structured export reports

`exporters/report.py` (new) — `ExportReport` (format, world_id/version,
entities_exported, entities_skipped + matching skip_reasons,
deterministic sha256 `content_hash`) — studio campaign Phase 7's
explicit ask, previously entirely missing: all three exporters silently
skipped unexportable entities with no visibility into which or why, and
had no way to verify two export runs agree byte-for-byte. Wired
additively into all three exporters — `export_to_gltf_with_report()`,
`export_to_usda_with_report()`, `export_to_blender_script_with_report()`
— alongside their existing, unchanged, still-tested `export_...()`
functions (zero behavior change to anything already tested). 9 tests
(`tests/test_export_reports.py`): correct exported/skipped
classification with per-entity reasons across all three formats,
hash determinism/reproducibility, hash sensitivity to real content
changes, malformed-report rejection (mismatched skip list lengths),
empty-world edge case.

## Completed this session (2026-09-13, latest): compiled bodies now collide via the real backend

`engine/compiler/physics_compiler.build_stepped_physics_world()` — loads
every `compile_physics_world()` result into a real `PhysicsWorld`
(`engine/physics/backend/simple_backend.py`, the existing mature
broadphase/narrowphase/contact-solver stack). Closes the gap flagged at
the end of the prior cycle: compiled bodies were only proven to
integrate solo. `tests/test_physics_compiler_e2e.py::
test_debris_collides_with_reconstructed_floor_via_real_backend` drops a
debris box above a reconstructed room's promoted floor and steps the
real backend 300 times (5s @ 60Hz): it lands and rests on the floor's
real AABB top surface rather than tunneling through — box-vs-box
collision between a dynamic compiled body and a static compiled body,
both derived from actual reconstruction geometry, resolved by the
engine's existing contact solver with zero new physics-engine code.

## Completed this session (2026-09-13, latest): geometric spatial index / query engine

`engine/scene_graph/spatial_index.py` — scale campaign Phases 31-33
(Query Engine, Spatial Indexing, World Search): `SceneGraph` already
answers relationship questions ("what's inside Room 4?") but nothing in
the repo could answer a *geometric* one ("what's near this point",
"what's inside this region", "nearest entity to X") because nothing
indexed entity positions. `SpatialIndex(world)` resolves each entity's
position from `transform.position` or (falling back) its first
geometry's real AABB centroid, and supports `nearest()`, `within_radius()`
(with a `predicate` filter on the resolved Entity), and `within_region()`
(point-in-region plus AABB-overlap for entities with real bounds, so a
wall spanning a region boundary is still found from either side).
Entities with neither a transform nor real geometry bounds are recorded
as `unlocalized` with an explicit reason, never given a fabricated
position. Deliberately a flat O(n) scan, not an octree/R-tree/BVH --
documented as a named tradeoff (Phase 32 itself says "benchmark before
choosing a structure"; nothing in this repo queries at a scale where
linear scan is provably wrong yet). 13 tests
(`tests/test_spatial_index.py`): position resolution from both sources,
unlocalized-entity honesty, nearest-k with deterministic tie-breaking,
predicate filtering, radius inclusivity/negative-radius rejection,
region point vs. AABB-overlap semantics, snapshot-not-live semantics
(matches `SceneGraph`'s contract), empty-world edge cases.

**Also confirmed by inspection (not new work):** the scale campaign's
Phase 1/2 (coordinate frames, transform graph) turned out to be
substantially pre-existing — `world_ir/coordinates.py`'s
`CoordinateRegistry` already does BFS path resolution between
registered `Frame` edges with rigid-transform composition/inversion,
and is wired into `engine/world/runtime.py`. Not re-audited line-by-line
against every campaign sub-bullet (temporal calibration, cycle
detection diagnostics specifically) — flagged here so a future session
doesn't rebuild it from scratch believing it's missing.

## Completed this session (2026-09-13, latest): WorldIR -> Physics compiler bridge

`engine/compiler/physics_compiler.py` — the bridge the simulation
campaign opens by demanding: before this, `engine/physics/` (a large,
real rigid-body/materials/collision stack) had ZERO references to
WorldIR anywhere, and `evidence/`/`engine/compiler/` had zero references
to physics. Every physics golden scene was hand-built `RigidBody`
instances disconnected from any reconstructed world.

`compile_physics_world(world) -> PhysicsCompileDiagnostics`: for every
entity with a geometry that has real `bounds_min`/`bounds_max` (BOX or
PLANE), derives a `RigidBody` with an AABB-sized `Box` shape (1cm floor
per axis), mass=0 (static) for structural/boundary EntityTypes (WALL/
FLOOR/CEILING/ROOF/STRUCTURE/BUILDING/TERRAIN/COLUMN/BEAM/ROAD/CURB/
SIDEWALK/INFRASTRUCTURE — INFERRED, not observed), and
`density x AABB volume` (ESTIMATED) for everything else. Material comes
from `entity.custom_properties["physics_material"]` when it names a
`CANONICAL_MATERIALS` entry, else a low-confidence (`0.1`) default —
explicit diagnostics, not a silent guess. Entities with no geometry or
no real bounds are skipped with an explicit `PhysicsCompileStatus`, not
fabricated. 10 unit tests (`tests/test_physics_compiler.py`) + 3 e2e
tests (`tests/test_physics_compiler_e2e.py`) that run the *real*
pipeline (RANSAC plane detection → orientation → promotion → physics
compile → `engine/physics/rigid/integrator.integrate()`) and prove a
compiled wall/floor stays fixed under gravity while a compiled debris
entity actually falls — not just structurally valid output, a running
simulation.

## Completed this session (2026-09-13, latest): WorldDiff + production audit

`world_ir/diff.py` — `diff_worlds(a, b) -> WorldDiff`: deterministic
structural diff between two WorldIR snapshots (added/removed/modified
entities and geometries, field-level changes for transform/type/name/
provenance/confidence/custom_properties/geometry bounds). Previously
MISSING entirely (repo-wide search found zero prior art). 12 tests in
`tests/test_world_diff.py` (determinism, no-mutation, dict-order
independence, serialization shape). Not yet wired into any caller
(export-fidelity check, branch comparison, regression harness) — that's
the natural next step.

Also produced `docs/REALITY_STUDIO_PRODUCTION_AUDIT.md`: a scoped,
honest re-inspection of the repo against a 33-phase production-readiness
brief. Headline finding: **Reality Studio has no UI** — `apps/studio/`,
`apps/cli/`, `apps/viewer/`, `apps/capture/` are all placeholder READMEs
only; everything working today (`engine/studio/session.py` etc.) is a
headless Python session object, not an application. This is now the
documented largest remaining gap, ahead of export/validation polish.

## Completed this session (2026-09-13): Blender export target

`exporters/blender/exporter.py` — `export_to_blender_script()` produces a
standalone `blender --background --python <file>` script recreating one
cube object per WorldIR Entity with a transform position and a BOX/PLANE
geometry. Unlike the existing gltf/usd exporters (unit-cube-only), this
one sizes the cube to the geometry's real `bounds_min`/`bounds_max` AABB
when set (always true for `evidence/promote_planes.py` output), flooring
each axis at 1cm so a degenerate zero-thickness plane bound still exports
a visible, non-fabricated mesh. Real WorldIR metadata (entity_id/type/
provenance/confidence) is written as Blender custom properties. 8 tests
in `tests/test_blender_exporter.py` (ast.parse syntax validity, real-vs-
default dimensions, degenerate-axis flooring, custom properties, write/
read round-trip, empty-world case). Row R updated in
docs/CAPABILITY_MATRIX.md. Honest ceiling: no Blender binary is installed
in this environment, so the script has never actually been run inside
Blender — that remains the next verification step once Blender is
available.

**Gap found and closed in the same session:** `evidence/promote_planes.py`
never set `Entity.transform`, so all three exporters silently exported
zero objects from a real compiler-produced WorldIR. Fixed:
`promote_plane_to_entity()` now sets `transform.position` to the AABB
centroid (midpoint of the already-computed `bounds_min`/`bounds_max` —
real data, never fabricated). `exporters/gltf/exporter.py` and
`exporters/usd/exporter.py` were also widened from BOX-only to
BOX+PLANE (they previously skipped every promoted plane entity even
with a transform). `tests/test_export_pipeline_e2e.py` (4 tests) proves
the closed loop with a *real* pipeline run — reconstruction -> RANSAC
plane detection -> orientation classification -> WorldIR promotion ->
all three exporters — not hand-built WorldIR fixtures: promoted
wall/floor/ceiling entities now produce real nodes/prims/cubes with
correct AABB-derived transforms and (for Blender) real dimensions.
Room entities (`evidence/promote_rooms.py`) still carry no geometry by
design (`geometry_ids=[]` — "the room's extent lives in its parts") and
correctly still export nothing.

## Milestone reached before this session

Foundation (WorldIR/provenance/physics/runtime/destruction/rain/water) +
evidence layer + merge + COLMAP reconstruction (real binary run: 3/4 images
registered, 228 points on a 4-photo test scene) + reconstruction validation
wired into promotion + Studio (outliner/selection/viewport/inspector) +
command pipeline (4 command kinds) + scene graph query engine + ontology
extension (10→24 EntityType values) + perception adapter interfaces
(depth/segmentation/instance-lifting, zero concrete backends) + glTF/USD
exporters (BOX-only) + quality report + geometric adjacency inference wired
into commands + tech registry (research only, nothing installed).

## Highest-value dependency-safe gap (from docs/REALITY_ENGINE_AUDIT.md)

The ontology now has WALL/FLOOR/CEILING vocabulary, but "no code anywhere in
this repo assigns these types automatically -- no plane detection, no
room-boundary inference, no geometric reasoning." Nothing derives
measurements from reconstructed geometry either.

## Completed this session (2026-09-12): first geometric-reasoning slice

`perception/geometry/planes.py` (deterministic RANSAC),
`perception/geometry/orientation.py` (wall/floor/ceiling classification
with camera-side disambiguation and honest UNKNOWN),
`evidence/promote_planes.py` (WorldIR promotion with ESTIMATED extent +
pair-based wall thickness measurements), `GeometryType.PLANE` schema
addition, 32 tests in `tests/test_geometric_reasoning.py`. Row T2 added
to `docs/CAPABILITY_MATRIX.md`; audit updated. Real bugs caught by tests
during this slice: majority-rule plane extraction deadlock, tilted
single-refit planes, same-facing-normal wall-pairing, plane-gap sign,
unnormalized-normal handling. All fixed and regression-tested.

## Completed 2026-09-12 (later session): evidence fusion core

Branch `evidence-fusion` off `main` (PR #3 with the geometric-reasoning
slice was MERGED; MiDaS depth backend also landed). New:
`reconstruction/fusion/fusion.py` — deterministic inverse-variance
fusion of scalar quantities with 5-sigma conflict detection, honest
conflict resolution (weighted mean of ALL sources + CONFLICT provenance
+ preserved conflicting pairs, never a winner-pick), passthrough
provenance, unit-mismatch refusal, `fused_to_measurement()` WorldIR
bridge. 20 tests (`tests/test_evidence_fusion.py`) incl. the spec's
LiDAR-3.17/photogrammetry-3.22 canonical scenario. Full suite: 761
passed, 1 skipped. Row T3 added to docs/CAPABILITY_MATRIX.md; audit got
a dated fusion update. NOTE: fusion is a deterministic core awaiting
callers — nothing in the repo yet holds two independent observations
and fuses them automatically.

## Completed 2026-09-12 (latest): room inference

`evidence/promote_rooms.py` — ROOM entities from closed wall∩floor
boundary rings (walls resting on the floor via signed lowest-point
test, shared top height, corner-support-validated ring of wall-plane
intersection lines oriented by the floor's inlier centroid and walked
by DIRECTED angle), CONTAINS/PART_OF edges, ESTIMATED area/extents/
height measurements (height = exact floor-to-ceiling plane distance
when the ceiling is observed). 18 tests; suite 798 passed / 2 skipped.
Bugs caught by tests: piercing-wall contact (min-|dist|), folded-angle
ring ordering (parallel walls adjacent), wall-top height undercount
(ceiling-plane preference now). Room queries flow through
`SceneGraph.contents_of()` end-to-end.

## Completed 2026-09-12 (latest): world compiler + validation gate

`engine/compiler/world_compiler.py` + `world_ir/validation.py`: one
deterministic call turns a ReconstructionResult into a validated
WorldIR (planes -> classification -> promotion -> rooms -> gate), with
CompileDiagnostics accounting for every input and a gate that refuses
corrupted state. 26 tests; suite 824 passed / 2 skipped. Real gap
closed: per-instance uuid4 main_branch_id broke byte-identical replay
-- compiler supplies stable identity now.

## Completed 2026-09-12 (latest): compiler -> pipeline -> Studio

`CompileWorldCommand` wires the world compiler through the command
pipeline (typed command -> validation -> permission -> STAGED compile ->
gate -> merge-on-success -> WorldCompiledEvent -> version bump).
Transaction semantics: gate failure leaves the session world untouched;
recompile is idempotent and overwrites corrupted state with honest
state. `StudioSession.compile_reconstruction()` is the user-visible
action; `processor.last_compile_diagnostics` exposes coverage. 8 tests;
suite 832 passed / 2 skipped.

## Completed 2026-09-13: evidence packages (deep-implementation sec-2 FIRST priority)

`evidence/packages.py` — the structured ingestion layer above the
append-only Session: `EvidenceSource`/`EvidenceAsset`/`EvidencePackage`/
`EvidenceReference`/`ObservationSet`/`DeterministicPackageBuilder`.
Content-derived deterministic ids (`ev-{seed}-{index}-{sha256[:16]}`;
content-derived package id; rebuild reproduces the package byte-for-byte
— tested), real corruption gate (min size + magic-byte signatures for
JPEG/PNG/TIFF/EXIV/TS/LAS/E57/PLY/PCD; a GIF named .jpg refuses at build
time), content-hash duplicate detection (`DuplicateEvidenceError` names
the existing asset id), frozen-asset processing history (reuse of
`session.ProcessingRecord` — one vocabulary). Two structured callers:
`ObservationSet` -> `fuse_quantity()` (LiDAR 3.17/photogrammetry 3.22
CONFLICT scenario tested with the evidence chain resolvable to real
assets) and `to_evidence_items()` -> reconstruction backends ->
compiler points' `source_evidence_ids` (end-to-end tested). 27 tests;
full suite **859 passed / 2 skipped** (832 baseline + 27, no
regressions). Row T6 added to docs/CAPABILITY_MATRIX.md; audit dated
update added. Labelled NOT built: disk/camera file importer, EXIF/GPS
decoding, payload blob storage.

## Completed 2026-09-13 (latest): media preprocessing (deep-implementation sec-3)

`evidence/importers.py` + `evidence/frames.py` — the real capture-to-
package path (package payloads are no longer caller-supplied bytes only):
disk import of photo/LAS/video folders feeding `DeterministicPackage
Builder`, EXIF/GPS/timestamp decode (DateTimeOriginal->acquired_at, GPS
DMS->signed decimal, exposure/ISO metadata), resolution from container
headers, measured quality (Laplacian-variance blur, luma mean, clipped
fraction — honestly unmeasured when decode fails), two-layer duplicate
detection (content-hash exact: skip-and-record; 64-bit dhash near-dup:
marked in report), and extensible deterministic video frame selection
(`IFrameSelectionStrategy` + `UniformTimeSamplingStrategy`, first+last
always included, hand-computed indices tested; selected frames become
DERIVED assets linked via `source_video_asset_id`). Zero-install honesty:
PIL/cv2/numpy probed at import (`ImporterCapabilityError` names what to
install; pyproject stays dependency-free). Deterministic ids/timestamps:
sorted path order, EXIF-only acquisition times, never wall clock;
byte-identical rebuild tested. End-to-end tested: folder -> package ->
`to_evidence_items()` -> reconstruction -> compiled WorldIR with point
`source_evidence_ids` resolvable to ingested assets; folder ->
`ObservationSet` -> `fuse_quantity()`. 35 tests; full suite **894 passed
/ 2 skipped** (859 baseline + 35, no regressions). Rows A/T7 in
docs/CAPABILITY_MATRIX.md; registry rows for Pillow/OpenCV/numpy added.
Labelled NOT built: RAW/HEIC decode, package->Session bridge, quality
GATE (signals only), camera-facing capture UI.

## Next tasks (dependency-safe, in order)

- Add a minimal CLI (`apps/cli/`) that is itself a client of `sdk.reality`
  — natural next step now that a real public surface exists to build a
  command line around, and still the biggest named gap from the studio
  campaign (Phase 17: headless workflow).
- Add `sdk.reality.spatial_index(world)` wrapping
  `engine.scene_graph.spatial_index.SpatialIndex` and `scene_graph(world)`
  wrapping `engine.scene_graph.graph.SceneGraph`, so query capability is
  reachable from the SDK too, not just compile/validate/diff/export.
- Wire `SpatialIndex` into a real caller — e.g. `evidence/promote_rooms.py`'s
  room-detection already does its own ad-hoc geometric neighbor logic;
  a coverage-analysis or nearest-wall-to-point Studio tool would be the
  first real consumer of this index rather than it sitting unused.
- Wall-vs-plane / wall-vs-wall collision uses Box-vs-Box only right now;
  `SimpleRigidBodyBackend` also supports Plane statics via `add_plane()`
  which `build_stepped_physics_world()` does not populate (compiled
  entities always become Box bodies, never Planes) -- fine for now since
  Box-vs-Box already resolves correctly, but worth noting if a future
  caller wants true infinite-plane floors instead of thick AABB slabs.
- Feed `compile_physics_world()` output into an event-timeline/replay
  test (`engine/physics/replay/`) -- the compiler produces bodies and
  the backend now steps/collides them; nothing yet records that as a
  replayable event sequence tied back to WorldIR provenance.
- Wire `world_ir.diff.diff_worlds()` into a real caller: an export
  fidelity check (compile → export → readback → diff against source, if
  a readback path existed) or a golden-scene regression test (compile
  twice / compile-then-recompile and assert an empty diff).
- Start a minimal CLI (`apps/cli/`) exposing at least `compile`,
  `validate`, and `export` as real subcommands over the existing library
  functions — the single highest-leverage step toward making Reality
  Engine usable without writing a Python script per session, and a
  prerequisite for any headless benchmark/CI workflow.
- Run the Blender exporter's generated script inside an actual Blender
  install once one is available in this environment, and visually
  inspect the result (geometry/transforms/hierarchy/custom properties) —
  the exporter itself is done and tested, but never opened in Blender.
- Real mesh/point-cloud geometry storage in WorldIR (`Geometry` currently
  stores only `vertex_count`, no actual vertex buffer) so exporters can
  emit real wall/floor rectangles or point clouds instead of a bounding
  box standing in for the shape.
- Compiler consumption of depth/segmentation/material evidence (currently
  planes+rooms only).
- Non-convex (L-shaped) room rings; DOOR/WINDOW/ROOF assignment; multi-room
  shared-wall ownership (room topology is now the foundation).
- Wire plane+room promotion into a Studio action so a user-visible flow exists.
- Evidence fusion across competing plane fits (multiple reconstructions) —
  the fusion core now exists (`reconstruction/fusion/fusion.py`); what
  remains is a caller that detects competing plane fits and feeds them in.
- Compiler consumption of depth/segmentation/material evidence (currently
  planes+rooms only).
- Non-convex (L-shaped) room rings; DOOR/WINDOW/ROOF assignment; multi-room
  shared-wall ownership (room topology is now the foundation).
- Wire plane+room promotion into a Studio action so a user-visible flow exists.
- Evidence fusion across competing plane fits (multiple reconstructions) —
  the fusion core now exists (`reconstruction/fusion/fusion.py`); what
  remains is a caller that detects competing plane fits and feeds them in.
- Wire plane promotion into a Studio action so a user-visible flow exists
  (detection currently runs as library calls).
- One real depth/segmentation backend per docs/TECHNOLOGY_REGISTRY.md
  (license check first) — NOTE: another agent's MiDaS backend work was
  observed in-flight in perception/depth/ during this session; coordinate
  before starting another depth backend.
- Real mesh/point-cloud geometry storage in WorldIR so exporters can emit
  something beyond BOX.

## Decisions affecting this work

- `WorldIR.created_at`/`modified_at` default to 0.0, never wall-clock.
- Reconstruction provenance is RECONSTRUCTED; geometric inference from
  those points is INFERRED, never silently OBSERVED.
- Empty/failed reconstruction raises; nothing fabricates.
- COLMAP Windows needs QT_QPA_PLATFORM=offscreen + use_gpu=0 (handled in
  colmap_backend).
