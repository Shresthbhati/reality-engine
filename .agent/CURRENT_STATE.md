# Reality Engine — Current State

**Updated:** 2026-09-13 (real geometry storage session complete)
**Branch:** `claude/reality-engine-audit-impl-25e738` (worktree off `main`, which already has the CLI + SDK-query PR merged)
**Verified baseline before this session:** 1084 passed, 1 pre-existing environment failure deselected.
**Verified after this session:** 1106 passed, 1 pre-existing failure deselected (observed 79.8s) — +22 tests, zero regressions.

**Known pre-existing environment failure (not caused by this session, not fixed):**
`tests/test_sam_backend.py::TestSAMSegmentationBackendIntegration::test_real_model_load_and_inference`
fails with `FileNotFoundError` for `hubconf.py` under `~/.cache/torch/hub/facebookresearch_segment-anything_main/`
— a corrupted/partial local torch.hub cache on this machine, not a code defect. Deselect it or clear that
cache directory to get a clean run; do not "fix" it in source.

## Completed this session (2026-09-13, latest): real geometry storage in WorldIR (P0.10/P0.11) — closes the last "no fake completion" gap in the export path

Closed the item this file has flagged since it was first written:
"`Geometry` currently stores only `vertex_count`, no actual vertex
buffer... a bounding box standing in for the shape."

New: `world_ir/geometry_data.py` (`PointCloudData` — a deterministic
binary payload format, magic + count + float64 xyz triples, order-
preserving) and `world_ir/artifact_store.py` (`ArtifactStore` ABC +
`MemoryArtifactStore` + `FileArtifactStore` — content-addressed,
sha256-keyed, dedupes identical bytes, `FileArtifactStore` sharded like
Git's object store and verified to survive a new store instance over
the same root). `Geometry.data_uri`/`data_hash` (schema_v1.py) already
existed for exactly this and were never written to until now.

Wired end-to-end, not left as an unused interface:
`evidence/promote_planes.py::promote_plane_to_entity` gained an optional
`artifact_store` param — when given, it stores the plane's REAL inlier
point positions (already computed for the AABB, previously discarded
after that) as a `PointCloudData` artifact and sets `data_uri`/`data_hash`
for real. `engine/compiler/world_compiler.py::CompileOptions` gained a
matching `artifact_store` field, threaded through. `exporters/gltf/exporter.py`
gained the consumer side: given the same store, `export_to_gltf()` now
builds a REAL per-entity POINTS-mode mesh from the stored positions
instead of the placeholder unit cube, for any entity whose geometry
resolves through the store — entities with no resolvable real data
still get the cube, honestly, never fabricated. `sdk.reality.export()`
threads `artifact_store` through for the gltf format specifically (usda/
blender don't consume real geometry yet — named, not hidden).

Fully additive: every new parameter defaults to `None`/omitted and
reproduces the exact prior behavior (`vertex_count`+bounds only, cube
mesh) — verified by an explicit regression test
(`test_omitting_artifact_store_reproduces_old_behavior`). Determinism
preserved through the new layer too (`test_two_compiles_of_the_same_evidence_produce_identical_artifacts`
— same seed -> byte-identical artifact hashes).

22 new tests in `tests/test_geometry_artifacts.py`: PointCloudData
round-trip (incl. empty, duplicate points, bad-magic rejection), both
ArtifactStore backends (put/get, content-addressed dedup, persistence
across a fresh FileArtifactStore instance, unknown-uri error), the full
promote_planes -> WorldIR wiring (real data matches vertex_count,
determinism), and the gltf export path (real mesh vs. cube fallback,
SDK-level threading). Full suite: 1106 passed (was 1084), zero
regressions.

Not done in this pass (named, not hidden): triangulated MESH storage
(only POINTCLOUD payloads exist — a real surface-reconstruction step is
a separate, larger follow-on); usda/blender exporters do not yet
consume real geometry (gltf only); `evidence/promote_objects.py` (object
entities) does not yet write real geometry artifacts, only
`promote_planes.py` does.

## Completed 2026-09-13: `sdk.reality.spatial_index()` / `scene_graph()` + `reality query` CLI

Added the two SDK query wrappers named as the next task after the CLI
landed: `sdk/reality.py` now exposes `spatial_index(world)` (wraps
`engine.scene_graph.spatial_index.SpatialIndex` — nearest/within_radius/
within_region) and `scene_graph(world)` (wraps
`engine.scene_graph.graph.SceneGraph` — edges_from/to, contents_of,
container_of), both direct pass-throughs, no new logic. Wired into a
real caller: `apps/cli/main.py` gained `reality query nearest <world>
<x> <y> <z> [--k N]` and `reality query contents <world> <entity-id>`,
both exercised by 3 new CLI tests. `tests/test_sdk_external_consumer.py`
(the file that proves the SDK boundary is real, importing nothing but
`sdk.reality`) gained a dedicated test using the two-room compiled-world
fixture, proving `nearest()` returns correctly ordered results and
`scene_graph().contents_of()`/`container_of()` resolve a real
CONTAINS/PART_OF edge produced by room promotion — not a stub. The
existing full-flow SDK test was also extended to touch both new
functions. Full suite: 1084 passed (was 1080), zero regressions.

Considered and explicitly NOT done: wiring `SpatialIndex` into
`evidence/promote_rooms.py`'s ad-hoc neighbor logic (the other option
named in the prior next-tasks list) — that logic runs on planes
*before* they exist as WorldIR entities, so `SpatialIndex` (which
requires a `WorldIR`) does not apply at that layer without a larger,
riskier restructuring. The CLI is a real, lower-risk caller that
exercises the same code paths honestly.

## Completed 2026-09-13: headless CLI (`apps/cli/`) — closes the biggest named studio-campaign gap

`apps/cli/main.py` — a `reality` command-line client of `sdk.reality`
(`ingest`, `reconstruct`, `validate`, `diff`, `export`, `physics`). Every
subcommand is a thin pass-through: `ingest` calls
`evidence.importers.import_folder` + `DeterministicPackageBuilder`;
`reconstruct` runs the real `ReconstructionOrchestrator` (COLMAP backend
first, fake-with-no-canned-data second) then `sdk.reality.compile_world_from_reconstruction`
in one step (`ReconstructionResult` has no stable serialization of its
own by design, so ingest/reconstruct are necessarily separate CLI verbs
sharing one process for the reconstruct step); `validate`/`diff`/`export`/
`physics` load/save WorldIR as plain `to_dict()`/`from_dict()` JSON and
call the matching `sdk.reality` function directly. Honesty preserved:
`reconstruct` exits non-zero with the orchestrator's full backend attempt
log on stderr (never fabricates a world) when no backend can honestly
produce geometry — verified live with two real flat-color JPEGs and no
COLMAP binary, which correctly refuses rather than inventing points.
`tests/test_cli.py` (12 tests) exercises every subcommand against real
files on disk: ingest determinism, validate against both a real compiled
room-scene world and a hand-built broken one, diff (identical + a real
removed-entity case), export to all three real formats (byte content
verified, not just exit code), physics compilation, and the honest-
refusal path for reconstruct. `python -m apps.cli.main --help` verified
live. `apps/cli/README.md` updated from "Not yet implemented" to the
real usage doc.

Not built in this pass (named, not hidden): no `console_scripts` entry
point in `pyproject.toml` (repo has no packaging setup for `apps.*` yet
— `[tool.setuptools.packages.find]` only includes `engine*`/`world_ir*`/
`provenance*`/`events*`); no `--seed`-stable ingest source id override
beyond folder name; no batch/watch mode. All were out of scope for "the
biggest named gap" and none block current use via `python -m apps.cli.main`.

## Completed 2026-09-13: object promotion into WorldIR — closes the pipeline

`evidence/promote_objects.py` — `promote_object_to_entity()`: writes a
`MergedObjectCandidate` (from `perception/instances/object_resolution.py`)
into a real WorldIR `Entity` + `Geometry`, mirroring
`evidence/promote_planes.py`'s established pattern. This was the
explicitly-named "next highest-value task" from the prior session: the
object-perception pipeline (lift -> merge -> measure) produced real,
tested, composable data but never reached WorldIR. Entities get
`EntityType.UNKNOWN` (the ontology has no furniture/object categories --
honest rather than guessed) with the real label preserved in
`semantic_labels`/`name`; measurements land on `custom_properties`; a
full provenance trail (evidence ids, region ids, observation count)
lives in `Observation` metadata. Refuses a zero-hypothesis candidate.

`tests/test_object_pipeline_e2e.py` (2 tests) proves the FULL chain
works together: two `PinholeCamera`s at different positions, two real
`DepthMap`+`SegmentedRegion` pairs of the same chair -> real
`lift_region_to_3d()` -> real `merge_hypotheses()` -> real
`promote_object_to_entity()` -> exactly one validated WorldIR entity
(`validate_world_ir().is_valid()` checked, not assumed). A second test
confirms two distinct objects correctly produce two separate entities.
Plus 8 unit tests (`tests/test_promote_objects.py`).

`docs/IMPLEMENTATION_DELTA.md` (new) records this delta against the
prior verified baseline. The object-understanding vertical slice is now
code-complete and self-consistent end-to-end; the only remaining gap is
running it against a real detector/segmenter and real photos (both
still blocked on missing model/checkpoint installs and a committed
dataset, unchanged from prior sessions).

## Completed this session (2026-09-13, latest): object measurement (master directive priority #11)

`perception/instances/measurement.py` — `measure_dimensions(candidate)`
(width/height/depth/volume from a `MergedObjectCandidate`'s union AABB)
and `measure_distance(a, b)` (centroid-to-centroid). Reuses the existing
`world_ir.schema_v1.Measurement` type rather than inventing a parallel
one -- the same currency `evidence/promote_planes.py`'s extent/thickness
measurements already use. Precision uses an explicitly-named
approximation (`value * (1 - confidence)`, floored at 1cm) since nothing
upstream yet propagates real per-axis geometric uncertainty -- honest
about being a heuristic, not a calibrated statistical model, per this
campaign's own "documented approximation" allowance. All measurements
are `Provenance.ESTIMATED`. 8 tests
(`tests/test_object_measurement.py`): extent/volume correctness,
provenance/confidence propagation, precision-vs-confidence monotonicity,
precision floor, Euclidean distance correctness and symmetry, weaker-
endpoint confidence rule.

Pipeline chain now real end-to-end at the code level (still untested
against real photos, per prior sessions' named blocker): `DepthMap` +
`SegmentedRegion` -> `lift_region_to_3d()` -> `ObjectHypothesis3D` ->
`merge_hypotheses()` -> `MergedObjectCandidate` -> `measure_dimensions()`
/ `measure_distance()` -> `Measurement`.

## Completed this session (2026-09-13, latest): multi-view object entity resolution (semantic perception campaign, Phase 5)

`perception/instances/object_resolution.py` — `merge_hypotheses(hypotheses,
distance_threshold_m)`: clusters `ObjectHypothesis3D` observations (from
`perception/instances/lifting.py`) of the same physical object across
frames into one `MergedObjectCandidate`, using union-find over same-
label + within-distance pairs so transitive chains of overlapping views
merge correctly regardless of input order. Never discards a source
hypothesis; merged bounds are the union AABB (never smaller than any
single view); confidence rises with independent agreement
(`1 - (1-best)^n`, capped at 1.0) rather than being averaged down.
Deliberately NOT appearance/embedding-based (explicit "similarity is
not identity" rule, same as `world_ir/entity_reid.py`) and NOT
multi-view-geometrically-verified (no epipolar check) — both named as
real, separate, larger follow-ons. 12 tests
(`tests/test_object_resolution.py`): merge/no-merge by distance and
label, transitive chain merging, bounds union, confidence-boost math,
single-hypothesis passthrough, determinism regardless of input order,
empty-input and invalid-threshold edge cases.

Per the user's stated priority (campaigns 11-32 come after 1-10 finish),
this closes another concrete stage of campaign 3 (semantic perception)
rather than starting the newly-listed campaigns 11+.

## Completed this session (2026-09-13, latest): depth -> point cloud (Phase 8, both convergence and hardening audits' named next step)

`reconstruction/depth_to_points.py` — `depth_map_to_points(depth, camera,
stride=1)`: unprojects every valid pixel of a `DepthMap` through the
real `PinholeCamera` into `ReconstructedPoint`s -- the SAME output type
`ReconstructionResult.points` already uses, so depth-derived points
compose directly with everything downstream (RANSAC plane detection,
world compiler) without a new point-cloud type. Refuses relative/non-
metric depth (DepthToPointsError); skips (never fabricates) non-finite/
non-positive-depth pixels; `stride` is honest pixel-decimation, named
explicitly as NOT voxel-grid downsampling. 9 tests
(`tests/test_depth_to_points.py`), including
`test_output_composes_with_real_ransac_plane_detection` -- runs the
*actual* `perception/geometry/planes.detect_planes()` over depth-derived
points and confirms a real plane is detected with the correct normal,
proving the integration point works end-to-end, not just in isolation.

This closes the "depth -> point cloud" gap flagged as the concrete next
step in both `docs/REAL_CAPTURE_VERTICAL_SLICE_AUDIT.md` and
`docs/RECONSTRUCTION_HARDENING_AUDIT.md`.

## Completed this session (2026-09-13, latest): 2D->3D lifting (semantic perception campaign, Prompt 3)

`perception/instances/lifting.py` — Phase 4. `lift_region_to_3d(region,
depth, camera)`: `SegmentedRegion` (2D mask) + `DepthMap` + the real
`PinholeCamera` (from the reconstruction-hardening session) ->
`ObjectHypothesis3D` (centroid + AABB of unprojected valid-depth mask
pixels). Real deterministic geometry, no ML model needed to run or test
it -- operates on backend output *types*, same pattern as
`evidence/promote_planes.py` not needing COLMAP installed to be tested.
Refuses (LiftingError) a relative/non-metric depth map rather than
silently treating it as meters; returns None (not a fabricated
hypothesis) when too few mask pixels have valid depth. 10 tests
(`tests/test_2d_3d_lifting.py`).

`docs/SEMANTIC_PERCEPTION_AUDIT.md` (new): confirmed by direct import
attempt that SAM cannot actually run in this environment right now
(torch 2.14.0 CPU is installed; `segment_anything` package and any
checkpoint are not) — that is the real, named blocker for Phases 1-2 of
the semantic-perception campaign, not a code gap. Detection
(`perception/detection/`) remains a bare README placeholder.

## Completed this session (2026-09-13, latest): real pinhole camera model (reconstruction hardening campaign, Prompt 2)

`reconstruction/calibration/camera.py` — Prompt 2 Phase 2 exactly.
`reconstruction/calibration/` was a README placeholder; nothing in the
repo could project a 3D point to a pixel or unproject a pixel+depth back
to 3D. `CameraIntrinsics` (fx/fy/cx/cy + Brown-Conrady k1/k2/p1/p2/k3,
validated), `CameraExtrinsics` (camera-to-world position+rotation,
matching `ReconstructedCameraPose`'s existing convention so a COLMAP
pose plugs in directly), `PinholeCamera.project/.unproject/.ray`.
Undistortion uses fixed-point iteration (no closed-form inverse for
Brown-Conrady). Added `Quat.conjugate()` to
`engine/physics/math3.py` (unit-quaternion inverse rotation) rather than
reimplementing it locally. 21 tests
(`tests/test_camera_calibration.py`): intrinsics validation, projection
geometry (behind-camera/at-plane rejection), round-trips with AND
without real distortion coefficients (sub-mm precision), ray casting,
translated/rotated camera sanity checks, dict round-trips.

This is the prerequisite Phase 8 (depth -> point cloud) needs:
`PinholeCamera.unproject(u, v, depth)` is exactly the per-pixel
operation a `DepthMap -> point cloud` converter would call in a loop —
not yet wired into one. `docs/RECONSTRUCTION_HARDENING_AUDIT.md` records
this as the concrete next step.

## Completed this session (2026-09-13, latest): Studio -> orchestrator -> compiler wiring (convergence campaign, Prompt 1 slice)

`StudioSession.reconstruct_and_compile(evidence, orchestrator, compile_options=None)`
(`engine/studio/session.py`) — closes the exact gap this file's own
prior entry flagged: `reconstruction/orchestrator.py`'s real backend
selection (COLMAP availability probing, fallback, attempt-log
diagnostics — landed on `main`, merged into this branch) had no path
into a Studio session. One call now: evidence -> orchestrator picks and
runs a real backend -> `compile_reconstruction()` -> validated WorldIR,
returning both the `ReconstructionRun` (backend used, attempt log) and
the compile's `CommandResult`. Raises `ReconstructionOrchestrationError`
on total backend failure -- verified the world stays empty (0 entities)
afterward, never a partial/corrupted result.

`docs/REAL_CAPTURE_VERTICAL_SLICE_AUDIT.md` (new) — honest phase-by-
phase status against the "convergence campaign" brief (build one real,
working, end-to-end capture pipeline). Headline honest gap: no real
photo fixture is committed to this repo, so the COLMAP path (previously
run once, documented, not reproducible from a committed asset) could
not be re-exercised this session; the 3 new tests use the real
`ReconstructionOrchestrator` class with `FakeReconstructionBackend`
(same orchestrator code path a COLMAP backend would flow through).
Depth/mesh/2D-3D-lifting remain entirely unintegrated into the compiler
-- still the biggest structural gap toward Prompt 1's full Definition
of Done.

## Completed this session (2026-09-13, latest): cross-session entity re-identification

`world_ir/entity_reid.py` — world-memory campaign Phases 3/4. Given two
WorldIR snapshots (`before`/`after`), classifies each `before` entity's
correspondence in `after` as MATCH / POSSIBLE_MATCH / NO_MATCH /
UNRESOLVED using real evidence already in WorldIR: EntityType +
geometric position (reusing `engine/scene_graph/spatial_index.py`'s
position resolution, renamed `_entity_position` -> public
`entity_position` to avoid duplicating it). Deliberately NOT
embedding-based -- no real embedding/vision model exists anywhere in
this repo, and the world-memory campaign explicitly forbids fake
embeddings, so that entire area (Phases 5-15, 20-30, 39, 48-51) stays
documented as blocked rather than faked. 11 tests
(`tests/test_entity_reid.py`): match/possible-match/no-match/unresolved
classification, type exclusivity (same position, different type never
matches), deterministic tie-breaking, determinism, threshold
validation, empty-world and no-mutation edge cases.

`docs/WORLD_MEMORY_LEARNING_AUDIT.md` (new) — honest phase-by-phase
snapshot against the 91-phase world-memory campaign; headline finding
is that the campaign's central ask (semantic embeddings/retrieval)
cannot be honestly built without first doing real model
selection/licensing work this session did not attempt.

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

## Completed 2026-09-13 (latest): reconstruction orchestrator (integration campaign sec 2)

PR #4 (evidence-fusion) was MERGED; this session works on `evidence-fusion`
ahead of main again. New: `reconstruction/orchestrator.py` — the single
layer that selects and executes reconstruction backends:
`validate_evidence` (empty/<2 images/duplicate ids refused before any
backend runs), availability probes vs acceptance gates as DISTINCT checks,
preference-ordered selection with fallback on decline/exception/None
(contract violation, captured)/registration-failed, exhaustion raising
`ReconstructionOrchestrationError` with the full attempt log, frozen
`BackendAttempt`/`ReconstructionRunDiagnostics` (to_dict, deterministic
order; durations are wall-clock telemetry), provenance stamping
(confidence preserved verbatim, note merged with backend=NAME), and
distinct display names for same-class fallback chains (stub / stub#2).
Runtime stays LLM-free — selection is explicit policy. 23 tests incl.
end-to-end: orchestrated run -> world compiler -> validated WorldIR
(gate clean, WALL/FLOOR, 1 room) and orchestrated run -> plane summaries
-> detect_rooms (5.625 m^2 exact, slab honestly NO_CLOSED_RING). Full
suite **917 passed / 2 skipped** (894 baseline + 23, no regressions).
Row T8 added to docs/CAPABILITY_MATRIX.md; audit dated update added.
Follow-on in the same session: COLMAP backend now wires its real gates
for the orchestrator (`availability_probe` = shutil.which on the binary,
shared truth with reconstruct()'s own check; `accepts` = >=2 image floor,
recorded as DECLINE not a failed run). Probe verified live on this
machine: COLMAP IS installed (C:\Users\shres\tools\colmap-extracted).
Machine-independent chain tests pin the gate wiring for both
environments (COLMAP present/absent). Full suite **920 passed /
2 skipped** (+3 gate tests).

## Next tasks (dependency-safe, in order)

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
- Route `StudioSession.compile_reconstruction()` through the backend
  orchestrator now that `availability_probe`/`accepts` are wired for real
  on the COLMAP backend (see the completed-work entry above this
  section) — the orchestrator's gates are real but nothing calls it yet.
- Run the Blender exporter's generated script inside an actual Blender
  install once one is available in this environment, and visually
  inspect the result (geometry/transforms/hierarchy/custom properties) —
  the exporter itself is done and tested, but never opened in Blender.
- Triangulated MESH storage: `world_ir/geometry_data.py` now has real
  `PointCloudData`, but no `MeshData` (vertices/indices/normals) yet —
  needs a real surface-reconstruction step (e.g. alpha-shape/Poisson
  over a plane's inlier points), not just a new payload format.
- Wire `evidence/promote_objects.py` (object entities) through
  `artifact_store` the same way `promote_planes.py` now is, so real
  object point clouds are storable too, not just structure planes.
- usda/blender exporters don't consume `artifact_store`/real geometry
  yet (only gltf does as of this session) — extend
  `exporters/usd/exporter.py` and `exporters/blender/exporter.py` the
  same way if real geometry in those formats is needed.
- Compiler consumption of depth/segmentation/material evidence (currently
  planes+rooms only).
- Non-convex (L-shaped) room rings; DOOR/WINDOW/ROOF assignment; multi-room
  shared-wall ownership (room topology is now the foundation).
- Wire plane+room promotion into a Studio action so a user-visible flow exists.
- Evidence fusion across competing plane fits (multiple reconstructions) —
  the fusion core now exists (`reconstruction/fusion/fusion.py`); what
  remains is a caller that detects competing plane fits and feeds them in.
- One real depth/segmentation backend per docs/TECHNOLOGY_REGISTRY.md
  (license check first) — NOTE: another agent's MiDaS backend work was
  observed in-flight in perception/depth/ during this session; coordinate
  before starting another depth backend.

## Decisions affecting this work

- `WorldIR.created_at`/`modified_at` default to 0.0, never wall-clock.
- Reconstruction provenance is RECONSTRUCTED; geometric inference from
  those points is INFERRED, never silently OBSERVED.
- Empty/failed reconstruction raises; nothing fabricates.
- COLMAP Windows needs QT_QPA_PLATFORM=offscreen + use_gpu=0 (handled in
  colmap_backend).
