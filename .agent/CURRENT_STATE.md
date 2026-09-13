# Reality Engine — Current State

**Updated:** 2026-09-13 (media-preprocessing session complete)
**Branch:** `evidence-fusion`
**Verified baseline:** 725/725 tests passing (`python -m pytest -q --ignore=tests/test_midas_backend.py`, observed 13.8s). The excluded file is another agent's in-flight work (MiDaS depth backend), not part of this session's changes.

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

- Wire `availability_probe` into the COLMAP backend (shutil.which probe +
  qt-offscreen env note) so the orchestrator's detection is real for the
  one production backend; then route StudioSession.compile_reconstruction
  through the orchestrator when a backend chain is configured.
- Blender export path (WorldIR -> .py/.json add-on input; Reality Engine
  -> WorldIR -> Blender adapter, Blender as consumer) -- now the
  highest-value missing export target.

- Blender export path (WorldIR -> .py/.json add-on input; Reality Engine
  -> WorldIR -> Blender adapter, Blender as consumer) -- now the
  highest-value missing export target.
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
