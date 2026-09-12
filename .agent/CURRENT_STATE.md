# Reality Engine — Current State

**Updated:** 2026-09-12 (geometric-reasoning session complete)
**Branch:** `studio-command-pipeline`
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

## Next tasks (dependency-safe, in order)

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
