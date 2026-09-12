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

## Next tasks (dependency-safe, in order)

- Room inference from connected floor/wall/ceiling plane structure (the
  planes now exist as typed entities; ROOM entities and CONTAINS/PART_OF
  edges from plane topology are the next geometric-reasoning step).
- Evidence fusion across competing plane fits (multiple reconstructions).
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
