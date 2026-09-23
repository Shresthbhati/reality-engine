# HANDOFF

Agent: Freebuff (Reality Engine core)
Branch: agent/freebuff-reconstruction-perception
Checkpoint: CORE_ENGINE_SIX_AREAS (P1-03, P3-02, P7-03, P7-05, P7-06, P14-01)
Status: CHECKPOINT_READY
Date: 2026-09-23

## What was closed this session

### P7-03 Architectural perception breadth -- windows + roofs (DONE)
- perception/architecture/windows.py: WindowFit measures an edge-bounded
  opening INSIDE a wall's own inlier coverage (reuses classify.py's bucketed
  scan discipline); refuses with WindowRefused carrying the measured fact
  when the gap touches the wall boundary (that is door-shaped, not a window)
  or has no interior margin.
- perception/architecture/roofs.py: RoofFit measures the up-facing plane set
  (min_axis_up_dot from the registry governs; wraps the same plane fit
  classify.py uses); refuses sloped-beyond-gate / non-up-facing sets via
  RoofRefused.
- Wired into build_component_observations -> union-find resolution ->
  promote_component_to_entity (EntityType.WINDOW/ROOF), confidence =
  fit confidence, custom_properties = measured facts, evidence ids = the
  fit's own point ids.
- 16 tests in tests/test_window_roof_detectors.py: known answers, 7+
  refusal classes, determinism, observation->promotion integration,
  refused-points-never-become-entities.
- classify.py's own limitation note ("sloped planes are UNKNOWN because no
  roof detector exists") is now resolvable via the new detector.

### P1-03 Depth integration -- EXR adapter (DONE)
- evidence/depth_exr.py + parse_depth_exr: OpenEXR single-channel depth
  decode -> DepthFrame (calibration, provenance, timestamp preserved; dtype
  EXR_HALF/FLOAT honored; invalid/NaN policy explicit). Routed from
  evidence/depth_frames.py by extension alongside PNG16/PFM.
- tests/test_depth_exr.py (7 tests) + full depth cluster re-run green.
- Remaining honest gap: no RGB-D hardware capture adapter was added --
  hardware verification is an external dependency and was NOT simulated.

### P3-02 VIO/localization federation -- honest status
- Docker Desktop was unavailable at session start; brought up, then probed
  the real-backend path: an OpenVINS run needs a ROS docker image (~2 GB
  pull, started from the local tools/open_vins checkout) AND a EuRoC MH_01
  bag (~2.3 GB download) -- the local repo ships only ground-truth
  CSV/TXT, no bags. Build/download did NOT complete inside this session.
- The adapter architecture's UNAVAILABLE path (exact reason + remediation)
  is what shipped states earlier; nothing was faked. EuRoC bag acquisition
  remains the single blocking external dependency for a real run.

### P7-05/P7-06 -- ledger reconciliation (were already done)
- Per-cell GSD budgets (P7-05) landed in PR #86 (47d2a6a); adaptive
  subdivision (P7-06) landed earlier (9/9 green). Stale "open" notes
  removed from .agent/TASKS.yaml; both marked with evidence pointers.

### P14-01 City World Compiler -- streaming compile (DONE)
- engine/compiler/streaming.py: compile_city_world_streaming(features,
  tile_size) yields (tile_key, WorldIR) per non-empty tile (sorted keys)
  then (None, report). Same canonical per-feature rules as
  compile_city_world (partitioning, not a second compiler); a feature
  spanning several tiles compiles into EVERY touched tile with
  spans_multiple_tiles stamped; deterministic fragment sequence
  (world.to_dict equality tested).
- Measured: 200k features / 320-tile grid -> 67.7 MB peak tracked memory,
  14.1 s, all 200000 entities emitted exactly once.
- 6 tests in tests/test_city_streaming_compile.py.

## Verification
- Affected clusters: 126 passed (depth frames/EXR, stair/column/beam/
  window/roof detectors, city compiler + streaming, subdivision, OSM
  features), plus detector/promotion integration in the same run.
- .agent/TASKS.yaml and .agent/CAPABILITIES.yaml parse (yaml.safe_load);
  city_streaming_compile added as registry entry #58.

## Remaining limitations (honest)
- Real VIO execution blocked on EuRoC bag download (external, ~2.3 GB).
- Real-capture verification of window/roof detectors is synthetic-fixture
  only; real captures remain an external dependency (same discipline as
  stairs/columns/beams).
- 3DCityDB export and dedicated door detector remain open (P14-01/P7-03).
- The interrupted desktop-redesign work (uncommitted UI scaffold) is
  preserved on branch agent/freebuff-desktop-workspace, NOT part of this
  engine checkpoint.

## Antigravity consumption
1. git fetch && git switch agent/freebuff-reconstruction-perception
2. python -m pytest tests/test_window_roof_detectors.py
   tests/test_depth_exr.py tests/test_city_streaming_compile.py -q
3. Per-tile compile: see compile_city_world_streaming docstring; consume
   the generator, persist each fragment via WorldStore, read the final
   (None, report) dict for measured counts.

