TASK: World Core convergence pass — fix real-data test skip discipline; close P14-01 OSM-ingestion remainder; unblock/verify the 58-plane real-data integration fixture
AGENT: Claude (World Core)
DATE: 2026-09-21
STATUS: DONE (all items independently verified)

UPDATE (same day, later in session): the 14 tests skip-fixed in the
first item below now RUN FOR REAL instead of skipping. Found the
gitignored fixture artifacts (datasets/room_capture/pipeline_out/,
datasets/real_room_capture/, datasets/real_room_capture_worldir/)
present and intact in the main checkout one directory up from this
worktree (a prior session's own note -- "copied from the main checkout
to unblock" -- was the clue). Verified worldir.json matches exactly
what the tests hardcode (58 entities, 58 geometries, struct-plane-000
present) before copying. `pytest tests/integration/
test_system_runtime_proof.py -q` -> **21 passed, 0 skipped, 0 failed**
(was 7 passed / 14 skipped). This is a LOCAL unblock only (gitignored,
not committed) -- see NEXT RECOMMENDED TASK for what would make this
durable across fresh clones.

COMPLETED:
- Verified WorldIR/WorldStore/provenance/uncertainty/spatial-tiling
  ledger entries (P9-01, P10-01, P10-02, P11-01, P12-01, P13-01,
  P13-02) against actual implementation and test runs, not just the
  ledger text — all confirmed genuinely passing.
- Found and fixed a real discipline gap: 14 tests in
  tests/integration/test_system_runtime_proof.py hard-failed with
  `AssertionError: Missing real pipeline worldir/reconstruction/
  evidence-package at <gitignored path>` on a fresh clone/worktree,
  instead of skipping cleanly like every other real-data-gated test in
  the repo. Root cause: three loader/inline checks used bare
  `assert path.exists()` instead of `pytest.skip(...)`.
- Closed P14-01's named-open OSM-ingestion seam:
  engine/compiler/osm_features.py translates evidence/city_import.py's
  OSM way records into engine/compiler/city_compiler.py's CityFeature
  via a documented tag->kind table (building/highway/natural=water/
  waterway/landuse=forest,wood/power/man_made/amenity->infrastructure).
  Unmapped tags and <3-point ways are skipped and recorded
  (unmapped_kind, too_few_points), never guessed. Height only from the
  way's own height/building:levels tags, never invented.

CHANGED:
- tests/integration/test_system_runtime_proof.py:
  `_load_real_room_reconstruction_result()`,
  `_load_real_structural_world()`, and
  `TestRealDatasetVerticalSlice.test_real_room_evidence_admission_and_compilation`'s
  inline check — assert -> pytest.skip, naming the missing path and
  scripts/generate_room_dataset.py as the way to produce it.
- engine/compiler/osm_features.py (new): OSM tag/record -> CityFeature
  bridge, with OsmCompileReport tracking compiled/unmapped_kind/
  too_few_points/not_osm_way.
- tests/test_osm_city_features.py (new): 16 tests over a synthetic OSM
  doc (tag classification, height derivation, skip/record paths, full
  city_import -> osm_features -> compile_city_world pipeline,
  determinism).
- .agent/TASKS.yaml: P14-01 `basis:` updated with this evidence.
- .agent/EXECUTION_STATE.md: dated entry with full verification detail.

VERIFIED:
- `pytest tests/integration/test_system_runtime_proof.py -q` ->
  7 passed, 14 skipped (was 14 failed, 7 passed before the fix).
- `pytest tests/ -k "worldstore or worldir" -q` -> 181 passed,
  2 skipped (was 1 failed, 181 passed, 1 skipped).
- `pytest tests/test_osm_city_features.py tests/test_city_import.py
  tests/test_city_compiler.py -q` -> 24 passed, 7 skipped (skips are
  the pre-existing missing real OSM dataset fixture, unchanged).

TESTS:
- 16 new (tests/test_osm_city_features.py).
- 0 removed, 0 weakened — the skip-discipline fix only changes the
  failure MODE (hard fail -> honest skip) on a fresh checkout; when the
  gitignored fixtures ARE present the same tests still run and assert
  real behavior unchanged.

- CLOSED THIS SESSION (was "NEXT RECOMMENDED TASK" below): wired
  engine/compiler/osm_features.py into a real CLI entry point —
  `reality city-compile --osm <path> [--geojson <path>] -o <world.json>`
  (apps/cli/main.py:cmd_city_compile). 5 new tests
  (tests/test_cli_city_compile.py), plus a manual real-process
  `python -m apps.cli.main city-compile ...` run to confirm actual
  reachability, not just in-process test-harness reachability.
  Verified: 65 passed, 7 skipped across the full OSM/city/CLI test set.

KNOWN LIMITATIONS:
- RESOLVED (same session, see UPDATE at top): the gitignored real-data
  fixtures were found intact in the main checkout and copied into this
  worktree — no longer a limitation for THIS worktree. Still true for
  any genuinely fresh clone that doesn't have access to a checkout that
  already has them: the skip-cleanly fix means it degrades honestly
  (14 skips, not 14 failures) rather than fixing the underlying
  fresh-clone gap. See NEXT RECOMMENDED TASK.
- P14-01 (city world compiler) stays PARTIAL by design: satellite/
  aerial/LiDAR ingestion and full CityGML/3DCityDB export are still
  open. `reality city-compile` currently only compiles OSM ways into
  entities (GeoJSON is imported as evidence but not yet translated into
  CityFeatures — honestly 0 entities from a GeoJSON-only run, not a
  crash).

OPEN ISSUES:
- None new from this pass.

DEPENDENCIES:
- None blocked.

- CLOSED THIS SESSION (was the "NEXT RECOMMENDED TASK" above):
  engine/compiler/geojson_features.py compiles GeoJSON Polygon
  features into CityFeature, reusing osm_features.classify_osm_tags
  directly on `properties` (one shared tag table). `reality
  city-compile` now combines OSM + GeoJSON sources into one WorldIR.
  9 new library tests + 3 new CLI tests; manually re-verified via a
  real process run combining both `--osm` and `--geojson`.

NEXT RECOMMENDED TASK:
- The gitignored real-data fixtures now exist and verify correctly
  (21/21 tests passing) but remain uncommitted and untracked by any
  worktree/clone that doesn't happen to sit next to a checkout that has
  them. Decide deliberately (product/process decision, not a routine
  one — flagging per Agent OS §51 rather than choosing unilaterally):
  either (a) commit datasets/room_capture/pipeline_out/,
  datasets/real_room_capture/, datasets/real_room_capture_worldir/ as a
  fourth exception in .gitignore (mirroring datasets/south_building/'s
  existing pattern), which would make these 14 tests run for real in
  every fresh clone/CI run at the cost of ~1.5MB+158KB+1.4MB of
  committed binary/JSON fixture data; or (b) leave them local-only and
  accept that fresh clones always see the 14 honest skips unless
  someone manually copies the artifacts (current state, lowest repo
  weight, weakest CI coverage).
