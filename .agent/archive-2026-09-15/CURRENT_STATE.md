<!--
ARCHIVED 2026-09-15 (P0-01 canonical-state consolidation).
SUPERSEDED: this file is point-in-time history and is NOT current truth.
Canonical files now: .agent/ENGINEERING_CONSTITUTION.md,
.agent/REALITY_ENGINE_MISSION.md, .agent/TASKS.yaml,
.agent/EXECUTION_STATE.md, .agent/CAPABILITIES.yaml, .agent/LICENSES.yaml.
Do not extend or edit this file; read it only for history.
-->

# Reality Engine — Current State

**Updated:** 2026-09-14 (mesh-stage fix + identity fixes + Priority 2: IMU/GNSS/telemetry/calibration sensor-stream normalization)
**Branch:** `claude/reality-engine-build-e6ac6e` (worktree off `main`, main already has dense-mesh/Poisson-mesh + glTF uint32-index fix + viewer real-mesh rendering + camera-envelope filter merged, PR #16)
**Verified baseline before this session:** 1298 passed, 1 pre-existing environment failure deselected.
**Verified after this session:** 1339 passed, 1 pre-existing failure deselected (63.4s) — +41 tests, zero regressions.

## Completed 2026-09-14 (continued Priority 2): added CALIBRATION and TELEMETRY sidecar parsing

Extended `evidence/sensors.py` with the two remaining well-defined
sidecar types named in Priority 2 (DEPTH was explicitly scoped out
last session and remains out -- see the module docstring's "what this
deliberately does not do" section for why: device-specific
image-format/scale-factor decisions, not a parsing-only problem like
the other three).

**Calibration**: `parse_calibration_json()` parses a single JSON
object (not JSONL -- a calibration is one artifact, not a time series)
into a `CalibrationRecord`. Deliberately reuses
`reconstruction.calibration.camera.CameraIntrinsics`/`CameraExtrinsics`
verbatim (their own `from_dict`/`to_dict`/`__post_init__` validation --
positive focal lengths, finite values, positive dimensions) instead of
reinventing a calibration schema, per the "don't rewrite working code"
principle -- a calibration sidecar file IS those types' own JSON shape
on disk. `extrinsics` is optional (a lab-calibrated lens may have no
known mount pose).

**Telemetry**: `parse_telemetry_jsonl()` -- a real, explicitly-scoped
drone-flight-controller-log schema (t, optional GPS position reusing
GNSS's own lat/lon/alt fields, optional attitude_deg/gimbal_attitude_deg
Euler angles in one named convention, optional battery_pct/flight_mode).
Documented as ONE specific convention, not a universal solution --
flight controllers disagree on Euler sign/axis conventions, named
explicitly rather than silently assumed compatible.

**Wiring**: `MultiSourceSession.sensor_streams()` now also dispatches
"telemetry" (added to `PARSERS_BY_COMPONENT`); new
`MultiSourceSession.calibrations(source_id)` method (separate from
`sensor_streams` because calibration parses to one record per file,
not a time-ordered sample stream).

13 new tests in `tests/test_sensor_streams.py` (telemetry parsing incl.
every honest-failure path and the "all fields but t are optional" case;
calibration parsing incl. reusing CameraIntrinsics' own validation
error, not a weaker reimplementation; both components' wiring through
`MultiSourceSession`). Verified additionally with a REAL end-to-end run
outside pytest: built an on-disk `rgb/`+`telemetry/`+`calibration/`
drone-capture-shaped folder, ran `add_source()` +
`sensor_streams("telemetry")` + `calibrations()`, confirmed correct
classification and correctly parsed values (position, attitude,
gimbal, battery, flight_mode; fx/width intrinsics). Full suite: 1339
passed (was 1326), zero regressions.

Priority 2 status now: IMU/GNSS/telemetry/calibration all have real
parsers wired end-to-end through `MultiSourceSession`. DEPTH remains
explicitly open (real format-decision blocker, not an oversight). No
time sync, no CRS conversion, no VIO -- those are Priorities 3/6/7.

## Completed 2026-09-14: Priority 2 -- real IMU/GNSS sensor-stream ingestion (evidence/sensors.py, new module)

Closed the gap the spec's Priority 2 named exactly: "recognizing imu/
is not the same thing as ingesting and using IMU measurements." Before
this, `evidence/multi_source.py`'s composite-capture detection found
`imu/`, `gps/`, `depth/`, `calibration/`, `telemetry/` sidecar
subdirectories and used their PRESENCE to classify a folder as
`COMPOSITE_CAPTURE`/`PHONE_CAPTURE`/`DRONE_CAPTURE` -- but the file
paths themselves were never surfaced anywhere a caller could reach,
and there was no parser for their contents at all.

**New module `evidence/sensors.py`**: real, typed, unit-explicit
parsers for IMU and GNSS sidecar files. Chose JSON Lines as the file
format (documented in full in the module docstring, including the
exact schema for both record types) since no convention existed
anywhere in this codebase to match against -- self-describing (a
missing/renamed field fails loudly, not a silently-shifted CSV
column), trivially streamable. `IMUSample` (t, accel_mps2 [specific
force, body frame, m/s^2, includes gravity], gyro_rps [rad/s],
optional mag_ut), `GNSSSample` (t, lat_deg/lon_deg [WGS84], alt_m
[ellipsoidal], optional accuracy_m/fix_type/satellites/velocity_mps),
both wrapped in a `SensorStream` (kind, source_path, samples,
provenance=OBSERVED, `.is_monotonic()` diagnostic). Honest failure:
`SensorParseError` names the file+line+reason for every malformed
record (missing field, wrong type, non-finite number, lat/lon out of
range, unrecognized fix_type) -- never silently dropped or coerced;
`OSError` (missing file) stays distinct from a parse error.

**Explicitly NOT done in this module (named, not hidden, matches the
spec's own "what this doesn't do" discipline)**: no WGS84->ECEF->ENU
coordinate conversion (spec Priority 6/sec 8, a separate CRS layer),
no cross-stream time synchronization (Priority 5/sec 10 -- `t` stays
per-file, per-sensor-clock), no IMU integration/trajectory estimation
(Priority 6/sec 7, VIO), no binary/vendor format support (only the
documented JSONL convention).

**Found and fixed a second, deeper bug while wiring this in**:
`evidence/multi_source.py::MultiSourceSession.add_source()` -- the
`SourceRecord.components` field existed on the dataclass (and
round-tripped through `to_dict()`/`from_dict()`) but was NEVER
actually populated during ingestion. `_source_type_of()` computed the
composite-component file manifest internally (to decide DATASET vs
`*_CAPTURE`) and threw it away. Fixed: `add_source()` now computes the
component dict once and threads it into every `SourceRecord`
construction path (INGESTED/FAILED/UNSUPPORTED). New method
`MultiSourceSession.sensor_streams(source_id, component)` resolves a
record's recorded relative sidecar paths against `original_path` and
parses them via the new module (on-demand, not eagerly during
`add_source`, so a non-JSONL sidecar file never fails an otherwise-good
import -- it just isn't parseable when asked for).

25 new tests in `tests/test_sensor_streams.py` (IMU/GNSS parsing incl.
every honest-failure path, comment/blank-line handling, round-trip,
component dispatch, and the full `MultiSourceSession` wiring incl. the
`components`-was-always-empty regression and a session
to_dict/from_dict round-trip proving parsing still works after
reload). Additionally verified with a REAL end-to-end run outside
pytest (per CLAUDE.md's "run the actual thing" rule): built an
on-disk `rgb/`+`imu/`+`gps/` phone-capture-shaped folder, ran
`add_source()` + `sensor_streams()` on it, confirmed correct
classification, correct component paths, and correctly parsed
IMU/GNSS values. `tests/test_multi_source_session.py` +
`tests/test_media_preprocessing.py` + `tests/test_sensor_streams.py`:
87 passed. Full suite: 1326 passed (was 1298 baseline), zero
regressions.

Not done (still open, named not hidden): DEPTH/CALIBRATION/TELEMETRY
sidecar parsing (spec Priority 2 names these too; only IMU/GNSS have
real parsers now -- depth needs an image-format decision, calibration
needs an intrinsics/extrinsics schema, telemetry has no universal
format at all, each a real separate design decision, not attempted
blind here). No time synchronization, no CRS conversion, no VIO. This
is normalization/ingestion only, exactly the scope the user asked for
this turn.

## Completed 2026-09-14 (continued Priority 1): closed the OTHER two `disk:`-basename identity sites in evidence/importers.py

Follow-on to the `MultiSourceSession.add_source()` identity fix above.
`evidence/importers.py::import_file()`/`import_folder()` also built a
default `EvidenceSource` with a filename-derived id (`disk:<basename>`)
when no caller-supplied `source` was given -- this is the LIVE path
`apps/cli/main.py`'s plain `reality ingest <folder>` command uses (no
`MultiSourceSession` involved, so no `SourceRecord` to reconcile
against, but still a real violation of "no filenames as identity" and
still collision-prone: two folders named `capture/` anywhere on disk
got the identical default source id).

Fixed with two new private helpers: `_default_source_id_for_file()`
(sha256 of the file's own bytes, `file:<hash16>`) and
`_default_source_id_for_folder()` (sha256 over every contained file's
`(relative_path, sha256)` pair, `folder:<hash16>` -- same guarantee
`evidence/multi_source.py::_content_hash_of_path` already gives
`SourceRecord`, computed once per `import_folder()` call over the
already-built sorted `paths` list, not recomputed per file). Both are
fallback-only: a caller-supplied `source=` is always used unchanged.

2 new tests: `tests/test_media_preprocessing.py::test_default_source_id_is_not_filename_derived`
(two differently-located same-named folders with different photo
bytes get different ids, and the id is no longer `disk:`-prefixed) and
the `MultiSourceSession` identity test from the prior fix. Verified
additionally with a REAL CLI-equivalent call (not just unit tests, per
CLAUDE.md's "run the actual command" rule): built two on-disk folders
named identically but with different photo bytes, called
`MultiSourceSession.add_source()` on both, and confirmed every
resulting `EvidenceAsset.source.source_id` matches its owning
`SourceRecord.source_id` exactly and the two records get distinct ids.
`tests/test_media_preprocessing.py`: all passing (+1). Full suite:
1301 passed (was 1300), zero regressions.

Not done (still open, named not hidden): `EvidencePackage`'s own
`sources` dict is already keyed by `source.source_id`, so it inherits
both fixes above automatically -- no separate change was needed there,
verified by inspection (`evidence/packages.py:319-320`,
`:438`). The full canonical `Source`/`Evidence`/`Observation`/
`Artifact` model the spec describes (explicit `content_identity`,
`acquisition_id`, `device_id`, `capabilities`, calibration references,
a dedicated `Observation` layer between Evidence and WorldIR) remains
unbuilt -- this and the prior session's fix closed the concrete
mismatch bugs the spec's own examples pointed at; they did not build
the larger identity architecture. That remains the next real slice of
Priority 1 if the user wants it continued further, OR the next phase
(Priority 2, sensor stream normalization) per the user's own
dependency order.

## Completed 2026-09-14: SourceRecord <-> EvidenceSource canonical identity (a real, narrow slice of the pasted spec's Priority 1)

The user's spec described a full Source/Evidence/Observation/Artifact
identity redesign. That is a multi-session architectural project, not
attempted here. What WAS confirmed as a real, concrete bug in this
codebase (not the hypothetical example the spec used, but the actual
equivalent): `evidence/multi_source.py::MultiSourceSession.add_source()`
gave the `SourceRecord` a collision-resistant, index+content-hash id
(`src-0000-<hash>`) but, when the caller didn't pass an explicit
`EvidenceSource`, built the DEFAULT `EvidenceSource` with a totally
different, basename-derived id (`disk:<basename>`) -- and that
`EvidenceSource` object is embedded directly in every `EvidenceAsset`
produced from the call (`EvidenceAsset.source`). Result: there was no
way to go from an asset's embedded source id back to the owning
`SourceRecord.source_id` without a caller already holding the
`SourceRecord` in hand, and two different source folders sharing a
basename (e.g. two "photos/" folders added from different parents)
would silently get the SAME default `EvidenceSource.source_id` despite
being different `SourceRecord`s.

Fixed: the default `EvidenceSource.source_id` is now the SAME
`source_id` as the `SourceRecord` it belongs to (unique per
`add_source()` call by construction: index + content hash). Callers
who pass an explicit `source=` keep full control -- this only changes
the fallback used when none is given. New regression test
`test_default_evidence_source_id_matches_the_source_record_id` proves
every `EvidenceAsset.source.source_id` produced by a call equals
`SourceRecord.source_id`. `tests/test_multi_source_session.py`: 26
passed (was 25). Full suite: 1300 passed (was 1299), zero regressions.

Not done (this is a slice, not the full spec item): `EvidencePackage`'s
own separate `sources` registry (`evidence/packages.py`), the
`evidence/importers.py` two other `disk:`-id call sites (used by the
single-package `evidence/session.py` / direct `import_folder`/
`import_file` path, not `MultiSourceSession`), and the full
canonical `Source`/`Evidence`/`Observation`/`Artifact` identity model
the spec describes (content_identity, acquisition_id, device_id,
capabilities, calibration refs, etc.) are all still open. This fix
closes the specific mismatch the spec's own example illustrated,
nothing broader.

## 2026-09-14: response to a 23-phase mega-roadmap request — scope note

The user pasted a ~40-section "principal engineer, do everything" directive
covering documentation reconciliation across 58+ READMEs, source/evidence
identity redesign, real multi-sensor ingestion, time sync, VIO/SLAM,
cross-source registration, dense MVS, uncertainty propagation, a full
WorldStore, Reality Studio, GIS/robotics, and CI/release engineering — in
one session. This is realistically weeks of work, not a single-session
task. Rather than fabricate progress across all 23 phases, this session
did ONE concretely-specified, independently verifiable item from the
directive (the mesh-stage gate-ordering bug, section 15) and is reporting
honestly rather than claiming broader completion. See "Fixed" section
below and `.agent/ROADMAP_2026-09-14.md` for the full remaining scope.

## Completed 2026-09-14: fixed the mesh-stage gate-ordering bug (real bug, not cosmetic)

`engine/pipeline/vertical_slice.py::_mesh_stage()` ran the COLMAP
`poisson_mesher_available()` capability probe BEFORE the
`len(points) < 100` point-count gate. Confirmed as a real, observable
bug, not merely a style nit: `tests/test_meshing_pipeline.py::test_too_few_points_skips`
was passing on this machine ONLY because real COLMAP happens to be
installed here (`C:\Users\shres\tools\colmap-extracted`, noted in an
earlier session) — the capability probe returned True and execution
fell through to the point-count check anyway, masking the ordering
bug. On a machine WITHOUT COLMAP, the exact same too-small point cloud
would have reported "COLMAP binary unavailable" instead of the real,
environment-independent reason ("too few fused points").

Fixed by reordering: point-count gate now runs immediately after the
disabled/artifact_store/metric-scale gates and BEFORE the meshing-deps
import and the COLMAP capability probe — matching `disabled? ->
artifact_store? -> metric? -> input quantity gate -> preprocess
viability -> backend capability -> execution`. Docstring updated to
document the gate order and why it's deliberate. New regression test
`test_too_few_points_skips_before_the_colmap_probe` monkeypatches
`poisson_mesher_available` to a spy that would fail the test if called
at all — proves the probe is never even reached when the cloud is too
small, so this can't silently regress back to being environment-
dependent. `tests/test_meshing_pipeline.py`: 28 passed (was 27). Full
suite: 1299 passed (was 1298), zero regressions.

## Roadmap dump received 2026-09-14 (see .agent/ROADMAP_2026-09-14.md for the full 47-item text)

The user pasted a comprehensive P0-P8 roadmap (packaging → docs → source
identity → sensor ingestion → sync → VIO/SLAM → registration → dense
MVS/mesh → perception/identity → evidence fusion/uncertainty/provenance
→ WorldStore → Studio → scale/GIS/robotics → CI/release). Saved
verbatim to `.agent/ROADMAP_2026-09-14.md` (a new file -- `.agent/PROMPT_LOOP.md`
is a separate, pre-existing loop file and was not touched) so it
survives context compaction. Working it top-down per priority number;
each item gets its own dated section here as it lands.

## Completed 2026-09-14: P0 item #1 -- fixed packaging (`pip install .` was broken)

Verified broken exactly as reported: `pyproject.toml`'s
`[tool.setuptools.packages.find].include` only listed `engine*`,
`world_ir*`, `provenance*`, `events*` -- `evidence`, `reconstruction`,
`perception`, `exporters`, `sdk`, `apps` were silently excluded from
the wheel, and there was no `[project.scripts]` entry, so no `reality`
executable existed after install (only `python -m apps.cli.main` worked,
and only from inside a checkout with the repo root on `PYTHONPATH`).

Fixed:
- `pyproject.toml`: broadened the include glob to cover all six missing
  top-level packages; added `[project.scripts]` `reality =
  "apps.cli.main:main"`.
- Added `__init__.py` to `reconstruction/`, `exporters/`, `apps/` --
  these worked as implicit PEP 420 namespace packages for every
  existing import site (`reconstruction.backend.interface`, etc., used
  throughout `evidence/`, `perception/`, `sdk/`, tests), so this changes
  no import path, but `setuptools.find_packages()` (invoked by
  `[tool.setuptools.packages.find]`) requires `__init__.py` to discover
  a directory as a package for wheel inclusion, and namespace-package
  discovery was not turned on. (`reconstruction/{features,matching,mvs,
  registration,sfm}` and `apps/{capture,studio,viewer}` remain README-only
  empty scaffolds with no Python code -- correctly still absent from the
  wheel; nothing to package there yet.)
- `dependencies = []` was also wrong: a clean-venv install surfaced
  `ModuleNotFoundError: numpy` the moment the CLI imported
  `reconstruction.backend.colmap_backend` (core reconstruct path, not
  the optional `perception` ML extra). Added `numpy>=1.24` and
  `scipy>=1.10` (used unconditionally by `reconstruction/meshing/preprocess.py`,
  the dense-mesh Poisson-reconstruction step) to `dependencies`.
  `cv2`/`PIL`/`torch`/`torchvision`/`segment_anything` were confirmed to
  live only under `perception/` and stay correctly gated behind the
  `perception` extra (not touched).
- `apps/cli/README.md` updated to document the installed `reality`
  entry point (the `python -m apps.cli.main` form still documented and
  still works, unchanged).

**Verified for real, not assumed:** built a throwaway venv
(`python -m venv`), ran `pip install .` against this worktree from a
clean environment, then from `/tmp` (outside the repo entirely, so
nothing could be resolving against a checkout on `PYTHONPATH`):
`reality --help` printed the full subcommand list (exit 0), and a
Python one-liner imported `sdk.reality`, `evidence.promote_rooms`,
`evidence.promote_planes`, `evidence.promote_objects`,
`reconstruction.backend.colmap_backend`, and all three exporters
(`exporters.gltf/usd/blender.exporter`) cleanly. Full in-repo test
suite re-run after the fix: 1298 passed, 1 pre-existing environment
failure deselected, zero regressions (105.9s) -- the `__init__.py`
additions and pyproject changes touch only packaging metadata, not
runtime behavior, and the suite confirms that.

Not done in this pass (named, not hidden): no CI wiring yet to catch a
future regression of this same bug automatically (P8 item #41/#42 in
the roadmap) -- this was a manual clean-venv verification, not an
automated one. `reality-engine` is still `version = "0.1.0"`,
unreleased; no PyPI/wheel distribution step exists yet.

## Completed 2026-09-14 (latest): wired room artifact_store through the compiler (CLI comes free)

Follow-on named at the end of the room-geometry feature session (same
day): `engine/compiler/world_compiler.py::compile_reconstruction_to_world`
now passes `options.artifact_store` into `promote_room_to_entity` (one
line, same pattern already used for `promote_plane_to_entity` two
blocks above) — a compiled room's boundary polygon now gets a real
`data_uri`/`data_hash`, not just promoted planes/objects.

CLI needed **no separate change**: `apps/cli/main.py::cmd_reconstruct`
already builds `CompileOptions(artifact_store=FileArtifactStore(...))`
for every `reality reconstruct` call (real-geometry-by-default since the
prior session's CLI work) and passes it straight through to
`compile_reconstruction_to_world` — once the compiler threads the store
into room promotion, the CLI path is fixed automatically. Verified by
running the full CLI test suite (`tests/test_cli.py`, 30 tests, all
pass unchanged) rather than assuming from reading the code.

2 new tests in `tests/test_geometry_artifacts.py`
(`TestPromoteRoomsWritesRealGeometry`): one proves a room compiled with
an `artifact_store` gets a resolvable `data_uri` whose decoded point
count matches `geometry.vertex_count`; one proves omitting the store
leaves `room.geometry_ids == []` (the additive/backward-compat
contract). Full suite: 1298 passed (was 1296), zero regressions.

## Completed 2026-09-14: room boundary polygon geometry (the one item the 2026-09-13 session flagged "do this one carefully, in-session, not delegated blind")

`evidence/promote_rooms.py::promote_room_to_entity()` gained an optional
`artifact_store` param, same pattern as `promote_planes.py`/
`promote_objects.py`: when given, the room's boundary ring — already
computed in floor-plane (u, v) coordinates by `_trace_ring_from_lines`
— is converted to world-space xyz via a new `_ring_world_positions()`
helper (origin = `-d*normal`, basis = the same `_floor_basis(normal)`
used to build the ring, so the conversion is the exact inverse of the
projection — no re-derivation, no coordinate-frame drift) and stored as
a real `PointCloudData` artifact. A `Geometry` (type `PLANE`, matching
the existing exportable-type set so gltf/usda/blender export it with no
exporter change, same as objects reusing `BOX` did) is attached to the
ROOM entity's `geometry_ids` — previously always `[]` ("the room's
extent lives in its parts, not new geometry"). Omitting the param
reproduces prior behavior exactly (regression test). New test also
proves every decoded vertex satisfies `n.p + d == 0` on the room's own
floor plane — the coordinate-frame-bug risk the prior session explicitly
named. 2 new tests in `tests/test_room_inference.py`. Full suite: 1296
passed (was 1294), zero regressions.

Not wired in this pass (named, not hidden): no caller (CLI/SDK/studio
session) passes `artifact_store` into `promote_room_to_entity` yet —
same follow-on gap `promote_planes`/`promote_objects` had before their
own callers were updated; wiring it through `engine/compiler/world_compiler.py`
and `apps/cli/main.py` is the natural next step for this specific line
of work.

## Completed this session (2026-09-13, latest): security fix + real geometry uniform across gltf/usda/blender in the SDK and CLI

**Security fix (Strix-flagged, MEDIUM, CWE-22):** `ArtifactStore.digest_of()` (`world_ir/artifact_store.py`) accepted any string after `artifact://` and handed it straight to `FileArtifactStore._path_for()`, which joins it onto the store root — a `data_uri` of `artifact://../../victim` could read files outside the store. `data_uri` is untrusted input (round-trips through WorldIR JSON, which can come from anywhere). Fixed: `digest_of()` now requires exactly 64 lowercase hex characters (`^[0-9a-f]{64}$`), enforced once on the shared `ArtifactStore` base so both backends are covered. 9 regression tests (traversal sequences, absolute paths, wrong-length/non-hex digests, explicit filesystem-untouched assertion for the file backend).

**Uniform real-geometry export:** `sdk.reality.export()` previously special-cased `artifact_store` threading to `format == "gltf"` only, from when gltf was the only exporter that supported it. Now that usda/blender accept the same `artifact_store` parameter (see prior session), the special-case was removed — `export()` threads it through uniformly for all three formats. `apps/cli/main.py::cmd_export` was updated the same way: it now reconnects to `<world>.artifacts/` for `--format usda`/`--format blender` too, not just gltf. Updated `tests/test_cli.py`'s `test_export_usda_blender_unaffected_by_artifacts_dir` (now misleadingly named) into two accurate tests: one proving usda/blender DO emit real geometry (`def Points "` / `from_pydata(` present) when a store is reconnected, one proving both still fall back to their placeholder shape without one. `apps/cli/README.md` updated.

## Completed this session (2026-09-13, latest): real geometry reaches every exporter + the CLI (dispatched as 3 parallel agents, disjoint file sets, reviewed and integrated centrally)

Closed three of the four remaining gaps this file listed after the plane/object real-geometry-storage work: gltf was the only consumer of `artifact_store`, usda/blender exporters didn't consume it at all, and the CLI never created or passed one through even though the engine supported it end-to-end.

**usda exporter** (`exporters/usd/exporter.py`): `export_to_usda`/`export_to_usda_with_report`/`write_usda_file` gained an optional `artifact_store` param; a resolvable geometry now emits a real USD `Points` prim (`point3f[] points = [...]`, local-space, order-preserved) instead of the placeholder `Cube`. Same honesty fallback as gltf. 6 new tests in `tests/test_usd_exporter.py`.

**blender exporter** (`exporters/blender/exporter.py`): same pattern — a resolvable geometry now generates real `bpy.data.meshes.new()` / `from_pydata()` / `bpy.data.objects.new()` script text (a real point-cloud mesh) instead of the AABB-scaled cube, in local space, same custom-property/collection-linking logic shared between both paths. 4 new tests in `tests/test_blender_exporter.py`. Neither exporter's real-geometry text has been validated against a real `pxr`/Blender install (both already carried that caveat before this session; unchanged).

**CLI** (`apps/cli/main.py`): `reality reconstruct` now defaults to real-geometry storage via a real, persistent `FileArtifactStore` rooted at `<output>.artifacts/` (e.g. `world.json` -> `world.json.artifacts/`) — real geometry is now the CLI default, not something a caller has to opt into. `--no-real-geometry` opts out, reproducing the exact prior CLI behavior (verified). `reality export --format gltf` automatically reconnects to `<world>.artifacts/` if present, so a `reconstruct` -> `export` round-trip across two separate CLI processes now emits real per-entity meshes; usda/blender exports are unaffected (those exporters don't consume the store from the CLI yet even though they now support the parameter directly — wiring that through is a small follow-on, see below). 6 new tests in `tests/test_cli.py`. `apps/cli/README.md` documents the new default and convention.

All three were dispatched as independent parallel subagents (disjoint file sets: `exporters/usd/`, `exporters/blender/`, `apps/cli/` + each one's own test file) via superpowers' dispatching-parallel-agents pattern, then reviewed and integrated in this session — no merge conflicts (verified: `git status` showed only the 7 expected files, no overlapping edits), diffs spot-checked for correctness against the gltf exporter's established pattern before accepting. Full suite run once after integration: 1127 passed (was 1111), zero regressions.

Not done in this pass (named, not hidden):
- A true end-to-end `reconstruct` (real backend, real photos) -> real `.artifacts/` directory round-trip is still unverified — no COLMAP install in this environment, same blocker every session has hit. The `--no-real-geometry` opt-out and the artifact-path-derivation/reconnection logic ARE tested directly for all three formats now; only the full real-backend path is unverified.
- Triangulated MESH storage is still open (point-clouds only, across all three formats now).

## Completed this session (2026-09-13, latest): real geometry storage for OBJECTS (P0.10/11 follow-on) — `evidence/promote_objects.py` now writes real points too, not just `promote_planes.py`

Closed the first item named in the prior session's "Not done" list:
`ObjectHypothesis3D` (`perception/instances/lifting.py`) already
computed real unprojected 3D points per mask pixel inside
`lift_region_to_3d()` and threw them away after their centroid/bounds/
count -- the exact same gap `promote_planes.py`'s inlier positions had
before the plane-geometry-storage work. Fixed at the source: added a
`points: Tuple[Vec3, ...] = ()` field (default-valued, so every
hand-built hypothesis in existing tests keeps working unchanged) and
populated it for real in `lift_region_to_3d()`.

`evidence/promote_objects.py::promote_object_to_entity()` gained the
same `artifact_store` parameter `promote_plane_to_entity()` has: when
given, it unions every contributing hypothesis's real points
(`candidate.source_hypotheses[*].points` -- nothing was discarded by
`merge_hypotheses()`, so this was reachable with zero changes to
`object_resolution.py`) into one `PointCloudData` artifact and sets
`data_uri`/`data_hash`. No exporter change was needed either:
`exporters/gltf/exporter.py`'s real-mesh path already covers `BOX`
geometry (`_EXPORTABLE_GEOMETRY_TYPES`) and doesn't filter by type when
resolving `data_uri` -- the machinery built for planes just worked for
objects too, confirmed by a full lift -> merge -> promote -> validate ->
export test.

5 new tests in `tests/test_geometry_artifacts.py` (points survive
lift->merge, backward-compat default, real data_uri written, the
additive-omit regression test, and the end-to-end export proof). Full
suite: 1111 passed (was 1106), zero regressions.

Not done in this pass (named, not hidden): `perception/instances/object_resolution.py`
still doesn't store a *merged* point cloud on `MergedObjectCandidate`
itself -- promotion unions the per-hypothesis points at promotion time
instead, which is correct but means any other future consumer of
`MergedObjectCandidate` before promotion still can't see the union
directly. Triangulated MESH storage and usda/blender real-geometry
export are still open, same as before this session.

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

## Completed 2026-09-13 (latest): simulation campaign — three milestones

Started from verified state: physics core/destruction/glass/debris/water/
rain REAL; 15 physics dirs (fire, fluids, smoke, structural, fracture...)
EMPTY scaffolds; no WorldIR->physics bridge.

1. **WorldIR->physics compiler** (`engine/physics/world_compiler.py`,
commit cd77720): compile_world_to_physics() turns reconstructed WorldIR
entities into SimpleRigidBodyBackend bodies — PLANE geometry -> static
infinite planes (bounds-thinnest-axis normal), BOX -> dynamic/static
bodies (structure types static). Honesty: density labelled
observed/estimated/fallback with derivation notes; unsupported geometry
skipped per entity with reason codes (NO_GEOMETRY/
UNSUPPORTED_GEOMETRY/BAD_BOUNDS); mass derived deterministically; the
infinite-plane OVER-restraint (doorway gaps do not pass bodies) is a
documented approximation with a pin test. End-to-end: reconstructed room
-> WorldIR -> compiled physics -> crate falls and rests on the compiled
floor plane (y≈0.19 vs 0.2 hand-computed). Engine characteristic
observed: resting bodies stay AWAKE (gravity applied before the sleep
timer check) — flagged, not silently patched. 13 tests.

2. **Temporal state layer** (`engine/simulation/temporal.py`, commit
2ce36f3): complements the existing Timeline/ReplayController (segments
+ playback — pre-existing, 520 lines, tested; NOT rebuilt). EventGraph
(ancestors/descendants/root_causes/entity/after queries over the real
EventBus causal field list, unknown ids raise); SnapshotStore (full
WorldIR to_dict captures, deep-copied + mutation-isolated, deterministic
ids snap-{branch}-{tick}-{ordinal}, in-place restore preserving world
identity); BranchManager (deep-copy inheritance — child mutation never
reaches parent (tested), registry semantics, baseline protected,
structural compare added/removed/changed with old->new values);
replay() on a deep copy against a fresh bus — non-destructive by
construction. 18 tests.

3. **Wind field** (`engine/environment/wind.py`, commit bcdab40):
frozen config + DeterministicRNG + Beaufort bands + optional
engine.world.events publishing (wind.band_changed), matching the
rain.py pattern. Boundary-layer power law (alpha in documented terrain
range), deterministic two-component sine gusts, quadratic drag
F=0.5*rho*Cd*A*v_rel^2 with relative-velocity form; apply_wind_loads()
impulses real dynamic bodies in the SimpleRigidBodyBackend (static
untouched) — wind speed -> force -> motion -> contacts, no scripted
timers. Hand-computed drag values tested. 20 tests.

Full suite after all three: **971 passed / 2 skipped** (933 baseline +
38 new). All committed on `evidence-fusion` (unpushed).

## Next tasks (dependency-safe, in order)

Pruned this session (audited against actual code, not assumed from the
list — all three were already done and just never removed here):
`world_ir.diff.diff_worlds()` already has a real caller (`reality diff`
in `apps/cli/main.py`); `StudioSession.reconstruct_and_compile()`
(`engine/studio/session.py`) already routes evidence through the
backend orchestrator then `compile_reconstruction()`; that same method
is already the "Studio action" for plane+room promotion (it calls
`compile_reconstruction_to_world()`, which promotes both). Lesson: this
file drifts behind actual code across sessions — verify a next-task
item against the source before spending effort on it, same discipline
this session applied before dispatching the three parallel agents above.

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
- Run the Blender exporter's generated script inside an actual Blender
  install once one is available in this environment, and visually
  inspect the result (geometry/transforms/hierarchy/custom properties,
  now including the real point-cloud mesh path added this session) —
  the exporter itself is done and tested, but never opened in Blender
  (no Blender binary in this environment, confirmed this session).
- Triangulated MESH storage: `world_ir/geometry_data.py` now has real
  `PointCloudData`, consumed by all three exporters (gltf/usda/blender)
  and both promotion modules (planes/objects), but no `MeshData`
  (vertices/indices/normals) yet — needs a real surface-reconstruction
  step (e.g. alpha-shape/Poisson over a plane's inlier points), not
  just a new payload format.
- Compiler consumption of depth/segmentation/material evidence (currently
  planes+rooms only).
- Non-convex (L-shaped) room rings; DOOR/WINDOW/ROOF assignment; multi-room
  shared-wall ownership (room topology is now the foundation).
- Room entities (`evidence/promote_rooms.py`) carry no geometry of their
  own (only relationships to their walls/floor/ceiling, which do have
  real geometry) — a real room-boundary polygon (the already-computed
  `_Ring` in floor-plane (u,v) coordinates) could become real geometry
  too, the same way planes/objects now do, but needs careful (u,v) ->
  world-space conversion via the floor's basis vectors to avoid a
  coordinate-frame bug — deliberately NOT attempted via a parallel
  agent this session because of that risk; do this one carefully,
  in-session, not delegated blind.
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
