# Engineering Design Decisions

Canonical decision log. Only decisions **already made and enforced in
the codebase** appear here; open questions live in
[`../implementation/PENDING_IMPLEMENTATION.md`](../implementation/PENDING_IMPLEMENTATION.md)
as BLOCKED entries. Format mirrors `docs/DECISIONS.md` (V11 decisions
1–16); numbering continues from there.

Statuses: `ACCEPTED` (enforced), `PROVISIONAL` (reversible; revisit
trigger named).

---

## Decision 017 — Canonical geometry representation

**Status: ACCEPTED**

`MeshData` (`reconstruction/meshing/mesh.py`) is the canonical runtime
mesh representation: vertices, triangle indices, normals, coordinate
frame, metadata. COLMAP/OpenMVS are backend representations; Blender,
glTF, USD are export representations. A bounding box is an approximation
of an object and is labeled `INFERRED` — never presented as mesh
geometry.

## Decision 018 — Canonical world representation

**Status: ACCEPTED**

WorldIR v1 (`world_ir/world_v1.py`) is the canonical semantic world.
The viewer/Studio is a client of WorldIR + ArtifactStore. Compiled
worlds reference geometry through `(data_uri, data_hash)` artifact
pointers rather than embedding payloads.

## Decision 019 — Runtime intelligence

**Status: ACCEPTED**

No LLM dependency in the canonical runtime. Perception comes from real
model backends (Mask R-CNN, MiDaS, SAM when unblocked), geometry, and
numerical methods. Detection/segmentation backends sit behind
interfaces (`IDetectorBackend`, `ISegmentationBackend`) so models are
replaceable without touching the pipeline.

## Decision 020 — First depth sidecar format

**Status: ACCEPTED**

16-bit PNG first; EXR second (floating-point); device-specific raw
formats via explicit adapters later. Canonical representation:
`DepthFrame` with explicit `depth_scale`, `invalid_value`, `units`,
`calibration_ref`. **Raw integer depth must never be interpreted as
meters without an explicit scale factor.** Rationale: lossless,
inspectable, trivially testable, ubiquitous in RGB-D capture; EXR adds
value only when sub-mm float precision matters (rare before P1).

## Decision 021 — Surface reconstruction backend

**Status: PROVISIONAL**

COLMAP `poisson_mesher` (CPU) is the surface-reconstruction backend:
mature, already a repo dependency, accepts a plain oriented point cloud.
The backend adapts Poisson depth to point density and retries with
`trim=0` when the dense-cloud default yields an empty mesh (observed:
COLMAP's trim=10 requires density sparse clouds never reach).
**Revisit trigger:** a CUDA-capable COLMAP (unblocking
`patch_match_stereo` true MVS) or an OpenMVS backend — whichever lands
first changes the dense-geometry path, not the `MeshData` contract.

## Decision 022 — Honest unavailability states

**Status: ACCEPTED**

When a backend/model/dependency is missing, stages report a skip with
the reason (report.json `"status": "skipped"` + note; CLI exit 1 with
diagnostics) — never a placeholder that pretends inference happened.
Corollaries already enforced: `lift_region_to_3d` returns `None` for
non-metric depth instead of guessing scale; the perception stage refuses
without trusted intrinsics; `reality compile` never prints green on
failed stages.

## Decision 023 — Metric scale anchoring

**Status: ACCEPTED**

Metric scale comes only from an explicit operator-measured baseline
(`ScaleReference`) between two registered cameras, or — for mono-depth —
from alignment of relative depth to the SfM sparse cloud, in which case
the world is labeled `metric-by-alignment`. Unanchored worlds stay
`relative` with unit-less distances. There is no code path that invents
a scale factor.

**Amendment 2026-10-31 (user decision, option 2 -- suggest, user accepts; plan item D2b in `docs/superpowers/plans/2026-10-30-room-reconstruction-fidelity.md`).**
Scale is NEVER applied without the user's consent. Besides the camera baseline,
two more `ScaleReference` methods exist, both in model space and both gated:

- `known_length` -- the user states the real length of one edge / two picked
  points. Treated as a measurement (state MEASURED). Several known lengths are
  combined by median and must agree within `RATIO_SPREAD_MAX` (5 %), otherwise
  they are refused. API: `POST /api/worlds/{id}/scale` with
  `{method:"known_length", distance_m, point_a, point_b}` (or `model_length`).
- `prior_accepted_by_user` -- when the world is RELATIVE and a door with a
  measured height was fitted, the engine SUGGESTS a typical size
  (`GET /api/worlds/{id}/scale/suggestion`: door 2.0-2.1 m, midpoint 2.05 m, the
  object it came from, implied m/unit). Nothing changes until the user accepts
  (`POST .../scale` with `method:"prior_accepted_by_user"`, `source_entity_id`).
  Rejecting (`POST .../scale/suggestion/reject`) leaves the world RELATIVE. No
  door => no suggestion; a default is never offered.

A prior-based scale is ALWAYS labelled ESTIMATED (`scale_basis =
"prior_accepted_by_user"`, `dimension_state = "ESTIMATED"` in the world's
`metadata.scale` and the version report `stages.scale`); every dimension of
that world inherits ESTIMATED, never MEASURED (plan D1 reads these keys). Any
real measurement (camera baseline or known length) replaces the prior: trust
order is camera baseline > known length > accepted prior. A model-space length
is tied to the model unit it was measured in by a camera-pair fingerprint; if a
rebuild changes the unit the reference is refused rather than misapplied.

## Decision 024 — Camera-envelope filtering for meshing

**Status: ACCEPTED**

The sparse SfM cloud is triangulated, trustworthy geometry; it defines
the plausible scene envelope (95th-percentile camera-distance). Depth-
derived points beyond it are unprojection artifacts and are removed
before normal estimation/meshing. Observed necessity: without it, depth
outliers (km-scale) overwhelmed global outlier statistics and Poisson
meshed garbage at 400 m+ scale.

## Decision 025 — Deterministic test seam for the CLI pipeline

**Status: ACCEPTED**

`REALITY_TEST_BACKEND=module:Class` (env var, tests only) injects a
synthetic reconstruction backend so `reality compile` is E2E-testable
offline. Production leaves it unset and always uses COLMAP; the CLI
raises a clear error on a malformed spec rather than silently ignoring
it.

## Decision — Candidate selection by world evolution, rollback as a pointer move

**Status: ACCEPTED**

1. Incremental vs full COLMAP candidates are compared as WORLDS against HEAD with the same `world_delta`
   machinery that gates adoption (`engine/pipeline/candidate_selection.py`), by an ordered, recorded
   hierarchy - never by registered-camera count and never by a scalar score. Ties keep the incremental
   candidate (coordinate-frame continuity). If both fail acceptance the adoption gate keeps HEAD.
2. Rollback moves only the HEAD pointer. Versions stay immutable, evidence is never removed, the COLMAP
   base is set aside (renamed) before the pointer moves, and the next version chains onto the rolled-back
   HEAD (its evidence is the union of everything the world holds, so photos later versions used are retried).
3. The COLMAP session is derived state that commits strictly after WorldStore adoption and read-back; a
   crash leaves it at most one version behind, which converges. No cross-store journal is required for that.

## Decision — Adoption protocol (WorldStore <-> DB) and coordinate-frame preservation

**Status: ACCEPTED (2026-10-02, final sprint)**

Found by probing the real code: a crash between `WorldStore.save_version` and the DB commit left an orphan version that
`resync_versions` later mirrored as a PHANTOM version, and a retry then minted a second one; and an in-process fault after
the HEAD `UPDATE` was persisted by the job runner's own failure-bookkeeping commit, leaving HEAD naming a version with no
mirror row.

1. Adoption order is: intent file (`adoption-intents/`) -> WorldStore version -> read-back verification (BEFORE the
   pointer moves; the DB never names a version the store cannot reconstruct) -> DB HEAD + mirror row in ONE transaction
   -> intent cleared. Both writers (`commit_version`, the entity-correction route) use it.
2. A version written but never adopted is QUARANTINED (`quarantine/`; record moved, artifact kept, nothing deleted):
   immediately on an in-process fault, or by `reconcile_adoptions` at startup after a hard kill. `resync_versions` never
   mirrors a version with a pending intent. Versions with no intent (CLI writers) remain legitimate history.
3. A failed adoption undoes its own pending HEAD move/row inside the caller's transaction (no rollback: that would expire
   the caller's loaded objects).
4. Candidate arbitration: with an established HEAD, a candidate that keeps HEAD's coordinate frame beats one that
   re-expresses the world in a new frame unless the latter places >= `FRAME_CHANGE_MIN_EXTRA_PHOTOS` (2) more photographs.
   The frame shift is measured from the similarity `world_delta` already fits (geometry rules are blind to it by design).
   Tolerances are PROVISIONAL.
5. Retention: quarantined versions are kept indefinitely; a pruning policy is not decided.

Supersedes the old expectation (test_commit_failed_db_write_rolls_back) that an orphan is "reconciled into the listing".

## Decision — The working prototype is judged by the core loop; verification has four honest levels

**Status: ACCEPTED (2026-10-03, prototype-completion sprint)**

1. **Prototype scope.** The prototype is judged by `evidence -> world -> more evidence -> same world -> better world ->
   version -> export`, not by an ecosystem-integration checklist. Its export story is glTF, USD(A), IFC, CityGML,
   CityJSON and Blender (P16-01, DONE). The GIS / ROS / Habitat / SUMO / Unreal / Godot writers (P16-02) exist and are
   structure-tested, but are `OPTIONAL_DEFERRED`: no further work until a user of one of those ecosystems needs it, and
   nothing claims they load. Ledger states are exactly DONE / PARTIAL / BLOCKED / OPTIONAL_DEFERRED.
2. **Verification levels** (`engine/core/verification.py`; API `/api/system/verification`; CLI `reality verification`;
   `.agent/CAPABILITIES.yaml` is checked against the registry by a test): IMPLEMENTED < SYNTHETICALLY VERIFIED <
   REAL-WORLD VERIFIED, with EXTERNAL VERIFICATION PENDING items listed beside a capability. A capability below
   REAL-WORLD may not name real data; a synthetic one must be worded as synthetic; every one cites evidence files that
   exist. RGB-D, VIO, indoor architecture and 3DCityDB are synthetically verified with their real counterparts pending.
3. **Synthetic data policy** (`synthetic/`): deterministic, seeded, with the ground truth it was built from, labelled in
   every manifest it writes. It verifies the CONTRACT around a sensor or backend (formats, scales, invalid pixels, clock
   offsets, dropped frames, occlusion) -- never the sensor, never a VIO algorithm, never a real building. The VIO
   "binary" is a stand-in that perturbs ground truth; it exists to run the real subprocess adapters.
4. **Known limits are pinned, not hidden.** Where a synthetic fixture measured the engine WRONG (rooms joined by doorways
   merge; phantom openings under realistic depth noise; stairs not wired into assembly; the stair detector refuses an
   unsegmented flight or one with unobserved treads) the test is a strict `xfail` carrying the measured numbers: it flips
   to a failure the day the engine improves, forcing the pin and the ledger to be updated.
5. **Trajectories reach a world as an observation record**, not a registration: `trajectories.worldir.attach_trajectory`
   keeps the poses, clock/sync state and drift (unknown stays unknown) and records `registered_into_world: False` for a
   gauge-fixed frame. A VIO "world" frame is session-local until registration anchors it.
6. **3DCityDB availability is three-valued** (`probe_citydb`: AVAILABLE / UNAVAILABLE / FAILED) and probe-only: AVAILABLE
   means the `citydb` CLI runs, never that a database was reached. Contract tests against a fake CLI are not real
   3DCityDB verification; that stays external.

## Decision — Axis convention

**Status: ACCEPTED**

Right-handed, Y-up is the canonical world frame (`ARCHITECTURE.md`, `docs/MODULE_OWNERSHIP.md`,
`reconstruction/frame.py`, the viewport's `three-scene.ts`). The procedural grammars
(`procedural/room_grammar.py`, `urban_grammar.py`) author Z-up in grammar-local axes (x east, y north, z up; specs and
`origin` stay local) and convert once, in `_entity`, with the rotation (x, y, z) -> (x, z, -y), so height is world +Y and
local +y is world -Z. **Latent inconsistency, not fixed here:** `exporters/boxes.py` states "WorldIR is Z-up", which
contradicts this decision; reconcile it separately (exporters were deliberately left untouched).

## Decision — Site composer is manual placement, not registration

**Status: PROPOSED (feature at FOUNDATION; recorded 2026-10-31)**

A Site groups several already-reconstructed worlds and records a `SitePlacement` (position/rotation on a metric grid)
that the USER chose. It never infers where a world belongs, and a placement is never presented as registered or
geo-referenced. Automatic cross-source / geo-registration remains the MISSING item in
`docs/future/registration/CROSS_SOURCE_REGISTRATION.md` (it needs a measurable residual and a valid correspondence
set, which manual placement does not provide). Material hypotheses (plan M3) are likewise kept ESTIMATED with
`asserted=False` and are deliberately not mapped to physics materials.


## Quality bands (2026-10-13)

A model score is not a probability (master document 14.2), so the product shows coarse bands.

- **Entity band** (`apps/api/place_summary.py::entity_band`): `unknown` when provenance is UNKNOWN or confidence is missing or not
  a number; `weak` when provenance is CONFLICT or confidence is below 0.5; `fair` from 0.5 up to 0.75; `good` from 0.75.
  A conflict is never good, whatever its score. Not measured is never shown as good or bad.
- **Place band** (`band_and_reason`): `unknown` with no model; `weak` when fewer than half of the photographs are placed or the
  engine fell back below multi-view; `good` when the build was graded SUCCESS and at least 80% of photographs are placed;
  otherwise `fair`. The reason is returned with the band so the UI can say why.
- There is **no per-point band**: the stored point cloud carries no per-point confidence.
- The thresholds (0.5, 0.75, 0.8) are tolerances, not calibrated constants.

## Decision — One Studio UI (the workstation); no Simple/Pro switch (2026-11-07)

Status: accepted by the product owner ("Aa" = option A). Plan task T-1.1 in `docs/blueprint/33_TURNAROUND_PLAN.md`.

- The world page renders `SpatialWorkstation` for everyone. The Simple "place viewer" (`components/simple/*`) and the
  per-viewer Pro-mode preference (`re-pro-mode`, `useProMode`, Settings toggle) are removed.
- Why: the first customer (architecture and interior design firms) needs measuring, the inspector and evidence in the main
  view; two UIs for one world was a third noun beside Session and World (audit `32_FRONTEND_AUDIT.md` D1).
- A simpler presentation for clients comes back later as a **view mode inside the Studio** (plan M4b read-only client link),
  not as a second app.
- Consequence: `lib/copy.ts` and `lib/places.ts` (consumer wording and honest job progress) are no longer used by any screen; they
  stay because contract tests cover them. Delete or reuse when the client view is built.

## Decision — Proxy-only browser transport (2026-11-07, implementation pending)

Status: accepted direction; implementation is plan task T-1.2 and is a T3 slice.

- Target: the browser talks only to same-origin Next route handlers, which add the server-side token. Rationale: a hosted product
  cannot ship `NEXT_PUBLIC_API_TOKEN` to browsers, and one transport removes the "which client do I use" duplication.
- Not done yet. Needs a design that covers streaming (job SSE), multipart uploads and the Electron shell, all of which
  currently go direct. Minimal first step: one catch-all handler `app/api/[...path]/route.ts` forwarding any method with the
  token; explicit handlers keep priority. Must be verified with the Playwright suite before the base URL flips.
- Until then both transports exist; the new guard test `test_every_proxy_targets_a_path_the_api_serves` keeps the proxies honest.
