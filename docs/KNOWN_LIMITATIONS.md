# Known Limitations

Generated from the per-requirement "Limitations" field in
[BUILD_LEDGER.md](BUILD_LEDGER.md) — nothing here is invented; this is
that data regrouped for scanning. Update this file whenever a
requirement's limitations change, not just when new ones appear.

## Blocking nothing today, but real gaps

| Requirement | Limitation | Upgrade path |
|---|---|---|
| REQ-007 Job System | Stub only, not integrated into runtime | Wire into `WorldRuntime.step()` once systems need parallel scheduling |
| REQ-011 Collision Detection | O(n²) broadphase; sphere/box/plane primitives only | BVH/spatial hash + capsule/mesh shapes when entity counts or geometry demand it |
| REQ-012 Contact Resolution | Single-pass solver (no iteration); 4-material database | Sequential-impulse iteration loop; expand material table as new materials are added |
| REQ-025 Basic Fracture | No mesh fragment generation; single-pass (no cascading fracture) | Add mesh-based fragmentation and multi-generation cascade once a mesh pipeline exists |
| REQ-026 Glass Physics | Frame attachment points tracked but not enforced; no mesh generation | Enforce attachment constraints in `fracture_pane()`; generate real shard meshes |
| REQ-027 Debris System | No inter-fragment collisions; fixed sleep threshold; no spatial partitioning | Add fragment-fragment collision pass; make sleep threshold configurable; spatial hash once fragment counts grow |
| REQ-028 Replay System | No automatic replay-determinism verification; snapshots don't capture all internal state; no compression | Add a replay-vs-live comparison test; extend `TimelineSnapshot` fields as needed |
| REQ-029 Viewport | Angle-cone frustum test only (no separate H/V FOV, no projection matrix); no occlusion culling | Real Mat4 projection + occlusion when a renderer consumes this |

## By design, not expected to change soon

| Requirement | Limitation | Why acceptable |
|---|---|---|
| REQ-003/018 Entity Registry | Single-threaded | No concurrency requirement yet |
| REQ-004 Serialization | Schema v1 only, no migrations | No v2 schema exists yet |
| REQ-005 Coordinate System | Only registered frames resolve, no auto-discovery | Explicit registration matches §10's "never silently mix coordinate systems" |
| REQ-006 Deterministic Clock | No wallclock sync, purely simulation-time | Correct for a deterministic simulation engine |
| REQ-008 Units | Lightweight `Quantity` wrapper, no dimensional analysis | Sufficient for current SI-consistency requirement |
| REQ-009 Deterministic RNG | `random.Random`-based, not cryptographic, platform-dependent floats | Determinism requires reproducibility, not cryptographic strength |
| REQ-010 Rigid Body | No rotational contact response; velocity-based sleeping only | Matches current golden-test scope |
| REQ-013 Numerics | Energy explosion threshold fixed at 1000x | No case yet requiring it to be configurable |
| REQ-014 Event Bus | In-memory only, no persistence | Durable log deferred to export/replay integration |
| REQ-015 Logging | No file rotation; stdout/stderr only | No production deployment yet needs rotation |
| REQ-016 Config | TOML only, no env var override | No deployment need for env overrides yet |
| REQ-017 Errors | Hard failures halt-on-first, no recovery | Matches §17's hard-failure classification intent |
| REQ-019 Component Registry | No component dependencies/ordering | Not yet needed by any system |
| REQ-020 Resource Manager | No pooling, no async loading | No asset pipeline exists yet to need it |
| REQ-021 World Lifecycle | No pause/resume states | Not yet requested by any consumer |
| REQ-022 Dependency Graph | Kahn's algorithm, no in-tier priority | No system yet needs same-tier ordering control |
| REQ-023 Caching | No adaptive sizing; substring pattern invalidation | Sufficient for current cache usage |
| REQ-024 WorldRuntime | No streaming, single-world only | Multi-world/streaming not yet a requirement |

## Fixed bug (found Step 17, fixed during the 2026-09-07 quick audit)

**`WorldRuntime` could not actually construct against the WorldIR it's
type-hinted for.** `WorldRuntime.__init__`/`get_entity_from_world`
iterated `world.entities` directly; that only yields `Entity` objects
for the legacy `world_ir.world.WorldIR`'s `EntityRegistry.__iter__` —
for the V1 schema (`world_v1.WorldIR.entities`, a plain
`dict[str, Entity]`), iterating the dict yields its string keys, and
`entity.id` raised `AttributeError` the instant a V1-schema world had
any entities. Fixed by a `_iter_world_entities()` helper that branches
on `isinstance(world.entities, dict)`, used at both call sites in
`engine/world/runtime.py`. Regression tests added in
`tests/test_world_runtime.py`. See [DECISIONS.md](DECISIONS.md) #14.

**Update (2026-09-07, P0 audit-repair)**: the transform-resolution gap
this section used to describe is now fixed — see
[DECISIONS.md](DECISIONS.md) #15. V1-schema `Entity.transform` dicts
resolve through `CoordinateRegistry` exactly like legacy `Transform`
objects, including multi-hop chains and inverse-direction resolution.
Inspector (REQ-030) still reads `world_v1.WorldIR` directly rather than
through `WorldRuntime` — that part of the original decision stands
independent of this fix, since `WorldRuntime`'s ECS side still has no
materials/geometries/measurements API — see [DECISIONS.md](DECISIONS.md) #12.

## Open spec ambiguity

- **Provenance taxonomy** — the implemented `Provenance` enum
  (`OBSERVED/RECONSTRUCTED/ESTIMATED/INFERRED/GENERATED/UNKNOWN/CONFLICT`)
  does not carry a separate `MEASURED` state distinct from `ESTIMATED`,
  which some directive text implies. See [DECISIONS.md](DECISIONS.md) #10.
  Not reconciled — flagged as `SPEC_AMBIGUITY` rather than silently resolved.

## Not yet started (honest gap, not a limitation of something built)

**CORRECTED 2026-09-23**: This section was updated to reflect the
2026-09-15 through 2026-09-23 implementation campaigns. The canonical
status ledger `.agent/TASKS.yaml` (see `.agent/EXECUTION_STATE.md` for
evidence) is authoritative over this file when they disagree.

**What exists now (verified by file count):**
- `reconstruction/` (34 .py files) — COLMAP backend, orchestrator, scale, depth, fusion, meshing
- `perception/` (45 .py files) — MiDaS, Mask R-CNN, SAM, lifting, fusion, quality, detail, tracking
- `apps/` (10 API backend .py files + CLI) — Full FastAPI backend (db/jobs/main/models/routes_*), reality CLI
- `exporters/` (13 .py files) — glTF, Blender, USDA, CityJSON, CityGML
- `benchmarks/` (13 .py files) — Competitive benchmark suite
- `frontend/` (12 API routes + 33 page components) — Next.js 16 + Three.js viewer, desktop + mobile
- `.github/workflows/ci.yml` — CI with multi-Python matrix, wheel build, 28+ test files

**Physics removed to child (2026-09-18):**
`engine/physics/*`, `engine/fire/*`, `engine/fluids/*`,
`engine/weather/*`, `engine/disasters/*` were moved to
`reality-engine-child`. Core no longer contains physics.

**Still genuinely empty/not started (matching TASKS.yaml):**
`gpu/`, `datasets/` (runtime data directory only), `plugins/`,
`shaders/`, `tools/`. Disaster-application logic intentionally excluded
from core (P0-02/P18-01 — platform/application boundary).

**Still unstarted per TASKS.yaml:**
Causal graph, branching/counterfactual engine, AI copilot,
natural-language query.

## Evidence-progressive product path (2026-09-29)

Details and evidence: [PROGRESSIVE_RECONSTRUCTION.md](PROGRESSIVE_RECONSTRUCTION.md).

| Area | Limitation | Upgrade path |
|---|---|---|
| Single-image bootstrap | Relative scale only; assumed camera when EXIF has no focal length; assumes a roughly level camera; confidence capped at 0.35. Opening detection is image-space rectangle *candidates* only (conf <= 0.2): measured on one South Building facade photo it found 2 of ~15 visible windows plus 1 fragment after the size filter, i.e. **low recall**, multi-pane windows are mostly missed, and a wall hidden behind foliage yields nothing. Corridor axis needs two walls on opposite sides of the camera plus a floor; it was **only verified on a constructed corridor**, no real corridor photograph exists in the repo. Role labels (floor/ceiling) on an *exterior* photo come from the level-camera assumption and are hypotheses | Learned window/door segmentation; a real corridor dataset; accept a user-supplied scale reference |
| Multi-view | Only the sparse COLMAP path is wired (Level 2); dense (Level 3) exists but is off by default; wide-baseline sets legitimately fail and fall back to Level 0 (every usable photo is kept as its own single-view hypothesis and laid out side by side for display only; their relative placement is unknown) | Enable dense MVS behind an explicit operator/GPU decision |
| Frame prior | Camera-up gravity prior assumes people hold cameras roughly upright and that orientations are diverse; sideways/rolled captures can mislead it (guarded by coherence + diversity thresholds) | Use EXIF/IMU gravity when present |
| Versions | V(n) is rebuilt from the union of evidence, so entity ids are re-derived; diffs can show remove+add for a refined structure | Wire `world_ir/entity_reid.py` into the rebuild |
| Studio | The status panel describes the *current* model while an older version is inspected | Serve the selected version's report/evidence |
| Datasets | The only real photographic golden dataset is South Building (an exterior); no real corridor or room dataset exists in the repo | Capture and document real corridor/room sets with rights |
| Video | Frames are sampled uniformly in time (24 max); no blur/overlap-aware selection | Use the contribution measure to pick frames |
| Confidence defaults | `Observation` / `Relationship` constructors still default to 1.0; depth-frame decode fidelity states 1.0 with its basis | Dedicated pass over constructor defaults |

