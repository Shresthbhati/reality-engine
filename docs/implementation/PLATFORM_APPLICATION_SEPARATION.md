# Platform / Application Separation — Scoping and Extraction Plan

Task: **P0-02** (`.agent/TASKS.yaml`). Principle (constitution Article XI,
mission "Application boundary"): disaster management is a **consumer** of
the world platform, not its definition. Core keeps the world platform and
generic primitives; application workflows live in application packages.

Inventory verified against the repository 2026-09-15 (grep + module reads,
not documentation claims).

---

## 1. What was actually found

### The feared disaster-application layer barely exists in code

- `engine/disasters/` — **README-only scaffold** ("Not yet implemented").
  Zero Python files.
- No evacuation orchestration, hazard-scenario engine, emergency-response
  logic, disaster dashboards, or domain scoring exists anywhere: repo-wide
  search for `disaster|evacuat|hazard|emergency|earthquake|flood_sim|
  scenario|drill|response|incident|casualt` over `engine/ apps/
  exporters/ sdk/ world_ir/` returns only (a) comments, (b) the generic
  event/response physics vocabulary, (c) the Inspector's generic
  `world.scenarios` listing (branch/versioning UI, not disaster logic).
- **Conclusion: there is no disaster application to extract.** The
  application-specific attack surface the mission worried about was never
  built — the separation problem is *preventive*, not curative.

### What exists that LOOKS application-flavored but is generic

| Module | Verdict | Why |
|---|---|---|
| `engine/physics/fire/` (FireSolver, fuel/ignition/spread) | **STAY (core)** | Causal physical process on material state (heat → ignition → combustion → spread); no disaster workflow inside. Mission lists "thermal/combustion primitives" as core. |
| `engine/physics/destruction/` (fracture/glass/debris) | **STAY (core)** | Mechanics of material failure; consumes physics state, emits events. Listed core ("destruction foundations"). |
| `engine/environment/` (wind/rain/water fields) | **STAY (core)** | Environmental fields are listed core primitives; deterministic, force-coupled, event-driven. |
| `world_ir/schema_v1.py` `NATURAL_HAZARD` entity type, `DESTRUCTION` observation type, `FRACTURED` geometry state | **STAY (core)** | Ontology entries describing *states of the world* — a hazard event is world state any application (insurance, robotics, urban) may consume. Not disaster workflow logic. |
| `engine/physics/disasters/`, `engine/physics/{weather,fluids,hydrology,thermal,atmosphere,smoke,particles,debris}/` | **EMPTY SCAFFOLDS** | Zero py files today. When populated they must hold *generic physics*, per Article XI. |

### Real layering violations found (small, concrete)

The world-platform chain reaches into engine internals in exactly four
files:

1. `reconstruction/calibration/camera.py` → `engine.physics.math3` (Quat/Vec3)
2. `perception/instances/{lifting,object_resolution}.py`,
   `perception/geometry/planes.py` → `engine.physics.math3`, `engine.core.rng`
3. `evidence/{session,dataset}.py` → `engine.core.logging`
4. `world_ir/entity_reid.py` → `engine.scene_graph.spatial_index`

None touch simulation domains (fire/destruction/environment) — verified:
the entire `evidence → reconstruction → perception → world_ir → exporters`
chain imports no simulation module. The coupling is to **basic math/logging
utilities**, not to application logic. This is a purity wart, not a
platform/application mixing, but it is the mechanism by which core would
rot if copied into `reality-apps/` — so it gets a named rule and an
enforcement test.

---

## 2. The extraction plan (when an application exists)

Trigger: an actual disaster application (dashboards, evacuation
orchestration, response scoring) is built. Then:

1. **New package** `reality-apps/` (separate top-level package in-repo now;
   separate repository when it needs an independent release cycle):
   ```
   reality-apps/
     disaster/
       workflows/      # scenario construction, response orchestration
       scoring/        # domain metrics (damage assessment, evacuation KPIs)
       dashboards/     # presentation only, reads WorldIR/WorldStore
     urban/
     robotics/
     construction/
     inspection/
   ```
2. **Dependency direction (enforced):** `reality-apps/* → world_ir,
   evidence, engine/*, sdk` — never the reverse. Applications read the
   world and drive it through the command pipeline (`WorldCommandProcessor`,
   `sdk/reality.py`); they never mutate WorldIR directly and never get
   imported by core.
3. **What moves:** only application-specific code written from that day
   forward (workflows, scoring, dashboards). Nothing moves today because
   nothing application-specific exists.
4. **What never moves:** physics primitives, event system, temporal state,
   branching/replay, materials, environmental fields, geometry, the
   simulation compiler, WorldIR, ArtifactStore — the mission's core list.

## 3. Rules adopted now (preventive)

- **R1 (boundary rule):** the world-platform chain
  (`world_ir, evidence, reconstruction, perception, exporters, sdk`) must
  not import simulation domains
  (`engine.physics.{fire,destruction}, engine.environment,
  engine.disasters`) — consumed via the physics compiler bridge
  (`engine/compiler/physics_compiler.py`) at the *simulation* boundary
  instead. **Enforced by `tests/test_platform_boundary.py`** (added this
  pass; the four math/logging warts above are explicitly allowlisted with
  follow-up notes).
- **R2:** empty scaffolds under `engine/` (disasters, physics/disasters,
  physics/weather, physics/fluids, …) may only ever receive generic
  primitives. A future `engine/disasters/evacuation.py` is rejected
  regardless of convenience (constitution Article XI) — it belongs in
  `reality-apps/disaster/workflows/`.
- **R3:** application packages must never appear in pyproject's package
  list until they exist; when created, `reality` core must install and
  test green with the application package absent.

## 4. Follow-ups registered

- F1: extract `math3` (Quat/Vec3) into a dependency-free module (or
  accept + document the small core); un-warts items 1–2.
- F2: move `get_logger` usage in evidence to a local logging shim or
  accept + document (item 3).
- F3: relocate `world_ir/entity_reid.py`'s spatial-index use behind an
  interface, or accept + document (item 4).

F1–F3 are hygiene, not separation: they are queued in TASKS.yaml notes,
not executed in this pass (change minimization; no behavior change was
requested or needed).

## 5. Verification of this pass

- Inventory: repo-wide greps + module reads (commands in session log).
- New enforcement test `tests/test_platform_boundary.py` green.
- Full suite: 1,392 passed / 1 skipped, zero regressions (pre-existing
  SAM env failures deselected).
- No runtime code changed other than the new test: docs/plan + test only.
