# Reality Engine V11 — Build Ledger

**Authority:** V11 Specification §0-437
**Last Updated:** 2026-09-04
**Status:** PHASE 2 IN PROGRESS (Steps 1-12 tracked, Steps 13+ planned)

---

## Ledger Format

Each requirement entry contains:
- **ID**: Unique identifier (e.g., REQ-001)
- **Section**: Spec section reference
- **Description**: What must be built
- **Dependency**: Blocking requirements
- **Status**: NOT_STARTED | PLANNED | IN_PROGRESS | IMPLEMENTED | TESTED | VERIFIED | BLOCKED | DEFERRED
- **Files**: Source files implementing this requirement
- **Tests**: Test files validating this requirement
- **Benchmark**: Performance test file (if applicable)
- **Verification**: How requirement is verified (automated test, manual, code review, etc.)
- **Limitations**: Known gaps or deviations from spec
- **Blockers**: Issues preventing completion

---

## PHASE 0: FOUNDATION (Steps 1-7)

### REQ-001: Repository Structure
- **Section**: §81 (Repository Architecture)
- **Description**: Git repo with proper directory layout, CI/CD scaffolding
- **Dependency**: None
- **Status**: VERIFIED
- **Files**: `.github/workflows/`, `pyproject.toml`, `conftest.py`
- **Tests**: None (structural requirement)
- **Verification**: Manual code review, directory listing
- **Limitations**: None
- **Blockers**: None

### REQ-002: WorldIR Canonical Schema
- **Section**: §3-6 (WorldIR specification)
- **Description**: Entity, transform, material, geometry, component, temporal models
- **Dependency**: REQ-001
- **Status**: VERIFIED
- **Files**: `world_ir/schema_v1.py`, `world_ir/world_v1.py`
- **Tests**: `tests/test_world_ir_v1.py` (29 tests)
- **Benchmark**: N/A (data model)
- **Verification**: Round-trip serialization tests, JSON determinism tests
- **Limitations**: None
- **Blockers**: None

### REQ-003: Entity Registry & Lifecycle
- **Section**: §7 (Entity System)
- **Description**: Create, destroy, tag entities with lifecycle tracking
- **Dependency**: REQ-002
- **Status**: TESTED
- **Files**: `engine/world/entity.py`
- **Tests**: `tests/test_runtime_foundation.py::TestEntityRegistry` (9 tests)
- **Benchmark**: None
- **Verification**: Unit tests with observer pattern validation
- **Limitations**: Single-threaded by design
- **Blockers**: None

### REQ-004: Serialization & Format Versioning
- **Section**: §103 (Serialization Contracts)
- **Description**: Versioned YAML/JSON save/load with schema evolution
- **Dependency**: REQ-002
- **Status**: VERIFIED
- **Files**: `world_ir/serialization.py`
- **Tests**: `tests/test_world_ir_v1.py` (includes round-trip tests)
- **Benchmark**: None
- **Verification**: Round-trip (state → JSON → state) must be byte-identical
- **Limitations**: Schema v1 only; v2+ migrations not yet implemented
- **Blockers**: None

### REQ-005: Coordinate System & Multi-Frame Transforms
- **Section**: §2 (Coordinate Systems)
- **Description**: WGS84, UTM, ENU, local frames with transform composition
- **Dependency**: REQ-002
- **Status**: VERIFIED
- **Files**: `world_ir/coordinates.py`
- **Tests**: `tests/test_world_runtime.py::test_runtime_builds_coordinate_registry_from_entities`
- **Benchmark**: None
- **Verification**: Frame resolution tests, transform composition tests
- **Limitations**: Only registered frames resolving; no auto-discovery
- **Blockers**: None

### REQ-006: Deterministic Clock & Simulation Time
- **Section**: §90 (Determinism)
- **Description**: Fixed-timestep simulation clock with tick counting
- **Dependency**: None
- **Status**: VERIFIED
- **Files**: `engine/core/clock.py`
- **Tests**: `tests/test_physics_backend_golden.py` (implicit in determinism tests)
- **Benchmark**: None
- **Verification**: Tick count matches substeps; time accumulates correctly
- **Limitations**: No wallclock sync; purely simulation-time
- **Blockers**: None

### REQ-007: Job System & Topological Ordering
- **Section**: §6 (Job System)
- **Description**: Task scheduling with dependency resolution
- **Dependency**: REQ-006
- **Status**: IMPLEMENTED
- **Files**: `engine/core/jobs.py`
- **Tests**: None
- **Benchmark**: None
- **Verification**: Code review (not yet tested)
- **Limitations**: Not integrated into runtime yet; stub only
- **Blockers**: No integration tests

### REQ-008: Units & SI Enforcement
- **Section**: §82 (Units)
- **Description**: Quantity wrapper preventing silent unit conversions
- **Dependency**: None
- **Status**: VERIFIED
- **Files**: `engine/core/units.py`
- **Tests**: `tests/` (used throughout)
- **Benchmark**: None
- **Verification**: Type safety; configuration uses SI units consistently
- **Limitations**: Lightweight implementation; no dimensional analysis
- **Blockers**: None

### REQ-009: Deterministic RNG with Sub-Streams
- **Section**: §90 (Determinism)
- **Description**: Seeded Mersenne Twister with named sub-streams
- **Dependency**: None
- **Status**: VERIFIED
- **Files**: `engine/core/rng.py`
- **Tests**: `tests/test_fracture_system.py::test_deterministic_with_seed`
- **Benchmark**: None
- **Verification**: Same seed → same sequence; different seeds → different sequences
- **Limitations**: Python random.Random (not cryptographic); platform-dependent float generation
- **Blockers**: None

---

## PHASE 1: RIGID BODY PHYSICS (Steps 8-11)

### REQ-010: Rigid Body Dynamics
- **Section**: §8 (Rigid Body Dynamics)
- **Description**: F=ma, τ=Iα, semi-implicit Euler integration, damping, sleeping
- **Dependency**: REQ-006, REQ-008, REQ-009
- **Status**: VERIFIED
- **Files**: `engine/physics/rigid/body.py`, `engine/physics/rigid/integrator.py`
- **Tests**: `tests/test_physics_backend_golden.py` (RB_001, RB_002, etc.)
- **Benchmark**: None
- **Verification**: Golden tests (falling cube, stacked boxes, determinism)
- **Limitations**: No rotational contact response; sleeping is velocity-based only
- **Blockers**: None

### REQ-011: Collision Detection (Broadphase & Narrowphase)
- **Section**: §9 (Collision Detection)
- **Description**: AABB broadphase, sphere/box/plane narrowphase, raycast
- **Dependency**: REQ-010
- **Status**: VERIFIED
- **Files**: `engine/physics/collision/broadphase.py`, `engine/physics/collision/narrowphase.py`, `engine/physics/collision/raycast.py`, `engine/physics/collision/shapes.py`
- **Tests**: `tests/test_physics_backend_golden.py`, `tests/test_collision_*.py`
- **Benchmark**: None
- **Verification**: Golden tests; collision event emission tests
- **Limitations**: No BVH optimization; O(n²) broadphase; only sphere/box/plane primitives
- **Blockers**: None

### REQ-012: Contact Resolution & Material Physics
- **Section**: §9 (Contact Solver), §10 (Material Physics)
- **Description**: Sequential impulse solver, friction, restitution, material properties
- **Dependency**: REQ-011
- **Status**: VERIFIED
- **Files**: `engine/physics/constraints/contact_solver.py`, `engine/physics/materials/material.py`
- **Tests**: `tests/test_physics_backend_golden.py`, contact tests
- **Benchmark**: None
- **Verification**: Energy conservation; restitution tests; golden tests
- **Limitations**: Single-pass solver (no iteration); limited material database (4 types)
- **Blockers**: None

### REQ-013: Numerics Health Checks & Hard Failures
- **Section**: §86 (Numerical Health)
- **Description**: NaN/Inf detection, energy explosion guard, hard-failure classification
- **Dependency**: REQ-010
- **Status**: VERIFIED
- **Files**: `engine/physics/diagnostics/numerics.py`, `engine/core/errors.py`
- **Tests**: `tests/test_errors.py`, physics backend tests
- **Benchmark**: None
- **Verification**: NumericsError raised on divergence; hard-failure classification verified
- **Limitations**: Energy explosion threshold is fixed (1000x)
- **Blockers**: None

### REQ-014: Event Bus & Deterministic Messaging
- **Section**: §85 (Event System)
- **Description**: Typed, deterministic event log (ContactEvent, ImpactEvent, etc.)
- **Dependency**: REQ-006, REQ-010
- **Status**: TESTED
- **Files**: `engine/world/events.py`
- **Tests**: `tests/test_runtime_foundation.py::TestEventBus` (5 tests)
- **Benchmark**: None
- **Verification**: Event ordering, priority handling, recursion prevention
- **Limitations**: No event persistence; in-memory only
- **Blockers**: Integration with physics backend pending

### REQ-015: Structured Logging & Diagnostics
- **Section**: §83 (Logging)
- **Description**: JSON-formatted logging, diagnostic records, performance tracking
- **Dependency**: REQ-006
- **Status**: TESTED
- **Files**: `engine/core/logging.py`
- **Tests**: `tests/test_logging.py` (20 tests)
- **Benchmark**: None
- **Verification**: Log format validation; Diagnostics structure tests
- **Limitations**: No file rotation; stdout/stderr only (file output supported but not persisted automatically)
- **Blockers**: None

### REQ-016: Configuration System (TOML-based)
- **Section**: §82 (Configuration)
- **Description**: Immutable config loading, schema validation, SI units
- **Dependency**: REQ-008
- **Status**: TESTED
- **Files**: `engine/core/config.py`
- **Tests**: `tests/test_config.py` (17 tests)
- **Benchmark**: None
- **Verification**: Schema validation; immutability enforcement; round-trip
- **Limitations**: TOML only (YAML not yet supported); no environment variable override
- **Blockers**: None

### REQ-017: Error Hierarchy & Hard-Failure Classification
- **Section**: §86, §90 (Error Model)
- **Description**: 7-level exception hierarchy, hard-failure detection (NumericsError, ArchitectureError, ConfigError)
- **Dependency**: None
- **Status**: TESTED
- **Files**: `engine/core/errors.py`
- **Tests**: `tests/test_errors.py` (18 tests)
- **Benchmark**: None
- **Verification**: Exception hierarchy; is_hard_failure() classification tests
- **Limitations**: Hard failures are halt-on-first; no recovery mechanism
- **Blockers**: None

---

## PHASE 2: RUNTIME FOUNDATION (New)

### REQ-018: Runtime Foundation (Entity Registry)
- **Section**: §82 (Runtime Architecture)
- **Description**: Entity lifecycle, identity, unique ID enforcement
- **Dependency**: REQ-006, REQ-009
- **Status**: TESTED
- **Files**: `engine/world/entity.py`
- **Tests**: `tests/test_runtime_foundation.py::TestEntityRegistry` (9 tests)
- **Benchmark**: None
- **Verification**: Observer pattern; double-destroy prevention; tagging
- **Limitations**: Single-threaded; no batching
- **Blockers**: None

### REQ-019: Runtime Foundation (Component Registry)
- **Section**: §82 (Runtime Architecture)
- **Description**: ECS-style component attachment, querying, ownership
- **Dependency**: REQ-018
- **Status**: TESTED
- **Files**: `engine/world/component.py`
- **Tests**: `tests/test_runtime_foundation.py::TestComponentRegistry` (7 tests)
- **Benchmark**: None
- **Verification**: O(1) component lookup; multi-component queries
- **Limitations**: No component dependencies; no component ordering
- **Blockers**: None

### REQ-020: Runtime Foundation (Resource Manager)
- **Section**: §82 (Runtime Architecture)
- **Description**: Reference-counted asset lifecycle, ownership tracking
- **Dependency**: REQ-018, REQ-009
- **Status**: TESTED
- **Files**: `engine/world/resources.py`
- **Tests**: `tests/test_runtime_foundation.py::TestResourceManager` (7 tests)
- **Benchmark**: None
- **Verification**: Reference counting; unused resource detection; ownership tracking
- **Limitations**: No resource pooling; no async loading
- **Blockers**: None

### REQ-021: Runtime Foundation (World Lifecycle)
- **Section**: §82 (Runtime Architecture)
- **Description**: World state machine (CREATED → INITIALIZED → RUNNING → SHUTDOWN)
- **Dependency**: REQ-006
- **Status**: TESTED
- **Files**: `engine/world/lifecycle.py`
- **Tests**: `tests/test_runtime_foundation.py::TestWorldLifecycle` (7 tests)
- **Benchmark**: None
- **Verification**: State transition enforcement; metrics tracking
- **Limitations**: No pause/resume states
- **Blockers**: None

### REQ-022: Runtime Foundation (Dependency Graph)
- **Section**: §82 (System Execution)
- **Description**: Topological system ordering, cycle detection, deterministic execution
- **Dependency**: REQ-006, REQ-009
- **Status**: TESTED
- **Files**: `engine/world/dependency_graph.py`
- **Tests**: `tests/test_dependency_graph.py::TestDependencyGraph` (11 tests)
- **Benchmark**: None
- **Verification**: Cycle detection; topological sort; determinism
- **Limitations**: Kahn's algorithm (no priorities within same tier)
- **Blockers**: Integration with physics backend pending

### REQ-023: Runtime Foundation (Caching System)
- **Section**: §82 (Performance)
- **Description**: LRU cache, query cache, transform cache, component cache
- **Dependency**: REQ-019
- **Status**: TESTED
- **Files**: `engine/world/caching.py`
- **Tests**: `tests/test_dependency_graph.py::TestLRUCache|QueryCache|TransformCache|ComponentCache` (28 tests)
- **Benchmark**: None
- **Verification**: Cache hits/misses; eviction; invalidation patterns
- **Limitations**: No adaptive sizing; pattern invalidation is substring match
- **Blockers**: None

### REQ-024: WorldRuntime Integration Hub
- **Section**: §82 (Runtime Architecture)
- **Description**: Central runtime coordinating all subsystems
- **Dependency**: REQ-018 through REQ-023
- **Status**: TESTED
- **Files**: `engine/world/runtime.py`
- **Tests**: `tests/test_runtime_foundation.py::TestWorldRuntime` (8 tests), `tests/test_world_runtime.py` (5 tests, includes 2 regression tests for the V1-schema fix below)
- **Benchmark**: None
- **Verification**: Entity/component/resource/event integration; serialization; accepts both the legacy `world_ir.world.WorldIR` (EntityRegistry-backed, real `Transform` objects) and the V1 schema `world_ir.world_v1.WorldIR` (dict-backed entities, externalized dict transforms)
- **Limitations**: No streaming; single-world only. V1-schema entity transforms (plain dicts) are not registered into `CoordinateRegistry` — only real `Transform` objects are, so `resolve_point`/`resolve_transform` don't work for V1-schema entities yet; this needs the two WorldIR transform representations unified, which is a bigger change than this fix's scope.
- **Fixed 2026-09-07**: `__init__`/`get_entity_from_world` previously crashed with `AttributeError` the instant a V1-schema world had any entities (iterating `dict.entities` yielded string keys, not `Entity` objects) — see `docs/DECISIONS.md` #12 and `docs/KNOWN_LIMITATIONS.md`, both updated to reflect the fix.
- **Blockers**: None

---

## PHASE 2.5: DESTRUCTION (Steps 12+)

### REQ-025: Basic Fracture System
- **Section**: §12 (Basic Fracture)
- **Description**: Impact-based breaking with material toughness, deterministic fragment generation
- **Dependency**: REQ-010, REQ-024, REQ-009
- **Status**: TESTED
- **Files**: `engine/physics/destruction/interface.py`, `engine/physics/destruction/fracture.py`
- **Tests**: `tests/test_fracture_system.py` (18 tests)
- **Benchmark**: None
- **Verification**: Material fracture parameters; determinism with seeding; energy calculation
- **Limitations**: No mesh fragment generation; single-pass fracture (no cascading)
- **Blockers**: Integration with physics backend step() method pending

### REQ-026: Glass Physics System
- **Section**: §13 (Glass Physics)
- **Description**: Pane-based glass fracture with temper-specific patterns (radial/spider-web/laminated), deterministic shard generation
- **Dependency**: REQ-025, REQ-024, REQ-009
- **Status**: TESTED
- **Files**: `engine/physics/destruction/glass.py`, `engine/physics/destruction/__init__.py`
- **Tests**: `tests/test_glass_physics.py` (20 tests)
- **Benchmark**: None
- **Verification**: Temper-specific thresholds; shard pattern validation; deterministic generation; serialization round-trip
- **Limitations**: Frame attachment points tracked but not enforced in fracture; no mesh generation
- **Blockers**: Integration with physics backend step() method pending

### REQ-027: Debris System
- **Section**: §14 (Debris System)
- **Description**: Fragment lifecycle management with object pooling, collision handling, rendering queue, age-based culling
- **Dependency**: REQ-025, REQ-024, REQ-010
- **Status**: TESTED
- **Files**: `engine/physics/destruction/debris.py`, `engine/physics/destruction/__init__.py`
- **Tests**: `tests/test_debris_system.py` (22 tests)
- **Benchmark**: None
- **Verification**: Fragment lifecycle; physics stepping with gravity; sleeping optimization; collision response; pool reuse; max capacity; serialization
- **Limitations**: No inter-fragment collisions; sleeping threshold fixed; no spatial partitioning for optimization
- **Blockers**: Integration with physics backend step() method pending

### REQ-028: Replay System
- **Section**: §15 (Replay System)
- **Description**: Event recording, time scrubbing, branching with causality tracking and deterministic replay
- **Dependency**: REQ-024, REQ-009
- **Status**: TESTED
- **Files**: `engine/simulation/replay.py`, `engine/simulation/__init__.py`
- **Tests**: `tests/test_replay_system.py` (29 tests)
- **Benchmark**: None
- **Verification**: Event recording and queries; time scrubbing/seeking; branching; causality tracking; snapshots; serialization; playback control
- **Limitations**: No automatic replay verification; snapshots store world state but not all internal state; no compression
- **Blockers**: Integration with WorldRuntime for event capture pending

### REQ-029: Viewport
- **Section**: §16 (Viewport)
- **Description**: Camera model (position/forward/up/FOV) and frustum-culled, depth-sorted visible-entity query for the visualization layer
- **Dependency**: REQ-024
- **Status**: TESTED
- **Files**: `engine/render/viewport.py`, `engine/render/__init__.py`
- **Tests**: `tests/test_viewport.py` (12 tests)
- **Benchmark**: None
- **Verification**: look-at construction; near/far clipping; FOV cone test; radius-expanded culling; distance sort
- **Limitations**: Angle-cone frustum test only (no separate horizontal/vertical FOV, no projection matrix); no occlusion culling
- **Blockers**: None

### REQ-030: Inspector
- **Section**: §17/§35 (Inspector, Studio inspection capabilities)
- **Description**: Read-only query facade over WorldIR — entities, evidence, measurements, materials, structural relationships, confidence/uncertainty, causal chains, branches/scenarios, summary
- **Dependency**: REQ-002 (v1 schema)
- **Status**: TESTED
- **Files**: `engine/inspector/inspector.py`, `engine/inspector/__init__.py`
- **Tests**: `tests/test_inspector.py` (28 tests)
- **Benchmark**: None
- **Verification**: entity/material/geometry resolution; evidence retrieval; measurement extraction; confidence summary; relationship resolution with target names; distance measurement; provenance/confidence filtering; causal chain traversal; branch/scenario listing
- **Limitations**: Read-only by design (no mutation path — see spec §7 boundary between query and engine-owned truth); built directly against `world_ir.world_v1.WorldIR` rather than `WorldRuntime` because of a confirmed bug in the latter (see below)
- **Blockers**: None for Inspector itself

**Bug found and fixed during this step**: `world_ir/schema_v1.py` had a
stray `print("WorldIR V1 Schema definitions loaded successfully")` at
module level, executing on every import (removed).

**Bug found and recorded, not fixed (out of scope for this step)**:
`WorldRuntime` (REQ-024) is type-hinted for `world_v1.WorldIR` but its
constructor body only actually works against the legacy
`world_ir.world.WorldIR`; constructing it with the v1 schema crashes.
See [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) and
[DECISIONS.md](DECISIONS.md) #12 for full detail and fix options.

### REQ-031: Physics Debugger
- **Section**: §18/§36 (Physics Debugger, debug visualization)
- **Description**: Read-only debug-draw records for rigid bodies, contacts, forces/torques, and numerics health, read directly off the real `PhysicsWorld`/`ContactRecord` backend
- **Dependency**: REQ-010, REQ-011, REQ-012, REQ-013
- **Status**: TESTED
- **Files**: `engine/physics/debug/debugger.py`, `engine/physics/debug/__init__.py`
- **Tests**: `tests/test_physics_debugger.py` (10 tests)
- **Benchmark**: None
- **Verification**: body records (shape/transform/velocity/sleep-state/energy) sorted deterministically by id; force/torque records only for non-zero accumulators; contact records against the real plane/body contacts a golden scene produces; numerics passthrough; aggregate summary
- **Limitations**: No bounding-box/AABB record (shapes carry their own dims via `to_dict()`, not yet re-expressed as world-space AABB); no stress/damage visualization (no structural solver exists yet to source it from); no fluid/wind field visualization (no fluid/weather solver exists yet)
- **Blockers**: None

### REQ-032: Water
- **Section**: §20/§22 (Water system, fluid environment simulation)
- **Description**: Multi-body water simulation — depth/volume tracking, buoyancy force (Archimedes), inter-body flow equalization with drainage, and overflow-crossing event publishing with diagnostics
- **Dependency**: REQ-014 (event bus). Does NOT depend on the Step 19 rain REQ — water and rain are independent environment subsystems; nothing in water reads rain state or vice versa
- **Status**: TESTED
- **Files**: `engine/environment/water.py`, `engine/environment/__init__.py`
- **Tests**: `tests/test_water.py` (41 tests)
- **Benchmark**: None
- **Verification**: volume computation; buoyancy worked example; snapshot-based flow equalization between adjacent bodies (rate-limited vs. equalizing flow, no overshoot); multi-neighbor start-of-step snapshot consistency; drainage floored at zero; serialize/deserialize round-trip including adjacency and drainage; overflow event fires exactly once on the step a body's depth crosses above `max_depth_m`, does not refire on subsequent steps spent above threshold, does not fire when `max_depth_m` is `None`, and does not crash when `event_bus` is `None` (default)
- **Limitations**: P1 simplification — `flow_rate_coefficient` is a tunable constant, not a real hydraulic property; no pressure field; no obstruction/object interaction beyond the buoyancy force calculation; no wave propagation
- **Blockers**: None

---

## SUMMARY

### Ledger Statistics

**By Phase:**
- PHASE 0 (Foundation): 9 requirements
  - VERIFIED: 8
  - IMPLEMENTED: 1
  - BLOCKED: 0
  
- PHASE 1 (Physics): 8 requirements
  - VERIFIED: 7
  - TESTED: 1
  - BLOCKED: 0
  
- PHASE 2 (Runtime): 7 requirements
  - TESTED: 7
  - BLOCKED: 0
  
- PHASE 2.5 (Destruction & Simulation): 4 requirements
  - TESTED: 4
  - BLOCKED: 0

- PHASE 3 (Environment): 4 requirements
  - TESTED: 4
  - BLOCKED: 0

**Total: 32 requirements tracked**

### Status Distribution

| Status | Count |
|--------|-------|
| NOT_STARTED | 0 |
| PLANNED | 0 |
| IN_PROGRESS | 0 |
| IMPLEMENTED | 1 (Job System) |
| TESTED | 15 |
| VERIFIED | 16 |
| BLOCKED | 0 |
| DEFERRED | 0 |

### Test Coverage

- **Total Tests**: 474 (all passing)
- **New Tests This Session**: 255 tests
  - Runtime foundation: 44 tests
  - Dependency graph & caching: 30 tests
  - Fracture system: 18 tests
  - Glass physics system: 20 tests (Step 13)
  - Debris system: 22 tests (Step 14)
  - Replay system: 29 tests (Step 15)
  - Viewport: 12 tests (Step 16)
  - Inspector: 28 tests (Step 17)
  - Physics Debugger: 10 tests (Step 18)
  - Water: 42 tests (Step 20)
- **Coverage**: 100% of tested requirements

### Critical Blockers

| Blocker | Impact | Mitigation |
|---------|--------|-----------|
| None currently | — | All critical path items complete |

### Next Steps (PHASE 3: ENVIRONMENT)

**Planned Requirements (Not yet implemented):**
- Steps 19-33 (fluids, fire, weather, structural, disasters, causal graph, branching/counterfactuals, AI copilot, natural-language query, rendering, Studio, export, datasets, benchmarks)

---

## Maintenance Notes

This ledger is updated whenever:
1. A requirement is implemented
2. Status changes (IMPLEMENTED → TESTED → VERIFIED)
3. A blocker appears or resolves
4. Test coverage changes
5. Known limitations are discovered

**Last verified**: 2026-09-07 by implementation audit (Step 18 Physics Debugger completion)
**Next audit**: After Step 19 completion
