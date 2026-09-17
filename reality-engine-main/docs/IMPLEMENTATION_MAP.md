# Reality Engine V10 — Implementation Map

**Document Authority:** `REALITY_ENGINE_MASTER_SPECIFICATION_V10_WORLD_PHYSICS_AND_AGENT_BUILD_BIBLE.md` (§0-437)

**Mission:** Convert authorized real-world evidence into a persistent, semantically understood, physically coherent, temporally versioned, game-engine-native world.

**Core Rule:** THE WORLD IS NOT JUST BEAUTIFUL. THE WORLD IS INTERNALLY CONSISTENT.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Four-Plane Architecture](#four-plane-architecture)
3. [Complete Subsystem Inventory](#complete-subsystem-inventory)
4. [Build Order & Phases](#build-order--phases)
5. [Dependency Graph](#dependency-graph)
6. [Canonical Data Models](#canonical-data-models)
7. [Core Interfaces & Contracts](#core-interfaces--contracts)
8. [Architectural Invariants](#architectural-invariants)
9. [Testing Requirements](#testing-requirements)
10. [Performance Gates](#performance-gates)
11. [Security & Trust Model](#security--trust-model)
12. [Risk Register](#risk-register)
13. [Known Ambiguities & Engineering Decisions Required](#known-ambiguities--engineering-decisions-required)

---

## Executive Summary

Reality Engine is a **world compiler and physics-powered simulation platform** designed to:

- **Ingest real-world evidence** (photos, video, LiDAR, sensor data) and convert it to a canonical world representation
- **Run multi-fidelity physics** (P0 visual → P4 engineering reference)
- **Enable destruction and disaster simulation** with causal coupling (rain→flood→damage→fire→collapse→debris)
- **Export** to Unreal, Blender, game engines, and GIS systems without rebuilding
- **Replay deterministically** with versioned state and branching

**Scope:** 33 sequential build steps, 4 architectural planes, 50+ physics/environment solvers, deterministic simulation replay, studio UI.

**Timeline:** Steps 1-11 (foundation) are ~1 person-month. Steps 12-33 (destruction/disasters/coupled) require significant additional work.

---

## Four-Plane Architecture

```
REALITY PLANE
  Evidence → Reconstruction → WorldIR (canonical)
         ↓
SIMULATION PLANE
  WorldIR → Physics → Environment → Destruction → Events → Recovery
         ↓
RUNTIME PLANE
  WorldIR → Streaming → Rendering → Interaction → Export
         ↓
INTELLIGENCE PLANE
  WorldIR → Search → Reasoning → Planning → Automation (AI Copilot)
```

**Key insight:** Everything flows through **WorldIR** — a versioned, immutable interchange representation. This is the source of truth that prevents coupling between planes.

---

## Complete Subsystem Inventory

### Foundation Layer (§1-7)
| Subsystem | Purpose | Status | Dependencies |
|-----------|---------|--------|--------------|
| **WorldIR** | Canonical interchange format (YAML/JSON) | Core | — |
| **Entity Registry** | Identity, relationships, lifecycle | Core | WorldIR |
| **Serialization** | Versioned save/load format | Core | Entity Registry |
| **Coordinate System** | Multi-frame transforms (WGS84, ENU, local) | Core | WorldIR |
| **Deterministic Clock** | Fixed-timestep simulation time | Core | — |
| **Job System** | Topologically sorted task execution | Core | Clock |
| **Provenance Graph** | Evidence-first tracking (OBSERVED/RECONSTRUCTED/ESTIMATED/…) | Core | Entity Registry |
| **Units & Quantities** | SI-enforced, no silent conversions | Core | — |

### Physics Layer (§7-30)
| Subsystem | Purpose | P-Tier | Status | Dependencies |
|-----------|---------|--------|--------|--------------|
| **Rigid Body Dynamics** | F=ma, τ=Iα, semi-implicit Euler | P1 | Done (§8) | Clock, Materials |
| **Collision Detection** | Broadphase (AABB), narrowphase (sphere/box/plane) | P1 | Done (§9) | Rigid Body, Geometry |
| **Contact Resolution** | Sequential impulse, friction, restitution, sleeping | P1 | Done (§9) | Collision |
| **Material Physics** | Density, friction, restitution, thermal/acoustic properties | P1-P4 | Partial (§9) | Rigid Body |
| **Explosion System** | Impulse propagation, radial blast, debris generation | P1-P3 | Not started (§10) | Rigid Body, Destruction |
| **Fracture Engine** | Precomputed graphs, runtime cutting, debris | P1-P4 | Not started (§11) | Rigid Body, Materials |
| **Glass Physics** | Thickness, frame attachment, pane fracture, shard gen | P2 | Not started (§12) | Fracture |
| **Debris System** | Parent tracking, lifetime, collision, navigation hazard | P1-P2 | Not started (§14) | Rigid Body, Destruction |
| **Thermal Engine** | Heat conduction, material degradation, ignition | P2-P4 | Not started (§29) | Materials, Fire |
| **Structural System** | Member graphs, load paths, failure cascades, collapse | P2-P4 | Not started (§24) | Rigid Body, Materials |
| **Particle System** | Generic particle dynamics (dust, sparks, debris) | P1-P2 | Not started | Rigid Body |
| **Soft Body / Cloth** | Constrained deformation, tearing | P2-P3 | Not started (§61) | Rigid Body |
| **Rope / Cable** | Length constraints, attachment, snapping | P2 | Not started (§62) | Rigid Body, Constraints |

### Environment & Coupled Systems (§13-68)
| Subsystem | Couples To | Status | Dependencies |
|-----------|-----------|--------|--------------|
| **Fire Engine** | Thermal, Structure, Smoke, Destruction | Not started (§13) | Thermal, Materials, Destruction |
| **Smoke/Gas** | Fire, Physics, Rendering | Not started (§14) | Particle System, Advection |
| **Rain Engine** | Hydrology, Rendering, Audio | Not started (§15) | Particle System, Climate |
| **Flood/Water** | Hydrology, Buoyancy, Navigation, Destruction | Not started (§16) | Particle System, Terrain |
| **Wind/Atmosphere** | Objects, Particles, Audio, Erosion | Not started (§23) | Field API, Particle System |
| **Hydrology** | Water, Terrain, Flood, Drainage, Saturation | Not started (§20) | Terrain, Water |
| **Composites:** | | |
| — Earthquake + Fire (§66) | Pipe rupture → gas → ignition → fire | Complex (§27) | Earthquake, Fire, Structural |
| — Hurricane + Flood + Structure (§67) | Wind → rain → surge → roof damage → glass → debris | Very Complex (§28) | Wind, Flood, Structure, Destruction |
| — Tornado + Debris (§68) | Wind field → detachment → acceleration → impact | Complex (§29) | Wind, Debris, Rigid Body |

### Disaster & Event Systems (§85-98)
| Subsystem | Purpose | Status | Dependencies |
|-----------|---------|--------|--------------|
| **Event Bus** | Typed, deterministic event log (ContactEvent, ImpactEvent, etc.) | Done (§11 built) | Clock, Simulation |
| **Causal Graph** | "Why did X happen?" — traces event chains | Not started (§85) | Event Bus, Provenance |
| **Scenario Composer** | Parameterized disaster definition and execution | Not started (§69) | Physics, All Disaster Solvers |
| **Replay System** | Deterministic state snapshots + event log playback | Not started (§35) | Event Bus, Serialization, Determinism |

### UI & Editing (§111-125)
| Subsystem | Purpose | Status | Dependencies |
|-----------|---------|--------|--------------|
| **Viewport** | 3D rendering of WorldIR entities | Not started (§16) | Rendering, Streaming |
| **Inspector** | Entity properties, evidence, confidence, provenance | Not started (§17) | Entity Registry, Provenance |
| **Measurement Tools** | Distance, height, angle; OBSERVED/ESTIMATED tracking | Implied (§70) | Entity Registry, Uncertainty |
| **Physics Debugger** | Contact visualization, force vectors, profiling | Not started (§18) | Physics, Rendering |
| **Timeline UI** | Temporal scrubbing, branch visualization | Implied (§35) | Replay, Event Bus |
| **Disaster Composer UI** | Parameter sliders, scenario execution | Not started (§69) | Scenario Composer |

### Ingestion & Reconstruction (§42, §109-114)
| Subsystem | Purpose | Status | Dependencies |
|-----------|---------|--------|--------------|
| **Calibration** | Camera intrinsics, distortion models | Not started (§42) | — |
| **Feature Detection/Matching** | SfM pipeline feeder | Not started | — |
| **Structure-from-Motion (SfM)** | Sparse camera poses and 3D points | Not started | Feature Detection |
| **Multi-View Stereo (MVS)** | Dense depth reconstruction | Not started | SfM |
| **Meshing** | Surface reconstruction from point clouds | Not started | MVS |
| **Registration** | Geo-alignment (to WGS84/UTM) | Not started | Meshing, Geodesy |
| **Semantic Segmentation** | Label assignment (building/tree/road/…) | Not started | Perception models |
| **Material Estimation** | Physical property inference from appearance | Not started | Perception, Material Physics |
| **One-Click World Build** | Auto-pipeline orchestration | Not started (§112) | All of above |

### Export & Integration (§54-56, §104-106)
| Subsystem | Purpose | Status | Dependencies |
|-----------|---------|--------|--------------|
| **Unreal Integration** | Actor/component mapping, physics asset export | Not started (§56) | WorldIR, Serialization |
| **Blender Integration** | Collection organization, material baking, custom properties | Not started (§55) | WorldIR, Materials |
| **OpenUSD Adapter** | Scene composition, geometry/physics schemas | Not started (§55) | WorldIR, Geometry |
| **GIS Export** | Georeferenced layers, shapefile/GeoJSON | Not started | Coordinates, Geodesy |
| **Game Engine Neutral** | Scene format (glTF-inspired) with metadata | Planned (§54) | WorldIR |
| **Python API** | SDK for scripting, batch operations | Not started (§105) | All of above |
| **C++ API** | High-performance embedding | Not started (§106) | All of above |

### Testing & Validation (§91-107)
| Subsystem | Purpose | Status | Dependencies |
|-----------|---------|--------|--------------|
| **Golden Test Suite** | RB_001 (falling cube), RB_002 (stacked boxes), …RB_NNN | Partial (RB_001/002) (§91) | Rigid Body, Collision |
| **Numerical Health Checks** | NaN/Inf detection, energy explosion guard (§86) | Done (§9) | Physics solvers |
| **Conservation Contracts** | Momentum, energy, mass conservation assertions (§87) | Implied | Physics solvers |
| **Determinism Tests** | Identical seed → byte-identical state (§90) | Done (§9) | All solvers, RNG |
| **Serialization Tests** | Round-trip (state → JSON → state) matches (§103) | Done (§4) | Serialization |
| **Benchmark Harness** | Timing, throughput, memory, GPU cost (§91-95) | Scaffolded | Physics, Rendering |
| **Regression Suite** | Metrics vs. baseline (step count, energy, contacts) | Proposed (§95) | Golden Tests, Benchmark |
| **Visual Regression** | Screenshot diff (viewport rendering) | Proposed (§94) | Viewport, Rendering |

### AI & Query (§77-78, §140-142)
| Subsystem | Purpose | Status | Dependencies |
|-----------|---------|--------|--------------|
| **World Query Language** | `SELECT WHERE`, `MEASURE`, `TRACE CAUSE`, `COMPARE` | Planned (§78) | Event Bus, Causality |
| **AI Copilot** | Natural language → intent compilation | Planned (§77) | World Query Language |
| **Navigation** | Traversability under disaster conditions | Not started (§139) | Terrain, Physics, Flooding |

---

## Build Order & Phases

### §108 FIRST IMPLEMENTATION ORDER (Canonical Sequence)

```
PHASE 0: FOUNDATION (Steps 1-7)
├─ 1. repository
├─ 2. WorldIR
├─ 3. entity system
├─ 4. serialization
├─ 5. coordinate library
├─ 6. job system
└─ 7. deterministic clock

PHASE 1: RIGID BODY PHYSICS (Steps 8-11)
├─ 8. rigid body backend
├─ 9. collision
├─ 10. material physics
└─ 11. event bus

PHASE 2: DESTRUCTION (Steps 12-18)
├─ 12. basic fracture
├─ 13. glass
├─ 14. debris
├─ 15. replay
├─ 16. viewport
├─ 17. inspector
└─ 18. physics debugger

PHASE 3: SINGLE PHENOMENA (Steps 19-24)
├─ 19. rain
├─ 20. water
├─ 21. fire
├─ 22. smoke
├─ 23. wind
└─ 24. structural graph

PHASE 4: DISASTER SOLVERS (Steps 25-30)
├─ 25. explosion
├─ 26. flood
├─ 27. earthquake
├─ 28. hurricane
├─ 29. tornado
└─ 30. remaining disasters

PHASE 5: INTEGRATION & EXPORT (Steps 31-33)
├─ 31. coupled scenarios
├─ 32. world reconstruction
└─ 33. game export
```

**Current Progress:** Steps 1-11 complete (as of Sep 2026). Steps 12-33 not started.

**Rationale for Order:**
- §108: "Do NOT begin with hurricanes" — foundations first, then isolated physics, then coupling.
- Each step is independently testable (§80: unit tests + integration tests + serialization + determinism + diagnostics).
- Coupled scenarios (step 31) depend on all base solvers (steps 19-30) working independently first.

---

## Dependency Graph

### Tier 0: Mandatory Foundations (must exist before anything else)
```
Units & Quantities
    ↓
WorldIR (entity schema, transform, bounds)
    ↓
Provenance (OBSERVED/RECONSTRUCTED/ESTIMATED tracking)
    ↓
Serialization (versioned save/load)
    ↓
Coordinate System (frame transforms)
```

### Tier 1: Core Simulation Infrastructure
```
Deterministic Clock
    ↓
Job System (depends on Clock for deterministic ordering)
    ↓
RNG (seeded, deterministic sub-streams)
```

### Tier 2: Physics Foundation
```
Rigid Body Dynamics (depends on Clock, Units, Job System)
    ↓
Collision Detection (depends on Rigid Body, Geometry)
    ↓
Contact Resolution (depends on Collision, Materials)
    ↓
Material Physics (full spec depends on Rigid Body)
    ↓
Event Bus (emits on contact, depends on Clock, Physics)
```

### Tier 3: Destruction & Fracture
```
Rigid Body + Contact Resolution
    ↓
Fracture Engine (depends on Rigid Body, Materials, Destruction)
    ↓
Glass Physics (depends on Fracture)
    ↓
Debris System (depends on Rigid Body, Fracture, Destruction)
```

### Tier 4: Environment & Coupled Solvers
```
Rigid Body Physics
    ↓
Particle System (generic; depends on Rigid Body)
    ↓
┌─ Rain (Particle System → Hydrology)
├─ Wind (Particle System → Objects)
├─ Fire (Thermal + Smoke → Destruction)
├─ Water (Hydrology → Buoyancy + Destruction)
└─ Structural (Load + Thermal → Capacity → Failure)

Tier 4.1: Composite Couplings
├─ Earthquake + Fire (Pipe rupture → Gas → Ignition)
├─ Hurricane + Flood + Structure (Wind → Rain → Surge → Damage)
└─ Tornado + Debris (Wind field → Detachment → Impact)
```

### Tier 5: Simulation & Export
```
All Physics + Environment + Coupled Solvers
    ↓
Scenario Composer (parameterized disaster)
    ↓
Replay System (deterministic event log + snapshot recovery)
    ↓
Causal Graph (traces "why did X happen?")
    ↓
Reconstruction Pipeline (photos → WorldIR)
    ↓
Export Adapters (Unreal, Blender, OpenUSD, GIS)
```

---

## Canonical Data Models

### WorldIR (§6)
```yaml
world:
  id: world-id-uuid
  version: revision-number
  coordinate_system: WORLD_FRAME
  entities: Entity[]          # id-indexed entity registry
  terrain: Terrain            # elevation grid, material layers
  environment: Environment    # weather state, atmospheric
  physics: PhysicsState       # active solvers, simulation config
  materials: Material[]       # canonical material properties
  semantics: SemanticLabels[] # instance segmentation
  observations: Observation[] # sensor-derived facts
  sessions: Session[]         # capture metadata
  temporal_state: TemporalModel
  provenance: ProvenanceGraph
  uncertainty: UncertaintyRecords[]
  branches: Branch[]          # simulation forks
  schema_version: 1
```

### Entity (§6)
```yaml
entity:
  id: entity-uuid
  type: BUILDING | WALL | FLOOR | ROOF | DOOR | WINDOW | VEHICLE | TREE | ...
  parent_id: entity-uuid-or-null  # hierarchy
  transform:
    position: Vec3
    orientation: Quat
    scale: Vec3
  bounds:
    aabb: AABB
    volume: float
  geometry_refs: [GeometryAsset]   # LOD pyramid: mesh, convex hull, primitives
  material_refs: [Material]
  semantic_labels: [SemanticLabel]
  relationships: Relationship[]    # supports, adjacent, contains, damages
  evidence: [EvidenceRef]
  physics:
    rigid_body: RigidBody | null
    collision_shape: CollisionShape
    material: Material
  destruction:
    damage_state: INTACT | DAMAGED | FRACTURED | DETACHED | DEBRIS | REMOVED
    fracture_graph: FractureGraph | null
  temporal_validity:
    created_at: Timestamp
    destroyed_at: Timestamp-or-null
  confidence: 0.0..1.0
  provenance: ProvenanceRef
```

### RigidBody (§8)
```yaml
rigid_body:
  id: body-uuid
  entity_id: entity-uuid
  mass: float              # 0.0 = static/infinite
  center_of_mass: Vec3
  inertia_tensor: Mat3     # diagonal in principal axes
  position: Vec3           # world frame
  orientation: Quat        # world frame
  linear_velocity: Vec3    # world frame
  angular_velocity: Vec3   # body frame
  force_accumulator: Vec3
  torque_accumulator: Vec3
  linear_damping: float
  angular_damping: float
  gravity_scale: float
  collision_shape: CollisionShape
  material: Material
  sleep_state: AWAKE | SLEEPING | STATIC
  sleep_timer: float
```

### MaterialPhysics (§9)
```yaml
material:
  id: material-uuid
  name: string
  # Base
  density: float                    # kg/m³
  friction_static: float            # μ_s
  friction_dynamic: float           # μ_k
  restitution: float                # coefficient of restitution
  
  # Structural (for fracture/collapse)
  youngs_modulus: float             # E, Pa
  poissons_ratio: float             # ν
  tensile_strength: float           # σ_t, Pa
  compressive_strength: float       # σ_c, Pa
  shear_strength: float             # τ, Pa
  fracture_energy: float            # G_f, J/m²
  
  # Thermal (for fire/degradation)
  thermal_conductivity: float       # W/(m·K)
  specific_heat: float              # J/(kg·K)
  ignition_temperature: float       # K
  melting_temperature: float        # K
  
  # Hydrology (for water coupling)
  moisture_capacity: float          # kg_water/m³
  permeability: float               # m/s
  porosity: float                   # %
  
  # Audio
  acoustic_absorption: float        # 0..1
  acoustic_transmission: float      # 0..1
```

### Event (§11, §85)
```yaml
event:
  event_id: string                  # evt-{seed}-{tick:08d}-{sequence:04d}
  type: ContactEvent | ImpactEvent | BreakEvent | FractureEvent | ...
  timestamp: float                  # world time (seconds)
  tick: uint64                      # simulation tick
  source_refs: [entity-id]          # actor(s) that caused this
  target_refs: [entity-id]          # entity/entities affected
  parameters:
    # ContactEvent / ImpactEvent:
    normal: Vec3
    penetration: float
    approach_speed: float
    # BreakEvent:
    fracture_pattern: string
    shard_count: int
    # etc.
  cause_event_ids: [event-id]       # chain causality
  severity: info | warning | error  # or magnitude scale
  confidence: 0.0..1.0
  branch_id: branch-uuid-or-null    # simulation branch this belongs to
  actor_id: string-or-null          # AI agent or user who triggered this
  deterministic_seed: int
  resulting_state_refs: [state-id]  # snapshots resulting from this event
```

### Provenance & Uncertainty (§1, §87)
```yaml
claim:
  id: claim-uuid
  property: string                  # "geometry", "material_density", "position", etc.
  entity_id: entity-uuid
  value: any
  confidence: 0.0..1.0
  state: OBSERVED | RECONSTRUCTED | ESTIMATED | INFERRED | GENERATED | UNKNOWN | CONFLICT
  evidence_refs: [observation-id]
  source: sensor-id | algorithm | user-input
  timestamp: Timestamp
  uncertainty:
    mean: float
    lower_bound: float
    upper_bound: float
    std_dev: float
    method: measurement-model-name
```

---

## Core Interfaces & Contracts

### §83: Simulation Context & Result
```cpp
struct SimulationContext {
    WorldHandle world;              // current WorldIR revision
    ScenarioHandle scenario;        // disaster parameters
    double time;                    // world time (seconds)
    double dt;                      // fixed timestep
    uint64_t tick;                  // simulation tick (frame number)
    uint64_t seed;                  // determinism seed
};

struct SimulationResult {
    bool success;                   // solver completed without error
    Diagnostics diagnostics;        // NaN/Inf counts, energy reports, warnings
    EventStream events;             // all events emitted this step
    WorldDelta delta;               // what changed (for incremental export)
};
```

### §7.2: Physics Backend Interface
```cpp
class IPhysicsBackend {
public:
    virtual PhysicsWorldHandle createWorld(const PhysicsWorldConfig&) = 0;
    
    virtual void step(PhysicsWorldHandle, double dt) = 0;
    
    virtual void queryContacts(PhysicsWorldHandle, vector<Contact>&) = 0;
    virtual void queryRaycast(PhysicsWorldHandle, 
                              Vec3 origin, Vec3 direction, float max_dist,
                              RaycastHit& hit) = 0;
    
    virtual void serialize(PhysicsWorldHandle, JSONValue&) = 0;
    virtual void deserialize(const JSONValue&, PhysicsWorldHandle&) = 0;
};
```

### Specialized Solver Interfaces
```cpp
class ISolver {
public:
    virtual void initialize(const SimulationContext&) = 0;
    virtual void step(const SimulationContext&) = 0;
    virtual void finalize(const SimulationContext&) = 0;
};

class IFractureSolver : public ISolver;     // for destruction
class IFluidSolver : public ISolver;        // water + buoyancy
class IFireSolver : public ISolver;         // thermal + ignition
class IWeatherSolver : public ISolver;      // wind + rain
class IStructuralSolver : public ISolver;   // load paths + capacity
class IDisasterSolver : public ISolver;     // earthquake, tornado, hurricane
class IDebrisSolver : public ISolver;       // debris lifecycle + obstacles
```

### §84: Field API (for spatial grids)
```cpp
template<class T>
class Field3D {
public:
    T sample(Vec3 position) const;          // bilinear/trilinear interp
    void set(Vec3 position, T value);       // grid update
    void advect(Vec3 velocity, double dt);  // semi-Lagrangian advection
    void resample(int resolution);          // LOD change
};

// Supported: CPU storage, GPU storage, sparse tiling, adaptive resolution
```

---

## Architectural Invariants

### §1: Non-Negotiable Principles

#### 1.1 Evidence First
- Every property carries a **state**: `OBSERVED`, `RECONSTRUCTED`, `ESTIMATED`, `INFERRED`, `GENERATED`, `UNKNOWN`, `CONFLICT`
- **Generated content** (simulation, cinematic) may exist in game branches but must never silently become canonical reality
- **Conflict records** must be preserved, not silently averaged or overwritten
- **Unknown** is a first-class entity state (not "missing" or omitted)

#### 1.2 Causality First
Prefer causal chains over direct effects:
```
CAUSE → PHYSICAL STATE → SECONDARY EFFECTS → VISUAL → AUDIO → GAMEPLAY
```
Example:
```
Rain
  → surface water (state)
  → reduced friction (physical)
  → reflections (visual)
  → runoff (physical)
  → puddles (geometry)
  → splash sound (audio)
  → drainage gameplay (interaction)
```
NOT: `Rain slider → darker texture` (direct cinematic shortcut).

#### 1.3 Canonical Reality is Immutable
- WorldIR uses **append-only versioning**
- New evidence creates a **new world version** or **branch**
- Existing observations remain **recoverable**
- Edits are only safe in **simulation branches** or **design branches**

#### 1.4 Physics Must Have Fidelity Tiers (P0-P4)
| Tier | Purpose | Latency | Accuracy | Use Case |
|------|---------|---------|----------|----------|
| P0 | Visual approximation only | <1ms | Not physical | Ambient, non-interactive |
| P1 | Gameplay physics | 16-33ms | ±5-10% | Player-relevant interactions |
| P2 | Constrained real-time | 33-100ms | ±2-5% | Local simulation, streaming |
| P3 | High-fidelity local | 100ms-1s | <±2% | Focused region, offline compute |
| P4 | Reference/engineering | >1s | <±0.1% | Offline validation, design tool |

**Runtime choice based on:** camera distance, interaction, player relevance, semantic importance, hardware, available compute, determinism requirements.

#### 1.5 Every Subsystem is Independently Testable
Every solver/backend must expose:
- **Input contract** (what data it consumes, units, ranges)
- **Output contract** (what it produces, semantics)
- **Numerical limits** (max entity count, max timestep, memory budget)
- **Deterministic mode** (fixed seed, reproducible)
- **Fallback behavior** (graceful degradation on OOM/timeout)
- **Diagnostics** (warnings, error codes, metrics)
- **Benchmark** (latency, throughput, memory, GPU cost)
- **Serialization format** (versioned, round-trip verified)

### §80: Agent Done Definition
A task is NOT done because code compiles. Done means:
```
implementation
  + unit tests
  + integration tests
  + serialization test (round-trip)
  + determinism test (identical seed → identical state)
  + error handling (graceful failures, not crashes)
  + diagnostics (logging, metrics, warnings)
  + documentation (what it does, invariants, constraints)
  + benchmark (latency, memory, throughput)
  + no regression (doesn't break existing tests)
```

### §90: Determinism
Deterministic mode requires:
- Fixed timestep (no variable dt)
- Stable iteration ordering (id-sorted, never dict-iteration order)
- Controlled random seeds (named sub-streams, reproducible)
- Deterministic task scheduling (job system topological order)
- Version-pinned algorithms (config records which solver version ran)
- Serialized configuration (so identical seed+config = identical output)

**Floating-point nondeterminism must be measured and bounded** (e.g., FLT_EPSILON tolerance).

---

## Testing Requirements

### §80 & §91-95: Testing Pyramid

#### Unit Tests (Foundation Layer)
- **Math (Vec3, Quat, Mat3, Mat4):** arithmetic, transforms, edge cases
- **Units (Quantity):** unit wrapping, conversion errors caught
- **Serialization:** round-trip JSON/YAML for each schema
- **RNG & Seeding:** sub-stream reproducibility
- **Job System:** topological ordering, dependency resolution

#### Unit Tests (Solvers)
Each solver gets:
- **Nominal case** (typical inputs, expected outputs)
- **Boundary case** (edge of valid domain: zero mass, max timestep, etc.)
- **Malformed input** (invalid state, out-of-range values)
- **Empty input** (no entities, no bodies, no events)
- **Extreme value** (very large/small numbers, near-singular matrices)

#### Integration Tests
- **Physics pipeline:** rigid body → collision → contact resolution → events
- **Multi-solver coupling:** water + destruction (buoyancy → fabric failure)
- **Serialization roundtrip:** load state → step → save → load → verify identical
- **Determinism:** run(seed=X) twice → byte-identical states and event logs
- **Regression:** metrics vs. baseline (energy, contact counts, positions)

#### Golden Scene Tests (§91-93)
Predefined scenes with expected behaviors:
- **RB_001 (Falling Cube):** cube falls 5m, settles with ±0.05m tolerance
- **RB_002 (Stacked Boxes):** two boxes stacked, stay standing after 8s
- **RB_003 (Bouncing Ball):** rubber ball bounces, reaches ±10% of initial height
- **HYD_001 (Rainfall):** rain accumulates in terrain depression, drains over time
- **STRUCT_001 (Beam Under Load):** load causes stress, material degradation, failure at thresholds
- **COUPLE_001 (Water + Structure):** flood → buoyancy → load increase → capacity exceeded → collapse

#### Visual Regression (§94)
- **Snapshot comparison:** render same scene, diff against baseline
- **Metadata extraction:** bounding boxes, semantic labels, measurements match
- **Cross-engine validation:** export to Unreal → re-import → geometry matches within tolerance

#### Performance Regression (§95)
Track metrics per subsystem:
- **Latency:** p50, p95, p99 milliseconds per solver step
- **Throughput:** entities/sec, contacts/sec, events/sec
- **Memory:** peak RSS, entity buffer size, scratch allocation
- **GPU:** VRAM usage, shader time (if applicable)
- **Baseline:** historical commit SHAs to regress against

---

## Performance Gates

### Frame Budget (Runtime Play)
| Component | Budget | Notes |
|-----------|--------|-------|
| **Physics step** | 16.67ms (60 Hz) | P1 fidelity, id-sorted |
| **Rendering** | 16.67ms | viewport + UI |
| **Streaming** | <5ms | chunk loads, unloads |
| **Audio** | <2ms | event spatialization |
| **Event propagation** | <1ms | causality graph walk |
| **Total frame** | 33ms (30 Hz) | P1 + rendering + UI |

### Load Budget (Editor / Offline)
| Operation | Target | Notes |
|-----------|--------|-------|
| **World load** | <5s | 50k entities, 100 MB JSON |
| **Scenario setup** | <1s | parameter validation |
| **Physics rebuild** | <30s | broadphase AABB recompute |
| **Export** | <60s | 10k entities → Unreal |
| **Reconstruction** | <300s | 500 photos → mesh |

### Memory Budget
| Component | Target | Notes |
|-----------|--------|-------|
| **Entity registry** | <100 MB | 50k entities × 2KB avg |
| **Physics bodies** | <50 MB | 5k dynamic, 45k static |
| **Geometry (LOD0)** | <500 MB | streaming may load on-demand |
| **Particle scratch** | <100 MB | temporary per-frame allocation |
| **Total scene** | <1 GB | fits on mobile/console |

### Determinism Constraints
- Floating-point operations must converge within **FLT_EPSILON × magnitude**
- No IEEE754 nondeterminism (no fast-math, no SIMD shuffle-order variance)
- No threading (or strict work-stealing with seed-based ordering)
- No system-dependent rounding (use round-to-nearest, ties-to-even everywhere)

---

## Security & Trust Model

### §99-100: Privacy & Rights

#### Privacy
- User-uploaded evidence (photos, video) stays on-premises unless explicitly exported
- Derived data (geometry, materials, semantics) inherits source privacy
- No telemetry without explicit opt-in
- GDPR/CCPA compliance: right to delete, data portability via export

#### Rights & Licensing
- Third-party reconstructed models inherit copyright from source evidence owner
- Procedurally generated content (destruction, effects) is user-owned
- Exported data respects source CRS (Coordinate Reference System) to avoid IPR surprises
- OpenUSD format for interchange (vendor-neutral)

### §118-125: Agent & Plugin Security

#### AI Agent Boundaries
- Agents **never edit database rows directly**; all changes via type-checked APIs
- Agents **compile NL intent to typed commands** before execution
- Commands **validated for permissions** (can this agent edit this entity type?)
- **Fallback on rejection** (don't cascade errors to GUI)

#### Plugin Model
- Plugins operate behind **versioned interfaces** (stable ABI)
- Plugin memory is **isolated** (sandbox or separate process)
- Plugins **cannot mutate canonical world** (only simulation branches)
- Plugins register with **symmetric unregister** (cleanup guarantee)

---

## Risk Register

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|-----------|
| **Floating-point divergence** | Simulation replay breaks after minor changes | High | FLT_EPSILON tolerance gates, reference test suite |
| **Circular dependencies in reconstruction** | Mesh → material → structural graph → physics → mesh loop | Medium | Acyclic dependency validation, retry limits |
| **Disaster coupling explosion** | Hurricane → flood → fire → structure → debris → fire again (cycles) | High | Event loop cycle detection, constraint normalization |
| **GIS coordinate system mismatch** | Exported data doesn't align with georeferenced basemaps | Medium | Explicit CRS tags, round-trip validation for known locations |
| **Physics backend compatibility** | NVIDIA PhysX versions diverge; behavioral changes | Medium | Backend abstraction (§7.2), version pinning, bench regression |
| **UI responsiveness under large worlds** | >100k entities → viewport stutters, inspector hangs | Medium | Streaming + LOD + async query (don't block frame) |
| **Thermal runaway in coupled sims** | Fire spreads to all objects instantly (integration too coarse) | Medium | Adaptive substeps, energy dissipation guards |
| **Replay corruption** | Event log diverges from saved state (non-deterministic step() buried in library) | High | Determinism audit per step, regression tests, seed versioning |
| **Semantic annotation drift** | "Building" labels become inconsistent over model versions | Low | Label version tracking, retroactive audit queries |
| **Export fidelity loss** | Complex fracture patterns or exotic collision shapes drop to simpler reps | Medium | Fidelity-profile UI (show what's being simplified), fallback warnings |

---

## Known Ambiguities & Engineering Decisions Required

### SPEC_AMBIGUITY: Coordinate Frame Convention
**Issue:** Specification lists "multiple frame support" (§89) but does not explicitly define the **right-hand rule handedness** or **axis ordering** (row-major vs. column-major matrix storage).

**Impact:** Transform composition order, cross products, rotation matrix conventions.

**Needs Decision:**
- [ ] Standard: Use **right-hand, X-forward, Y-up (Unreal convention)** across all frames?
- [ ] Or adopt **ENU (East-North-Up) for geo frames** and local right-hand for entity-local?
- [ ] Explicit documentation in all interface signatures (`T_A_B` means "A ← B", compose as `T_AB * v_B`)?

---

### SPEC_AMBIGUITY: Destruction State Machine
**Issue:** §11 (Fracture) lists states `INTACT → DAMAGED → FRACTURED → DETACHED → DEBRIS → REMOVED` but does not define **transition thresholds** or **reversibility**.

**Impact:** Is `DAMAGED` reversible (repair) or one-way? Does `FRACTURED` → `DEBRIS` auto-spawn child entities?

**Needs Decision:**
- [ ] **Immutable cascade:** Once FRACTURED, always fragmented (no repair)?
- [ ] **Branched state:** DAMAGED can heal in simulation but not canon?
- [ ] **Explicit repair events:** DAMAGED → INTACT requires an event/action?

---

### SPEC_AMBIGUITY: Fidelity Selection Policy
**Issue:** §1.4 lists P0-P4 tiers and factors to choose them (camera distance, interaction, etc.) but does not define **the algorithm** or **priority order** if multiple factors conflict.

**Impact:** If camera is far (→ P0) but player is interacting (→ P2), which wins?

**Needs Decision:**
- [ ] **Priority ranking:** interaction > semantic importance > camera distance > hardware?
- [ ] **Per-entity override:** allow user to pin an entity to P2 despite distance?
- [ ] **Adaptive hysteresis:** switch fidelity only if distance changes >N meters to avoid flicker?

---

### SPEC_AMBIGUITY: Certainty Threshold for Export
**Issue:** §1.1 & §99 say "never invent canonical geometry" and "CONFLICT is preserved," but exporters (Unreal, Blender) may not support multiple competing geometries.

**Impact:** How to handle a building with **geometry: RECONSTRUCTED (80% confidence)** vs **ESTIMATED (60% confidence)** when exporting?

**Needs Decision:**
- [ ] **Export-time filter:** Only export OBSERVED + RECONSTRUCTED (confidence > threshold)?
- [ ] **Metadata fallback:** Export the high-confidence version + side data listing alternatives?
- [ ] **User-guided:** Let the exporter UI ask which version to use?

---

### SPEC_AMBIGUITY: Event Causality Depth
**Issue:** Causal graphs (§85) can grow very large (explosion → debris → impact → damage → failure → cascade). Specification does not define **max depth**, **cycle detection**, or **summary coarsening**.

**Impact:** Query "why did this building collapse?" could return 10,000 event chain links.

**Needs Decision:**
- [ ] **Transitive closure:** Compute the full chain or stop at direct causes?
- [ ] **Visualization:** How deep to show in UI before truncating with "... and N more events"?
- [ ] **Summarization:** Group rapid cascades (fire spreading tree→tree→tree) into one "fire propagation" event?

---

### SPEC_AMBIGUITY: Streaming LOD Strategy
**Issue:** §50 (Streaming Architecture) and §135 mention LOD and streaming but do not detail **mesh simplification ratios**, **shadow LOD** vs **physics LOD**, or **pop-in thresholds**.

**Impact:** When to hide vs. simplify vs. remove geometry; physics representation may need to diverge from visual.

**Needs Decision:**
- [ ] **Deterministic LOD:** Use consistent distance brackets or viewport-relative size?
- [ ] **Physics fidelity:** Do shadows/distant objects still have collision at P0, or is physics culled too?
- [ ] **Async loading:** Streaming happens in background; LOD geometry loads on the main thread; coordinate how?

---

### SPEC_AMBIGUITY: Multi-Agent Coherence
**Issue:** §162-164 describe multi-agent workflows and handoff but do not define **conflict resolution** if two agents edit the same entity simultaneously.

**Impact:** Agent A locks entity while B queues an edit; lock timeout? Merge?

**Needs Decision:**
- [ ] **Pessimistic locking:** One agent at a time per entity?
- [ ] **Optimistic merge:** Both agents commit; merge at save time if no conflict?
- [ ] **Operational transform (CRDT):** Support concurrent edits via commutative operations?

---

### SPEC_AMBIGUITY: Reconstruction Truth Provenance
**Issue:** §42-43 (Perception & Reconstruction) describe pose estimation and scale inference but do not formally link **uncertainty in pose** to **bounding box precision** to **entity placement confidence**.

**Impact:** A wall's position depends on camera calibration uncertainty; what confidence do we assign?

**Needs Decision:**
- [ ] **Backward propagation:** Compute uncertainty in entity pose from camera intrinsics + SfM covariance?
- [ ] **Conservative:** Assume ESTIMATED for all reconstruction, no matter confidence?
- [ ] **User audit:** Let expert approve which claims → RECONSTRUCTED vs. ESTIMATED?

---

## First Implementation Milestone

**Objective:** Build the **first integrated demo** following §108 steps 1-11 + visual inspection (steps 16-17).

**Components:**
1. **WorldIR** — entity registry, serialization, versioning
2. **Coordinates** — transform composition, multi-frame support
3. **Rigid Body Physics** — semi-implicit Euler, sphere/box/plane primitives
4. **Collision** — AABB broadphase, narrowphase contact generation
5. **Contact Solver** — impulse resolution, friction, resting contact
6. **Materials** — density, friction, restitution (subset of full spec)
7. **Event Bus** — ContactEvent, ImpactEvent emission on new contacts
8. **Deterministic RNG** — seeded sub-streams, reproducible from seed
9. **Replay** — serialize/deserialize physics state, run identical seed→identical log
10. **Unit Tests** — math, collision, contact, determinism, round-trip
11. **Integration Tests** — falling cube settles, stacked boxes stand, ball bounces
12. **Golden Tests** — RB_001 (falling), RB_002 (stacking) pass with tolerance ranges

**Success Criteria:**
- ✓ 112+ unit tests passing
- ✓ Deterministic replay (identical seed → byte-identical serialization)
- ✓ Golden scenes pass (falling cube within ±0.05m, stacking holds, bounces to >0.6m)
- ✓ No NaN/Inf in physics output
- ✓ Energy stays bounded (no explosion)
- ✓ Inspector can view entity properties, materials, evidence confidence
- ✓ Export to Unreal: sphere + box + plane entities appear in editor

**Timeline:** ~1 person-month (if not already complete per prior work).

---

**This map is authoritative as of 2026-09-04. Specification version V10. Build order per §108.**
