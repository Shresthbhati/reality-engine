# Reality Engine V11 — Concrete Repository Architecture

**Authority:** V11 Specification §81-82 (Repository Structure, Physics Subdirectory)
**Based on:** `docs/IMPLEMENTATION_MAP.md` dependency analysis
**Date:** 2026-09-04
**Status:** Architectural Framework (no implementation yet)

---

## Table of Contents

1. [Module Dependency Rules](#module-dependency-rules)
2. [Directory Structure & Ownership](#directory-structure--ownership)
3. [Module Boundaries & Invariants](#module-boundaries--invariants)
4. [Public API Surfaces](#public-api-surfaces)
5. [Configuration System](#configuration-system)
6. [Logging & Diagnostics](#logging--diagnostics)
7. [Error Model](#error-model)
8. [Event Model](#event-model)
9. [Serialization & Versioning](#serialization--versioning)
10. [Testing Structure](#testing-structure)
11. [Benchmark Structure](#benchmark-structure)
12. [Plugin Architecture](#plugin-architecture)

---

## Module Dependency Rules

### The Dependency DAG (must be acyclic)

```
Tier 0 (no dependencies)
├── core/units           (physical quantities, SI enforcement)
├── core/rng             (seeded, deterministic randomness)
└── core/config          (TOML/YAML configuration loading)

Tier 1 (depends on Tier 0)
├── core/math            (Vec3, Quat, Mat3, Mat4 — no physics)
├── world_ir/schema      (entity, transform, provenance data model)
├── provenance/          (OBSERVED/RECONSTRUCTED tracking)
└── core/clock           (fixed-timestep simulation time)

Tier 2 (depends on Tier 0-1)
├── core/jobs            (topological task scheduling)
├── world_ir/registry    (entity lifecycle, relationships)
└── world_ir/serialization (versioned save/load, JSON/YAML)

Tier 3 (depends on Tier 0-2)
├── world_ir/coordinates (transform composition, multi-frame)
├── world_ir/runtime     (loads WorldIR, coordinate registry, frame resolution)
└── physics/core/        (IPhysicsBackend interface, SimulationContext)

Tier 4 (depends on Tier 0-3, can depend on each other within this tier)
├── physics/rigid/       (RigidBody, inertia, integration, sleep)
├── physics/collision/   (shapes, broadphase, narrowphase, raycast)
├── physics/materials/   (PhysicsMaterial, canonical materials)
├── physics/constraints/ (contact solver, impulse, friction)
├── engine/physics/backend/simple (SimpleRigidBodyBackend concrete impl)
└── events/              (EventBus, Event schema, handler dispatch)

Tier 5 (depends on Tier 0-4, can depend on each other)
├── physics/destruction/ (fracture, debris, damage tracking)
├── physics/fluids/      (NOT YET)
├── physics/fire/        (NOT YET)
├── physics/weather/     (NOT YET)
├── physics/structural/  (NOT YET)
└── engine/world/        (WorldRuntime, simulation loop coordination)

Tier 6 (depends on Tier 0-5)
├── apps/cli/            (command-line interface, SDK wrapper)
├── exporters/           (Unreal, Blender, OpenUSD, GIS adapters)
└── reconstruction/      (photogrammetry pipeline — NOT YET)
```

### Forbidden Dependencies

- **No backwards edges:** Tier N can never depend on Tier N+1
- **No sibling cycles:** Within a tier, cycles are prohibited (e.g., physics/rigid → physics/collision is OK; physics/collision → physics/rigid is NOT)
- **No app-layer → engine leakage:** `apps/` can depend on `engine/`; `engine/` must never depend on `apps/`
- **No test code in production:** `tests/` lives separately; production code never imports from `tests/`
- **No plugin → core cycles:** Plugins can depend on core; core must never depend on plugins

---

## Directory Structure & Ownership

```
reality-engine/
│
├── .github/
│   ├── workflows/           # CI/CD pipelines (GitHub Actions)
│   └── CODEOWNERS           # Team ownership by file pattern
│
├── .claude/
│   ├── CLAUDE.md            # Agent instructions (if used)
│   ├── launch.json          # Dev server configuration
│   └── skills/              # Local skill definitions (if any)
│
├── engine/
│   │
│   ├── core/                # Tier 0-1 foundations
│   │   ├── __init__.py
│   │   ├── units.py         # Quantity wrapper, SI enforcement
│   │   ├── rng.py           # DeterministicRNG, seeded sub-streams
│   │   ├── clock.py         # SimulationContext, fixed timestep
│   │   ├── jobs.py          # JobSystem, topological ordering
│   │   ├── math/            # Vec3, Quat, Mat3, Mat4 (NOT physics yet)
│   │   │   ├── __init__.py
│   │   │   ├── vec.py
│   │   │   └── quat.py
│   │   ├── config.py        # Config loading (TOML, YAML, JSON)
│   │   ├── logging.py       # Structured logging, diagnostics
│   │   └── errors.py        # Base exception types
│   │
│   ├── world_ir/            # Tier 1-3: WorldIR canonical model
│   │   ├── __init__.py
│   │   ├── schema.py        # Entity, transform, bounds, provenance
│   │   ├── entity.py        # EntityRegistry, lifecycle
│   │   ├── coordinates.py   # Frame, Transform, CoordinateRegistry
│   │   ├── serialization.py # save_world, load_world, versioning
│   │   ├── runtime.py       # WorldRuntime, frame resolution
│   │   └── tests/           # (see Testing Structure below)
│   │
│   ├── physics/             # Tier 3-5: Physics solvers
│   │   ├── __init__.py
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── interface.py # IPhysicsBackend, ISolver, SimulationContext
│   │   │   └── math3.py     # Vec3/Mat3/Quat specific to physics (NOT core/math)
│   │   │
│   │   ├── rigid/           # Rigid body dynamics (Tier 4)
│   │   │   ├── __init__.py
│   │   │   ├── body.py      # RigidBody class, inertia tensor
│   │   │   ├── integrator.py # semi-implicit Euler, damping, sleep
│   │   │   └── tests/
│   │   │
│   │   ├── collision/       # Collision detection (Tier 4)
│   │   │   ├── __init__.py
│   │   │   ├── shapes.py    # Sphere, Box, Plane, shape_from_dict
│   │   │   ├── broadphase.py # AABB overlap, O(n²), id-sorted
│   │   │   ├── narrowphase.py # sphere_vs_sphere, box_vs_plane, etc.
│   │   │   ├── raycast.py   # ray_vs_sphere, ray_vs_box, ray_vs_plane
│   │   │   └── tests/
│   │   │
│   │   ├── materials/       # Material physics (Tier 4)
│   │   │   ├── __init__.py
│   │   │   ├── material.py  # PhysicsMaterial, canonical materials dict
│   │   │   └── tests/
│   │   │
│   │   ├── constraints/     # Contact & constraint solving (Tier 4)
│   │   │   ├── __init__.py
│   │   │   ├── contact_solver.py # resolve_contact, impulse, friction
│   │   │   └── tests/
│   │   │
│   │   ├── diagnostics/     # Numerical health checks (Tier 4)
│   │   │   ├── __init__.py
│   │   │   └── numerics.py  # check_world, NaN/Inf/energy guards
│   │   │
│   │   ├── backend/         # Physics backend implementations (Tier 4-5)
│   │   │   ├── __init__.py
│   │   │   ├── interface.py # IPhysicsBackend interface (pure ABC)
│   │   │   ├── simple_backend.py # SimpleRigidBodyBackend concrete
│   │   │   └── tests/
│   │   │
│   │   ├── destruction/     # Fracture, debris (Tier 5 — NOT YET)
│   │   │   ├── __init__.py
│   │   │   ├── fracture.py  # IFractureSolver (stub)
│   │   │   ├── debris.py    # IDebrisSolver (stub)
│   │   │   └── tests/
│   │   │
│   │   ├── fluids/          # Water, buoyancy (Tier 5 — NOT YET)
│   │   │   ├── __init__.py
│   │   │   ├── fluid.py     # IFluidSolver (stub)
│   │   │   └── tests/
│   │   │
│   │   ├── fire/            # Thermal, ignition, spread (Tier 5 — NOT YET)
│   │   │   ├── __init__.py
│   │   │   ├── fire.py      # IFireSolver (stub)
│   │   │   └── tests/
│   │   │
│   │   ├── weather/         # Wind, rain, atmosphere (Tier 5 — NOT YET)
│   │   │   ├── __init__.py
│   │   │   ├── weather.py   # IWeatherSolver (stub)
│   │   │   └── tests/
│   │   │
│   │   ├── structural/      # Load paths, capacity, collapse (Tier 5 — NOT YET)
│   │   │   ├── __init__.py
│   │   │   ├── structural.py # IStructuralSolver (stub)
│   │   │   └── tests/
│   │   │
│   │   └── tests/           # Physics integration tests
│   │
│   ├── world/               # Tier 5: World runtime & simulation loop
│   │   ├── __init__.py
│   │   ├── runtime.py       # WorldRuntime, step() orchestration
│   │   └── tests/
│   │
│   └── rendering/           # Tier 5-6: Viewport rendering (NOT YET)
│       ├── __init__.py
│       ├── viewport.py      # (stub)
│       └── tests/
│
├── events/                  # Tier 4: Event bus
│   ├── __init__.py
│   ├── event.py             # Event class, schema
│   ├── bus.py               # EventBus, subscriptions
│   ├── types.py             # KNOWN_EVENT_TYPES registry
│   └── tests/
│
├── world_ir/                # (moved to engine/world_ir, but link kept for legacy?)
│   └── [symlink or import from engine/world_ir]
│
├── provenance/              # Tier 1: Provenance tracking
│   ├── __init__.py
│   ├── provenance.py        # Provenanced, Uncertainty, evidence states
│   └── tests/
│
├── exporters/               # Tier 6: Export adapters (NOT YET)
│   ├── __init__.py
│   ├── unreal.py            # Unreal Engine exporter
│   ├── blender.py           # Blender exporter
│   ├── openusd.py           # OpenUSD adapter
│   ├── gis.py               # GIS/shapefile export
│   └── tests/
│
├── reconstruction/          # Tier 6: Reconstruction pipeline (NOT YET)
│   ├── __init__.py
│   ├── calibration.py       # Camera intrinsics
│   ├── sfm.py               # Structure-from-Motion
│   ├── mvs.py               # Multi-View Stereo
│   └── tests/
│
├── apps/                    # Tier 6: Applications
│   ├── cli/                 # Command-line interface
│   │   ├── __init__.py
│   │   ├── main.py          # Click/argparse entry point
│   │   ├── commands/        # Subcommand modules
│   │   └── tests/
│   │
│   ├── studio/              # Desktop UI (NOT YET)
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   ├── viewer/              # Viewport viewer (NOT YET)
│   │   ├── __init__.py
│   │   └── main.py
│   │
│   └── capture/             # Mobile capture app (NOT YET)
│       ├── __init__.py
│       └── main.py
│
├── plugins/                 # Plugin host & examples (NOT YET)
│   ├── __init__.py
│   ├── plugin_interface.py  # IPlugin ABC, versioning
│   ├── plugin_registry.py   # Plugin discovery, loading
│   └── examples/
│
├── tests/                   # Top-level test suite
│   ├── __init__.py
│   ├── conftest.py          # pytest fixtures, markers
│   ├── integration/         # Cross-module tests
│   │   ├── test_world_ir_physics.py
│   │   ├── test_serialization_determinism.py
│   │   └── test_export_pipeline.py
│   ├── golden/              # Golden scene tests (§91)
│   │   ├── rb_001_falling_cube.py
│   │   ├── rb_002_stacked_boxes.py
│   │   └── fixtures/
│   └── benchmarks/          # (see Benchmark Structure below)
│
├── benchmarks/              # Benchmark suite (§91-95)
│   ├── __init__.py
│   ├── conftest.py          # benchmark fixtures
│   ├── physics/
│   │   ├── bench_rigid_body.py
│   │   ├── bench_collision.py
│   │   └── bench_contact.py
│   └── profiling/
│       └── profile_golden_scenes.py
│
├── docs/
│   ├── IMPLEMENTATION_MAP.md (created earlier)
│   ├── ARCHITECTURE.md       (this file)
│   ├── API.md               # Public API reference
│   ├── BUILD_ORDER.md       # Progress tracking
│   ├── physics_primer.md    # Physics overview, equations
│   ├── world_ir_spec.md     # WorldIR format details
│   ├── event_model.md       # Event system design
│   └── examples/            # Usage examples
│
├── pyproject.toml           # PEP 517 build config, dependencies
├── setup.py                 # (legacy, if needed)
├── setup.cfg                # (legacy, if needed)
├── Makefile                 # Common tasks (test, lint, bench, build)
├── .pre-commit-config.yaml  # Pre-commit hooks (ruff, mypy, etc.)
├── pyproject.toml           # (tool.pytest, tool.ruff, tool.mypy)
├── pytest.ini               # (pytest config; can go in pyproject.toml)
├── .gitignore               # Standard Python ignores
├── README.md                # Project overview
├── CONTRIBUTING.md          # Developer guide
└── LICENSE
```

---

## Module Boundaries & Invariants

### engine/core/units

**Purpose:** Enforce SI units; prevent silent unit mismatches (e.g., meters vs. feet).

**Owns:**
- `Quantity` wrapper: value + unit
- Unit enum: `METER`, `KILOGRAM`, `SECOND`, `NEWTON`, `PASCAL`, `JOULE`, `KELVIN`
- Canonical conversions (meter → feet raises error)

**Must Never Own:**
- Physics equations (that's physics/core)
- Entity data (that's world_ir)

**Public API:**
```python
class Quantity:
    def __init__(self, value: float, unit: Unit)
    def to(self, target_unit: Unit) -> float
    def in_si(self) -> float
    
# Usage:
mass = Quantity(5.0, Unit.KILOGRAM)
meters = Quantity(10.0, Unit.METER)
feet = meters.to(Unit.FOOT)  # raises: cannot convert METER to FOOT
```

**Dependencies:** None (Tier 0)

**Invariants:**
- All public physics APIs use `Quantity`, never bare floats
- Silent unit mismatches are FORBIDDEN
- Conversions within same dimension (length → length) always succeed
- Cross-dimension conversions (mass → length) always raise

**Test Boundary:**
- Unit conversion correctness
- Rejection of invalid conversions
- Integration: physics solvers use Quantity everywhere

---

### engine/core/math (NOT physics)

**Purpose:** Pure linear algebra (Vec3, Quat, Mat3, Mat4). No physics semantics.

**Owns:**
- `Vec3`: 3D vector, dot/cross/normalize/rotate
- `Quat`: quaternion, rotate, integrate (but NOT physics-aware)
- `Mat3` / `Mat4`: rotation/translation matrices, composition
- No reference frames, no physics interpretations

**Must Never Own:**
- Physics mass, inertia, velocity (that's physics/rigid or physics/core/math3)
- Transforms with frame semantics (that's world_ir/coordinates)
- Determinism/seeding (that's core/rng)

**Public API:**
```python
class Vec3:
    x, y, z: float
    def dot(self, other: Vec3) -> float
    def cross(self, other: Vec3) -> Vec3
    def normalize(self) -> Vec3
    def rotate(self, q: Quat) -> Vec3

class Quat:
    w, x, y, z: float
    def rotate(self, v: Vec3) -> Vec3
    def multiply(self, other: Quat) -> Quat
    def integrated(self, angular_velocity: Vec3, dt: float) -> Quat
```

**Dependencies:** core/units

**Invariants:**
- All operations preserve numeric stability
- No side effects (all methods return new objects or scalars)
- No reference to physics coordinate frames

**Test Boundary:**
- Arithmetic correctness
- Numerical stability (epsilon tests)
- Composition laws (associativity, distributivity)

---

### engine/world_ir/ (Tiers 1-3)

#### world_ir/schema.py

**Purpose:** Data class definitions for entities, transforms, provenance.

**Owns:**
- `Entity`: id, type, transform, geometry_refs, material_refs, physics, evidence, confidence
- `Transform`: position, orientation, scale
- `GeometryRef`: LOD pyramid references
- `MaterialRef`: material assignment
- `ProvenanceRef`: link to evidence

**Must Never Own:**
- Physics behavior (that's physics/rigid)
- Serialization (that's world_ir/serialization)
- Frame semantics (that's world_ir/coordinates)

**Public API:**
```python
@dataclass
class Entity:
    id: str
    type: EntityType
    transform: Transform
    geometry_refs: List[GeometryRef]
    material_refs: List[MaterialRef]
    evidence: List[EvidenceRef]
    confidence: float
    # ... etc.

class EntityType(Enum):
    BUILDING, WALL, FLOOR, WINDOW, VEHICLE, TREE, ...
```

**Dependencies:** core/units, provenance

**Invariants:**
- Entity IDs are globally unique within a world
- All numeric fields use Quantity or strict types
- Confidence is always in [0.0, 1.0]
- Relationships reference existing entities (validated by EntityRegistry)

**Test Boundary:**
- Schema validation
- Dataclass serialization (asdict)
- Cross-field invariants (e.g., if type==WINDOW, geometry must be planar)

---

#### world_ir/entity.py

**Purpose:** EntityRegistry: lifecycle, relationships, validation.

**Owns:**
- `EntityRegistry`: add, remove, get, query
- Relationship validation: no dangling refs
- Hierarchical queries (children of parent)

**Must Never Own:**
- Transform composition (that's world_ir/coordinates)
- Physics simulation (that's engine/world)
- Serialization format (that's world_ir/serialization)

**Public API:**
```python
class EntityRegistry:
    def add_entity(self, entity: Entity) -> None
    def remove_entity(self, entity_id: str) -> None
    def get_entity(self, entity_id: str) -> Entity | None
    def entities_of_type(self, entity_type: EntityType) -> List[Entity]
    def children_of(self, parent_id: str) -> List[Entity]
    def validate_relationships(self) -> List[ValidationError]
```

**Dependencies:** core/units, world_ir/schema

**Invariants:**
- After add/remove, registry state is valid (no dangling refs)
- IDs are immutable
- Parent-child cycles are forbidden

**Test Boundary:**
- CRUD operations
- Relationship integrity
- Query correctness

---

#### world_ir/coordinates.py

**Purpose:** Multi-frame transforms, composition, resolution.

**Owns:**
- `Frame`: enum (WORLD, CAMERA, SENSOR, LOCAL, ENU, UTM, WGS84, ECEF)
- `Transform`: as above, but with frame semantics
- `CoordinateRegistry`: T_A_B lookup, composition by BFS

**Must Never Own:**
- Physics simulation (that's physics/)
- Entity hierarchy (that's world_ir/entity)

**Public API:**
```python
class Frame(Enum):
    WORLD, CAMERA, SENSOR, LOCAL_BUILDING, ENU, UTM, WGS84, ECEF

class CoordinateRegistry:
    def register_transform(self, frame_a: Frame, frame_b: Frame, T: Mat4) -> None
    def transform_point(self, p: Vec3, from_frame: Frame, to_frame: Frame) -> Vec3
    def compose_path(self, frame_a: Frame, frame_b: Frame) -> List[Tuple[Frame, Frame]]
```

**Dependencies:** core/math, core/units

**Invariants:**
- All transforms are in right-hand, X-forward, Y-up (§ engineering decision)
- T_A_B means "A ← B" (transforms B-frame points into A-frame)
- No cycles in transform graph (DAG only)
- Composition is associative

**Test Boundary:**
- Transform composition correctness
- Point transformation across frames
- Cycle detection

---

#### world_ir/serialization.py

**Purpose:** Save/load WorldIR in versioned JSON/YAML format.

**Owns:**
- `save_world(world, path)`: JSON/YAML writer with manifest
- `load_world(path)`: JSON/YAML parser, version check
- Format versioning (format_version=1 in manifest)

**Must Never Own:**
- Entity schema (that's world_ir/schema)
- Coordinate transforms (that's world_ir/coordinates)
- Physics state (that's physics/)

**Public API:**
```python
def save_world(world: WorldIR, path: str) -> None
    # Writes world.ir (JSON) + manifest (JSON with format_version, created_at, etc.)

def load_world(path: str) -> WorldIR
    # Reads manifest, checks format_version, parses world.ir
    # Raises WorldFormatVersionError if version mismatch
```

**Dependencies:** core/units, world_ir/schema, world_ir/coordinates

**Invariants:**
- Round-trip: load(save(world)) == world
- Format version pinning (unknown versions rejected)
- All Quantity fields serialize with units (not bare floats)

**Test Boundary:**
- Round-trip fidelity
- Format compatibility
- Version rejection

---

#### world_ir/runtime.py

**Purpose:** Load WorldIR, resolve transforms, prepare for simulation.

**Owns:**
- `WorldRuntime`: holds loaded WorldIR + coordinate registry
- Frame resolution on startup
- Lazy entity lookups

**Must Never Own:**
- Physics simulation (that's engine/world)
- Event dispatch (that's events/)

**Public API:**
```python
class WorldRuntime:
    def __init__(self, world: WorldIR)
    def get_entity(self, entity_id: str) -> Entity
    def transform_point(self, p: Vec3, from_frame: Frame, to_frame: Frame) -> Vec3
    def entities_in_region(self, aabb: AABB) -> List[Entity]  # spatial query
```

**Dependencies:** core/units, world_ir/schema, world_ir/coordinates, world_ir/serialization

**Invariants:**
- All queries return consistent views (no mid-query mutations)
- Transform cache is updated when WorldIR changes (or raise NotImplementedError)

**Test Boundary:**
- Load + query workflow
- Transform correctness across frames
- Spatial query correctness

---

### engine/physics/core/interface.py

**Purpose:** Abstract physics backend interface; defines all physics contracts.

**Owns:**
- `IPhysicsBackend`: ABC (createWorld, step, query, serialize)
- `ISolver`: ABC (initialize, step, finalize)
- `SimulationContext`: all data a solver needs per step
- `SimulationResult`: all data a solver produces per step

**Must Never Own:**
- Concrete implementations (that's physics/backend/simple_backend)
- RigidBody state (that's physics/rigid)
- Entity registry (that's world_ir)

**Public API:**
```python
class IPhysicsBackend(ABC):
    @abstractmethod
    def create_world(self, config: PhysicsWorldConfig) -> PhysicsWorldHandle
    
    @abstractmethod
    def step(self, world: PhysicsWorldHandle, dt: float) -> SimulationResult
    
    @abstractmethod
    def query_contacts(self, world: PhysicsWorldHandle) -> List[Contact]
    
    @abstractmethod
    def query_raycast(self, world: PhysicsWorldHandle, 
                      origin: Vec3, direction: Vec3, max_distance: float
                      ) -> RaycastHit | None
    
    @abstractmethod
    def serialize(self, world: PhysicsWorldHandle) -> dict
    
    @abstractmethod
    def deserialize(self, data: dict) -> PhysicsWorldHandle

class ISolver(ABC):
    @abstractmethod
    def initialize(self, ctx: SimulationContext) -> None
    
    @abstractmethod
    def step(self, ctx: SimulationContext) -> SimulationResult
    
    @abstractmethod
    def finalize(self, ctx: SimulationContext) -> None
```

**Dependencies:** core/units, core/math, world_ir/schema, core/clock

**Invariants:**
- All backends are deterministic given same seed + input state
- All backends must implement serialization (round-trip testing)
- No implicit global state

**Test Boundary:**
- Backend contract verification (golden tests apply to all backends)

---

### engine/physics/backend/simple_backend.py

**Purpose:** Concrete P1-tier rigid body physics backend (semi-implicit Euler, sphere/box/plane collision).

**Owns:**
- `SimpleRigidBodyBackend`: implements IPhysicsBackend
- `PhysicsWorld`: holds bodies, planes, solver state
- Integration loop (bodies → contacts → resolution → events)

**Must Never Own:**
- Entity registry (that's world_ir)
- Coordinate transforms (that's world_ir/coordinates)
- Fracture logic (that's physics/destruction)

**Public API:**
```python
class SimpleRigidBodyBackend(IPhysicsBackend):
    # Implements all abstract methods from IPhysicsBackend
    pass
```

**Dependencies:** core/units, core/math, core/clock, physics/core/*, physics/rigid, physics/collision, physics/materials, physics/constraints, events

**Invariants:**
- Step is deterministic: identical seed → identical state
- step(world, dt) modifies world in-place (returns SimulationResult)
- Serialization round-trip: load(serialize(step(world))) continues identically

**Test Boundary:**
- RB_001/RB_002 golden tests pass
- Determinism tests (seed → byte-identical)
- Serialization roundtrip

---

### events/

**Purpose:** Event bus: type-scoped subscriptions, deterministic ids, event log.

**Owns:**
- `Event`: dataclass (event_id, type, timestamp, tick, source_refs, target_refs, parameters, cause_event_ids, severity, confidence)
- `EventBus`: emit(type, **fields) → Event, subscribe(type, handler), events list
- Deterministic ID generation: f"evt-{seed}-{tick:08d}-{sequence:04d}"

**Must Never Own:**
- Physics solver (that's physics/)
- Entity registry (that's world_ir)

**Public API:**
```python
class EventBus:
    def emit(self, type: str, timestamp: float, tick: int, **fields) -> Event
    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None
    def events_of_type(self, event_type: str) -> List[Event]
    
class Event:
    event_id: str
    type: str
    timestamp: float
    tick: int
    source_refs: tuple[str, ...]
    target_refs: tuple[str, ...]
    # ... etc
```

**Dependencies:** core/units, core/clock

**Invariants:**
- Event IDs are deterministic: same seed + tick + sequence → same ID
- Events are append-only (no mutation)
- Subscribers are called synchronously (no async)

**Test Boundary:**
- Event emission and subscription
- Deterministic ID generation
- Serialization (to_dict, from_dict)

---

## Public API Surfaces

### What each module exports

Each module's `__init__.py` explicitly exports its public API:

```python
# engine/core/__init__.py
from .units import Quantity, Unit
from .rng import DeterministicRNG
from .clock import SimulationContext
from .jobs import JobSystem
from .math import Vec3, Quat, Mat3, Mat4
from .logging import Logger, get_logger
from .errors import RealityEngineError

__all__ = [
    "Quantity", "Unit",
    "DeterministicRNG",
    "SimulationContext",
    "JobSystem",
    "Vec3", "Quat", "Mat3", "Mat4",
    "Logger", "get_logger",
    "RealityEngineError",
]
```

**Rule:** If it's not in `__all__`, it's internal; clients must never import it directly.

**Rule:** Internal imports (things not in `__all__`) may change at any time; public imports have semver guarantees.

---

## Configuration System

### Config Files

**Location:** `config/` directory (not in repo yet)

**Files:**
- `config/physics.toml` — physics solver parameters
- `config/simulation.toml` — global simulation settings
- `config/logging.toml` — logging configuration

### Config Loading

```python
# engine/core/config.py
class Config:
    @staticmethod
    def load(config_path: str) -> dict
        # Loads TOML, validates schema, returns dict
        # Raises ConfigError if schema mismatch
    
    @staticmethod
    def from_dict(data: dict) -> PhysicsConfig | SimulationConfig | LoggingConfig
        # Type-safe config construction
```

### Example: physics.toml

```toml
[physics]
gravity = { value = -9.81, unit = "m/s^2" }
timestep = { value = 0.01, unit = "s" }
max_contacts_per_body = 8
linear_damping = 0.01
angular_damping = 0.05
sleep_threshold_linear = { value = 0.05, unit = "m/s" }
sleep_threshold_angular = { value = 0.05, unit = "rad/s" }
sleep_time = { value = 0.5, unit = "s" }

[broadphase]
aabb_margin = { value = 0.1, unit = "m" }

[numerics]
nan_inf_hard_failure = true
energy_absolute_floor = { value = 1.0, unit = "J" }
energy_explosion_relative_threshold = 1000.0
```

### Invariants

- All numeric config uses Quantity (with units)
- Config is immutable after load
- Unknown keys raise error (no silent ignores)
- Config versioning: `config_version = "1.0"`

---

## Logging & Diagnostics

### Logging API

```python
# engine/core/logging.py
class Logger:
    def debug(self, msg: str, **kwargs) -> None
    def info(self, msg: str, **kwargs) -> None
    def warning(self, msg: str, **kwargs) -> None
    def error(self, msg: str, **kwargs) -> None
    def critical(self, msg: str, **kwargs) -> None

def get_logger(name: str) -> Logger
    # Returns logger for module name
```

### Structured Logging Format

All logs are JSON (no plaintext):

```json
{
  "timestamp": "2026-09-04T12:34:56.789Z",
  "level": "ERROR",
  "module": "engine.physics.collision",
  "message": "Invalid narrowphase result: NaN in contact normal",
  "context": {
    "body_a_id": "sphere_1",
    "body_b_id": "box_2",
    "expected_normal": "[1.0, 0.0, 0.0]",
    "actual_normal": "[NaN, NaN, NaN]"
  }
}
```

### Diagnostic Categories

Every solver exposes a `Diagnostics` report at step-end:

```python
@dataclass
class Diagnostics:
    nan_count: int              # NaN values encountered
    inf_count: int              # Inf values encountered
    solver_iterations: int      # Contact solver passes
    energy_kinetic: float       # Total KE this step (Joules)
    energy_potential: float     # Total PE this step
    warnings: List[str]         # Non-fatal issues
    timing_us: Dict[str, int]   # Subsystem timings (microseconds)
```

---

## Error Model

### Exception Hierarchy

```python
# engine/core/errors.py

class RealityEngineError(Exception):
    """Base exception for all Reality Engine errors."""
    pass

class ArchitectureError(RealityEngineError):
    """Invariant violation, architectural constraint broken."""
    pass

class ConfigError(RealityEngineError):
    """Configuration loading or validation failed."""
    pass

class SerializationError(RealityEngineError):
    """Serialization or deserialization failed."""
    pass

class PhysicsError(RealityEngineError):
    """Physics solver error."""
    pass

class NumericsError(PhysicsError):
    """NaN, Inf, or numerical divergence detected."""
    pass

class CoordinateFrameError(RealityEngineError):
    """Frame resolution failed, transform invalid."""
    pass

class EntityError(RealityEngineError):
    """Entity registry invariant violated."""
    pass
```

### Error Handling Policy

- **Hard failures:** NaN/Inf in physics output → raise `NumericsError` immediately (spec §86)
- **Architectural violations:** Missing interface implementation → raise `ArchitectureError`
- **Graceful degradation:** Non-critical solver step failure → log warning, skip that contact/entity, continue
- **Never silent:** No "shrug and assume zero" — always fail or warn loudly

---

## Event Model

### Event Taxonomy (§85)

Currently implemented:
- `ContactEvent`: two bodies in contact (approach speed ≤ 2 m/s)
- `ImpactEvent`: two bodies colliding (approach speed > 2 m/s)

Future (stubs only):
- `BreakEvent`, `FractureEvent`, `ExplosionEvent`, `FireEvent`, `FloodEvent`, `WindEvent`, etc.

### Event Payload Structure

```python
@dataclass
class Event:
    event_id: str                           # evt-{seed}-{tick:08d}-{sequence:04d}
    type: str                               # ContactEvent, ImpactEvent, ...
    timestamp: float                        # world time (seconds)
    tick: int                               # simulation tick
    
    source_refs: tuple[str, ...]            # actor(s) that caused this
    target_refs: tuple[str, ...]            # entity/entities affected
    
    parameters: dict                        # solver-specific:
                                            # ContactEvent: {normal, penetration, approach_speed}
                                            # BreakEvent: {fracture_pattern, shard_count}
    
    cause_event_ids: tuple[str, ...]        # causal chain (what triggered this)
    severity: str                           # info, warning, error
    confidence: float                       # [0.0, 1.0]
    
    branch_id: str | None                   # simulation branch (None = canon)
    actor_id: str | None                    # AI agent or user who triggered
    deterministic_seed: int                 # seed this event used
    resulting_state_refs: tuple[str, ...]   # snapshot ids resulting from this
```

### Invariants

- Event IDs are deterministic: `seed + tick + sequence` uniquely identifies an event
- Once emitted, events are immutable
- `cause_event_ids` chain must be acyclic
- `timestamp` must be monotonically increasing within a simulation run

---

## Serialization & Versioning

### Versioning Strategy (§103)

**Format version:** Embedded in manifest (JSON + world.ir):

```json
{
  "format_version": 1,
  "schema_version": "1.0",
  "created_at": "2026-09-04T12:34:56Z",
  "reality_engine_version": "0.1.0"
}
```

**Rules:**
- Major version bump (1→2): breaking change, old files unloadable
- Minor version bump (1.1→1.2): backward compatible
- Unknown versions → `WorldFormatVersionError`, refuse to load
- Every subsystem that serializes must version its format

### Deterministic Serialization

```python
# All floats use fixed-precision JSON
# E.g., 1.5 meters →
{
  "position": {
    "value": [1.5, 2.0, 3.0],
    "unit": "meter"
  }
}

# Never bare floats; always {value, unit}
# No implicit conversions
```

### Round-Trip Guarantee

```python
def test_roundtrip(world: WorldIR):
    serialized = world.serialize()
    restored = WorldIR.deserialize(serialized)
    assert world == restored  # bit-identical
```

Every serializable object must pass this test (or explicitly document why not).

---

## Testing Structure

### Test Organization

```
tests/
├── conftest.py              # pytest fixtures, markers
├── unit/                    # Unit tests per module
│   ├── test_units.py        # engine/core/units
│   ├── test_math.py         # engine/core/math
│   ├── test_rng.py          # engine/core/rng
│   ├── test_entity.py       # engine/world_ir/entity
│   ├── test_coordinates.py  # engine/world_ir/coordinates
│   ├── test_serialization.py # engine/world_ir/serialization
│   ├── physics/
│   │   ├── test_rigid_body.py
│   │   ├── test_collision.py
│   │   ├── test_contact.py
│   │   └── test_materials.py
│   └── events/
│       └── test_event_bus.py
│
├── integration/             # Cross-module tests
│   ├── test_world_physics.py    # WorldRuntime + PhysicsBackend
│   ├── test_serialization_roundtrip.py
│   └── test_determinism.py
│
├── golden/                  # Golden scene tests (§91-93)
│   ├── rb_001_falling_cube.py
│   ├── rb_002_stacked_boxes.py
│   ├── rb_003_bouncing_ball.py
│   └── fixtures/
│       ├── cube.json
│       └── boxes.json
│
└── bench/                   # (see Benchmark Structure)
```

### Test Markers

```python
# conftest.py defines pytest marks for categorization:

@pytest.mark.physics           # Physics solver tests
@pytest.mark.golden           # Golden scene tests (determinism checks)
@pytest.mark.serialization    # Serialization roundtrip tests
@pytest.mark.determinism      # Reproducibility tests
@pytest.mark.numerics         # Numerical health checks
@pytest.mark.slow             # Long-running tests (>1s)
@pytest.mark.regression       # Benchmark regression tests
```

### Run Tests

```bash
# All tests
pytest tests/

# Only unit tests
pytest tests/unit/

# Only golden scenes
pytest tests/golden/

# Only fast tests (skip slow)
pytest -m "not slow"

# With coverage
pytest --cov=engine --cov=events --cov-report=html tests/
```

### Test Fixtures

```python
# conftest.py

@pytest.fixture
def deterministic_rng():
    """RNG with fixed seed 42."""
    return DeterministicRNG(seed=42)

@pytest.fixture
def simple_backend():
    """Concrete physics backend."""
    return SimpleRigidBodyBackend()

@pytest.fixture
def world_runtime():
    """Loaded WorldIR with default config."""
    world_ir = WorldIR.load("tests/golden/fixtures/minimal_world.json")
    return WorldRuntime(world_ir)

@pytest.fixture
def physics_config():
    """Physics config from TOML."""
    return Config.load("config/physics.toml")
```

---

## Benchmark Structure

### Benchmark Organization

```
benchmarks/
├── conftest.py              # fixtures shared by benchmarks
├── physics/
│   ├── bench_rigid_body.py  # RigidBody integration performance
│   ├── bench_collision.py   # Broadphase + narrowphase
│   ├── bench_contact.py     # Contact solver
│   └── bench_serialization.py # Roundtrip latency
│
└── profiling/
    ├── profile_golden_scenes.py # Flame graph generation
    └── memory_bench.py          # Peak memory usage

```

### Benchmark Format (pytest-benchmark)

```python
# benchmarks/physics/bench_rigid_body.py

def test_integrate_sphere(benchmark, simple_backend, deterministic_rng):
    """Benchmark semi-implicit Euler integration of a sphere."""
    # Setup: create 1000 falling spheres
    world = setup_falling_spheres(count=1000)
    
    # Run one step
    result = benchmark(simple_backend.step, world, dt=0.01)
    
    # Assertions
    assert result.success
    assert result.diagnostics.nan_count == 0

def test_broadphase_scaling(benchmark, simple_backend):
    """Broadphase latency vs. entity count (scaling curve)."""
    for n_bodies in [100, 1000, 5000]:
        world = setup_bodies(n_bodies)
        result = benchmark(
            lambda: simple_backend.query_contacts(world),
            name=f"broadphase_{n_bodies}"
        )
        assert result.timing_us['broadphase'] < 1000 * n_bodies  # linear upper bound
```

### Run Benchmarks

```bash
# Run all benchmarks, store baseline
pytest benchmarks/ -v --benchmark-only --benchmark-save=baseline

# Compare against baseline
pytest benchmarks/ -v --benchmark-only --benchmark-compare=baseline

# Generate flame graph
python benchmarks/profiling/profile_golden_scenes.py

# Memory profiling
python -m memory_profiler benchmarks/profiling/memory_bench.py
```

---

## Plugin Architecture

### Plugin Interface

```python
# plugins/plugin_interface.py

class IPlugin(ABC):
    """All plugins must implement this interface."""
    
    @property
    @abstractmethod
    def name(self) -> str
        """Plugin name, used for registration."""
    
    @property
    @abstractmethod
    def version(self) -> str
        """Semantic version: X.Y.Z"""
    
    @property
    @abstractmethod
    def api_version(self) -> str
        """Minimum engine API version this plugin requires."""
    
    @abstractmethod
    def initialize(self, engine: PluginHost) -> None
        """Called on load. Store references, register handlers."""
    
    @abstractmethod
    def shutdown(self) -> None
        """Called on unload. Clean up resources."""
    
    @abstractmethod
    def on_world_loaded(self, world: WorldIR) -> None
        """Hook: world loaded."""
    
    @abstractmethod
    def on_step_complete(self, result: SimulationResult) -> None
        """Hook: physics step complete, events emitted."""
```

### Plugin Host

```python
# plugins/plugin_registry.py

class PluginHost:
    """Manages plugin lifecycle, permissions, sandboxing."""
    
    def register_event_handler(self, 
                               event_type: str, 
                               handler: Callable[[Event], None]) -> None
        """Register a handler for event type."""
    
    def query_world(self, entity_type: EntityType) -> List[Entity]
        """Query world (read-only access)."""
    
    def emit_diagnostic(self, msg: str, level: str) -> None
        """Plugin can emit diagnostic output."""
```

### Isolation Model

- Plugins load in **separate processes** (optional: sandboxing)
- Plugins can **read** world state via queries
- Plugins **cannot mutate** world state directly
- Plugins communicate via **events and handlers**
- Plugin **crash does not crash** engine (isolated)

---

## Summary: Dependency Rules in Practice

| From → To | Allowed? | Reason |
|-----------|----------|--------|
| `engine/physics/rigid` → `engine/core/math` | ✓ | Rigid uses math primitives |
| `engine/core/math` → `engine/physics/rigid` | ✗ | Math is foundational; no physics back-deps |
| `engine/world_ir` → `engine/physics/backend` | ✗ | WorldIR is Tier 2; backend is Tier 4 |
| `engine/physics/backend` → `engine/world_ir` | ✓ | Backend consumes WorldIR entities |
| `apps/cli` → `engine/physics` | ✓ | Apps depend on engine |
| `engine/physics` → `apps/cli` | ✗ | Engine must not know about apps |
| `plugins` → `engine/core` | ✓ | Plugins depend on core interfaces |
| `engine/core` → `plugins` | ✗ | Core must not know about plugins |

---

**This is the engineering foundation. Before implementing §12+ (fracture, disasters), all Tier 0-4 must be complete and tested. No circular dependencies, no god modules, no leaky abstractions.**
