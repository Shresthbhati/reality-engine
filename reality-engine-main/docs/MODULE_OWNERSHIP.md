# Reality Engine V11 — Module Ownership & Boundary Contracts

**Authority:** ARCHITECTURE.md, V11 §81-82
**Status:** Architectural specification (binding on implementation)
**Last Updated:** 2026-09-04

---

## Ownership Matrix

Each module has a single owner responsible for its public API, invariants, and dependencies.

| Module | Tier | Purpose | Owner | Invariants | Can Depend On | Must NOT Depend On |
|--------|------|---------|-------|-----------|---------------|--------------------|
| `engine/core/units` | 0 | SI unit enforcement | Physics Lead | All floats in APIs use Quantity | (none) | Any physics/app module |
| `engine/core/rng` | 0 | Seeded PRNG, sub-streams | Simulation Lead | Same seed → byte-identical sequence | (none) | physics, app, export |
| `engine/core/clock` | 0 | Fixed timestep simulation time | Simulation Lead | Ticks are integers; dt is immutable per run | (none) | physics, app |
| `engine/core/jobs` | 0 | Job scheduling, topological order | Simulation Lead | DAG only; IDs unique; results deterministic | (none) | physics, app |
| `engine/core/config` | 0 | TOML/YAML loading, schema validation | Infrastructure | Unknown keys rejected; immutable after load | (none) | physics, app |
| `engine/core/math` | 1 | Vec3, Quat, Mat3, Mat4 (no physics) | Physics Lead | Numeric stability; no side effects | `units`, `rng` | physics/rigid, physics/collision |
| `engine/core/logging` | 1 | Structured JSON logging, diagnostics | Infrastructure | All logs are JSON; levels enforced | (none) | physics/backend, app |
| `engine/core/errors` | 1 | Exception hierarchy, NumericsError | Physics Lead | NumericsError raised on NaN/Inf only | (none) | app |
| `world_ir/schema` | 1 | Entity, Transform, Provenance dataclasses | Data Lead | Entity IDs globally unique; confidence ∈ [0,1] | `units`, `provenance` | physics, app |
| `world_ir/entity` | 2 | EntityRegistry, lifecycle, relationships | Data Lead | No dangling refs after add/remove; IDs immutable | `schema` | physics, app |
| `world_ir/coordinates` | 2 | Frame, CoordinateRegistry, transform composition | Data Lead | No cycles in transform graph; T_A_B semantics | `math`, `units` | physics, app |
| `provenance` | 1 | Provenanced, Uncertainty, evidence states | Reconstruction Lead | Confidence ∈ [0,1]; canonical check deterministic | (none) | physics, app |
| `world_ir/serialization` | 3 | save_world, load_world, format versioning | Data Lead | Round-trip fidelity; format_version pinned | `schema`, `entity`, `coordinates` | physics, app |
| `world_ir/runtime` | 3 | WorldRuntime, frame resolution, entity queries | Data Lead | Query results consistent; transforms cached | `serialization` | physics, app |
| `events` | 4 | EventBus, Event schema, deterministic IDs | Simulation Lead | Event IDs deterministic; append-only log | `units`, `clock` | app (OK) |
| `physics/core/interface` | 3 | IPhysicsBackend, ISolver, SimulationContext | Physics Lead | All backends implement full contract | (none) | physics/rigid, physics/collision |
| `physics/core/math3` | 3 | Vec3/Mat3/Quat physics-aware (inertia frame docs) | Physics Lead | Body-frame angular velocity documented | (none) | (none) |
| `physics/rigid` | 4 | RigidBody, inertia, integrator, sleep | Physics Lead | Semi-implicit Euler with gyro; P0 fidelity | `core/interface`, `math3`, `materials` | physics/collision (OK sibling) |
| `physics/collision` | 4 | Shapes, broadphase, narrowphase, raycast | Physics Lead | Sphere/box/plane only; O(n²) AABB; sorted pairs | `rigid`, `math3` | (none) |
| `physics/materials` | 4 | PhysicsMaterial, canonical materials dict | Physics Lead | All materials in dict; density > 0 | (none) | (none) |
| `physics/constraints` | 4 | Contact solver, impulse, friction, sleep | Physics Lead | REST_VELOCITY_THRESHOLD; Baumgarte correction | `rigid`, `materials` | (none) |
| `physics/diagnostics` | 4 | Numerics health checks (NaN/Inf/energy) | Physics Lead | ENERGY_ABSOLUTE_FLOOR threshold | (none) | (none) |
| `physics/backend/simple_backend` | 4 | SimpleRigidBodyBackend concrete impl | Physics Lead | Deterministic; event_bus optional; serialize roundtrip | `rigid`, `collision`, `constraints`, `materials`, `diagnostics`, `events` | app (only at backend level, NOT from rigid/collision) |
| `physics/destruction` | 5 | Fracture, debris (stubs for step 12+) | Physics Lead | Subinterfaces TBD | `backend/interface` | (none) |
| `physics/fluids` | 5 | Water, buoyancy (stubs for step 14+) | Physics Lead | Subinterfaces TBD | `backend/interface` | (none) |
| `physics/fire` | 5 | Thermal, ignition, spread (stubs for step 15+) | Physics Lead | Subinterfaces TBD | `backend/interface` | (none) |
| `physics/weather` | 5 | Wind, rain, atmosphere (stubs for step 16+) | Physics Lead | Subinterfaces TBD | `backend/interface` | (none) |
| `physics/structural` | 5 | Load paths, capacity, collapse (stubs for step 17+) | Physics Lead | Subinterfaces TBD | `backend/interface` | (none) |
| `engine/world` | 5 | WorldRuntime orchestration, simulation loop | Simulation Lead | step() coordination; deterministic | `world_ir/runtime`, `physics/backend`, `events` | app |
| `exporters` | 6 | Unreal, Blender, OpenUSD, GIS adapters | Export Lead | Lossy export OK; fidelity tiers respected | `engine/world`, `world_ir` | (none) |
| `reconstruction` | 6 | Photogrammetry, SfM, MVS pipeline | Reconstruction Lead | Provenanced entities; confidence tracking | `world_ir`, `provenance` | (none) |
| `apps/cli` | 6 | Command-line interface | Apps Lead | SDK wrapper; error messages clear | `engine/world`, all lower tiers | (none) |
| `plugins` | 6 | Plugin host, discovery, sandboxing | Infrastructure | Isolated process; read-only queries | `engine/core`, `events` | (none) |

---

## Dependency Validation Rules (MUST be enforced)

### Rule 1: Tier Monotonicity
- Module in Tier N may only import from Tiers 0 to N
- No backwards edges (N → N+1 forbidden)

### Rule 2: Sibling Cycles Forbidden
- Within a tier, cycles are prohibited
- Example: `physics/rigid` ↔ `physics/collision` allowed (one-way) but NOT both directions

### Rule 3: App Layer Isolation
- `apps/` can depend on anything below
- `engine/` must NEVER import from `apps/`
- `events/` must NEVER import from `apps/`

### Rule 4: Test Code Separation
- `tests/` lives separately; no production code imports from `tests/`
- Test utilities go in `tests/conftest.py` or test fixtures

### Rule 5: Plugin → Core Only
- Plugins can depend on `engine/core` and `events/`
- Plugins can query world state (read-only)
- Plugins must NEVER mutate engine state or be imported by engine

### Rule 6: Backend Abstraction
- `physics/rigid`, `physics/collision`, `physics/materials`, `physics/constraints` implement `physics/core/interface`
- `physics/backend/simple_backend` is the ONLY concrete backend in Tier 4
- All other backends (Tier 5+: destruction, fluids, fire, weather, structural) must also implement `physics/core/interface`

---

## Public API Boundaries

Each module exposes a **public API** via `__all__` in its `__init__.py`. Clients import only from `__all__`.

### Example: engine/core/__init__.py
```python
from .units import Quantity, Unit
from .rng import DeterministicRNG
from .clock import SimulationContext
from .math import Vec3, Quat, Mat3, Mat4
from .logging import Logger, get_logger
from .errors import RealityEngineError

__all__ = [
    "Quantity", "Unit",
    "DeterministicRNG",
    "SimulationContext",
    "Vec3", "Quat", "Mat3", "Mat4",
    "Logger", "get_logger",
    "RealityEngineError",
]
```

**Enforcement:** Code importing `engine.core.units.Quantity` directly (not via `engine.core`) is a violation. Linter/mypy should reject.

---

## Invariant Contracts (per module)

### engine/core/units

**Must Hold:**
- All numeric fields in public physics APIs use `Quantity`, never bare floats
- Conversions across dimensions (meter → kilogram) raise `UnitError`
- Conversions within dimension (meter → foot) always succeed
- No silent assumptions about unit (e.g., assume SI)

**How to Verify:**
```python
# OK
mass = Quantity(5.0, Unit.KILOGRAM)
gravity = Quantity(9.81, Unit.METER_PER_SECOND_SQUARED)

# NOT OK
mass = 5.0  # bare float in physics API
```

### engine/core/math

**Must Hold:**
- All operations preserve numeric stability (unit tests with epsilon checks)
- No side effects (immutable — all methods return new objects)
- No reference frames, no physics interpretation
- composition is associative: (A * B) * C == A * (B * C)

### world_ir/schema

**Must Hold:**
- Entity IDs are unique within a world
- Confidence is always in [0.0, 1.0]
- All numeric fields use Quantity with units
- Relationships reference existing entities (checked by EntityRegistry)

### world_ir/entity

**Must Hold:**
- After `add_entity(e)`, `get_entity(e.id)` returns the same entity
- After `remove_entity(id)`, `get_entity(id)` returns None
- Parent-child cycles are forbidden
- No dangling parent_id refs in any entity

### world_ir/coordinates

**Must Hold:**
- All transforms are in right-hand, X-forward, Y-up frame
- T_A_B means "A ← B" (transforms B-frame points into A-frame)
- Transform graph is acyclic (no cycles)
- Composition is associative

### events

**Must Hold:**
- Event IDs are deterministic: `evt-{seed}-{tick:08d}-{sequence:04d}`
- Same seed + sequence → identical ID across runs
- Events are append-only (immutable log)
- Subscriber calls are synchronous (no async)

### physics/backend (all implementations)

**Must Hold:**
- Deterministic: identical seed + input state → identical output state
- Every step produces a `SimulationResult` with diagnostics
- Serialization round-trip: `load(serialize(state))` continues identically
- All contacts/events use deterministic IDs

---

## Circular Dependency Detector

**To run the detector** (manual):

```python
# detect_cycles.py
import ast
import os

def find_imports(filepath):
    with open(filepath) as f:
        tree = ast.parse(f.read())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[1])  # grab tier
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[1])  # grab tier
    return imports

# Build graph: module → {imports}
# Run DFS for cycles
```

**In CI/CD:** Add to `pre-commit` or GitHub Actions:
```yaml
- name: Check for circular dependencies
  run: python detect_cycles.py engine/ events/ apps/
```

---

## Migration Paths for Steps 12-33

When implementing a new module (step 12 = Fracture):

1. **Define interface** in existing higher-tier module:
   - `physics/core/interface.py` adds `IFractureBackend`
   - All methods are abstract (ABC)

2. **Implement in new module** (Tier 5):
   - `physics/destruction/fracture.py` implements `IFractureBackend`
   - Import only from Tiers 0-4 + own tier

3. **Test independently**:
   - `tests/destruction/test_fracture.py` imports only `physics/destruction`
   - Golden tests use full backend

4. **Wire into backend** (if Tier 4):
   - `physics/backend/simple_backend.py` can now call fracture solver
   - No circular deps: backend depends on destruction, not vice versa

5. **Update ARCHITECTURE.md**:
   - Move module from "NOT YET" to "implemented"
   - Update dependency table
   - Add ownership info

---

**Enforcement:** This document is binding. Violations must be fixed before merge.
