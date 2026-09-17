# Reality Engine V11 — Cross-Cutting Systems Design

**Authority:** V11 §82-90 (configuration, logging, error handling, events, serialization, versioning)
**Status:** Binding specification for implementation
**Date:** 2026-09-04

---

## Table of Contents

1. [Configuration System](#configuration-system)
2. [Logging & Diagnostics](#logging--diagnostics)
3. [Error Handling](#error-handling)
4. [Event System](#event-system)
5. [Serialization & Format Versioning](#serialization--format-versioning)
6. [Implementation Checklist](#implementation-checklist)

---

## Configuration System

### Design Principles

- **Single source of truth:** Config files (TOML) are the authoritative source
- **Type-safe loading:** TOML parser outputs strongly-typed dataclass
- **SI units everywhere:** All numeric config uses `Quantity(value, unit)`
- **Fail fast:** Unknown config keys raise error immediately (no silent ignores)
- **Immutable after load:** Config object cannot be mutated at runtime

### File Structure

```
config/
├── physics.toml         # Physics solver parameters
├── simulation.toml      # Global simulation settings
├── logging.toml         # Logging configuration
└── default/             # Example/default configs
    ├── physics.toml
    ├── simulation.toml
    └── logging.toml
```

### physics.toml Specification

```toml
[format]
config_version = "1.0"

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

[contact_solver]
max_iterations = 5
baumgarte_factor = 0.2
rest_velocity_threshold = { value = 0.5, unit = "m/s" }

[numerics]
nan_inf_hard_failure = true
energy_absolute_floor = { value = 1.0, unit = "J" }
energy_explosion_relative_threshold = 1000.0
max_velocity_cap = { value = 100.0, unit = "m/s" }
```

### simulation.toml Specification

```toml
[format]
config_version = "1.0"

[simulation]
fixed_timestep = { value = 0.01, unit = "s" }
max_substeps = 10
deterministic_seed = 42
enable_event_logging = true

[coordinate_frame]
world_frame = "X_FORWARD_Y_UP"
handedness = "RIGHT"
```

### logging.toml Specification

```toml
[format]
config_version = "1.0"

[logging]
level = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
format = "JSON"  # JSON (structured) or TEXT (for debugging only)
output = "stdout"  # stdout, stderr, or file path

[json_output]
include_timestamp = true
include_module = true
include_thread_id = false

[diagnostics]
enable_performance_tracking = true
enable_numerics_checks = true
report_interval_seconds = 10
```

### Loading API

```python
# engine/core/config.py

from dataclasses import dataclass
from typing import Optional
import tomllib

@dataclass
class PhysicsConfig:
    """Immutable physics configuration."""
    gravity: Quantity
    timestep: Quantity
    max_contacts_per_body: int
    linear_damping: float
    angular_damping: float
    sleep_threshold_linear: Quantity
    sleep_threshold_angular: Quantity
    sleep_time: Quantity
    aabb_margin: Quantity
    max_solver_iterations: int
    baumgarte_factor: float
    rest_velocity_threshold: Quantity
    nan_inf_hard_failure: bool
    energy_absolute_floor: Quantity
    energy_explosion_threshold: float
    
    def __setattr__(self, name, value):
        raise RuntimeError("PhysicsConfig is immutable")

@dataclass
class SimulationConfig:
    fixed_timestep: Quantity
    max_substeps: int
    deterministic_seed: int
    enable_event_logging: bool
    world_frame: str  # "X_FORWARD_Y_UP"
    handedness: str  # "RIGHT"

@dataclass
class LoggingConfig:
    level: str  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    format: str  # JSON or TEXT
    output: str
    include_timestamp: bool
    include_module: bool
    include_thread_id: bool
    enable_performance_tracking: bool
    enable_numerics_checks: bool
    report_interval_seconds: float

class Config:
    @staticmethod
    def load_physics(path: str) -> PhysicsConfig:
        """Load physics.toml, validate schema, return immutable config."""
        with open(path, 'rb') as f:
            data = tomllib.load(f)
        # Validate format_version
        if data['format']['config_version'] != "1.0":
            raise ConfigError(f"Unsupported config version: {data['format']['config_version']}")
        # Parse into PhysicsConfig (with Quantity conversion)
        return _parse_physics_config(data['physics'])
    
    @staticmethod
    def load_simulation(path: str) -> SimulationConfig:
        # Similar to load_physics
        pass
    
    @staticmethod
    def load_logging(path: str) -> LoggingConfig:
        # Similar to load_physics
        pass
```

### Invariants

1. **Format version must match:** Unknown versions raise `ConfigError`
2. **All numeric fields use Quantity:** No bare floats in the config object
3. **Unknown keys raise error:** No silent ignores
4. **Config is immutable:** Attempt to mutate raises `RuntimeError`
5. **Defaults provided for optional fields:** If a field is not in TOML, provide sensible default

---

## Logging & Diagnostics

### Design Principles

- **Structured JSON by default:** All logs are JSON objects (not plaintext)
- **Deterministic:** Same event produces identical JSON across runs
- **Severity levels:** DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Module context:** Every log includes its originating module
- **Timestamps:** ISO 8601 UTC
- **Diagnostics bundled:** Each physics step emits a `Diagnostics` record

### JSON Log Format

```json
{
  "timestamp": "2026-09-04T12:34:56.123456Z",
  "level": "WARNING",
  "module": "engine.physics.collision.narrowphase",
  "message": "Sphere-vs-sphere contact produced degenerate normal",
  "context": {
    "body_a_id": "sphere_1",
    "body_b_id": "sphere_2",
    "overlap": 0.0001,
    "normal": [0.0, 0.0, 0.0],
    "penetration": 0.0001
  }
}
```

### API

```python
# engine/core/logging.py

import json
import sys
from dataclasses import dataclass
from datetime import datetime

class Logger:
    def __init__(self, name: str):
        self.name = name
    
    def debug(self, msg: str, context: dict = None) -> None:
        self._log("DEBUG", msg, context)
    
    def info(self, msg: str, context: dict = None) -> None:
        self._log("INFO", msg, context)
    
    def warning(self, msg: str, context: dict = None) -> None:
        self._log("WARNING", msg, context)
    
    def error(self, msg: str, context: dict = None) -> None:
        self._log("ERROR", msg, context)
    
    def critical(self, msg: str, context: dict = None) -> None:
        self._log("CRITICAL", msg, context)
    
    def _log(self, level: str, msg: str, context: dict = None) -> None:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level,
            "module": self.name,
            "message": msg,
        }
        if context:
            log_entry["context"] = context
        json.dump(log_entry, sys.stdout)
        sys.stdout.write("\n")

def get_logger(name: str) -> Logger:
    return Logger(name)

# Usage:
logger = get_logger("engine.physics.rigid")
logger.info("Body integrated", context={"body_id": "box_1", "velocity": "[1.0, 2.0, 3.0]"})
```

### Diagnostics Record

Every physics step emits a `Diagnostics` record (bundled into `SimulationResult`):

```python
@dataclass
class Diagnostics:
    nan_count: int                  # NaN values encountered
    inf_count: int                  # Inf values encountered
    solver_iterations: int          # Contact solver passes this step
    energy_kinetic: float          # Total KE (Joules)
    energy_potential: float        # Total PE
    contacts_resolved: int         # Number of contact pairs processed
    warnings: list[str]            # Non-fatal issues
    timing_us: dict[str, int]      # Subsystem timings in microseconds
```

**Report to diagnostics logger at end of step:**

```python
logger = get_logger("engine.physics.diagnostics")
if diagnostics.nan_count > 0 or diagnostics.inf_count > 0:
    logger.error(
        "NaN/Inf detected in physics state",
        context={
            "nan_count": diagnostics.nan_count,
            "inf_count": diagnostics.inf_count,
            "tick": world.tick,
        }
    )
```

---

## Error Handling

### Exception Hierarchy

```python
# engine/core/errors.py

class RealityEngineError(Exception):
    """Base exception for all Reality Engine errors."""
    pass

class ConfigError(RealityEngineError):
    """Configuration loading or validation failed."""
    pass

class ArchitectureError(RealityEngineError):
    """Module invariant or interface contract violated."""
    pass

class SerializationError(RealityEngineError):
    """Serialization or deserialization failed."""
    pass

class PhysicsError(RealityEngineError):
    """Physics solver encountered an error."""
    pass

class NumericsError(PhysicsError):
    """NaN, Inf, or numerical divergence detected (HARD FAILURE per §86)."""
    pass

class CoordinateFrameError(RealityEngineError):
    """Frame resolution failed, transform invalid."""
    pass

class EntityError(RealityEngineError):
    """Entity registry invariant violated (e.g., dangling ref, duplicate ID)."""
    pass
```

### Error Handling Policy (per §86)

| Condition | Action | Exception |
|-----------|--------|-----------|
| NaN in velocity/position | Raise immediately | `NumericsError` |
| Inf in any float | Raise immediately | `NumericsError` |
| Energy explosion (relative > 1000x above floor) | Raise immediately | `NumericsError` |
| Unknown config key | Raise immediately | `ConfigError` |
| Dangling entity ref | Raise on add/remove | `EntityError` |
| Circular parent-child | Raise on add | `EntityError` |
| Mismatched frame assumption | Raise on transform | `CoordinateFrameError` |
| Missing implementation (ABC method) | Raise immediately | `ArchitectureError` |
| Non-critical solver issue | Log WARNING, skip contact, continue | (no exception) |

### Usage

```python
# Hard failure on NaN
if math.isnan(velocity.x) or math.isnan(velocity.y) or math.isnan(velocity.z):
    raise NumericsError(f"NaN velocity in body {body_id}: {velocity}")

# Graceful degradation
try:
    contact = resolve_contact(body_a, body_b)
except PhysicsError as e:
    logger.warning(f"Contact resolution failed, skipping: {e}")
    continue
```

---

## Event System

### Design Principles

- **Deterministic IDs:** `evt-{seed}-{tick:08d}-{sequence:04d}`
- **Append-only:** Events never mutate; immutable log
- **Type-scoped subscriptions:** Handlers subscribe to specific event types
- **Causal chain tracking:** Events link to their causes via `cause_event_ids`
- **Severity levels:** info, warning, error

### Event Schema (per V11 §930)

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Event:
    """Immutable event record."""
    event_id: str                           # evt-{seed}-{tick:08d}-{sequence:04d}
    type: str                               # ContactEvent, ImpactEvent, BreakEvent, ...
    timestamp: float                        # world simulation time (seconds)
    tick: int                               # simulation tick counter
    
    source_refs: tuple[str, ...]            # actor(s) that caused this event
    target_refs: tuple[str, ...]            # entity/entities affected
    
    parameters: dict                        # solver-specific fields
                                            # ContactEvent: {normal, penetration, approach_speed}
                                            # ImpactEvent: {impulse_magnitude, energy_dissipated}
                                            # BreakEvent: {fracture_pattern, shard_count}
    
    cause_event_ids: tuple[str, ...]        # causal chain (what triggered this)
    severity: str                           # info, warning, error
    confidence: float                       # [0.0, 1.0] likelihood of correctness
    
    branch_id: str | None                   # simulation branch (None = canon)
    actor_id: str | None                    # AI agent or user who triggered
    deterministic_seed: int                 # seed used to generate this event
    resulting_state_refs: tuple[str, ...]   # snapshots resulting from this event
```

### Event Bus API

```python
# events/bus.py

class EventBus:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self._events = []
        self._subscribers = {}  # type -> [handlers]
        self._sequence = 0
    
    def emit(self, type: str, timestamp: float, tick: int, **fields) -> Event:
        """Emit an event with deterministic ID."""
        event_id = f"evt-{self.seed}-{tick:08d}-{self._sequence:04d}"
        self._sequence += 1
        event = Event(event_id=event_id, type=type, timestamp=timestamp, tick=tick, **fields)
        self._events.append(event)
        self._notify_subscribers(event)
        return event
    
    def subscribe(self, event_type: str, handler) -> None:
        """Register a handler for event_type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
    
    def events_of_type(self, event_type: str) -> list[Event]:
        """Return all events of a given type."""
        return [e for e in self._events if e.type == event_type]
    
    def _notify_subscribers(self, event: Event) -> None:
        """Synchronously call all handlers for this event type."""
        if event.type in self._subscribers:
            for handler in self._subscribers[event.type]:
                handler(event)
```

### Known Event Types (per V11 §85)

```python
# events/types.py

# Tier 4 (Physics-core): currently implemented
CONTACT_EVENT = "ContactEvent"
IMPACT_EVENT = "ImpactEvent"

# Tier 5 (Destruction): step 12
BREAK_EVENT = "BreakEvent"
FRACTURE_EVENT = "FractureEvent"

# Tier 5 (Fire): step 15
FIRE_EVENT = "FireEvent"
IGNITION_EVENT = "IgnitionEvent"

# Tier 5 (Fluids): step 14
FLOOD_EVENT = "FloodEvent"

# Tier 5 (Weather): step 16
WIND_EVENT = "WindEvent"

# Tier 5 (Structural): step 17
STRUCTURAL_FAILURE_EVENT = "StructuralFailureEvent"

# Tier 5 (General)
DEBRIS_EVENT = "DebrisEvent"
DAMAGE_EVENT = "DamageEvent"
REPAIR_EVENT = "RepairEvent"
EXPLOSION_EVENT = "ExplosionEvent"

KNOWN_EVENT_TYPES = frozenset([
    CONTACT_EVENT,
    IMPACT_EVENT,
    # Add others as their solvers are implemented
])
```

**Rule:** Do NOT add to `KNOWN_EVENT_TYPES` until the solver exists and emits that event.

---

## Serialization & Format Versioning

### Design Principles

- **Format versioning:** Every serialized data structure includes a version
- **Round-trip fidelity:** `load(save(X)) == X` (bit-identical)
- **Explicit unit handling:** No bare floats; all numeric fields carry units
- **Forward-incompatible by default:** Unknown versions raise error (fail fast)
- **Backward compatibility:** Version N can understand N-1 (migration logic per version)

### Format Version Strategy

```
MAJOR.MINOR.PATCH

Major: Breaking change (incompatible format, old files unloadable)
Minor: Backward-compatible (new optional fields, old readers still work)
Patch: Non-format (documentation, tool fixes)
```

**Examples:**
- V1.0 → V1.1: Add optional field (backward compatible)
- V1.0 → V2.0: Change field type (breaking change)

### Manifest Structure

```json
{
  "format_version": "1.0",
  "schema_version": "1.0",
  "created_at": "2026-09-04T12:34:56Z",
  "reality_engine_version": "0.1.0",
  "world_ir_id": "world_canonical_2026_sep_04_1234",
  "checksum_sha256": "abc123..."
}
```

### World IR Serialization (world.ir JSON)

```json
{
  "format_version": "1.0",
  "entities": [
    {
      "id": "building_1",
      "type": "BUILDING",
      "transform": {
        "position": {
          "value": [100.0, 50.0, 0.0],
          "unit": "meter"
        },
        "orientation": {
          "value": [1.0, 0.0, 0.0, 0.0],
          "unit": "quaternion"
        }
      },
      "geometry_refs": [],
      "material_refs": ["concrete_standard"],
      "evidence": [],
      "confidence": 1.0
    }
  ],
  "materials": [
    {
      "id": "concrete_standard",
      "density": { "value": 2400, "unit": "kg/m^3" },
      "friction": 0.7,
      "restitution": 0.1,
      "young_modulus": { "value": 30e9, "unit": "Pa" }
    }
  ]
}
```

### Implementation

```python
# world_ir/serialization.py

import json
from pathlib import Path

class WorldFormatVersionError(SerializationError):
    pass

def save_world(world: WorldIR, path: str) -> None:
    """Save world to path with manifest."""
    base_path = Path(path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Save manifest
    manifest = {
        "format_version": "1.0",
        "schema_version": "1.0",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "reality_engine_version": "0.1.0",
        "world_ir_id": world.id,
    }
    with open(base_path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    # Save world IR
    world_data = {
        "format_version": "1.0",
        "entities": [entity.to_dict() for entity in world.entities],
        "materials": [mat.to_dict() for mat in world.materials],
    }
    with open(base_path / "world.ir", "w") as f:
        json.dump(world_data, f, indent=2)

def load_world(path: str) -> WorldIR:
    """Load world from path with version check."""
    base_path = Path(path)
    
    # Load and check manifest
    with open(base_path / "manifest.json") as f:
        manifest = json.load(f)
    
    if manifest["format_version"] != "1.0":
        raise WorldFormatVersionError(
            f"Unsupported format version: {manifest['format_version']}. "
            f"This engine supports 1.0 only."
        )
    
    # Load world IR
    with open(base_path / "world.ir") as f:
        world_data = json.load(f)
    
    if world_data["format_version"] != "1.0":
        raise WorldFormatVersionError(
            f"Unsupported world IR format: {world_data['format_version']}"
        )
    
    # Deserialize
    entities = [Entity.from_dict(e) for e in world_data["entities"]]
    materials = [PhysicsMaterial.from_dict(m) for m in world_data["materials"]]
    
    return WorldIR(id=manifest["world_ir_id"], entities=entities, materials=materials)
```

### Round-Trip Testing

```python
# tests/test_serialization_roundtrip.py

def test_world_ir_roundtrip():
    world = WorldIR.create_test_world()
    serialized = save_world(world, "/tmp/test_world")
    restored = load_world("/tmp/test_world")
    assert world == restored  # bit-identical
```

---

## Implementation Checklist

- [ ] `engine/core/config.py`: Config loading with immutability
- [ ] `engine/core/logging.py`: JSON structured logging
- [ ] `engine/core/errors.py`: Full exception hierarchy
- [ ] `events/event.py`: Event dataclass with immutability
- [ ] `events/bus.py`: EventBus with deterministic ID generation
- [ ] `world_ir/serialization.py`: save_world, load_world with versioning
- [ ] `.pre-commit-config.yaml`: Enforce no circular imports (custom hook or ruff rule)
- [ ] `docs/SYSTEMS_DESIGN.md`: This document
- [ ] `docs/MODULE_OWNERSHIP.md`: Ownership matrix and rules
- [ ] All modules updated to use logging API for diagnostics
- [ ] All backends emit Diagnostics records
- [ ] Integration tests for serialization round-trips
- [ ] Integration tests for determinism (same seed → same events)

---

**Status:** This is a binding specification. All implementation must conform.
