# REALITY ENGINE — Mission and Engineering Directive

**Authority:** this file states what Reality Engine IS and what the
critical chain is. The engineering method that serves it is
`ENGINEERING_CONSTITUTION.md`; the ordered work queue is `TASKS.yaml`.

Ratified: 2026-09-15.

---

## North star

Build a **universal Reality Compiler + persistent WorldOS**:

```text
ANY REAL-WORLD SOURCE
(phone / drone / RGB-D / LiDAR / video / imagery / GNSS / IMU / GIS / …)
        ↓
Evidence
        ↓
Time + Calibration
        ↓
Localization
        ↓
Cross-source Registration
        ↓
Reconstruction
        ↓
Dense Geometry
        ↓
Perception
        ↓
Evidence Fusion
        ↓
Uncertainty + Provenance
        ↓
WorldIR
        ↓
WorldStore
        ↓
Persistent / Versioned World
        ↓
World Compilers
 ┌──────┼────────┬────────┬─────────┐
 ▼      ▼        ▼        ▼         ▼
BIM    GIS     Robotics  Runtime   Simulation
IFC    CityGML ROS       USD/glTF   Physics
                                │
                                ▼
                         Applications
                 disaster / urban / robotics /
                 construction / inspection / …
```

## Core architectural idea

**Reality Engine owns the world. Specialist systems provide
capabilities.**

Do not unnecessarily reinvent mature technologies. Prefer
adapters/federation for COLMAP, Open3D, OpenSfM, OpenMVS,
OpenDroneMap, AliceVision/Meshroom, ORB-SLAM3, OpenVINS, Basalt,
Nerfstudio, RoomFormer, SAM/detection models, IfcOpenShell, 3DCityDB,
SUMO, Habitat, Unreal, Godot, Blender, USD, and GIS tooling — where
their licenses and integration constraints permit the intended usage.

License discipline: before reusing source code (1) inspect the license,
(2) record it in LICENSES.yaml, (3) determine redistribution
obligations, (4) choose subprocess/library/adapter integration
appropriately. Algorithmic ideas may be independently implemented;
source code is treated according to its actual license.

## The real problem

The major challenge is **not producing one more mesh**. It is creating
**one coherent world** from evidence coming from different sensors,
timestamps, coordinate frames, capture sessions, reconstruction
systems, and uncertainty levels. Therefore the most important missing
chain is:

```text
TIME → TRAJECTORY → REGISTRATION → FUSION
```

This chain has priority over unrelated feature development.

Why each layer exists:

- Without TIME, sensors cannot be reliably correlated.
- Without TRAJECTORY, observations cannot be consistently localized.
- Without REGISTRATION, phone/drone/LiDAR/RGB-D results remain separate
  worlds.
- Without FUSION, we have disconnected geometry, not one measured world.
- Without PERCEPTION, geometry but not meaningful world entities.
- Without WorldIR, outputs rather than a canonical world.
- Without WorldStore, a model rather than a persistent world.
- Without incremental compilation, no maintainable world over time.
- Without compilers, the world cannot be consumed downstream.

## WorldIR

WorldIR is the canonical semantic world representation. Entities carry
identity, geometry, semantics, topology, materials, observations,
evidence, provenance, uncertainty, temporal history, coordinate frame,
external references, and representations where applicable. Every
important world statement is classifiable as:

```text
OBSERVED / INFERRED / DERIVED / PREDICTED / SIMULATED / PROCEDURAL
```

Do not mix these states. Observed reality must remain distinguishable
from simulation. (Maps to the provenance taxonomy in `provenance.py`;
DERIVED/PREDICTED/PROCEDURAL extensions are WorldIR 2.0 scope,
P9-01.)

## WorldStore

The world is versioned. New evidence produces a new world
state/version rather than silently destroying previous observed state
(append-only, see constitution Article IX).

## Application boundary

Disaster management is a downstream application, not the platform's
definition. Core keeps generic primitives (physics, materials,
collision, environmental fields, thermal/combustion primitives,
temporal events, branching, replay, simulation state); application
workflows live in application packages (constitution Article XI).

## Competitive strategy

Do not try to individually become the best COLMAP, the best SLAM
system, the best BIM tool, the best city generator, the best
simulator. Build the system that can:

1. select the appropriate specialist backend
2. execute it
3. evaluate its result
4. retry or switch backends when appropriate
5. fuse outputs
6. preserve evidence and provenance
7. quantify uncertainty
8. construct one canonical world
9. persist that world
10. compile it into downstream systems

The platform wins through **integration + world representation +
persistence + evidence + fusion + interoperability** — measured
against specialists (P19-01), not claimed.

## Anti-failure rules

Never:

- invent test results, benchmarks, or metrics
- claim completion without verification
- create fake implementations or placeholder backends
- hide unavailable backends behind success
- silently fall back to a semantically different algorithm
- replace real dense reconstruction with a fake dense result
- replace metric geometry with relative geometry without labeling it
- confuse confidence with uncertainty
- overwrite observed reality with simulated state
- add application-specific systems to the core platform

## Continuation rule

After finishing a task: update EXECUTION_STATE.md → check TASKS.yaml →
begin the next priority automatically. Sustained verified engineering
progress is the objective. DO THE WORK.
