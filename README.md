# Reality Engine

> **ARCHIVED README — superseded 2026-09-23.** This README references
> the obsolete "V10" specification and incorrectly describes the project
> state. Physics, destruction, fire, and disaster simulation were moved
> to `reality-engine-child` on 2026-09-18. A full browser-based Studio
> UI (Next.js 16 + Three.js) is now implemented. See
> `docs/REALITY_ENGINE_CURRENT_STATUS.md` for the authoritative status.

World Compiler + Universal Physics + Destruction + Disaster + Studio UI.

Full spec: `REALITY_ENGINE_MASTER_SPECIFICATION_V10_WORLD_PHYSICS_AND_AGENT_BUILD_BIBLE.md`
(not versioned in this repo — keep it alongside as the standing reference).

## Status

Steps 1-11 of 33 (spec §108). See [`docs/BUILD_ORDER.md`](docs/BUILD_ORDER.md)
for exactly what's implemented versus scaffolded, and for the honest
list of physics-backend gaps (no CCD, no angular contact response —
read it before building anything on top of the physics layer). No
rendering, reconstruction, or disaster simulation yet — deliberate;
the spec itself says not to start there.

A "V11 DETAILED" upload (5 files, ~1.1M lines) was checked and is
overwhelmingly machine-generated template filler with one substantive
220-line prose section (V11 §900-969) that's been incorporated — see
`docs/BUILD_ORDER.md` for specifics before trusting anything else from
that upload set.

> **OUTDATED:** The following "Implemented" section reflects a
> pre-2026-09-15 state. See `docs/REALITY_ENGINE_CURRENT_STATUS.md`
> for current status.

Implemented:

- **WorldIR** (`world_ir/`) — the canonical world representation: entity
  system with relationship-integrity checks, coordinate frames/transforms,
  a versioned save/load format.
- **provenance** (`provenance/`) — the evidence-first model (OBSERVED /
  RECONSTRUCTED / ESTIMATED / INFERRED / GENERATED / UNKNOWN / CONFLICT)
  that every value in WorldIR carries.
- **engine/core** — canonical units with mismatch detection, a
  deterministic fixed-timestep clock, a dependency-ordered job system,
  and a seeded RNG with reproducible sub-streams.
- **engine/world** — a runtime that loads a `WorldIR` package and resolves
  points between coordinate frames.
- **engine/physics** — **MOVED TO `reality-engine-child` 2026-09-18**
  (rigid body backend, collision, contact solver).
- **events** (`events/`) — a deterministic event bus

> **CURRENTLY IMPLEMENTED (2026-09-23):**
> - **reconstruction/** — COLMAP backend, orchestrator, scale, depth, fusion, meshing
> - **perception/** — MiDaS, Mask R-CNN, SAM, lifting, fusion, quality, detail, tracking
> - **apps/** — Full `reality` CLI + FastAPI backend (10 modules)
> - **exporters/** — glTF, Blender, USDA, CityJSON, CityGML (13 modules)
> - **benchmarks/** — Competitive benchmark suite (13 modules)
> - **frontend/** — Next.js 16 + Three.js viewer, 12 API routes, 33 pages
> - **.github/workflows/ci.yml** — CI with multi-Python matrix, wheel build

Everything else under `engine/` (physics moved to child),
`reconstruction/`, `perception/`, `apps/`, `exporters/`, `frontend/`
now contains real code — see `docs/REALITY_ENGINE_CURRENT_STATUS.md`.
`gpu/`, `datasets/`, `plugins/`, `shaders/`, `tools/` remain
placeholder/empty.

## Setup

```bash
cd reality-engine
python -m venv .venv
.venv/Scripts/activate  # or: source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Repository layout

See spec §81 (repository structure) and §82 (physics subdirectory) for
the full rationale; `docs/BUILD_ORDER.md` tracks what's real vs. stubbed.
