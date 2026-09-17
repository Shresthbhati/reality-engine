# Reality Engine V10

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
- **engine/physics** — a P1-fidelity ("gameplay physics", §1.4) rigid
  body backend behind the `IPhysicsBackend` interface (§7.2):
  semi-implicit Euler integration, sphere/box/plane collision (§44
  LOD_0/1), sequential-impulse contacts with friction/restitution/
  sleeping, numerical health checks (§86), and a versioned
  serialize/deserialize format, and the full Euler rotational equation
  (gyroscopic term included, V11 §915). See `docs/BUILD_ORDER.md` for
  what it deliberately does not do yet.
- **events** (`events/`) — a deterministic event bus (§85, V11 §930
  schema) wired into the physics backend: a new contact emits
  `ContactEvent` or `ImpactEvent` (by approach speed), never re-fired
  for an ongoing resting contact.

Everything else under `engine/`, `reconstruction/`, `perception/`,
`apps/`, etc. is a scaffolded directory with a placeholder `README.md`
noting the spec section it corresponds to.

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
