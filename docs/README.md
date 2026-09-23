# Documentation Map

Start here. Three sources of truth, by question:

| Question | Where |
|---|---|
| What exists now (verified)? | [`implementation/IMPLEMENTATION_STATUS.md`](implementation/IMPLEMENTATION_STATUS.md), [`implementation/CAPABILITIES.md`](implementation/CAPABILITIES.md) |
| What comes next, in what order? | [`.agent/TASKS.yaml`](../.agent/TASKS.yaml) (canonical queue), [`implementation/ROADMAP.md`](implementation/ROADMAP.md), [`implementation/PENDING_IMPLEMENTATION.md`](implementation/PENDING_IMPLEMENTATION.md) |
| How are unfinished systems supposed to be built? | [`future/`](future/) — per-subsystem engineering specs |
| Why is the architecture the way it is? | [`engineering/DESIGN_DECISIONS.md`](engineering/DESIGN_DECISIONS.md) (decisions 17+), [`DECISIONS.md`](DECISIONS.md) (V11 decisions 1–16) |
| How do the pieces fit? | [`architecture/`](architecture/) — system, data model, pipeline, coordinate frames |
| How is quality enforced? | [`engineering/TESTING_STRATEGY.md`](engineering/TESTING_STRATEGY.md), [`engineering/VERIFICATION.md`](engineering/VERIFICATION.md), [`engineering/SCIENTIFIC_METHODS.md`](engineering/SCIENTIFIC_METHODS.md) |

Operational state for autonomous sessions lives in
`.agent/EXECUTION_STATE.md` (must be updated at session end — see
CLAUDE.md §54); the canonical work queue is `.agent/TASKS.yaml`, the
governing doctrine `.agent/ENGINEERING_CONSTITUTION.md` and
`.agent/REALITY_ENGINE_MISSION.md`, capability/license facts in
`.agent/CAPABILITIES.yaml` / `.agent/LICENSES.yaml`.

## Legacy documents

Root-level audit/ledger/matrix documents (REALITY_ENGINE_AUDIT.md,
BUILD_LEDGER.md, CAPABILITY_MATRIX.md, TEST_MATRIX.md, BUILD_ORDER.md,
IMPLEMENTATION_MAP.md, REALITY_STUDIO_PRODUCTION_AUDIT.md, DECISIONS.md,
SYSTEMS_DESIGN.md, WORLD_IR_IMPLEMENTATION.md, INTEGRATION_AUDIT.md,
WIRING_POINTS.md, …) are point-in-time audit records: historically
accurate for their date, but many are now **ARCHIVED/SUPERSEDED** by
the 2026-09-15 through 2026-09-23 implementation campaigns:

- Physics (`engine/physics/*`, `engine/fire/*`, `engine/fluids/*`,
  `engine/weather/*`, `engine/disasters/*`) was moved to
  `reality-engine-child` on 2026-09-18.
- Frontend/Studio is implemented: Next.js 16 + Three.js viewer,
  12 API proxy routes, 33 page components (desktop + mobile).
- API backend is implemented: FastAPI with SQLite + async worker.
- Exporters include CityJSON/CityGML (not just glTF/USDA/Blender).
- CI exists: `.github/workflows/ci.yml` with multi-Python matrix.

When a root document and the canonical tree disagree, the canonical
tree wins — and the root document should be treated as an artifact of
its date. New status claims belong in
`REALITY_ENGINE_CURRENT_STATUS.md` (authoritative) and
`implementation/IMPLEMENTATION_STATUS.md`, not in new root-level files.
