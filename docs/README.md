# Documentation Map

Start here. Three sources of truth, by question:

| Question | Where |
|---|---|
| What exists now (verified)? | [`implementation/IMPLEMENTATION_STATUS.md`](implementation/IMPLEMENTATION_STATUS.md), [`implementation/CAPABILITIES.md`](implementation/CAPABILITIES.md) |
| What comes next, in what order? | [`implementation/ROADMAP.md`](implementation/ROADMAP.md), [`implementation/PENDING_IMPLEMENTATION.md`](implementation/PENDING_IMPLEMENTATION.md), `.agent/execution/TASKS.yaml` |
| How are unfinished systems supposed to be built? | [`future/`](future/) — per-subsystem engineering specs |
| Why is the architecture the way it is? | [`engineering/DESIGN_DECISIONS.md`](engineering/DESIGN_DECISIONS.md) (decisions 17+), [`DECISIONS.md`](DECISIONS.md) (V11 decisions 1–16) |
| How do the pieces fit? | [`architecture/`](architecture/) — system, data model, pipeline, coordinate frames |
| How is quality enforced? | [`engineering/TESTING_STRATEGY.md`](engineering/TESTING_STRATEGY.md), [`engineering/VERIFICATION.md`](engineering/VERIFICATION.md), [`engineering/SCIENTIFIC_METHODS.md`](engineering/SCIENTIFIC_METHODS.md) |

Operational state for autonomous sessions lives in
`.agent/execution/STATE.md` (must be updated at session end — see
CLAUDE.md §54).

## Legacy documents

Root-level audit/ledger/matrix documents (REALITY_ENGINE_AUDIT.md,
BUILD_LEDGER.md, CAPABILITY_MATRIX.md, TEST_MATRIX.md, …) are
point-in-time audit records: historically accurate, not maintained as
canonical. When a root document and the canonical tree disagree, the
canonical tree wins — and the root document should be treated as an
artifact of its date. New status claims belong in
`implementation/IMPLEMENTATION_STATUS.md` and the pending-implementation
registry, not in new root-level files.
