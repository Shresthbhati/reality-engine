# Reality Engine — Engineering Constitution

**Authority:** this file governs how engineering work is decided, executed,
and accepted on this repository. It wins over conflicting guidance in older
documents. The mission it serves is `REALITY_ENGINE_MISSION.md` (same
directory); the work queue is `TASKS.yaml` (same directory); what exists is
`docs/implementation/IMPLEMENTATION_STATUS.md`; what each backend can do is
`CAPABILITIES.yaml`.

Ratified: 2026-09-15 (canonical-state consolidation, P0-01).

---

## Article I — Mission precedence

The mission (REALITY_ENGINE_MISSION.md) defines the north star and the
critical chain:

```text
Evidence → Time → Pose → Registration → Fusion → Perception
        → World → Persistence → Incremental World → Compilation
```

Every task is judged by whether it strengthens this chain. Work that does
not strengthen it is secondary and must not displace it. "Disaster is a
consumer, not the definition of the platform" is a standing architectural
ruling (see mission, Application boundary).

## Article II — One source of execution truth

- The work queue is `.agent/TASKS.yaml`. Nothing else defines priority.
- Session state lives in `.agent/EXECUTION_STATE.md`, updated at session
  end (what was verified, what stopped, exact next step).
- Capability facts live in `.agent/CAPABILITIES.yaml`; license facts in
  `.agent/LICENSES.yaml`.
- Status documentation lives in `docs/implementation/`; future-system
  specifications in `docs/future/`; decision records in
  `docs/engineering/DESIGN_DECISIONS.md`.
- Other files (`.agent/CURRENT_STATE.md`, `.agent/PROMPT_LOOP.md`,
  `.agent/ROADMAP_2026-09-14.md`, root `IMPLEMENTATION_STATUS.md`) are
  **archived history**: they record what previous sessions did and may
  contain superseded claims. Never treat them as current truth, never
  extend them. `docs/README.md`'s legacy rule extends to all of them.

## Article III — Repository is the source of truth

Documentation does not prove implementation. Precedence: executable code
→ runtime behavior → tests → build/package → configuration →
documentation → prior claims. Before implementing, inspect. Before
marking complete, run.

## Article IV — Verification before completion

- A task is DONE when implementation + integration + tests + failure
  paths + diagnostics + status/docs all exist and the named verification
  in TASKS.yaml has actually run. Code that compiles is not done.
- Full-suite regression is the gate for any change (suite status per
  EXECUTION_STATE.md). Known environment failures are recorded there and
  never silently reclassified.
- Deterministic offline coverage is mandatory for pipeline behavior;
  real-data verification is additionally required to advance perception/
  reconstruction statuses (recorded per task in TASKS.yaml).
- Never claim what was not observed. Distinguish OBSERVED / INFERRED /
  ASSUMED / NOT VERIFIED in reports and docs.

## Article V — Status discipline

`MISSING → FOUNDATION → PARTIAL → EXPERIMENTAL → IMPLEMENTED → VERIFIED`
transitions require the evidence named in TASKS.yaml and the registry
update in the same change. Blocked items record
decision-required/options/tradeoffs/recommendation. Never inflate; never
downgrade working code because docs are stale.

## Article VI — Priority authority

The TASKS.yaml order is the WHAT; the agent owns the HOW. Do not invent a
roadmap, do not reorder by preference, do not skip hard tasks for easy
ones. Finish a priority → begin the next automatically. If blocked by a
missing dependency, implement the minimum robust dependency, verify it,
return. If blocked by a genuine architecture/product decision, record it
and move to the next independent priority only if dependency order is
preserved.

## Article VII — Federation over reimplementation

Reality Engine owns the world; specialists provide capabilities. Before
reimplementing a mature algorithm, check CAPABILITIES.yaml and prefer
adapter/subprocess federation where license and integration constraints
permit. Never vendor source without a recorded license decision
(LICENSES.yaml). Never let a federated backend's failure fabricate a
result: honest unavailability (skip with reason / BACKEND_UNAVAILABLE)
is the contract, placeholders are forbidden.

## Article VIII — No invented facts, no silent conversions

- No fabricated test results, benchmarks, metrics, geometry, or coverage.
- Relative depth is never silently metric (Decision 023); raw integers
  are never meters without explicit scale (Decision 020); unknown GPS
  stays unknown, never 0,0; a missing mesh is never a labeled-as-real
  cube (Decision 022).
- Confidence is not uncertainty. Do not substitute one for the other.
- Observed reality is never overwritten by simulated state; simulation
  results are derived, versioned artifacts.

## Article IX — Provenance and immutability

Source identity, original files, and observation history survive every
transformation (append-only discipline). Deleting evidence tombstones;
it never rewrites lineage. Every derived artifact carries producer,
inputs, and content hash through the ArtifactStore.

## Article X — Change minimization and honest scope

Smallest correct change; no unrelated refactors; no silent redesign of
working systems. Partial completions are recorded as partial with the
exact remainder — never rounded up to done to close a session.

## Article XI — Core platform boundary

Core contains the world platform: evidence, time, calibration,
localization, registration, reconstruction, dense geometry, perception,
fusion, uncertainty, provenance, WorldIR, WorldStore, incremental
compilation, world compilers, and generic simulation primitives.
Application-specific logic (disaster workflows, dashboards, domain
scoring) belongs to application packages, not core. New application
features are rejected from core regardless of convenience.

## Article XII — Amendment

Changing this constitution or the mission requires updating the file and
recording the reason in EXECUTION_STATE.md in the same change. Silent
divergence is treated as a bug.
