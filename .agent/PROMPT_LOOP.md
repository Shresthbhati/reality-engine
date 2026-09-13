# Reality Engine — Agent Prompt Loop

> **This file IS the loop.** Every session (human or agent) starts here, not
> from a raw chat prompt. It makes the user's standing instruction operational:
> *always read CLAUDE.md, always target exact files, keep all campaigns moving
> through one queue.* Update it after every milestone — never let it go stale.

## THE LOOP (run every session, in order)

1. **READ `CLAUDE.md`** (root, then any subdirectory CLAUDE.md governing the
   target files). Non-negotiable, every session, before any tool call that
   edits code. Then read `.agent/CURRENT_STATE.md` and the queue below.
2. **VERIFY, DON'T TRUST**: `git status` + `git log --oneline -8` (concurrent
   agents commit on this lineage). Run the relevant test subset. Re-inspect
   the target directory — an empty dir means NOT STARTED regardless of what
   any doc says.
3. **PICK ONE MILESTONE** from the active campaign queue below — highest-value,
   dependency-safe, matching the user's latest instruction. If the user gave a
   campaign prompt, its FIRST section names the milestone.
4. **TARGET EXACT FILES** before writing: name the directory, files, symbols,
   seams you will compose with, acceptance criteria, and verification commands.
5. **IMPLEMENT → TEST → REGRESS** (full `python -m pytest -q`) → repair any
   test-caught bug at root cause (never weaken a test).
6. **RECORD**: commit locally (never push unless asked), update
   `.agent/CURRENT_STATE.md` + `docs/CAPABILITY_MATRIX.md`, tick the queue item
   here with the commit hash and observed numbers.
7. **CONTINUE** to the next queue item while budget lasts. Do not stop after
   one milestone.

## CAMPAIGN QUEUE (user-supplied, in order)

### Campaign A — Simulation, Physics & Digital Twin  *(current)*
Status legend: [x] done (commit) · [~] in progress · [ ] next · (—) blocked/none

- [x] Sec 1 WorldIR→Physics compiler — `cd77720` (13 tests; density
      provenance labels, honest skips, infinite-plane approximation pinned)
- [x] Sec 18–21 Event graph / temporal state / branching / replay — `2ce36f3`
      (18 tests; complements existing `engine/simulation/replay.py`, not a
      rebuild)
- [x] Sec 11 Wind field with drag coupling to physics — `bcdab40` (20 tests)
- [ ] **Sec 9 FIRE — ignition/fuel/heat/spread/suppression on physical
      material state (ignition_temperature/specific_heat already in WorldIR
      schema), wind-coupled, emitting causal events** ← CURRENT
- [ ] Sec 10 SMOKE (source/density/temperature/dissipation; sim state separate
      from rendering)
- [ ] Sec 12 STRUCTURAL GRAPH (foundations→columns→beams→walls→roof; loads,
      damage, failure; confidence/provenance on inferred structure)
- [ ] Sec 5+13 Wind→glass failure and explosion causal chains (compose:
      drag force → GlassPhysicsSolver → debris pool → event graph)
- [ ] Sec 14 FLOOD on reconstructed geometry (buoyancy/flow via water.py +
      physics compiler bodies)
- [ ] Sec 15–17 EARTHQUAKE / HURRICANE / TORNADO composed from wind+rain+
      structural+destruction (no scripts)
- [ ] Sec 25 Studio simulation workspace (play/pause/step/branch/compare over
      real state)
- [ ] Sec 26 Canonical simulation scenes + Sec 28 benchmark timings
      (`benchmarks/` harness exists — add simulation benches to it)

### Campaign B — Reality Studio, Export, Validation & Production Integration
Key targets: `docs/REALITY_STUDIO_PRODUCTION_AUDIT.md` (phase 0), WorldIR
hardening (phase 2), WorldDiff (phase 6 — reuse `BranchManager.compare`),
export reports (phase 7 — glTF/USD exporters exist, BOX-scope), Blender
adapter (phase 8), golden scenes 001–006 (phase 10), CLI/headless (phase 17),
production readiness report (phase 31). Start after Campaign A's fire/structural
core lands or on direct user instruction.

### Campaign C — Perception Fusion & Real-World Reconstruction
Key targets: `docs/REALITY_PERCEPTION_FUSION_AUDIT.md` (phase 0), scale
estimation (phase 6), depth+camera consistency (phase 8), 2D→3D lifting +
entity resolution (phases 14–16 — SAM backend exists; MiDaS depth backend
exists; fusion core `reconstruction/fusion/fusion.py` exists), conflict
resolution + world completeness (phases 25–26), quality gates (phase 33),
failure-first fixtures (phase 40). The 20–50-photo vertical slice lives here.

## STANDING RULES (compressed from the user's campaign prompts)

- LLM-free runtime. Deterministic algorithms only inside the engine.
- WorldIR is canonical; evidence immutable; derived data carries provenance;
  uncertainty survives transforms; conflicts stay inspectable; no silent
  anything (defaults, conversions, deletions, resolutions, failures).
- Simplifications allowed only when documented + labelled (spec sec 29
  pattern: approximation note + pin test).
- Never rebuild a validated system: check the queue's "done" items and the
  actual code before starting.
- Tests are the judge: every milestone carries tests with hand-computable
  expectations; failures are root-caused, not softened.
- Concurrent agents commit on `evidence-fusion`: stage only your files,
  never theirs; never push without being asked.

## SESSION LOG (append one line per session)

- 2026-09-13: loop file created; Campaign A secs 1/18–21/11 already landed
  (cd77720, 2ce36f3, bcdab40, 971 passed/2 skipped); starting FIRE.
