# Reality Engine Blueprint — start here

This folder is the **north star** for Reality Engine. It describes the product
as if we were building it from scratch, then maps every part onto the code that
already exists. Any coding agent should be able to pick the next task, build it,
and check the result against these documents without asking anyone.

> Status of things lives in `docs/implementation/` and `.agent/`.
> **What the product must be** lives here. When they disagree, this folder wins
> on *intent*; the status docs win on *what currently works*.

## Reading order

| # | Doc | Read it when |
|---|-----|--------------|
| 1 | [00_VISION.md](00_VISION.md) | Always. One page: what we are building and the demo that proves it. |
| 2 | [01_PRINCIPLES.md](01_PRINCIPLES.md) | Always. The 14 rules every change is checked against. |
| 3 | [02_GLOSSARY.md](02_GLOSSARY.md) | Always. Words like *Observation*, *Entity*, *OBSERVED* have one meaning. |
| 4 | [10_SYSTEM_ARCHITECTURE.md](10_SYSTEM_ARCHITECTURE.md) | Before touching more than one module. |
| 5 | [11_WORLDIR_CONTRACT.md](11_WORLDIR_CONTRACT.md) | Before writing anything that reads or writes world data. |
| 6 | [12_PIPELINE.md](12_PIPELINE.md) | Before touching any reconstruction or perception stage. |
| 7 | [13_BACKEND_AND_PERSISTENCE.md](13_BACKEND_AND_PERSISTENCE.md) | Before touching the API, database, storage or jobs. |
| 8 | [14_FRONTEND_UX.md](14_FRONTEND_UX.md) | Before touching any screen. |
| 9 | [15_ML_MODEL_STRATEGY.md](15_ML_MODEL_STRATEGY.md) | Before adding or swapping any model. |
| 10 | [16_MEDIA_GEO_VIDEO.md](16_MEDIA_GEO_VIDEO.md) | Before touching ingestion, video, GPS or coordinate frames. |
| 11 | [17_ONTOLOGY.md](17_ONTOLOGY.md) | Before adding a new object/surface class. |
| 12 | [18_VALIDATION_AND_BENCHMARKS.md](18_VALIDATION_AND_BENCHMARKS.md) | Before claiming anything works. |
| 13 | [20_REUSE_MAP.md](20_REUSE_MAP.md) | Before writing new code — the thing may already exist. |
| 14 | [30_BUILD_PLAN.md](30_BUILD_PLAN.md) | To pick your next task. |

## How an agent uses this folder (the loop)

1. Open [30_BUILD_PLAN.md](30_BUILD_PLAN.md). Take the **first task whose
   status is not DONE and whose dependencies are DONE**.
2. Read the docs that task links to. Read the code the task names.
3. Check [20_REUSE_MAP.md](20_REUSE_MAP.md): extend what exists, do not rebuild it.
4. Build the smallest complete change that meets the task's **Done when**.
5. Run the task's **Verify** steps. Paste the real output into the PR / handoff.
6. Run the **Vision check** at the bottom of this file.
7. Mark the task DONE in [30_BUILD_PLAN.md](30_BUILD_PLAN.md) with the commit
   hash and the evidence, and update `docs/implementation/IMPLEMENTATION_STATUS.md`.

## Vision check (run before every merge)

Answer each with *yes / no / not applicable*. Any wrong answer blocks the merge.

1. Does this keep **one world per physical place** (no duplicate rooms/walls/objects)? — P1, P2
2. Did any ML output reach the authoritative world **without** passing the ledger + geometric check? — P3 (must be *no*)
3. Is every new property tagged with an **evidence state** and **provenance**? — P4, P7
4. Are uncertain or ambiguous results **kept visible** instead of hidden or forced? — P5
5. Is every new number shown to a user backed by a **scale source** or labelled relative? — P10
6. Does the UI show **real backend state only** (no fake progress, no placeholder data)? — P14
7. Was it **verified by running it** (test output, API call, browser screenshot)? — Verification rule
8. Did you **reuse** an existing module or OSS project instead of writing a new one where possible? — Reuse rule

## Maturity labels (used everywhere)

`MISSING → FOUNDATION → PARTIAL → EXPERIMENTAL → IMPLEMENTED → VERIFIED`

- **FOUNDATION**: interfaces/contracts exist, no real behaviour.
- **PARTIAL**: some real behaviour, known gaps.
- **EXPERIMENTAL**: works end-to-end on test data only.
- **IMPLEMENTED**: works through the real app, acceptance evidence incomplete.
- **VERIFIED**: has a reproducible test **and** was shown working through the
  real application workflow (API + browser where applicable).

A function existing, a unit test passing, and a user journey working are three
different things. Only the last one earns VERIFIED.
