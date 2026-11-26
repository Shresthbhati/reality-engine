# 12-Day Execution Report

Date: 2026-11-25. Baseline: `main` @ `8975eb2e`.

This report states what was done, what was verified, and what was **not** done. It
does not claim the 12-day plan is complete. Three of the twelve work packages were
executed in this session (Day 1 audit, Day 2 orchestration, Day 3 evidence fusion
into the canonical store); Days 4–12 were not executed.

## Work completed

### Day 1 — audit

`docs/audits/12_DAY_BASELINE_AUDIT.md`. Findings F1–F13 and gaps G1–G10, each
naming the file and fact it rests on. Baseline test run before any change:
`tests/test_multiengine_foundation.py`, `test_multi_engine_adapters.py`,
`test_multi_engine_comparator.py`, `test_multiengine_evidence_flow.py`,
`test_multiengine_worldir_bridge.py` → 68 passed.

### Day 2 — unified engine orchestration

Acceptance criteria from the plan, and where each is now enforced:

| Criterion | Where |
|---|---|
| Dependency chains execute in the correct order | `MultiEngineOrchestrator._order` (Kahn, priority tie-break) |
| CPU-only execution does not defer CPU-capable engines | `_compute_vram_budget` returns `None` without CUDA; `EngineSpec.cpu_capable` |
| GPU scheduling respects available memory | budget check, one retry after cache release, then `unavailable` with the measured budget |
| Multiple jobs cannot contaminate one another | registry lock, per-call context copy, orchestrator lock |
| Engine failures remain visible | `EngineExecution` for every engine; `MultiEngineReport.engine_executions` |
| No engine is marked successful without a valid output | `_validate_output` → `invalid_output` |

Also added: per-engine `timeout_s` (a timed-out call is *abandoned* — Python cannot
interrupt it — and the contract records that), `should_cancel`, transient-only
retry (`max_attempts`), and `EngineSpec.run_condition` respected by dispatch.

### Day 3 — real multi-engine evidence fusion

* `perception/engine_matrix_bridge.py` converts `UnifiedEngineOutput` observations
  into the canonical `EngineObservation` + `EvidenceContext` + `RunRecord` store.
  Content-keyed idempotency (re-import is a no-op); no calibrated confidence is
  claimed; a reload through a **new** `EvidenceMatrix` reader is part of the import
  result.
* `apps/api/evidence_world_stage.py` runs the multi-engine pass **before**
  `persist_world`, imports into the run's matrix, writes
  `multi_engine_evidence.json`, and `persist_world` reads that file into
  `world.metadata["multi_engine_evidence"]`. A reloaded world therefore retains the
  engine lineage, and re-persisting the same run reproduces the same
  content-addressed version instead of creating a new one.
* Contradiction detection now distinguishes "two engines, one object" (overlapping
  boxes on one image) from "two objects" (no overlap), which is the plan's
  requirement to distinguish same-object from different-object observations.
* Fabrications removed: LightGlue confidence was `match_count / match_count`, i.e.
  exactly 1.0 for every match; `compare_observations` reported relative and
  SfM-gauge position deltas as metres.

## Files changed

| File | Change |
|---|---|
| `perception/engine_orchestrator.py` | rewritten execution layer + `EngineExecution` contract, `unregister_engine` |
| `perception/engine_evidence.py` | `bbox_iou`; frame/scale-aware `compare_observations` |
| `perception/engine_adapters.py` | fabricated confidence removed; VRAM telemetry fixed |
| `perception/engine_matrix_bridge.py` | new: engine evidence → canonical Evidence Matrix |
| `perception/multi_engine_stage.py` | honest aggregate status; matrix import; matrix_root parameter |
| `apps/api/evidence_world_stage.py` | multi-engine pass moved before persistence; lineage written and read; cancellation guard |
| `tests/test_unified_engine_orchestration.py` | new, 23 tests |
| `tests/test_engine_matrix_bridge.py` | new, 10 tests |
| `docs/status.yaml`, `docs/STATUS.md` | S5-23 recorded as `IMPL-UNVERIFIED` |
| `docs/audits/12_DAY_BASELINE_AUDIT.md`, this file | new |

No commit was created (the plan requires separate branches and incremental commits
but does not authorise merging, and the working tree's branch was not changed).

## Tests executed

| Suite | Result |
|---|---|
| `tests/test_unified_engine_orchestration.py` | 23 passed |
| `tests/test_engine_matrix_bridge.py` | 10 passed |
| `tests/test_multiengine_foundation.py` + 6 sibling suites + the two new files | 122 passed |
| `tests/test_evidence_world_persistence.py` | 1 failed → fixed → passed |
| `tests/test_evidence_world_job.py` (5 tests) | passed after the cancellation guard |
| `tests/test_evidence_world_failure_semantics.py` | passed |
| `python -m ruff check --select F821 …` on all changed files | clean |

Two failures were caused by this change set and were fixed at the cause, not
worked around:

1. *Same world persisted twice no longer reused its version* — putting
   timestamp-derived observation ids in `world.metadata` made the world content
   differ per run. Fixed by making the persisted lineage content-derived
   (`observation_content_ids_by_engine`) and reading it from the run directory, so
   a retry reproduces the same version. Verified by the existing
   `test_persisting_the_same_world_again_reuses_the_version_instead_of_duplicating`.
2. *Operator-cancel test saw extra engine run records* — the multi-engine pass
   started after the flow had been cancelled. Fixed: a cancelled run starts no new
   engine work and records `multi_engine: {"status": "cancelled"}`. Verified by
   `test_operator_cancel_during_the_stage_stops_the_job_before_adoption_and_keeps_finished_evidence`.

## Hardware measurements

None. No GPU run was performed, so there is no VRAM or runtime figure to report for
the 6 GB target. `torch 2.14.0+cu130` is installed on this machine and
`_get_vram_peak()` was verified to return `0.0` rather than raising.

## Actual engine integrations

The orchestrator, the adapters and the matrix import are integrated code paths.
**None of them was executed against real photographs or real model weights in this
session**, so no engine output here is a measured reconstruction result. The
adapters used in tests were deterministic stubs.

## New production capabilities

1. A persisted World now carries traceable engine lineage, and a fresh process can
   recover the underlying observations from the run's Evidence Matrix.
2. An unavailable, cancelled, skipped or timed-out engine is a first-class,
   reported state — it can no longer look like a success.
3. On a CPU-only machine the pipeline runs the engines it can instead of deferring
   all of them.

## Outstanding blockers

Carried from the audit (unchanged by this work):

* **No metric ground truth.** `RELEASE.md`: six measured rooms configured,
  `total_verified: 0`, `overall_status: BLOCKED`. No accuracy claim possible.
* **No confidence calibration.** `apps/api/evidence_world_stage.py`
  `ENABLEMENT_CRITERIA` marks `confidence_calibration` unmet.
* **No real-image engine run recorded** for the multi-engine pass
  (`real_multi_photo_job_run` is unmet in the same table).
* **No CUDA verification** on the 6 GB target.

## Next work package

Days 4–12 of the plan were not attempted. The smallest next step that has a real
verification path is a **Day 4 subset**: run the existing COLMAP/LightGlue
registration on `datasets/south_building` (which already exists in the repository)
and record per-engine status, VRAM peak and runtime through the new
`EngineExecution` contract, before touching depth or architecture inference. That
produces the first measured evidence for the plan's camera-registration
acceptance criteria instead of more unverified code.
