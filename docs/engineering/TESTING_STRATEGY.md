# Testing Strategy

## The three tiers

| Tier | Requirement | Runs in CI | Examples |
|---|---|---|---|
| **Deterministic offline** | No network, no weights, no external binaries. Must always be green. | Yes (target) | `tests/test_cli_compile.py` (synthetic backend), `tests/test_meshing_pipeline.py`, `tests/test_mapping_spine_perception.py` (fake detector) |
| **Marked real-model** | Needs weights (download once, then cached) / torch. Skip **honestly with reason** when unavailable — never silently. | No (local/opt-in) | `tests/test_maskrcnn_real_model.py`, `tests/test_midas_backend.py` |
| **Real-data acceptance** | Needs COLMAP binary + dataset + ~30 min. Manual/detached today; to become marked-opt-in. | No | flagship runner on `datasets/room_capture` |

Baseline at last reconciliation: **1279 passed, 1 skip** (excluding the
pre-existing SAM torch-hub environment failure, `tests/test_sam_backend.py`
— BROKEN in this environment, unrelated to mapping work).

## Rules

1. **Never weaken a test to go green.** First determine: implementation
   bug, stale expectation, or environment problem.
2. **Environment failures are reported, not hidden.** A skipped
   real-model test names its missing dependency; a broken cache is a
   BROKEN status in the backlog, not a deleted test.
3. **Tests exercise real paths.** The deterministic CLI test asserts
   report structure, exit codes, metric anchoring from a measured
   baseline, honest skip notes, and artifact presence on disk — not
   mocks of the SDK.
4. **Failure cases are first-class:** corrupt manifests, missing images,
   insufficient points, unavailable binaries, wrong `module:Class` specs.
5. **Regression policy:** targeted tests for the touched subsystem first,
   then the affected suite, then the full suite before pushing.

## Current gaps (tracked in backlog)

- P0.18: the real-data acceptance test is manual — needs a marked,
  committed, opt-in test file.
- P7.29: no CI workflows exist; the deterministic tier is CI-ready.
- P7.28: wheel-contents/clean-install verification not yet performed.

See also [`VERIFICATION.md`](VERIFICATION.md) for what counts as
evidence for each status transition.
