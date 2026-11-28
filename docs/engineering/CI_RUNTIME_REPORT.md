# CI Runtime and Reliability Report

Measured 2026-11-28 on the maintainer workstation (16 logical CPUs, 16 GB RAM with 1-4 GB free because another session was
running; Windows 11), torch-free Python 3.13 venv, shards run **sequentially**, `-m "not slow"`, before the fixes in this PR.
Machine-readable data: `docs/engineering/ci_diagnostics_baseline.json` (produced by `scripts/ci_diagnostics.py`).
GitHub-hosted runners are faster per shard (CI log: shards took 1.5-5.2 minutes each); the *shape* (balance, slowest files) is
what transfers, absolute seconds do not.

## 1. Measurements

| quantity | value |
|---|---|
| test files / collected tests | 804 files, about 7,400 collected (`--collect-only`: 60 s locally, one pass) |
| tests executed in the 8 shards | 6,972 (6,800 passed, 39 failed, 2 errors, 131 skipped = 1.9 % skipped, budget 30 %) |
| shard wall seconds (local, sequential) | 685, 296, 524, 215, 450, 486, 501, 388 |
| shard balance | max/min **3.18**, max/mean **1.55** (the size-weighted split balances *file bytes*, not run time) |
| imports / process start | `import apps.api.main` 5.8 s; COLMAP process start (`colmap help`) 0.1 s |
| slowest tests | `test_room_refine` (68.5 s, 31.7 s, 14.8 s), `test_technologies_honesty::test_report_has_no_unconditional_verification_claim` (23 s), `test_job_progress::test_reads_stay_fast_while_a_job_blocks` (21 s), `test_engine_reachability...::test_measurement_runs_and_is_self_consistent` (21 s), `test_spatial_association_cache::test_the_96_view_case...` (20 s) |
| slowest files per shard | `test_application_api.py` 179 s (shard 1), `test_room_refine.py` 124 s, `test_upload_integrity_contract.py` 99 s (shard 3), `test_reconstruction_reliability.py` 94 s (shard 5), `test_evidence_world_api.py` 79 s (shard 6) |
| release gate (3 tests, real COLMAP, CUDA) | 462 s under `run_isolated.py` (its header claims 10-15 min on CPU) |
| native startup / GPU contention | GPU was idle (191 MiB) outside the release gate; concurrent GPU workloads were deliberately avoided (another session's benchmark). GPU contention is **not verified** |

## 2. Reliability findings

* Before this PR a single collection error interrupted a whole shard (4 shards in run 46124614788); the `core` matrix
  cancelled the other Python versions on the first failure; the collection step could not fail. All three are fixed.
* Failure inventory and classification: `CI_FAILURE_BASELINE.md`. Flaky candidates: `FLAKY_TEST_REPORT.md`.
* Skips are visible and budgeted: `ci_skip_report.py` (30 %), reasons in the diagnostics (top: torch/transformers not
  installed x28, "collection skipped" x25, missing local datasets x17). A requirement that matters is made a failure with
  `RE_STRICT_REQUIREMENTS=1`.

## 3. What CI now emits (per shard, as artefacts)

`junit.xml`, `shard_files.txt` (the exact manifest), `ci_diagnostics_<n>.json` (totals, slowest tests/files, skip reasons,
wall time). `ci-integrity` verifies the manifests partition `tests/`, the registry is current and equals what pytest
collects, and runs the dependency/tooling contract tests. Join the per-shard JSON with
`python scripts/ci_diagnostics.py junit1.xml ... junit8.xml --out ci_diagnostics.json` for shard balance.

## 4. Recommendations (not done: each needs a decision or a measured follow-up)

1. **Rebalance shards by measured duration**, not file size: shard 1 is 3.2x shard 4. Needs a committed `file -> seconds`
   table refreshed from the diagnostics artefacts; changing the algorithm moves every file between shards, so it should be a
   deliberate, reviewed change (the registry deliberately does not commit shard numbers).
2. `test_application_api.py` (179 s) and `test_room_refine.py` (124 s) are the two files worth splitting or marking `slow`.
3. Run the native/GPU/physical-data lane on a self-hosted runner with `RE_STRICT_REQUIREMENTS=1` and `run_isolated.py`; the
   GitHub-hosted runners have no COLMAP or GPU.
4. Python 3.10 and 3.12 results are unmeasured locally (the 3.11 launcher is broken on the workstation); rely on the `core`
   matrix (now `fail-fast: false`) for them.
