# Flaky Test Report

Method: `pytest --count=N` (pytest-repeat, one process) and concurrent pytest processes on the same machine, torch-free
venv (Python 3.13), on the branch after merging `origin/main` (includes #418 "concurrent same-world commit"). No arbitrary
sleeps were added. **Rates below are observed counts, not estimates; where the sample is small it is said.**

## 1. Candidates and measured failure rates

| test | scenario | runs | failures | verdict |
|---|---|---|---|---|
| `test_furniture_commands_and_undo::test_cross_world_undo_isolation` | solo, repeated | 15 | 0 | not reproducible on current main; failed on pre-#418 branches (`KeyError: 'entities'`), fixed by #418 |
| `test_reconstruction_reliability::test_same_world_concurrent_completion_single_head` | solo, repeated | 15 | 0 | same: failed when two RECONSTRUCT jobs raced on one world before #418; not reproducible now |
| `test_adoption_transaction::test_a_concurrent_head_move_orphans_nothing...` | solo, repeated | 15 | 0 | stable |
| `test_refine_world_api::test_a_registered_engine_is_used_and_its_result_is_committed` | solo, repeated | 15 | 0 | passes alone; **failed once inside a full local shard 8** (order / shared-state dependent; not captured in detail) |
| `test_application_api::test_reconstruct_succeeded_grading_deterministic` | solo, repeated | 15 | 0 | |
| same | 2 processes concurrently (15 each) | 30 | **at least 1 process failed**: `sqlite3.DatabaseError: database disk image is malformed` on `GET /api/jobs/<id>` | reproduced once |
| same | 3 processes concurrently (10 each) | 30 | 0 | not reproduced |
| `test_application_api` (whole file) | inside local shard 1 | 1 | 1 (the same "malformed" error) | seen at baseline, local only |

Total for the SQLite-malformed signature: seen in 2 of about 9 concurrent-session executions of that file/test and 0 of 45
solo repetitions. It has **not** been seen on GitHub runners (each shard is its own machine).

## 2. What is established, what is not

* The malformed-database error appears only when several pytest sessions share one machine (or one very loaded shard). Every
  app test points `DATABASE_URL` at its own `tmp_path`, so a shared database file is *not* the cause; pytest's numbered
  base-temp directories (`pytest-of-<user>/pytest-N`, retention 3) and background worker threads that outlive a test are the
  suspects. **Root cause not proven; not fixed.** Mitigation in place: `scripts/run_isolated.py` runs risky workloads in
  their own process; the CI lanes run one pytest session per machine.
* Next step to prove or disprove: repeat the 2-process experiment with a unique `--basetemp` per session; if the error
  disappears, the shared base-temp is the cause.

## 3. Items on the requested list

| area | finding |
|---|---|
| concurrent world commits | fixed by #418 (retry/rebase); 0/15 failures now |
| distinct-world job isolation | `test_cross_world_undo_isolation` 0/15; covered |
| undo / redo | `test_furniture_commands_and_undo` stable (0/15 for the isolation test) |
| upload dedup | not flaky, but its test was **stale** after #415 (same-session upload is idempotent): fixed, see `CI_FAILURE_BASELINE.md` #2 |
| DB transactions | `test_adoption_transaction` 0/15 |
| native lifecycle | no native crash observed (`NATIVE_RUNTIME_DIAGNOSTICS.md`) |
| GPU contention | `test_13_gpu_memory_exhaustion...` depended on the host GPU (deterministic per host, not random): fixed by pinning the hardware probe |

## 4. Tooling added so flakiness is measurable

`scripts/ci_diagnostics.py` (per-shard wall time, slowest tests, skip reasons), `pytest-timeout` (a hung test fails with a
name instead of burning the job), `scripts/run_isolated.py` (crash/timeout classification). To measure a candidate:

```powershell
python -m pytest "<nodeid>" --count=20 -q -p no:cacheprovider --timeout=300 -rf        # pip install pytest-repeat
```
