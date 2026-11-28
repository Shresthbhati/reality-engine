# Native Runtime Diagnostics (COLMAP release-gate crash investigation)

Status 2026-11-28: **the native crash was NOT reproduced** on the maintainer workstation. What was established, with
evidence, is below. No cause is claimed that was not observed. This is a bounded external diagnosis plus two real defects
found on the way (a crash-vs-failure ambiguity, and the release gate's dependence on free RAM / engine set).

## 1. Environment (observed)

| item | value |
|---|---|
| OS | Windows 11 Home 10.0.26300, AMD64, 16 logical CPUs, 16 GB RAM (1.1-3.7 GB free during the runs: a second session was running a GPU benchmark) |
| GPU / driver | NVIDIA GeForce RTX 4050 Laptop, 6141 MiB, driver 621.40 (torch CUDA runtime 13.0, cuDNN 9.24) |
| COLMAP | 4.2.0 (commit be5e291, 2026-08-31, **with CUDA**), `C:/Users/shres/tools/colmap-extracted/bin/colmap` |
| Python | 3.14.7 (`C:\Python314`) for the release-gate runs; 3.13.0 venv for the torch-free CI reproduction |
| OpenCV / NumPy / pycolmap / torch | 5.0.0 / 2.5.3 / 4.2.1 / 2.14.0+cu130 |
| thread env | `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `CUDA_VISIBLE_DEVICES`, `RE_COLMAP_GPU` all unset |
| repo | branch `ci/verification-infrastructure` (origin/main ff73ea90 + merge of be06ee37); code commit is recorded in each job's `run_record` |

## 2. Experiments and results

| # | experiment | result |
|---|---|---|
| 1 | `tests/integration/test_release_gate.py` under pytest, faulthandler on, first run | no crash; failed after 128 s with `job.status == "partial"` (stage `done_with_warnings`), the test asserts `succeeded` |
| 2 | same, through `scripts/run_isolated.py` (child process, 1700 s tree-kill timeout) | outcome `test_failure`, exit 1, **no `Fatal Python error`**, 462 s; 1 passed, 2 failed (`..._to_corrected_v2_and_honest_failure`, `..._same_input_reproduces`); no hang, no native trace |
| 3 | the same upload+reconstruct job run directly, without pytest's `RE_MEMORY_ADMISSION=record` | job `failed` after 3 attempts: `admission refused: insufficient memory after waiting 30s: RAM: need 2500 MB + 768 MB reserve, 1710 MB available` |
| 4 | the same job under pytest's environment (`REALITY_ENV=test`, `RE_MEMORY_ADMISSION=record`) | job `partial`, `model_state=PARTIAL`, verdict `ACCEPT`, level 3 *dense reconstruction*, 6 cameras registered, 56 494 points, `adopted` and `validated` true; stages `pose=degraded`, `layout=degraded`, `perception/mesh/quality_report=skipped`, all others `succeeded` |
| 5 | real COLMAP 4.2.0 `feature_extractor` on 3 South Building photos, workspace path 110 chars | exit 0, database written (2.3 s) |
| 6 | same with a 352- and a 371-character workspace path | **exit 1, clean `Invalid options provided` message, no crash, no database** (the Windows MAX_PATH hazard is a clean error in 4.2.0 at this stage) |
| 7 | repeated and concurrent runs of unrelated app tests (`FLAKY_TEST_REPORT.md`) | one SQLite "disk image is malformed" under concurrent sessions; not a native crash |

Stage reached: every run that reached COLMAP completed sparse (`reconstruct=succeeded`) and dense (`dense=succeeded`);
the last completed stage before the gate's assertion is `changeset_commit`.

## 3. Findings

1. **No evidence of a COLMAP, pycolmap, OpenCV, NumPy-ABI, CUDA or OpenMP crash.** Thread counts were default, CUDA was in
   use (GPU idle between runs), teardown completed, the child exited normally with the test verdict. Candidate causes that
   remain *open* (not excluded, not supported by evidence):
   * memory exhaustion when free RAM is below ~2.5 GB. The engine refuses cleanly when admission is enforced (experiment 3),
     but root `conftest.py` sets `RE_MEMORY_ADMISSION=record`, which turns the refusal into a measurement under pytest, so a
     low-RAM machine runs COLMAP anyway under test;
   * a Windows long-path failure: known and guarded (`WorkspacePathTooLong`, limit 170 characters,
     `reconstruction/backend/colmap_backend.py::_check_workspace_path`); experiment 6 shows COLMAP 4.2.0 itself exits 1 cleanly.
2. **A real crash is indistinguishable from a pytest internal error by exit code on Windows.** Measured:
   `python -c "import faulthandler; faulthandler._sigsegv()"` exits with **3** and prints `Fatal Python error: Segmentation fault`;
   `os.abort()` exits 3 with `Fatal Python error: Aborted`. Exit code 3 is also pytest's `INTERNAL_ERROR`, so a crash that kills
   pytest reads as "pytest broke" in a CI log. `scripts/run_isolated.py` classifies by exit code *and* the `Fatal Python error:`
   marker. Conversely `ctypes.string_at(0)` on Windows is turned into an ordinary `OSError` (exit 1) although faulthandler still
   prints `Windows fatal exception: access violation`, so that line alone must not mean "crash"; the classifier ignores it
   (regression tests in `tests/test_ci_tooling.py`).
3. **The release gate is not independent of machine state.** Its strict assertion (`status == "succeeded"`) depends on free
   RAM and on which optional engines exist (`pose`/`layout` degrade, `perception`/`mesh` skip). The job grades `PARTIAL` here.
   **Owned by the reconstruction workstream (Cline); not changed by this work.** Either the gate declares the engine set it
   needs (`@pytest.mark.requires(...)`, strict lane) or the pipeline's grading needs review. Evidence: `run_record.stages` in
   experiment 4.
4. The test header documents "about 10-15 minutes on CPU"; measured 7.7 minutes with CUDA available.

## 4. How to reproduce / catch it next time

```powershell
$env:RE_STRICT_REQUIREMENTS = "1"      # a missing COLMAP / dataset is a failure here, never a skip
python scripts/run_isolated.py --timeout 1700 --report release_gate.json -- python -X faulthandler -m pytest tests/integration/test_release_gate.py -v -p no:cacheprovider
```

`release_gate.json` records outcome (`native_crash` / `test_failure` / `timeout`), exit code in hex, interpreter, platform and
the last 40 output lines including any faulthandler trace. On timeout the whole process tree (COLMAP included) is killed, so a
hang cannot leave COLMAP holding the GPU.

If a native crash does occur, capture before changing anything: the report JSON, `colmap --version`, `nvidia-smi -q`, free RAM
at start, the workspace path length, and the Windows Event Viewer "Application Error" entry (its faulting module separates
COLMAP / OpenCV / CUDA / OpenMP).

## 5. Not verified

* Linux CI behaviour of COLMAP (no COLMAP on the GitHub runners; the native lane is workstation-only).
* Python 3.10-3.12 behaviour (not installed locally).
* Behaviour with the benchmark session's GPU job running concurrently (deliberately avoided).
