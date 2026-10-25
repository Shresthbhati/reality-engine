# Known Limitations — Reality Engine Capability-Fusion Investigation

These are real, observed limitations recorded honestly (not hidden). They are split into (1) pre-existing Reality Engine limitations confirmed against the code/tests, and (2) limitations of this investigation session. Pre-existing failures were NOT introduced by this session and are NOT "fixed" to appear healthy.

## 1. Pre-existing Reality Engine limitations (confirmed vs code/tests)

Source: re-runs of test subsets this session plus the repo's `test_results.txt` (baseline: 35 failed, 75 passed in 41.73s).

| Area | Failure | Evidence | Status |
|---|---|---|---|
| Room inference (`test_room_inference.py`) | `AttributeError: 'PlaneSummary' object has no attribute 'plane'` in `evidence/promote_rooms.py::_wall_lines_in_floor` | Multiple TestDetectRooms / TestPromoteRoom tests fail | Pre-existing, unfixable within this session's scope without a real plane-fixture dataset |
| World compiler (`test_world_compiler.py`) | TestCompilerEndToEnd / TestValidationGate assertions fail | Downstream of the room-inference `plane` attribute error | Pre-existing, cascades from room inference |
| Compile command (`test_compile_command.py`) | 3 failures: `assert 10 == 8` (compiler produces 10 entities, tests expect 8) | TestCompileWorldCommand tests | Pre-existing entity-count mismatch |
| Reconstruction orchestrator (`test_reconstruction_orchestrator.py`) | TestEndToEnd failures | Downstream of compiler/plane pipeline | Pre-existing |
| WorldIR v2 (`test_world_ir_v2.py`) | 4 failures: `WorldIR` has no attribute `temporal_history`; external-reference round-trip issues | test_world_ir_v2.py | Pre-existing v2 schema gaps |
| WorldStore concurrency (`test_world_store_concurrency.py`) | Hangs: multi-process race spawns subprocesses that exceed the 30s command timeout | `test_world_store_concurrency.py::TestConcurrentMultiProcessWriters` (and lock tests) | Environment-dependent; in-process thread tests pass |
| Artifact integrity (`test_artifact_integrity.py`) | 11 fixture-related errors | Fixture setup issues | Pre-existing |

**None of these were introduced or changed by this session.** Per the mission's honesty rule, they are documented here rather than concealed or papered over.

## 2. Investigation-session limitations (environment / scope)

| Limitation | Detail | Blocker | Unblock condition |
|---|---|---|---|
| Neural depth (MiDaS / Depth Anything V2) cannot run | `MiDaSDepthBackend`/`SAMSegmentationBackend` written but blocked at runtime — checkpoints not cached and cannot be downloaded (PyPI `files.pythonhosted.org` unreachable; `pip install` times out); torch is CPU-only; no GPU | No weights, no GPU, no PyPI access | GPU + cached/checkpointed weights + PyPI access + per-checkpoint license review |
| Open3D python package unavailable | `codes/Open3D` cloned (MIT, `b6c5e19`) but `pip install open3d` fails (PyPI unreachable); cannot use ICP/registration/meshing at runtime | PyPI unreachable | PyPI access + open3d wheel |
| REA live binary analysis blocked | REA cloned (MIT, `b22b0ef`); no Ghidra/IDA/Hopper/Frida installed; Node 26.8.1 below REA's 26.12.1 minimum so full `npm ci && build` not attempted | No commercial providers; Node version | Ghidra/IDA/Hopper (BYOS) + supported Node; used as source-level research only |
| No model weights / datasets downloaded | Per mission rules, no weights or large datasets auto-downloaded; none justified by a specific runnable experiment | Policy + no PyPI | A specific, justified experiment with licensed weights |
| Full test suite not run in one shot | 30s per-shell-call timeout + hanging concurrency test prevented a clean full `pytest tests` run; run in fast subsets instead | Command timeout + concurrency hang | Longer command window or excluding the hanging test |

## 3. Scope-deferred (not limitations of built code, but of current charter)

- SLAM/VIO (ORB_SLAM3, VINS-Fusion, OpenVINS rejected on GPLv3 + no robotics slice; GTSAM deferred).
- BIM/CAD (IFC/USD deferred; license mix to verify).
- Geospatial visualization (CesiumJS/3D Tiles/MapLibre/GDAL/PROJ deferred).
- Multi-session pose-graph alignment / loop closure / co-observation resolver (deferred; not a defined slice).

## 4. What this session did NOT do (honesty boundary)
- Did not clone every candidate repository (only REA, COLMAP, Open3D; others deferred with reasons).
- Did not run REA's compiled runtime (blocked, documented).
- Did not download any model weights or datasets (blocked by policy and network).
- Did not run COLMAP on a real dataset (no dataset; binary available but no directed capture; deferred to a benchmark with a legal small dataset).
- Did not add new RE source code (no gap justified a rewrite; existing components are real and tested).
