# Benchmarks

**Status: measured benchmark infrastructure exists.** This document was
previously a "zero benchmarks exist" placeholder; it is reconciled here
against the actual code (the doc must follow implementation, never lead
it).

## What exists now

| Subsystem | File | What is measured | Status |
|---|---|---|---|
| Timing harness | `benchmarks/harness.py` | stdlib perf_counter, real wall clock, crashing benchmark = failing benchmark | working, tested |
| Physics stepping | `benchmarks/physics_bench.py` | rigid-body stepping at 2 scales, command pipeline throughput | working, tested |
| Quality report scaling | `benchmarks/quality_bench.py` | O(entities) scan of `compute_quality_report` (regression catcher) | working, tested |
| Architectural | `benchmarks/architectural.py` | full perception stack over a ReconstructionResult; CAPTURE_PENDING recorded honestly | working, tested |
| **Descriptor suite** | `benchmarks/suite.py` | descriptor-driven (P19-01, directive item 18): any scene runs through the SAME quality → discovery → ROI pipeline with no per-benchmark special cases; per-stage wall-clock timings + measured counts; `ground_truth=None` descriptors report accuracy as None, never a guess | working, tested |

Run: `python -m benchmarks.run_all` (physics/quality); the descriptor
suite is invoked programmatically (see `tests/test_benchmark_suite.py`)
— a CLI wrapper is not built yet.

## Measured records (reproducible)

Synthetic planar-wall descriptors (this branch, Windows, i7; the exact
scene builder is `tests/test_benchmark_suite.py::_scene`):

| Descriptor | points | quality | discovery | roi | candidates | ROIs | GSD | tier |
|---|---|---|---|---|---|---|---|---|
| synthetic-wall-400 | 400 | 357.7 ms | 0.6 ms | 0.5 ms | 4 | 1 | 3.19 mm/px | fine |
| synthetic-wall-4000 | 4000 | 56.5 ms | 7.2 ms | 60.3 ms | 22 | 1 | 3.21 mm/px | fine |

Real-data detail-spine records (PR #43 branch `claude/detail-real-run`,
294,345-point CUDA COLMAP MVS fused.ply): GSD 1.169 mm/px (fine), 103
discovery cells (96 structure), 12 ROIs, 12 refined / 0 refused, full
chain 182.7 s — see `.agent/TASKS.yaml` P7-06 `real_data_verification`
and `.agent/EXECUTION_STATE.md` entry 4 for the full measured table.

## Not yet benchmarked (honest gaps)

Accuracy/ATE/RPE/mesh-quality comparisons against COLMAP/OpenSfM/ODM/
Open3D/OpenMVS/AliceVision/ORB-SLAM3/RoomFormer/Nerfstudio and the
other systems named in ledger P19-01 remain future work: they require
public datasets with ground truth wired into the suite as descriptors
(`ground_truth=` populated). The suite's descriptor contract is that
seam — no accuracy number may be published for a descriptor whose
ground_truth is None.

Known algorithmic ceilings worth benchmarking first (from
DECISIONS.md / KNOWN_LIMITATIONS.md):

| Subsystem | Ceiling | File |
|---|---|---|
| Collision broadphase | O(n²) pair checks | `engine/physics/collision/broadphase.py` |
| Contact solver | Single-pass, no iteration | `engine/physics/constraints/contact_solver.py` |
| Detail discovery | per-cell Python loop over all points | `perception/detail/discovery.py` |
