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

## District generation (C5 perf gate, measured 2026-10-18)

`python -m benchmarks.district_bench` -- a 500-building district (25 x 20
grid, 24 m lots, 3 seeded building families), Python 3.14.7, Windows 11,
Intel64 Family 6 Model 186 (single run of 3 repeats; min of 3 shown).
`generate_district` is the pure grammar; `spec_store.generate` adds
stamping every entity; `regenerate` re-runs the whole district and
fingerprints before/after (cost scales with total entities, not with the
one edited parcel).

| Detail policy | Entities | Geometries | WorldIR JSON | generate_district | spec_store.generate | regenerate (1 parcel) |
|---|---|---|---|---|---|---|
| footprint_only (max_full=0) | 1,048 | 547 | 0.91 MB | 32.4 ms | 38.9 ms | 141.0 ms |
| default cap (max_full=25) | 3,623 | 3,929 | 3.62 MB | 65.4 ms | 86.2 ms | 237.4 ms |
| all full (max_full=None) | 63,317 | 84,164 | 67.11 MB | 752.8 ms | 895.2 ms | 5,017.6 ms |

API path (`POST /api/worlds/{id}/districts` through FastAPI `TestClient`,
temp SQLite, one run, includes validation + generation + WorldStore
commit): max_full=0 -> 119 ms (1,048 entities added); max_full=25 -> 467 ms
(3,623 added). `PUT` of one parcel: 305 ms (2 changed / 1,046 unchanged)
and 1,198 ms (2 changed / 4,669 unchanged) respectively. The all-full case
was not run through the API.

NOT measured: browser viewport frame time, LOD/instancing (none exists),
memory, concurrent requests. The all-full 500-building WorldIR is 67 MB of
JSON -- generating it is fast, but it is not a sensible thing to ship to a
viewer, which is exactly why the footprint_only policy exists. No
"district scale" claim is made for rendering until a browser frame-time
record exists.

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
