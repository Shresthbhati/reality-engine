# Sprint 10 — Production reconstruction and world integrity

Every number below was measured on this machine. Where a claim could not be measured it is marked **NOT MEASURED**
rather than estimated. Nothing here is described as physically accurate: no independent physical ground truth was
used (`physical_accuracy: NOT_MEASURED` in every run).

Baseline code: `d79df667` plus the Sprint 1C branch (`feat/reconstruction-sprint1c`, PR #380).

---

## 1. Production execution graph (P0.1, P0.2)

The real user-facing path is `POST /api/sessions/{id}/reconstruct` → a `RECONSTRUCT_SESSION` job → `apps/api/jobs.py`
→ `run_stages(build_stages("interior_survey_standard"), profile p108-ladder)`.

There are **two nested ladders**. The 15 coarse stages come from
`engine/pipeline/stages/registry.py::PIPELINE_PROFILES`; the fine stages run *inside* `reconstruct`
(`engine/pipeline/stages/ladder.py::fine_chain`) and are reported afterwards as `LadderReport` stages.

```
evidence_resolve → evidence_quality → reconstruct{ pose → scale → depth → layout → perception
                                                  → mesh → dense → detail → finalize }
   → quality_report → compile_validate → changeset_commit
```

Stage facts are persisted in the run record (`engine/pipeline/stages/run_record.py`) and in the job payload; the
evidence-world stage (opt-in) writes its own snapshot, manifest and WorldStore version.

| Stage | Kind | Inputs | Outputs / artifacts | Failure behaviour | Observed in `job_d1c9a0a05ac6` |
|---|---|---|---|---|---|
| `evidence_resolve` | coarse segment | session evidence rows | resolved evidence list | job fails | succeeded |
| `evidence_quality` | coarse segment | resolved evidence | per-image quality facts | degraded, not fatal | succeeded |
| `reconstruct` | coarse segment | evidence, options | COLMAP model, ladder report; `ctx.state["ladder"]` | degrade → single-image bootstrap | degraded (partial registration) |
| `pose` | fine (ladder) | evidence, COLMAP model | `intrinsics_adopted` (fx/fy/cx/cy, source, trust) | ladder falls back | degraded (focal estimated this run) |
| `scale` | fine | poses, sparse points | metric/gauge calibration record | ladder falls back | succeeded |
| `depth` | fine | evidence, cameras | depth maps + `metric3d` records | ladder falls back | succeeded |
| `layout` | fine | compiled frame | layout/world scaffold | ladder falls back | succeeded |
| `perception` | fine | images, cameras | detections/observations | ladder falls back | succeeded (0 canonical entities) |
| `mesh` | fine | sparse/dense cloud | mesh artifact | recorded, not fatal | **skipped** |
| `dense` | fine | sparse model | MVS depth + fused cloud | recorded, not fatal | **skipped** |
| `detail` | fine | fused points | fine detail entities | recorded, not fatal | succeeded |
| `finalize` | fine | stage outputs | assembled `VerticalSliceResult` | ladder falls back | succeeded |
| `quality_report` | coarse | `ctx.state["ladder"]` | quality facts | recorded | **skipped** (no mesh rung) |
| `compile_validate` | coarse | world draft | `world_ir` artifact (sha256 in stage outputs) | job fails on invalid world | succeeded (`ad854098…`) |
| `changeset_commit` | coarse | validated world | new world version | job fails | succeeded |

Opt-in layer (`RE_EVIDENCE_WORLD=1`, `apps/api/evidence_world_stage.py`), off by default:

```
detection_pipeline → depth (MiDaS/DepthAnything, sequential) → spatial lift (align to SfM points)
   → surfaces (RANSAC planes + TSDF fused mesh) → WorldIR bridge → validate → WorldStore version → reload check
```

Measured stage durations for `p0run10` (10 real photos, 344.1 s total):

| Phase | Window | Duration |
|---|---|---|
| `analyzing_evidence` (detection + depth + evidence world) | 6.2 s → 71.4 s | 65.2 s |
| `reconstructing` (COLMAP + fine ladder) | 71.4 s → 263.0 s | 191.6 s |
| `committing_version` | 263.0 s → 344.0 s | 81.0 s |

**Gap found and fixed in this sprint:** the run record persisted only `name`, `status`, `inputs_hash` and `outputs`
per stage — no duration, warnings, error, metrics or peak memory, so a completed run could not say how long a stage
took or why it degraded. `run_record._stage_record` now records all of those (fields the runner never set stay absent
rather than being invented as zero). Covered by `tests/test_run_record.py::test_stage_record_carries_duration_*`.
**Not yet observed in a completed job run** — see §5.

## 2. Modules that exist but are not reached by the production path (P0.3)

| Module / path | Why it does not run |
|---|---|
| `surfaces` + TSDF fusion + evidence-world WorldIR persistence | whole stage is behind `RE_EVIDENCE_WORLD=1`; the default `/reconstruct` never persists the fused surface |
| `mesh`, `dense`, `quality_report` stages | reported `skipped` with a reason: the multi-view rung's fine stages ran, but these read a mesh/gate the capture did not produce |
| `MAPPER_ATTEMPTS[1..2]` (seed 1, relaxed seed 2) | **only** used by the session backend path (`colmap_backend.py:790`); the non-session `reconstruct()` path runs `MAPPER_ATTEMPTS[0]` alone (line 540) |
| `guided_matching` | implemented, default off — measured on real repetitive architecture to add cameras while corrupting poses (documented at `DEFAULT_GUIDED_MATCHING`) |
| SAM segmentation (`segment_anything`) | not installed here → `detection_pipeline` reports `segmentation: unavailable`, `masks: 0`; lifting is box-only |
| `perception` canonical entities | the stage succeeded but produced 1 raw detection → 0 refined boxes → 0 canonical entities |

## 3. Camera-registration diagnosis (P1)

**Dataset identity.** `datasets/south_building` — derivative of the COLMAP *south-building* example set (South
Building, UNC Chapel Hill; 128 real photographs, single Panasonic DMC-TZ3), original archive sha256
`d210016bd2de20936a5f02b87fd38a76bf0440c42d045231218372cf9db9a7a1`. This checkout keeps 32 photographs sampled at
stride 4; **no contiguous frames exist in it**, so the wide baseline is a property of the dataset, not of a selection
bug. Frames used (bytes, sha256 prefix): `P1180141` 246,122 `b3cb3cbf…`, `P1180145` 324,805 `daa214a0…`,
`P1180149` 300,414 `653247a8…`, `P1180153` 308,078 `b4bb75bc…`, `P1180157` 329,713 `4030559677…`, `P1180161`
336,269 `607ac724…`. No EXIF (Make/Model/FocalLength all null); 1024×768.

Tool: `scripts/sprint1/registration_diagnosis.py` (extended in this sprint with per-pair inliers, connected
components, COLMAP's own failure lines and a `--robust-sift` mode matching the production default).

| Config (6 frames) | Keypoints | Verified pairs (of 15) | Inliers | Median | Connected components | Cameras |
|---|---|---|---|---|---|---|
| Plain SIFT | 58,689 | 8 | 15, 18, 42, 49, 54, 63, 66, 103 | 51.5 | 1 (all six) | **0** |
| Robust SIFT (production default: affine shape + DSP) | 58,526 | 6 | 31, 45, 46, 55, 65, 87 | 50.5 | 1 (all six) | **0** |

COLMAP's own words: `Registering initial image pair #3 and #4` → `Discarding reconstruction due to bad initial pair`
→ `=> No good initial image pair found` → `Failed to create any sparse model`. Raw matches are ~70 per pair (563
over 8 pairs) — **the matcher finds almost nothing to verify**, so the mapper has no well-conditioned pair to start
from, and no threshold change would create matches that do not exist.

10-frame run of the same set: 4/10 cameras registered (`registration_status: partial`) — consistent with the
reported "4–6 of 11" range on this capture.

**Negative result worth keeping:** the repository comment above `DEFAULT_ROBUST_SIFT` records "+22% verified points"
with affine+DSP on a 32-photo subset. On these six frames the same configuration produced **fewer** verified pairs
(6 vs 8) and a lower maximum inlier count (87 vs 103), and registered no more cameras. The default is not
universally better; it is better on some subsets.

**What was deliberately not done:** no ratio-test, RANSAC, `min_model_size`, `init_min_tri_angle` or
`abs_pose_min_*` threshold was lowered, and guided matching was not enabled, because each would manufacture
registered cameras without adding real overlap. The capture, not the threshold, is the constraint.

## 4. Before / after on real data (deliverable 3)

| | Before — `prod10` (code `e46e0f1a`) | After — `p0run10` (this branch) |
|---|---|---|
| Photographs | 10 real south-building frames | same 10 |
| Cameras registered | 3 / 10 | 4 / 10 |
| Fused TSDF mesh | 42,972 v / 78,869 t built, then **lost** | 42,235 v / **75,700 t** persisted in the world |
| Surfaces stage | `skipped: no aligned depth maps` | ran: 4 walls, 1 support surface, 1 fused surface, 1 plane correctly `skipped` |
| Persisted world | `not_persisted` (0 entities) | `v-d621237151fa524d`, 6 entities / 6 geometries, reload verified |
| Export | resolved geometry from an empty directory → placeholder boxes | 75,724 real triangles, 6 exported / 0 skipped |
| Run record | no stage timing/warnings | stage duration, warnings, error, metrics, peak RSS (new) |

New in this sprint: a geometry that **names** a payload the store cannot produce no longer becomes the placeholder
cube. The entity is skipped and named in the export report, because a cube exactly where the reconstruction's surface
belongs is a fabricated surface, not a rendering fallback (`_unresolvable_payload` tests; a resolvable mesh still
exports its real triangles).

## 5. Failure and recovery (P5, deliverable 4)

| Scenario | Observed behaviour | Verdict |
|---|---|---|
| Job exceeds its deadline (1800 s) under GPU contention | `RuntimeError: job handler timed out after 1800s`; no evidence snapshot was written; job failed and was retried | honest failure, no partial success reported |
| Sticky `CUDA_ERROR_OUT_OF_MEMORY` in the detection stage | job failed; the scene gate recorded `unavailable` with the error | recorded, not hidden |
| Capture registers 0 cameras (`p0run7`, 6 frames) | depth produced, lifting `skipped: no registered cameras`, surfaces `skipped`, 0 entities → `not_persisted` | **no fake success** |
| SAM absent | `detection_pipeline` status partial with `segmentation did not run: sam unavailable`; masks 0 | recorded |
| Missing / corrupt geometry payload at export | entity skipped and named in `skip_reasons`; no cube emitted | fixed this sprint (2 tests) |
| Missing/corrupt WorldStore artifact, invalid world, immutable version reuse, malformed package, unsupported format | `tests/integration/test_failure_injection.py` → **12 passed** | covered |
| Fresh process reload + export of a finished run | `scripts/sprint1/p0_proof.py` → **PASS** (verify, hash, PLY counts, world API, GLB triangles) | covered |
| Interrupted fusion / worker termination | **NOT MEASURED** this sprint | open |
| Fresh job to observe the new stage telemetry | two attempts (`p0run6`, `p0run11`) died mid-`reconstructing`: the job row stayed `running` with a frozen heartbeat, no COLMAP process remained, and no `run.json` was written | **NOT MEASURED** (change is unit-tested only) |

## 6. Resource profile — RTX 4050 Laptop (deliverable 5)

| Measurement | Value |
|---|---|
| GPU | RTX 4050 Laptop, 6,141 MiB VRAM |
| Peak CUDA allocated (torch, this process) | **692.4 MB** |
| Peak resident memory (this process; COLMAP subprocess excluded) | **5,901.2 MB** |
| Job wall time | 344.1 s (10 photos) |
| Depth engines | 2 (MiDaS, Depth-Anything), loaded and unloaded **sequentially**, one at a time |
| Depth cache | disabled: `checkpoint hash unknown` → every image a miss, model reloaded per job |
| Export (GLB, 6 entities) | 963,444 bytes |

The 5.9 GB peak RSS is close to what this machine sustains; running while another perception benchmark held ~3.7 GB
of VRAM produced the 1800 s timeout and a sticky CUDA OOM. Peak VRAM for the depth models (~0.7 GB) is *not* the
constraint — system RAM and contention are.

## 7. Remaining blockers (deliverable 6)

1. **Registration is the binding constraint.** 4/10 on this capture, 0/6 on a 6-frame subset, with the cause measured
   (wide baseline in the dataset itself, repetitive brick, moving foliage). No threshold change is acceptable and none
   was made.
2. **The evidence-world stage is opt-in.** Persisted surfaces/fusion/WorldIR only happen under
   `RE_EVIDENCE_WORLD=1`; `ENABLEMENT_CRITERIA` still lists unmet criteria (real furnished-interior run, detector
   accuracy benchmark, calibration, quality-gate thresholds).
3. **No masks in production** (`segment_anything` missing) → box-only lifting, extents include background.
4. **`mesh` / `dense` / `quality_report` still skipped** on real captures because registration does not pass the gate.
5. **No physical ground truth** → every scale stays SfM gauge; `NOT_MEASURED` everywhere.
6. **Openings and room connectivity are not produced**: surfaces are observed hulls with no room closure, no
   door/window subtraction, no floor-ceiling-wall joins.
7. **Interrupted-fusion and worker-kill recovery are untested** (P5 remainder).
