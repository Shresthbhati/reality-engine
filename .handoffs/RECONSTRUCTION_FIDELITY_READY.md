# HANDOFF

Agent: Freebuff (reconstruction + perception owner)
Branch: agent/freebuff-reconstruction-perception
Commit: (see git log — "feat(fidelity): canonical absolute-pose evaluation + measured benchmark fidelity report")
Checkpoint: RECONSTRUCTION_FIDELITY_READY
Status: CHECKPOINT_READY

## What exists

The P1 RECONSTRUCTION FIDELITY mission outcome: benchmark fidelity is
now MEASURED with a gauge-correct metric instead of asserted with a
gauge-contaminated one, and every benchmark's honest status is recorded
in a committed report.

1. `reconstruction/evaluation.py` — the canonical absolute-pose
   evaluator. Prior GT comparisons compared quaternions directly
   against the reference model and claimed "rotation is comparable
   without alignment" — false (incremental SfM lives in an arbitrary
   gauge; the old 8.94° median was ~98% gauge, not error). The
   evaluator least-squares-aligns estimate→reference (Horn quaternion
   rotation over orientation frames; scale/translation from
   camera-center geometry, robust to planar/collinear references),
   then measures per-camera rotation + center error. Point fidelity
   (`evaluate_point_cloud`, cKDTree one-way est→GT) reuses the SAME
   single gauge transform. Refuses on <3 common cameras, coincident/
   collinear centers, empty sides — never guesses.

2. `tests/test_pose_evaluation.py` — 13 measured-property tests:
   gauge invariance (known Sim(3) → ~0 error), noise honesty
   (per-camera perturbation measured back at injected magnitude;
   CONSTANT perturbation correctly absorbed as gauge), refusal
   paths, determinism, Horn/Umeyama primitive exactness, point-cloud
   offset measured back exactly.

3. `scripts/run_south_building_e2e.py` — wired to the canonical
   evaluator; dead duplicated comparison block removed; poses renamed
   to original filenames via frozen dataclass rebuild (the evaluator
   matches on evidence_id).

4. `docs/reconstruction/BENCHMARK_FIDELITY_REPORT.md` — per-benchmark
   INPUT/PIPELINE/OUTPUT/KNOWN FAILURE/METRICS/REMAINING GAP for:
   south-building (REAL, measured), Victoria Memorial (NO_CAPTURE_YET),
   Burj Khalifa (NOT_YET_CAPTURED — no tower primitives, no stand-ins),
   room_capture/room_capture_mvs (SYNTHETIC EVIDENCE / REAL BACKEND).

## Measured headline (real data, committed run record)

`datasets/south_building/runs/run_20260920T175402_gpu.json`
(COLMAP 4.2.0 CUDA, 50.2 s total / 48.1 s reconstruction):

- 32 real photos → 22/32 registered → DEGRADED (machine-readable reason)
- pose fidelity AFTER gauge alignment: rotation median **0.214°**
  (was 8.94° pre-fix — that number was gauge, not error), max 0.653°
- center error median 0.0215 (reference units); scale ratio 1.092
- point→reference distance: median **6.8 mm**, p90 16 mm, max 135 mm
  (4,453 est pts vs 49,608 reference pts)
- fusion provenance CONFLICT preserved (198 conflicts kept)

Interpretation: WHERE it registers, the pipeline is genuinely faithful
on real photos; the honest gap is coverage (10 unregistered cameras,
~9% point coverage), which is evidence/matching-recall, not metric.

## Files changed

- reconstruction/evaluation.py (new, 380 lines)
- tests/test_pose_evaluation.py (new, 13 tests)
- scripts/run_south_building_e2e.py (canonical evaluator wiring)
- docs/reconstruction/BENCHMARK_FIDELITY_REPORT.md (new)
- datasets/south_building/runs/run_20260920T175402_gpu.json (measured record)

## Public contracts

- `evaluate_absolute_poses(poses, gt_images, min_cameras=3) -> dict`
  with `refused`/measured fields + `gauge_transform` ((R, t, s),
  estimated→reference, reusable for point-clouds).
- `evaluate_point_cloud(points, gt_points, transform, max_points=20000) -> dict`
  (deterministic stride subsample above 20k).
- `horn_rotation`, `umeyama_sim3`, `quat_to_matrix`, `quat_angle_deg`
  primitives.

## How to consume (Antigravity / any fidelity consumer)

- Import `evaluate_absolute_poses` for any benchmark claiming absolute
  pose accuracy; pass COLMAP images.txt rows as
  {name: (qw,qx,qy,qz,tx,ty,tz)} (world→cam) — conversion handled.
- Reuse the returned `gauge_transform` for all geometric comparisons
  against the same reference model.
- Trust `refused: true` reports as genuine blocks, not failures to
  paper over.

## Tests run

- tests/test_pose_evaluation.py: 13/13.
- Full unbounded suite: **2143 passed / 3 skipped / 0 failed** (300 s).

## Runtime verification

Real COLMAP 4.2.0 CUDA end-to-end twice on the committed 32-photo
dataset (fix iteration + final record); GT comparison went 0-common →
22-common → measured fidelity across those iterations.

## Dependencies

None new (numpy/scipy already core).

## Known limitations

- Coverage on the real benchmark remains the gap (22/32 cameras).
- Burj Khalifa / Victoria Memorial remain UNCAPTURED; report states
  this plainly — the frontend corpus's "COMPILED WorldIR" claim for
  Victoria Memorial is not backed by committed evidence.
- Point fidelity is one-way (est→GT) by design; no symmetric claim.

## Next agent

Exact action: any benchmark work must run through
`reconstruction/evaluation.py`; direct quaternion comparison against
reference models is now a known-wrong pattern (see report §0).
