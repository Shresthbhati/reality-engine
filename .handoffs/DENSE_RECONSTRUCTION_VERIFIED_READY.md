# HANDOFF

Agent: Freebuff (reconstruction + perception owner)
Branch: agent/freebuff-reconstruction-perception
Commit: (see git log — "feat(dense): verified real dense MVS path — backend continuation, canonical artifact ingestion")
Checkpoint: DENSE_RECONSTRUCTION_VERIFIED_READY
Status: CHECKPOINT_READY

## What exists

The CRITICAL CURRENT PRIORITY outcome: the real dense reconstruction
path is detected → validated → consumed → provenance-preserved →
entered into the canonical pipeline → reached WorldIR geometry →
persisted as a content-addressed artifact, verified by an actual real
COLMAP CUDA dense run on the committed real-photo dataset. No orphaned
artifact; no fake success.

1. `reconstruction/backend/interface.py` — `ReconstructionResult`
   carries two new optional fields:
   - `dense_points`: the fused cloud parsed into canonical
     `ReconstructedPoint`s (track ids `dense:<i>`, provenance =
     the REGISTERED evidence ids — only cameras the sparse model
     actually solved);
   - `dense_report`: the dense run's observed facts
     (`DenseMVSRun.to_dict()` + `sparse_model` continued,
     `parse_facts`, `artifact_uri`/`artifact_sha256`/
     `artifact_vertex_count`/`provenance` when a store was configured,
     or an explicit `"artifact": "NOT PERSISTED …"` honesty record
     when not).
   Defaults `None`: sparse-only results are byte-for-byte untouched.

2. `reconstruction/backend/colmap_backend.py` —
   `ColmapReconstructionBackend(dense_mvs=True, artifact_store=…)`:
   - capability PROBED before any compute (`dense_mvs_available`:
     the binary's own help must list `patch_match_stereo`); refusal
     carries the dense-specific reason;
   - after mapping, the reference sub-model continues through the
     canonical dense chain IN THE SAME WORKSPACE;
   - the fused.ply is parsed and ingested through
     `reconstruction/dense_ingest.ingest_fused_ply` INTO the
     content-addressed ArtifactStore INSIDE the workspace lifetime —
     the first real run proved the file otherwise dies with the temp
     dir (orphaned artifact by construction);
   - a dense stage failure RAISES; it never degrades into a
     sparse-only success;
   - without a store the cloud still travels on `dense_points` and
     the report records `NOT PERSISTED` honestly.

3. `reconstruction/dense_pipeline.py` — CLI-drift discipline. The
   installed COLMAP 4.2 build rejects `--PatchMatchStereo.use_gpu`
   (GPU selection moved out of the option manager) and stereo_fusion
   lost the `--StereoFusion.` prefix (bare `--input_type` /
   `--output_path`). Each stage's `--help` is now PROBED and only
   options the binary actually accepts are passed; `gpu_flag_applied`
   records whether a GPU request could actually be applied. The old
   hard-coded flags were written blind and had never executed — the
   first real attempt failed on exactly this.

4. `reconstruction/orchestrator.py` — `_stamp_provenance` now carries
   `merge_report`/`dense_points`/`dense_report` through the frozen
   rebuild (the same loss class as the 2026-09-20 merge_report loss;
   regression test added).

5. `scripts/run_south_building_e2e.py --dense` — the reproducible
   real dense run: verifies dense capability upfront, enables the
   backend continuation with a PERSISTENT store under
   `datasets/south_building/runs/artifacts/`, and records dense facts
   + artifact verification + sparse-vs-dense nearest-neighbor
   fidelity in the run record.

## Measured real run (REAL, not synthetic)

`datasets/south_building/runs/run_20260921T164048_gpu.json`
(COLMAP 4.2.0 CUDA, RTX 4050, committed 32-photo south-building
subset — reproducible via `scripts/fetch_south_building.py verify` +
`python scripts/run_south_building_e2e.py --dense --gpu`):

- sparse: 22/32 cameras, 5,402 points; GT rotation median 0.176°
  (max 0.383°), zero outliers (gauge-correct evaluator)
- dense: 246,816 fused points (45.7× sparse);
  image_undistorter 0.39 s, patch_match_stereo 343.7 s,
  stereo_fusion 4.5 s; total run 626.8 s
- artifact: `artifact://7caea004…642` — resolved, decoded
  (246,816 vertices), round-trip byte-identical, sha256 recorded
- sparse↔dense agreement: nearest-neighbor median 3.8e-3 model
  units, p95 2.0e-2 (COLMAP SfM scale, honestly recorded — no
  metric anchor)
- honesty: outcome stays `degraded` (`registration_status=partial`);
  the dense cloud did NOT upgrade an incomplete registration
- `gpu_flag_applied: false` recorded (4.2 removed the option)

## Public contracts

- `ReconstructionResult.dense_points / dense_report` (Optional, default None)
- `ColmapReconstructionBackend(…, dense_mvs: bool = False, artifact_store=None)`
- `DenseMVSRun.gpu_flag_applied` (added to `to_dict()`)
- run-record section `"dense"` with `artifact_verification`

## How to consume

- Backend-only dense continuation:
  ```python
  from world_ir.artifact_store import FileArtifactStore
  backend = ColmapReconstructionBackend(
      use_gpu=True, dense_mvs=True,
      artifact_store=FileArtifactStore(root="artifacts"))
  result = backend.reconstruct(evidence_items)
  # result.dense_report["artifact_uri"] -> content-addressed WorldIR-ready cloud
  ```
- Full reproducible real run: `python scripts/run_south_building_e2e.py --dense --gpu`
- Pipeline-level dense (vertical slice, pre-existing): `VerticalSliceOptions(dense_mvs_enabled=True)` — unchanged, its gates still apply.

## Tests run

- `tests/test_colmap_backend_dense.py` (7): dense off by default;
  fused points + report attached with provenance; artifact persisted
  inside workspace lifetime and decodable; no-store honesty record;
  dense failure raises; non-dense binary refused before any compute
  (fake COLMAP writes COLMAP's REAL output formats, not mocks).
- `tests/test_dense_pipeline.py` (11): updated for the probe
  contract — stage order, probed args, honest failures, timeout wrap.
- `tests/test_reconstruction_orchestrator.py` (27): stamp preserves
  merge + dense diagnostics.
- Full dense/backend cluster: 67 green. Orchestration/dense suites:
  45 green.

## Test results

All green (no skips introduced; the only skips in the broader suite
remain the pre-existing real-data gates).

## Runtime verification

Real COLMAP 4.2.0 CUDA dense chain executed on real photos (numbers
above); persisted artifact decoded and round-trip verified from disk
after the run.

## Dependencies

COLMAP CUDA build on PATH (dense-capable probe), RTX-class GPU for
practical patch_match runtimes; scipy for the fidelity measurement;
no new packages.

## Known limitations

- Scale remains COLMAP's own SfM scale (no metric anchor); recorded
  everywhere it matters, rescale path exists (`scale_factor`).
- Dense coverage bounded by patch_match's texture dependence; 22/32
  cameras still unregistered (matching recall) — outcome `degraded`.
- CLI drift is handled by probing, not version pinning; an exotic
  COLMAP build with a help output lacking all known option names
  fails loudly rather than guessing.
- `use_gpu` cannot be FORCED on COLMAP ≥ 4.2 dense stages; the
  request and its non-application are both recorded.

## Known bugs

None open. Two real defects found and fixed during this work (stamped
diagnostics loss; orphaned fused.ply), both regression-tested.

## Next agent (Antigravity / WorldStore + Studio consumers)

The canonical artifact is addressable and decodable:

1. `artifact://<sha256>` blobs under
   `datasets/south_building/runs/artifacts/<sha[:2]>/<sha>.bin` decode
   via `PointCloudData.from_bytes` — ready for WorldStore POINTCLOUD
   geometry binding and Studio viewport streaming.
2. `run.result.dense_report` carries everything a viewer needs
   (counts, durations, provenance, sha256) — render diagnostics from
   it, never invent them.
3. Do NOT treat the dense cloud as upgrading registration status —
   `registration_status`/outcome remain the authority.

## Exact action for next agent

Bind `artifact_uri` → WorldStore entity geometry for the
south-building world, then surface `dense_report` in the Studio
diagnostics panel (real numbers exist now; no mock values needed).
