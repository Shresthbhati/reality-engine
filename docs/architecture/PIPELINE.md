# Reality Engine — Pipeline

Stage-by-stage reference for the capture→world mapping pipeline
implemented in `engine/pipeline/vertical_slice.py`, invoked by both
`scripts/run_vertical_slice.py` (flagship runner) and
`reality compile <dataset>` (CLI; PR #19).

```
reality compile ./capture
  0. dataset load            engine/pipeline/dataset.py
     manifest.json + images/ → EvidenceItems, scale refs, intrinsics
     (missing/invalid manifest → DatasetError, never empty dataset)
  1. reconstruction          reconstruction/orchestrator.py
     COLMAP SfM → ReconstructionResult (points, poses, status)
     availability ≠ acceptance; registration_status never fabricated
  2. metric scale            reconstruction/scale.py
     measured baseline → meters_per_unit; honest RELATIVE fallback
  2.5 frame canonicalization reconstruction/frame.py
     dominant plane → floor; +Y up; world axes pinned
  3. depth                   perception/depth/
     MiDaS → DepthMap; SfM sparse fit → metric-by-alignment
     per-pixel world points via trusted intrinsics (P0.5)
  3.5 perception             perception/detection/ + instances/
     Mask R-CNN detect+segment → lift_region_to_3d (metric only,
     honest None) → multi-view merge → promote → re-validate (P0.8)
  3.6 mesh                   reconstruction/meshing/
     fuse sparse+depth points → voxel downsample → outlier filter →
     camera-envelope filter → PCA normals → COLMAP poisson_mesher →
     MeshData artifact (SHA256) → WorldIR Geometry(MESH) (P0.11–13)
  4. compile                 engine/compiler.py
     ReconstructionResult + ArtifactStore → WorldIR v1
     planes/rooms/promotions converge here (P0.14)
  5. validate                world_ir/validation.py
     fail loudly on invariant violations
```

## Output set (`reality compile`)

```
output/
  worldir.json      canonical WorldIR
  report.json       per-stage status, counts, warnings, outputs
  points.ply        fused real points
  mesh.ply          reconstructed surface mesh (when mesh stage ran)
  cameras.json      registered camera poses
  artifacts/        content-addressed geometry payloads
  exports/scene.gltf
```

## Honesty contract

- Every stage reports status into `report.json`; skipped stages carry
  the reason (e.g. "no trusted intrinsics — refusing to guess a camera
  model").
- Exit code 0 only for success or partial-with-world; 1 when a stage
  refuses. No green success when required stages failed.
- Deterministic testing: `REALITY_TEST_BACKEND=module:Class` injects a
  synthetic backend (tests only); production always uses COLMAP.
