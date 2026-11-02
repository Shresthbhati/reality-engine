# Known limitations (observed)

- torch is a CPU build: SAM takes ~29 s/image (vit_b, 16 pts/side); RTX 4050 unused by torch. COLMAP has no CUDA.
- 3D Tiles: box geometry only; single LOD (no geometric-error ladder); not opened in CesiumJS.
- WorldIR up-axis is documented inconsistently: exporters/boxes.py says Z-up, exporters/godot/profile.py says Y-up.
- Pose graph: scipy backend, dense-ish for large graphs (finite-difference Jacobian with sparsity); no IMU, no loop
  detection, not wired into WorldIR/registration; priors are isotropic position-only.
- Baseline: 2 room-geometry tests failing (contract disagreement), 1 flaky WorldStore concurrency test.
- Stale docs: docs/implementation/PENDING_IMPLEMENTATION.md (last reconciled 2026-09-14) understates exporters and
  overstates SAM breakage.
- Worktree lacks datasets/ (present only in the main checkout); real-photo tests/benchmarks must point there.
- cross_session.align_session can accept a grossly wrong alignment (measured 4.3 m off at rmse 0.55 m on a ~10 m scene).
  Pose-graph refinement only helps when other alignments corroborate the rest; a lone 3-session loop is "ambiguous".
- Rotation sigma for session alignments is a heuristic (sigma_t / 10 m); ICP does not measure it.
- The refined registration is recorded on a WorldIR (`metadata.registration.cross_session`) by a library call only;
  no API route persists it into a world version, and nothing applies the transforms to the world's geometry.
- /api/registration/* are stateless: callers supply the alignments; the API does not run ICP on stored sessions.
