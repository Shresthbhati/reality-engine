# Reconstruction Hardening Audit (Evidence, Camera, Depth, 3D)

**Date:** 2026-09-13. Direct repository inspection.

Scoped against Prompt 2 (Evidence, Camera, Depth & 3D Reconstruction Hardening). Builds on `docs/REAL_CAPTURE_VERTICAL_SLICE_AUDIT.md`'s baseline.

| Phase | Status | Evidence |
|---|---|---|
| 1 — Evidence registry | IMPLEMENTED (pre-existing) | `evidence/importers.py`, `evidence/packages.py` — content-addressed ids, corruption gates, EXIF/GPS, quality signals. Unchanged this session. |
| 2 — Camera calibration | **IMPLEMENTED (new this session)** | `reconstruction/calibration/camera.py` — real pinhole `CameraIntrinsics` (fx/fy/cx/cy + Brown-Conrady k1/k2/p1/p2/k3), `CameraExtrinsics` (camera-to-world position+rotation, matching `ReconstructedCameraPose`'s convention), `PinholeCamera.project`/`.unproject`/`.ray`. Previously `reconstruction/calibration/` was a README placeholder — nothing in this repo could turn a 3D point into a pixel or a pixel+depth back into 3D. 21 numerical tests: projection geometry, behind-camera/at-plane rejection, round-trips with and without distortion (sub-mm precision), ray casting, coordinate-frame translation/rotation sanity, dict round-trip. |
| 3 — Feature pipeline (matching) | MISSING | No feature extraction/matching code exists outside COLMAP's own subprocess (which handles this internally and opaquely to this repo). |
| 4 — SfM hardening | UNCHANGED | `reconstruction/backend/colmap_backend.py` real, `reconstruction/orchestrator.py` real (prior sessions). Not touched this session. |
| 5 — Scale | UNCHANGED | No dedicated metricization module exists; COLMAP's own scale (unconstrained without a scale reference) is used as-is. |
| 6 — Depth | UNCHANGED, still UNVERIFIED in this session | MiDaS backend exists (`perception/depth/`, another agent's work) implementing `IDepthBackend`. Not exercised here — this session's camera model is model-agnostic and depth-backend-agnostic by design (it operates on any `(u, v, depth)` triple, real or from any backend), but no depth backend was actually run this session. |
| 7 — Depth/camera consistency | **Enabled, not yet built** | The new `PinholeCamera` is the prerequisite this check needs (reproject a depth-derived 3D point and compare pixel coordinates) — the checking logic itself is not built this session. |
| 8 — Point cloud (depth -> points) | **Enabled, not yet built** | `PinholeCamera.unproject(u, v, depth)` is exactly the per-pixel operation a depth-map-to-point-cloud converter would call in a loop over a `DepthMap`'s valid pixels — not yet wired into such a loop. |
| 9 — Mesh | MISSING (unchanged) | No TSDF/Poisson/volumetric reconstruction exists; `Geometry` still has no real vertex buffer. |
| 10-13 — Quality gates, benchmarking, regression, end-to-end | MISSING/UNCHANGED | Not attempted this session. |

## This session's change

**`reconstruction/calibration/camera.py`** — a real, tested pinhole camera model (intrinsics validation, Brown-Conrady distortion forward/inverse, projection, unprojection, ray casting), closing Prompt 2's Phase 2 exactly and unblocking Phase 8 (depth → point cloud) as the next concrete step, since unprojection is the core primitive that phase needs and did not exist anywhere in the repo before this.

Also added `Quat.conjugate()` to `engine/physics/math3.py` (a genuinely missing, generically useful operation — a unit quaternion's own inverse rotation) rather than reimplementing an inverse-rotation helper locally inside the new camera module, avoiding a duplicated abstraction.

**Verification:** `tests/test_camera_calibration.py`, 21 tests, all passing on first run, including round-trip precision checks with real (nonzero) distortion coefficients. Full suite: 1002 passed, 1 skipped (981 baseline + 21), zero regressions — including all existing physics tests, confirming `Quat.conjugate()` is additive and safe.

## Honest gap against Prompt 2's Definition of Done

Depth backend integration, point-cloud generation, mesh reconstruction, and quality gates remain entirely unbuilt. This session closed exactly one prerequisite (the camera model) rather than the full hardening campaign — the next concrete, dependency-safe step is wiring `PinholeCamera.unproject()` into an actual `DepthMap -> list[Vector3]` point-cloud function (Phase 8), which this camera model now makes straightforward.
