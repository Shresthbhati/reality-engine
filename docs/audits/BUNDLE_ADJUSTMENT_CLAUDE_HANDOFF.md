# Bundle Adjustment — Technical Handoff for Claude (Geometry Owner)

**Author:** Buffy (perception/reconstruction engineering agent)
**Date:** 2026-11-26
**Branch context:** `feat/day4-real-geometry-measurements` @ `43ca2b4a` (PR #395 MERGED per gh)
**Audience:** Claude — canonical geometry owner (camera pose graph, bundle adjustment, reconstruction accuracy)
**Purpose:** Hand off the untracked `perception/bundle_adjustment.py` investigation so Claude can decide whether and how to make it the canonical BA, without me creating a second canonical geometry pipeline.

---

## 1. What I investigated

`perception/bundle_adjustment.py` — an untracked (never committed) standalone bundle adjustment module, 765+ lines, found in the working tree during the six-agent recovery audit. I read it in full and compared it against the existing tested `BundleAdjustmentOptimizer` in `reconstruction/sfm/core_pipeline.py`.

I did NOT commit it. I did NOT wire it into the pipeline. I did NOT add tests. Per AGENTS.md §52 and the geometry-ownership boundary, I am handing the investigation to the canonical geometry owner rather than introducing a competing canonical implementation.

---

## 2. The two implementations, side by side

### 2.1 Existing tested implementation — `reconstruction/sfm/core_pipeline.py::BundleAdjustmentOptimizer`

**Status:** Committed, tested (`tests/test_sfm_pipeline_rule4.py`), integrated into the pipeline.

**What it actually does:** A coordinate-descent-style point-position refiner. For each iteration it:
- Projects each 3D point into each observing camera,
- Computes the pixel residual `err = uv_meas - proj`,
- Converts the pixel delta to a world-delta approximation via an inverse-Jacobian heuristic (`d_world = R.T @ [(err[0]/fx)*z, (err[1]/fy)*z, 0]`),
- Accumulates per-point deltas weighted by observation count,
- Steps 3D point positions by `step_scale * 0.25 * (delta / weight)` and accepts if RMSE improves, else reverts,
- Damps on stagnation.

**What it does NOT do (i.e. what makes "BundleAdjustmentOptimizer" a misnomer):**
- **No rotation parameters.** Camera rotations `R` are held fixed throughout; only 3D point positions move. A true bundle adjustment jointly optimizes camera pose (6 DOF: 3 rotation + 3 translation) and point position (3 DOF).
- **No analytic Jacobian.** The "backprojection ray" heuristic is a first-order approximation for point updates, not the true derivative of reprojection error w.r.t. the parameter vector. There is no sparse Jacobian matrix `J` of size `(2*n_obs) x (6*n_cam + 3*n_pt)`.
- **No proper solver.** It is a hand-rolled coordinate descent with RMSE accept/revert and a damping multiplier, not a Levenberg-Marquardt (or any other) nonlinear least-squares solver operating on a well-defined `rug = ||J^T W J||` normal-equation system.
- **No gauge fixing.** No camera is pinned; there is no handle on the similarity-gauge freedom (7 DOF: 3 rotation + 3 translation + 1 scale, or 6 DOF rotation+translation if scale is already fixed by a metric reference).
- **No covariances.** No posterior uncertainty is estimated; `BundleAdjustmentResult` carries `initial_rmse_px`, `final_rmse_px`, `iterations`, `converged`, `num_cameras`, `num_points`, `num_observations`, `outlier_count` — diagnostics, not a proper BA result.
- **Huber "loss" is post-hoc weighting, not in the normal equations.** When `err > huber_delta_px`, the residual is set to `sqrt(2*delta*err - delta^2)` — a scalar weight applied to the residual, but the optimizer does not form the robustified normal equations `(J^T W J) dx = J^T W r` with a weight matrix `W` derived from the Huber influence function. The "Huber" label describes the scalarization, not a robust LM solve.

**Bottom line:** `BundleAdjustmentOptimizer` is a point-position smoother that reduces reprojection RMSE by nudging points along approximate backprojection rays. It is useful as a lightweight refinement, but it is NOT a bundle adjustment in the SfM sense. Calling it "bundle adjustment" overstates what it does.

### 2.2 Untracked implementation — `perception/bundle_adjustment.py::run_bundle_adjustment`

**Status:** Untracked, NOT committed, NOT tested, NOT imported by any committed code.

**What it actually does:** A correct, rigorous sparse Levenberg-Marquardt bundle adjustment.

**Parameterization:**
- Camera pose = world→camera: `X_cam = R @ X_world + t`. Rotation stored as a rotation vector `w` in so(3) (the LM parameter), converted to matrix via `rotation_exp(w)`; conversion back via `rotation_log(R)`. Translation `t` is a 3-vector.
- Each optimisable camera contributes 6 parameters (3 rotation + 3 translation).
- Each 3D point contributes 3 parameters (its world position).
- **Camera 0 (first in sorted order) is the gauge:** its pose is held at the initial value and is NOT part of the parameter vector. This fixes the similarity gauge (translation + rotation of the whole scene), leaving 6 DOF per other camera + 3 per point.

**Jacobian (the key thing):**
- Each observation `(camera_id, point_id, u, v, sigma_px)` produces 2 residual rows `(r_u, r_v)` = `(proj_u - u)/sigma, (proj_v - v)/sigma`.
- The analytic Jacobian `J` is sparse: each of the 2 rows has at most 9 nonzeros (6 camera params + 3 point params) when the observation is in a non-gauge camera, or 3 nonzeros (point params only) when the observation is in camera 0 (gauge, no camera columns).
- The derivatives are computed analytically through the full chain:
  - `X_cam = R @ P + t`
  - `dXc_drot = R @ skew(P)` (derivative w.r.t. rotation vector, via the so(3) exponential map — `skew(P)` is the cross-product matrix)
  - `dXc_dt = R` (derivative w.r.t. translation)
  - `dXc_dP = R` (derivative w.r.t. point position)
  - `dproj_dXc = [[fx/z, 0, -fx*x/z^2], [0, fy/z, -fy*y/z^2]]` (derivative of pinhole projection w.r.t. camera-frame coordinates)
  - Full chain: `dproj_dparam = dproj_dXc @ dXc_dparam / sigma_px`
- The sparse matrix is assembled in CSR format and passed to `solve_sparse_lm` from `reconstruction/spatial/sparse_lm.py`.

**Solver:** `solve_sparse_lm(residuals, jacobian, x0, loss="cauchy", f_scale=1.0, max_nfev=100)` — the exact sparse LM from `reconstruction/spatial/sparse_lm.py`. Cauchy loss (configurable) down-weights outliers; `f_scale` is the Cauchy threshold.

**Visibility filtering:** Points with `Z_cam <= 1e-6` (behind or on the camera plane) are excluded from that view's residual (set to 0) and from the Jacobian (the 18 or 6 entries for that observation are skipped). This is the standard "point behind camera" guard.

**Result:** `BundleAdjustmentResult` with:
- `status`: "optimised" | "failed" | "unavailable"
- `cameras`: dict[str, CameraResult] — `camera_id`, `R` (3x3), `t` (3,), `cov` (6x6 or None; None for gauge camera 0), `n_observations`, `median_reprojection_px`
- `points`: dict[str, PointResult] — `point_id`, `position` (3,), `cov` (3x3 or None), `n_observations`, `median_reprojection_px`
- `observations`: dict[str, float] — `(camera_id|point_id) -> reprojection_px`
- `median_reprojection_px`, `n_cameras`, `n_points`, `n_observations`, `n_kept`
- `seconds`, `nfev`, `optimality`, `message`, `lm_result`: LMResult | None
- **Per-camera 6x6 and per-point 3x3 posterior covariances** from `(J^T J)^{-1}` at the solution (with 1e-12 diagonal regularisation; falls back to None on singular). Camera 0 (gauge) has `cov=None`.

**API surface:**
- Dataclasses: `CameraObservation(camera_id, point_id, u, v, sigma_px)`, `CameraParams(camera_id, intrinsics, R, t)`, `Point3D(point_id, position)`, `CameraResult`, `PointResult`, `BundleAdjustmentResult`.
- `build_bundle_problem(cameras, points, observations) -> dict` — assembles the problem (parameter vector `x0`, residuals fn, jacobian fn, sparsity, indices).
- `run_bundle_adjustment(cameras, points, observations, *, loss="cauchy", f_scale=1.0, max_nfev=100) -> BundleAdjustmentResult`.

**Dependencies:** `perception.camera_pose_graph.rotation_exp/rotation_log`, `reconstruction.spatial.sparse_lm.solve_sparse_lm/LMResult`, `world_ir.camera.PinholeCamera/CameraIntrinsics/quat_to_matrix`, numpy, scipy sparse.

---

## 3. The core finding

These are NOT equivalent implementations and NOT duplicates. They are different algorithms at different rigor:

| | BundleAdjustmentOptimizer (committed) | bundle_adjustment.py (untracked) |
|---|---|---|
| Optimises | 3D point positions only | 6-DOF cameras + 3-DOF points |
| Rotation params | None (R fixed) | Yes (so(3) vector, 3/camera) |
| Jacobian | Heuristic backprojection-ray approx | Analytic sparse J (full chain rule) |
| Solver | Coordinate descent + RMSE accept/revert | Exact sparse LM |
| Gauge | None | Camera 0 fixed |
| Robust loss | Huber scalarization (post-hoc) | Cauchy (in the solver, configurable) |
| Covariances | None | Per-camera 6x6 + per-point 3x3 from (J^T J)^-1 |
| Result | rmse/iterations/converged/outliers | status/nfev/optimality/message/lm_result/per-obs errors/covariances |
| Lines | ~120 (optimize) | ~765 (full module) |
| Tested | Yes (test_sfm_pipeline_rule4.py) | No |
| In pipeline | Yes | No (no imports) |

**The important point for Claude:** the existing `BundleAdjustmentOptimizer` does not actually perform bundle adjustment. If the codebase wants a true BA (joint camera+point optimization with a proper sparse Jacobian, gauge, and covariances), the untracked file is a candidate implementation that already does it — but it is untested and unwired, and committing it as-is would introduce an unverified implementation alongside the tested one, which is exactly what the mandate's BA-consolidation instruction says NOT to do.

---

## 4. What I did NOT do (and why)

- **Did not commit `perception/bundle_adjustment.py`.** No tests, no integration, no consumers. Committing an untested parallel BA would add unverified code and leave two implementations that both claim to do "bundle adjustment" without clarity.
- **Did not wire it into the pipeline.** The pipeline currently calls `BundleAdjustmentOptimizer.optimize(camera_poses, points_3d, observations, camera_model)` signature. `run_bundle_adjustment` has a different signature (`CameraParams`/`CameraObservation`/`Point3D` dataclasses). Wiring requires an adapter + tests + deciding which result shape the downstream consumers expect — that's a geometry-owner decision, not a perception-agent decision.
- **Did not add tests.** Tests require knowing the intended contract (which result fields matter to downstream consumers, what tolerance to assert, what synthetic inputs are meaningful). That's a geometry-owner decision.
- **Did not modify `reconstruction/sfm/core_pipeline.py`.** That's canonical geometry; I don't touch it without coordination.

---

## 5. Recommendation (my analysis, not a decision — Claude owns the call)

The untracked file is a *better* implementation of "bundle adjustment" than the existing `BundleAdjustmentOptimizer` — IF the goal is a true joint camera+point BA. But "better algorithm" is not the only consideration; the tested, integrated, consumed implementation has value too.

**Option A (integrate the rigorous BA into the canonical pipeline):**
- Write an adapter from the pipeline's current call signature `(List[(R,t)], np.ndarray (N,3), List[(cam_idx, pt_idx, uv)], CameraModel)` to `CameraParams`/`CameraObservation`/`Point3D`.
- Call `run_bundle_adjustment` from the pipeline (likely from `BundleAdjustmentOptimizer.optimize` or a new entry point), replacing or supplementing the coordinate-descent point refiner.
- Add tests: synthetic input with known ground-truth (e.g. a small set of cameras + points + observations where the optimum is computable), assert the result shape (cameras with cov, points with cov, per-obs errors, nfev/optimality), assert reprojection error decreases, assert gauge invariance (camera 0 unchanged), assert covariance realism (positive semi-definite, reasonable magnitude).
- Update `tests/test_sfm_pipeline_rule4.py` (or add a new test file `tests/test_bundle_adjustment.py`) to cover the real BA.
- Decide what to do with the existing `BundleAdjustmentOptimizer`: retain as a lightweight point refiner (rename to reflect what it actually does, e.g. `PointRefiner`), or retire it if the real BA fully supersedes it.
- **Risk:** the existing `BundleAdjustmentOptimizer` is consumed somewhere; replacing its result shape (it returns `(cams, pts, BundleAdjustmentResult)` where `BundleAdjustmentResult` has `initial_rmse_px`/`final_rmse_px`/`iterations`/`converged`/`outlier_count`) with the new result shape (which has `cameras`/`points`/`observations`/`nfev`/`optimality`/`message`/`lm_result` and per-camera/per-point `cov`) could break consumers. Claude needs to audit consumers before changing the result type.

**Option B (document the distinction, track the wiring as a task, defer):**
- Record in the canonical docs that `BundleAdjustmentOptimizer` is a point refiner (not a true BA) and that `run_bundle_adjustment` is a real BA entry point (untested, unwired).
- Add a tracked task: "Wire `run_bundle_adjustment` into the reconstruction pipeline with tests" — owned by Claude, with the analysis above as the spec.
- Do NOT commit `perception/bundle_adjustment.py` until the task is scoped with tests.
- This avoids introducing unverified code now while preserving the option to integrate later.

**Option C (archive the untracked file with an explanation):**
- Move `perception/bundle_adjustment.py` to a tracked research location (or discard it) with a note: "Investigated; superior algorithm but untested and the existing tested point refiner is the production path; integrate via Option A if a true BA is desired."
- This clears the working-tree untracked file without introducing unverified code.

**My recommendation:** Option A is the right engineering outcome IF Claude agrees the pipeline should have a true BA (joint camera+point, gauge, covariances). Option B is the safe intermediate if Claude wants to scope it properly first. Option C is appropriate only if a true BA is not wanted in the canonical pipeline.

---

## 6. Things Claude needs to verify before integrating (regardless of option)

1. **Consumers of `BundleAdjustmentOptimizer.optimize`.** Audit every caller — what do they do with the returned `(cams, pts, BundleAdjustmentResult)`? Do they use `final_rmse_px`, `converged`, `outlier_count`, `iterations`? If so, the new result shape must preserve those fields (or the callers must be updated).
2. **The existing test `tests/test_sfm_pipeline_rule4.py::test_bundle_adjustment_optimization`.** Read it. What does it assert today? If it asserts on `final_rmse_px`/`converged`/`outlier_count`, an integration must preserve those assertions (or deliberately change them with a documented reason).
3. **`solve_sparse_lm` contract.** The untracked file calls `solve_sparse_lm(residuals, jacobian, x0, loss, f_scale, max_nfev)` and expects a `sol` with `.x`, `.success`, `.nfev`, `.optimality`, `.message`, `.jac` (the sqrt(W)*J at solution, used for covariance). Claude should read `reconstruction/spatial/sparse_lm.py` to confirm the interface and that `.jac` is what the covariance code expects.
4. **Gauge appropriateness.** Camera 0 is the gauge. Is that the right choice for the pipeline's use case? If the pipeline already has a metric scale reference (operator-measured baseline), the gauge may need to be different (e.g. fix scale too). The current `BundleAdjustmentOptimizer` has no gauge at all.
5. **Covariance consumers.** Does anything consume per-camera/per-point covariances? If yes, the new result's `cov` fields are a feature. If no, they're just not-yet-used diagnostics (still correct to compute).
6. **`sigma_px` / observation quality.** `CameraObservation` carries `sigma_px` (pixel noise estimate). Where does that come from in the pipeline today? If the pipeline doesn't have per-observation noise estimates, the Cauchy loss still works (default `f_scale=1.0` treats all observations equally), but the `/ sigma_px` Jacobian scaling would need a default (e.g. `sigma_px=1.0` for all) unless the pipeline can supply it.
7. **Rotation representation.** The untracked file uses `rotation_exp`/`rotation_log` from `perception.camera_pose_graph` (so(3) vector <-> matrix). Confirm those are the right ops and that the pipeline's camera poses are representable the same way.

---

## 7. Dependencies / interfaces to preserve

- **Do NOT** introduce a second canonical camera-geometry or reconstruction-architecture pipeline. The handoff is about making the existing canonical `BundleAdjustmentOptimizer` either correct (as a true BA) or correctly-described (as a point refiner), not about adding a competing pipeline.
- **Do NOT** change `perception/camera_pose_graph.py` (cheirality constraints added this session) without coordinating — that's now committed geometry.
- **Do NOT** change `reconstruction/spatial/sparse_lm.py` without reading it first — the untracked file depends on its interface.
- **Coordinate with me (Buffy) or the perception agent** before changing `engine/multiengine/evidence_fusion.py` or the multi-engine observation contract — that's perception-layer, not geometry.

---

## 8. Related context from this session

- **Recovery report:** `docs/audits/FREEBUFF_200K_RECONSTRUCTION_REPORT.md` §4.2 (bundle_adjustment.py description) + §4.3 (full BA comparison) — contains the detailed write-up.
- **Delivery ledger:** `docs/engineering/FREEBUFF_DELIVERY_LEDGER.yaml` section "Comparison verdicts (this session)" — contains the verdict table.
- **EXECUTION_STATE.md:** `.agent/EXECUTION_STATE.md` §Comparison verdicts — records the finding and the Option A/B/C recommendation.
- **CAMERA_POSE_GRAPHS:** `perception/camera_pose_graph.py` was extended this session with cheirality constraints (`points3d` optional param, `(R_c @ P + t_c).z > 0` residuals) — committed, all 30 tests pass. This is relevant if the BA is to consume the pose-graph output.
- **The stash is NOT the source of bundle_adjustment.py.** `perception/bundle_adjustment.py` is a separate untracked file in the working tree, not from `stash@{6}`. The stash's `apps/api/jobs.py` contains `_run_reconstruct_session` work (unrelated to BA).

---

*End of handoff.*
