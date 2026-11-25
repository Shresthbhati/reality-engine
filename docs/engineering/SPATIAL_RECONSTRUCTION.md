# Spatial reconstruction: how it works today

Reference for what is implemented in `reconstruction/spatial/` (not a plan). Everything below names the file that implements it and the test or benchmark that exercises it.

## Boundaries
| Layer | Owner | Where |
|---|---|---|
| Perception evidence (detections, lifted segments, poses) | Gemini / perception | produces `spatial_inputs.v1`; **no producer exists in this repository yet** (no branch contains one) |
| Geometric inference (association, optimization, constraints) | spatial compiler | `association.py`, `multiview.py`, `sparse_lm.py`, `constraints.py`, `refinement.py` |
| Canonical geometry | WorldIR + WorldStore | `compiler.py::to_worldir`, `worldstore/` |
| Semantic interpretation | beliefs on entities (`spatial_beliefs`) | `model.py`, `uncertainty.py` |
| Presentation | Studio / API | reads `job.payload.spatial_global` and the world |

An observation is not a canonical object: observations live in `world.metadata["spatial_evidence"]` and `["spatial_optimization"]["raw"]`; entities cite them by evidence id.

## Pipeline
`spatial_inputs.v1` -> `stage.parse_inputs` (structured refusals) -> `compile_multiview`:
1. `optimize_multiview`: observations are in IMAGE frames; priors are plan-view poses with sigmas. Rounds of associate -> joint least squares over (pose corrections, wall lines) with a robust loss; only the perpendicular residual is used (the one dimension a wall observation measures). Unobservable pose dimensions stay prior-dominated and are reported (`observed_dims`). Raw measurements are preserved; corrected copies keep the same evidence ids.
2. `compile_spatial`: association -> junction constraints (evidence-gated; contradictions surfaced) -> topology (rooms = bounded faces of the wall graph; severities error / contradiction / incomplete / info) -> completeness (separate axes).
3. `classify_reconstruction`: invalid / uncertain / incomplete / valid_partial / valid_complete plus every flag that applies.
4. `to_worldir` (+ `stable_id_map` when a parent version exists) -> `WorldStore.save_version(parent=...)`.
Optional pipeline stage `spatial_global` (profile `interior_survey_spatial`, flag `RE_SPATIAL_GLOBAL`, default OFF, optional: failure => PARTIAL). With no producer it is recorded as `skipped` with the reason.

## Solvers
`solver = auto | dense | sparse` (`MultiViewConfig`). Dense: SciPy TRF with an analytic dense Jacobian. Sparse: analytic CSR Jacobian (5 nonzeros per observation row, index arrays built once per round) and `sparse_lm.py`, an exact sparse-LU Levenberg-Marquardt with IRLS robust weights and Nielsen damping. LSMR trust-region steps were tried first and rejected: on a reproduced 96-view case they stalled (optimality 39 -> 27 in 400 evaluations; dense converged in 38). `auto` = dense up to `auto_sparse_above` (16) images, sparse above. The chosen solver, per-round termination status/message, evaluations, optimality and cost are returned in `MultiViewResult.diagnostics`; a non-converged run is reported, never silently retried. Posterior covariance is (J^T W J)^-1 for both solvers (one definition, first order).

## Association cache
`_cluster_walls` memoises each cluster's line fit by the sorted evidence ids it contains. The cache is local to one call, so observations cannot change under it; `AssociationConfig.memoize_fits=False` recomputes everything, and results are bitwise identical (`tests/test_spatial_association_cache.py`). Hits/misses are in `AssociationResult.cache_stats` and the job facts. Effect (48 views): ~87% hit rate, 3x faster association. Association, not the solver, dominated runtime before this change.

## Frames and scale
`frames.py::Sim3` (scale, rotation, translation) with invertible, composable, validated transforms (invalid or non-orthonormal input is refused); `FrameRegistry` records provenance per transform, refuses unknown paths and mixed frames, and checks loop consistency. The stage refuses any `scale_state != "metric"` (`SCALE_NOT_METRIC`). Every compiled entity carries `coordinate_frame` and `scale_state`; `to_worldir` without a stated scale writes `"unspecified"`, never metric.

## Uncertainty
Seven dimensions per entity (existence, position, dimensions, orientation, geometry, topology, semantic), each a `Belief` {status, sigma, confidence, support}. Statuses: directly_observed, geometrically_inferred, semantically_inferred, weakly_supported, conflicting, unknown. Missing confidence stays `None`. Position sigma is propagated through pose optimization (posterior covariance of the image poses -> observation sigma), constraint solving (posterior end sigma), openings (inherit the host) and room area (first-order). `sigma_parts` splits measurement / optimization / model; association ambiguity is a count. Calibration is measured only against simulated ground truth (`benchmarks/results/*multiview*`); nothing is claimed calibrated for real data.

## Identity
Entity ids are a hash of the evidence set, so they change when evidence is added (measured: 71 of 85 established walls). `stable_id_map` carries ids from the parent version by one-to-one geometric assignment (same type and storey, end points within 0.6 m); nothing is merged and unmatched entities keep their own id.

## Failure taxonomy
`benchmarks/spatial/failure_analysis.py`: 20 categories, per-failure scene, seed, entities, observations, residual, outlier score, evidence, root cause, severity, resolution (`unresolved`, `excluded_tentative`, `surfaced`, `mitigated`; never "resolved" because a residual vanished), `excluded`, `tentative`. `depth_inconsistency` is not evaluable: the contract carries no depth.

## Quality (no aggregate score)
`job.payload.spatial_global.quality` has five axes: geometric (accuracy vs truth is "not evaluable in a job"), structural, evidence (direct vs inferred belief counts), topological, processing (solver, convergence, timings).

## Benchmarks
```
python -m benchmarks.spatial.run_spatial_benchmark      # association + compile vs baselines (seeds 0-11 dev; --seeds)
python -m benchmarks.spatial.run_multiview_benchmark    # existing vs optimized vs optimized+constraints; evaluation seeds >= 100
python -m benchmarks.spatial.run_solver_benchmark       # dense vs sparse by view count, with parity
python -m benchmarks.spatial.failure_analysis           # taxonomy report (json + txt)
```
All observations are simulated over known truth (surveyed room plans supply real laser-measured ground truth only); no result here is a real-photograph accuracy claim.
