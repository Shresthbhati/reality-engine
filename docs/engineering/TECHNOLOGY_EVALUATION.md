# Technology evaluation (Master Directive Rule 2)

Status of every figure below: **measured** (this machine, this session), **verified** (read from the repo/model card on 2026-11-15)
or **assessment** (my judgement, not a measurement). Accuracy against ground truth is **not measured for any candidate**: no
annotated or surveyed data exists yet. Raw measurements: `technology_probe.json`. Machine: RTX 4050 Laptop, 6 GB VRAM,
torch 2.14.1+cu130, Python 3.14.

## 1. What is installed or cached here (observed)

Installed: torch, transformers, OpenCV, timm, trimesh, COLMAP (used by the pipeline).
**Not installed:** `segment_anything`, `sam2`, `open3d`, `pycolmap`, `ultralytics`. No VGGT / DUSt3R / MASt3R / CoTracker /
Florence-2 / SAM weights are cached. Cached weights: Grounding DINO tiny, OWL-ViT, Depth-Anything-V2 Small and Metric-Indoor
Small, DPT_Hybrid, MiDaS small, Mask R-CNN, DINOv2 ViT-S/14, UPerNet ConvNeXt-tiny, ALIKED / DISK / LightGlue.

Note: the repo's own catalog (`engine/technologies/catalog.py`, 50 entries) contains **none** of the directive's candidates
(no Grounding DINO, SAM 2, Florence-2, COLMAP, VGGT, DUSt3R/MASt3R, Depth Anything, Metric3D, Open3D, TSDF, 3DGS). It lists
detectors/segmenters/SLAM libraries that were not chosen. This table is therefore a separate record for the directive's list.

## 2. Measured on one 1672x941 interior image (warm median of 3)

| Model | Latency | Peak VRAM | Notes |
|---|---|---|---|
| Grounding DINO tiny, fp32, 2 prompts | 0.82 s | 3.4 GB | **current default** (batch 2). The previous batch of 7 spilled the 6 GB card (3.4 s/prompt) |
| Grounding DINO tiny, fp16 (`.half()`), 2 prompts | 0.46 s | 1.7 GB | **not used**: it passed in an isolated script but crashed inside the API job (`grid_sample`: half values vs a float32 grid) and silently produced 0 objects. Autocast was correct but slower than fp32 |
| DPT_Hybrid (current depth) | 0.16 s | 0.68 GB | |
| Depth Anything V2 Small, fp16 | 0.044 s | 0.42 GB | relative (inverse) depth |
| Depth Anything V2 Metric-Indoor Small, fp16 | 0.040 s | 0.24 GB | metric depth, trained on Hypersim (synthetic indoor) |
| Mask R-CNN R50-FPN | 0.17 s | 1.55 GB | COCO classes only |

## 3. Consistency of depth models with our SfM geometry (one scene, 9 registered views; NOT accuracy)

Relative residual = median |depth - SfM depth| / median SfM depth, after the fit each model needs. SfM is itself an estimate.
**Reproducibility:** measured with a one-off script on a saved reconstruction snapshot (cameras + sparse points of the 10-photo
living room, intrinsics fx = 1070.1 px) kept outside the repo; it is not reproducible from the repo alone until a snapshot and
the script are checked in as a fixture (next step).

| Model | Fit used | Median relative residual | Per-view range |
|---|---|---|---|
| DPT_Hybrid (current) | scale per view | 0.199 | 0.137-0.318 |
| Depth Anything V2 Small | scale per view | 0.291 | 0.131-0.367 |
| **Depth Anything V2 Metric-Indoor Small** | scale per view | **0.117** | 0.093-0.153 |
| same, **one global scale for all views** | 1.552 SfM units per metre | **0.118** | 0.095-0.142 |

The metric model's per-view scale to SfM units is nearly constant (1.51-1.58, coefficient of variation 1.6%), while the
current pipeline's per-view scales (from the earlier run's logged alignments) spanned 0.30-0.60 for views aligned on ~1,000
points (about 2x; one view aligned on only 115 points gave 0.16). DPT's units are arbitrary per image, so each view needs its
own scale.

### 3a. Follow-up controlled A/B: does a metric model with one global scale fix the layered floors? **No (not supported).**

Same saved sparse model, same compile; only the depth source differs (A = DPT_Hybrid per-view scale, B = metric depth with one
global scale, C = metric depth with a scale per view):

| | A | B | C |
|---|---|---|---|
| Dense points | 120,399 | 147,973 | 147,640 |
| Horizontal-plane share of plane inliers | 0.66 | 0.64 | 0.68 |
| Share of below-camera horizontal points within +-0.15 of the dominant level | 0.19 | 0.22 | 0.22 |
| p10-p90 height spread of those points (working units) | 1.95 | 1.88 | 1.88 |
| Height peaks | 12 | 10 | 13 |
| Floor-role plane candidates before consolidation | 972 | 946 | 890 |
| Entities after compile (wall/floor/ceiling) | 6/3/2 | 7/3/3 (+corridor) | 11/2/3 |

The height distribution of horizontal surfaces is essentially the same whatever the depth source, so per-view scale freedom
is **not** what produced the many "floor" levels. My earlier suggestion that it was is retracted. The levels are
geometry-intrinsic: a furnished living room with a sofa seat, coffee table, shelves and stairs really has many upward-facing
surfaces, and the classifier labels every upward-facing plane a floor candidate. The pipeline's own floor estimator (needs a
2 cm-tight mode) also found no sharp floor peak in any variant.

What the experiment does show: metric depth fits the SfM geometry better (0.117 vs 0.199) and needs one scale, not one per
view. With that global scale (1.552 SfM units per metre) the cameras sit 1.86-2.06 m above the lowest horizontal surfaces, versus
a typical 1.5-1.7 m: plausible but ~20-30% high, so the scale carries an unknown bias and is not metric without a reference.
One scene, one sparse model, consistency with SfM, no ground truth. **Decision: no change to the default depth path.**

## 4. Candidate table (directive list)

Licence column: repo licence from the GitHub API or LICENSE file; weights from the Hugging Face model card. **Weights are
what matters for use**, and they are licensed separately from the code.

| Candidate | Solves | Does not solve | Licence (verified) | Here | Replaces / complements (assessment) | Integration (assessment) |
|---|---|---|---|---|---|---|
| Grounding DINO | open-vocabulary boxes from text | masks; tiny objects at coarse scale; class discovery (needs a prompt list) | code Apache-2.0; HF weights Apache-2.0 | cached, in pipeline | in use | done |
| SAM 2 | promptable masks, video propagation | classes; needs boxes/points | code Apache-2.0; HF weights Apache-2.0 | not installed | complements (boxes -> masks) | medium: install + ~1-2 GB weights |
| Florence-2 | captions, detection, grounding | metric 3D | HF weights MIT | not cached | possible complement to Grounding DINO | medium (remote code) |
| YOLO (Ultralytics) | fast closed-set detection | open vocabulary | **repo AGPL-3.0** | not installed | would complement; AGPL is a commercial blocker unless licensed | low, but licence |
| Mask2Former | semantic/instance/panoptic masks | open vocabulary | code MIT; HF COCO weights licence "other" (unresolved) | not cached | complements | medium |
| CoTracker | point tracking across frames | multi-view geometry | code CC BY-NC 4.0; CoTracker3 weights CC-BY-NC-4.0 | not cached | non-commercial only | n/a for a commercial product |
| COLMAP / PyCOLMAP | SfM, MVS | semantics; scale | COLMAP BSD (its dependencies licensed separately); pycolmap not checked | COLMAP used; pycolmap absent | in use | done (binary) |
| VGGT | cameras, depth, point maps from images | scale; huge scenes on 6 GB (untested) | **VGGT License v1 (custom, text not fully read); VGGT-1B weights CC-BY-NC-4.0** | not cached | candidate complement to COLMAP; non-commercial weights | medium; legal review first |
| DUSt3R / MASt3R | dense two-view geometry, matching | metric scale | **CC BY-NC-SA 4.0 (code)** | not cached (repo has a matcher adapter) | non-commercial only | n/a commercially |
| Depth Anything V2 | monocular depth | metric scale (Small/Base/Large relative variants) | code Apache-2.0; **Small weights Apache-2.0; Base and Large weights CC-BY-NC-4.0**; Metric-Indoor-Small card declares no licence (unconfirmed) | Small + Metric-Indoor Small cached | candidate replacement for DPT_Hybrid depth (section 3) | low: HF `AutoModelForDepthEstimation` |
| Metric3D | metric depth | | code BSD-2-Clause; weights not checked | not cached | possible alternative to Metric-Indoor | medium |
| Open3D | ICP, registration, TSDF, Poisson, point-cloud ops | learned perception | MIT | not installed | complements trimesh/numpy | low (pip) |
| PCL | C++ point-cloud processing | | BSD | not installed | low value next to Open3D | high (C++) |
| 3D Gaussian Splatting | photoreal novel views | accurate editable geometry | Inria/MPII custom licence (excerpt read; commercial terms not confirmed) | not installed | complements for appearance only | high |
| NeRF (nerfstudio) | view synthesis | editable geometry | Apache-2.0 | not installed | likely unnecessary | high |
| LightGlue, DINOv2 | matching; embeddings for re-identification | | Apache-2.0 both | weights cached | complements association (DINOv2 not wired) | low |

## 5. Licence consequences

* **Blocked for a commercial product as published:** VGGT-1B weights, CoTracker3 weights and code, DUSt3R/MASt3R, Depth Anything V2
  Base/Large, Ultralytics YOLO (AGPL) unless separately licensed. This removes the two geometry candidates the directive
  singles out (VGGT, MASt3R) from a commercial path unless terms change or are negotiated.
* **Usable on the licences read:** Grounding DINO, SAM 2, Florence-2, Depth Anything V2 Small, DINOv2, LightGlue, Open3D, COLMAP, Metric3D code.
* **Unresolved, needs a read before adoption:** Depth Anything V2 Metric-Indoor Small (no declared licence), Mask2Former COCO weights ("other"),
  Metric3D weights, VGGT License v1 full text, the Inria Gaussian Splatting terms.

## 6. Spatial reasoning compiler (`reconstruction/spatial`, 2026-11-25)

What the compiler needs and what was chosen. Observed on this machine: numpy 2.5.3, scipy 1.18.1, shapely 2.1.2, networkx 3.6.1,
trimesh 5.2.0, open3d 0.20.0, scikit-learn 1.9.1 (installed != declared: only numpy/scipy/open3d are core dependencies in `pyproject.toml`).

| Need | Chosen | Why | Cost / notes |
|---|---|---|---|
| Constraint solving (junction + prior least squares, robust loss, posterior covariance) | `scipy.optimize.least_squares` (trf, soft-L1), numpy `pinv` for (JᵀJ)⁻¹ | scipy is already a core dependency, BSD-3 (licence text read from the installed distribution); no new dependency | dense Jacobian: O((4n)²) memory, fine to a few hundred walls per storey; sparse `jac_sparsity` is the upgrade path |
| Spatial index / nearest neighbour | none added (O(n²) wall pairing per storey) | storeys have tens to low hundreds of walls | `scipy.spatial.cKDTree` is available when a storey exceeds ~500 walls |
| Planar graph / room faces | own half-edge traversal (~60 lines) | needs evidence-gated junctions and spur pruning; `shapely.polygonize` would also close gaps that are not evidenced and pulls GEOS (LGPL) | networkx `cycle_basis` would find cycles but not faces |
| Connected components / graph reasoning | `scipy.sparse.csgraph` available, not needed yet | | networkx is only in the optional `meshing` extra |
| Robust line / plane fitting | own weighted TLS + Huber/MAD (reject_k=3) | closed form, deterministic, gives the sigma the compiler needs; RANSAC adds randomness and no covariance | |
| Point cloud / mesh processing | `open3d` already used by `reconstruction/pointcloud_clean.py` (MIT per `.agent/LICENSES.yaml`) | consumed only through `ReconstructionResult` points | |
| Frame alignment, ICP, loop closure, bundle adjustment | **not re-implemented**: `registration/` (`cross_session`, `landmarks`, `pose_graph`), `reconstruction/sfm`, COLMAP via the perception stack | owned by FreeBuff's side; `refinement.frame_transform_2d` adapts a `RigidTransform` to the plan view and refuses tilted frames | |

Considered and not adopted: CGAL (GPL/commercial), python-igraph (GPL), Ceres/g2o/GTSAM (heavy native builds, nothing here needs a factor-graph solver yet),
pyransac3d (licence not read). Licences of shapely/networkx/trimesh above come from upstream metadata only; read them before any adoption.

## 7. Multi-view optimization layer (`reconstruction/spatial/multiview.py`, `frames.py`, 2026-11-25)

Availability observed on this machine: OpenCV 5.0.0, pycolmap 4.2.1, Open3D 0.20.0, SciPy 1.18.1, PyTorch 2.14; **not installed:** GTSAM, PyPose, scikit-sparse, Ceres.

| Candidate | Decision | Reason |
|---|---|---|
| COLMAP / pycolmap | consume, do not rewrite | already the SfM backend (`reconstruction/sfm`, `colmap_*`); poses enter as `ImagePose` priors with their sigmas |
| OpenCV | not used here | feature matching / PnP / homography belong to the perception side; no plan-view need |
| Bundle adjustment (full 3D) | not implemented | owned by COLMAP; this layer solves the *structural* problem: pose corrections + wall lines from perpendicular residuals |
| Ceres / GTSAM | not adopted | not installed, heavy native builds; the problem is ~3 params/image + 2/wall, solved in <1 s up to ~30 images by `scipy.optimize.least_squares` with an analytic Jacobian (finite differences were 3x slower; an `lsmr` sparse solve failed to converge and was removed) |
| PyPose | not adopted | not installed; would help only for GPU-batched SE(3) manifolds, unnecessary for a plan-view (SO(2) x R^2) problem |
| Open3D | consumed via `ReconstructionResult` points only | point-to-plane evidence already enters through `refinement.refine_walls_with_points` |
| Robust least squares | adopted: Cauchy (default) / Huber / soft-L1 via scipy | downweights a grossly wrong pose without hard rejection |

Scaling measured (corridor scene, analytic Jacobian, cold / warm restart): 8 images 0.6 s, 32 images 2.2 s, 64 images 7.6 s (warm restart 3.3 s, 16 vs 44 function evaluations). Dense (JᵀJ)⁻¹ covariance is the remaining super-linear term.

## 8. Sparse multi-view solve and pipeline stage (2026-11-25)

Measured (`benchmarks/results/spatial_solver_benchmark.json`; 3 scenes x 4 evaluation seeds = 12 runs per row; this machine only):
the analytic **sparse** Jacobian (5 nonzeros per observation row, index arrays reused across iterations) with SciPy `least_squares`
`tr_solver="lsmr"`, plus a sparse-LU posterior covariance for the pose block, reproduces the dense solution (wall offset identical to 1e-4 m,
pose error within 2e-3 m, one 16-view run with a 4e-2 cost difference) but is **slower below ~20 images** (more LSMR iterations) and faster above:
24 views 0.83 -> 0.78 s, 48 views 2.0 -> 1.4 s, 96 views 11.8 -> 5.0 s, peak traced memory 27.8 -> 4.5 MB at 96 views. One sparse run at 96 views failed to converge
(none dense). Decision: `solver="auto"` (dense up to 16 images, sparse above), both retained because neither dominates.
The larger finding: association, not the solver, dominated wall time (90% at 48 images); memoising cluster fits cut 48 views from 3.3 s to 0.54 s.
Ceres / GTSAM: not adopted (not installed; SciPy's sparse trust-region solves this problem structure); revisit only if a block Schur / marginalisation
solver is needed beyond ~500 images.

## 9. Exact sparse LM, per-technology decision table (2026-11-25)

Correction to section 8: the first sparse path (SciPy TRF with `tr_solver="lsmr"` and an analytic sparse Jacobian) was not merely slower below 20 views; it
failed to converge in 1/12 runs at 96 views (irregular room, seed 101: optimality 39 -> 27 over 400 evaluations, cost ~317 vs the dense 293.5). Cause:
inexact LSMR trust-region steps on a badly scaled system (radians vs metres vs a 1e-4 gauge prior), not the iteration limit. Replaced by
`sparse_lm.py` (sparse-LU Levenberg-Marquardt, IRLS weights, Nielsen damping, scipy-default tolerances). On 36 DEVELOPMENT-seed problems
(seeds 0-2; 2/4/12/96 views x 3 scenes): 0 non-converged (6 with the first LM settings, 2 with default tolerances), sparse cost never above dense. In the 2-4 view
problems dense TRF stops early (xtol) at a higher objective than the LM reaches, yet lower objective did not mean better geometry there (2 views: wall offset
0.114 dense vs 0.119 sparse in the v2 table), which is why `auto` keeps dense for small problems.

| Technology | Purpose | Licence (source) | Dependency / hardware | Measured benefit here | Decision |
|---|---|---|---|---|---|
| SciPy `least_squares` (TRF), `sparse.linalg.splu` | dense solve; sparse LU for LM + covariance | BSD-3 (read from the installed distribution) | core dependency, CPU | baseline solver; sparse-LU is the exact linear solve the LM needs | adopted |
| Own sparse LM (`sparse_lm.py`, ~120 lines) | exact sparse robust solve | repo code | none | fixes LSMR stall; 96 views 2.5 s vs dense 13.7 s on the reproduced case | adopted (justified by the measured failure) |
| Ceres Solver | large sparse NLLS, Schur complement | Apache-2.0 (upstream, not re-read) | native build, CPU | not measured; not installed | rejected for now: no measured need below ~500 images |
| GTSAM | factor graphs, incremental iSAM2 | BSD (upstream, not re-read) | native build, CPU | not measured; not installed | rejected for now; revisit for true incremental SLAM-style updates |
| PyPose | GPU-batched Lie-group optimization | Apache-2.0 (upstream, not re-read) | PyTorch (installed), GPU optional | not measured; not installed | rejected: problem is SO(2) x R^2, no manifold need |
| COLMAP / pycolmap 4.2.1 | SfM poses and intrinsics | BSD (per repo registry; pycolmap installed) | CPU/GPU | consumed as pose priors, not re-implemented | adopted as the pose source |
| OpenCV 5.0.0 | features, PnP | Apache-2.0 (upstream) | installed, CPU | not needed for plan-view structure | not used here |
| Open3D 0.20.0 | point-cloud cleanup | MIT (`.agent/LICENSES.yaml`) | installed, CPU/GPU | wall-point refinement consumes `ReconstructionResult` points only | adopted (existing) |
| Shapely 2.1.2 / NetworkX 3.6.1 | polygons, graphs | not re-read | installed; shapely bundles GEOS | would also close unevidenced gaps / is only an optional extra | not used: own half-edge faces keep junctions evidence-gated; `scipy.sparse.csgraph` covers components |
