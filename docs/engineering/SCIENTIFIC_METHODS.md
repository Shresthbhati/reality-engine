# Scientific Methods

The mathematics behind each mapping stage: model, assumptions, and the
diagnostics that must accompany them. Scientific code never invents
plausible-looking math — each method here is the one implemented.

## Metric scale anchoring (`reconstruction/scale.py`)

**Model.** SfM recovers geometry up to similarity transform. Given a
measured baseline `d_meas` between two registered cameras whose model
distance is `d_model`: `meters_per_unit = d_meas / d_model`.

**Assumptions.** Both endpoints registered; the measurement is between
the camera centers.

**Diagnostics.** `ScaleState` (`metric`/`relative`), `meters_per_unit`,
anchoring note. Unanchored → `relative`, never an invented factor.

## Mono-depth metricization (`perception/depth`)

**Model.** Relative depth from MiDaS; scale-and-shift alignment to the
SfM sparse cloud: minimize `‖s·d_rel(x) + t − d_sfm(x)‖` over the sparse
points visible in the frame (least-squares fit per view).

**Assumptions.** The sparse cloud is locally trustworthy; mono-depth has
consistent *relative* structure. This is **metric-by-alignment** —
labeled approximate everywhere, never sensor-grade.

**Diagnostics.** Fit residuals in the stage facts; the label
`metric-by-alignment` flows into provenance.

## Unprojection (`reconstruction/depth_to_points.py`)

**Model.** Pinhole back-projection: `p_cam = d(x) · K⁻¹·[u, v, 1]ᵀ`,
then rigid transform to world via the registered pose.

**Assumptions.** Trusted intrinsics `K`; depth is along the optical
axis (z-depth, per COLMAP convention); pose registered.

**Diagnostics.** Invalid-depth masking, explicit frame conversion, skip
with reason when `K` is untrusted.

## Normal estimation (`reconstruction/meshing/preprocess.py`)

**Model.** Local PCA: for each point, take k nearest neighbors
(scipy cKDTree), compute the smallest-eigenvalue eigenvector of the
local covariance — the surface normal.

**Orientation.** Sign resolved toward the observing camera center
(`n·(c − p) > 0`) — required by Poisson; multiple observers vote.

**Assumptions.** Local planarity at the neighborhood scale; oriented
cameras exist. Deterministic (no RNG).

## Camera-envelope filtering (`reconstruction/meshing/preprocess.py`)

**Model.** The sparse SfM cloud bounds the plausible scene: envelope
radius = 95th-percentile distance from sparse points to their nearest
camera center. Depth-derived points beyond it are unprojection
artifacts and are removed.

**Assumptions.** Sparse points are triangulated (trustworthy); cameras
surround the scene. **Observed necessity:** depth outliers at km scale
defeated global statistical filtering and produced garbage meshes.

## Surface reconstruction (`reconstruction/meshing/surface.py`)

**Model.** Screened Poisson reconstruction (COLMAP `poisson_mesher`):
solves for an indicator function over an adaptive octree from oriented
points, then extracts the zero level-set.

**Observed constraints (diagnostics-driven).** Solver depth is bounded
by point density (sparse clouds support shallow depth); the dense-cloud
default `trim=10` prunes everything on sparse input — the backend
defaults `trim=0` and retries at the observed max depth when a run
yields an empty mesh.

**Assumptions.** Oriented normals present; density sufficient for the
requested octree depth.

## Point-cloud preprocessing

- **Voxel downsample:** one representative point per voxel (mean),
  deterministic bucketing.
- **Statistical outlier removal:** distance to k-th neighbor; points
  beyond `mean + α·std` of the distribution removed. Insufficient
  alone against far outliers — hence the envelope filter above.

## Lifting 2D masks to 3D (`perception/instances/lifting.py`)

**Model.** For mask pixels with valid metric depth: `p = K⁻¹·[u,v,1]·d`
in camera frame → world via pose; hypothesis = point set + centroid +
bounds.

**Assumptions.** Metric depth (absolute refusal otherwise — returns
`None`), registered pose, mask on the source image plane.

**Diagnostics.** Valid/invalid pixel counts; `INFERRED` provenance for
AABB approximations; per-observation evidence ids preserved.
