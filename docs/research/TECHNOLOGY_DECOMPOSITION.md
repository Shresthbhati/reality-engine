# Technology Decomposition — Reality Engine Candidate Investigation

Each candidate is decomposed into individual capabilities (per the mission's decomposition rule), not treated as an indivisible product. For each capability: problem solved, algorithmic approach, data structures/interfaces, inputs/outputs, compute/hardware requirements, accuracy/robustness/failure modes, strengths vs Reality Engine (RE), weaknesses/integration cost, license, and the RE decision (RETAIN / WRAP / ADAPT / REIMPLEMENT / REPLACE / DEFER / REJECT).

Evidence basis: direct source inspection of the cloned repos (`codes/rea`, `codes/colmap`, `codes/Open3D`) and of the RE codebase; pinned commit SHAs recorded in `REPOSITORY_INVENTORY.md`. No model weights or datasets were downloaded (PyPI unreachable in this environment).

---

## A. REA — Reverse Engineer Anything (morluto/rea, b22b0ef, MIT)

REA is a Node.js/TypeScript reverse-engineering platform (CLI + MCP server), v6.7.0. It is a **development-time research instrument**, not a 3D/reconstruction library and not a RE runtime dependency.

### Capability A-1: Static JavaScript/asar application analysis
- **Problem:** Recover a module graph, facts, and evidence from a JavaScript application directory or Electron ASAR without source maps.
- **Algorithmic approach:** AST-only bundle projection; directory/ASAR reads; source-map codec port; file-location helpers. Runs in a dedicated worker process.
- **Data structures/interfaces:** `src/domain/javascript/*`, `src/application/javascript/javascriptApplicationAnalysis.ts`, `src/cli/webModuleCommands.ts`, `sourceToBundleCommand.ts`. Evidence projected through `inspect-analysis-view` (summary/page/javascript-facts views).
- **Inputs/outputs:** Input = app directory or `.asar`; Output = Evidence JSON (application graph), projectable into summary/page/facts views.
- **Compute/hardware:** Node.js; single process per invocation; memory bounded by the file reader, not inline input.
- **Accuracy/robustness/failure:** Static only (no execution); honors empty-path/NUL rejection; no silent fabrication. Coverage reported explicitly; simulated providers do not establish real-provider behavior.
- **Strengths vs RE:** RE has no JS/app-analysis capability (out of RE's core domain); the *design* (domain/application/cli layering, evidence projection views) is relevant.
- **Weaknesses/cost:** Requires Node 22+/24+/26+ (this env has 26.8.1, below 26.12.1 minimum, so `npm ci && build` was not attempted); commercial deep-analysis providers (Ghidra/IDA/Hopper/Frida/Kuna) are bring-your-own and not installed.
- **License:** MIT (REA-authored); third-party material retains own licenses (THIRD_PARTY_NOTICES.md).
- **Decision:** ADAPT patterns only (provider registry, evidence ledger, unknown ownership, honest-failure contracts). Do not embed REA in RE. Status: research/source inspection only.

### Capability A-2: Deep binary analysis provider registry (Ghidra / IDA / Hopper / Frida / Kuna)
- **Problem:** Decompile, inspect CFG/references, and trace runtime behavior of native binaries through a provider.
- **Algorithmic approach:** `AnalysisProviderRegistry` = sorted deep candidates with deterministic selection and NO fallback; `SessionProviderRouter` + `BinarySession` = one immutable deep binding per target; `Composition` layer = typed provider/session factories; `EvidenceLedger` + `Unknown` ownership; bounded lifecycle, cleanup retry, attributed failures.
- **Data structures/interfaces:** provider contracts in `src/contracts/*`; providers under `src/providers/{ghidra,ida,hopper,kuna,...}`; bridges (`bridge/ghidra/ReaGhidraBridge.java`, `bridge/hopper_bridge.py`).
- **Inputs/outputs:** Input = binary + provider selection; Output = decompilation/references/CFG/runtime trace as structured Evidence.
- **Compute/hardware:** Provider-specific (Ghidra/IDA/Hopper are heavy desktop apps; Frida needs a running target process).
- **Accuracy/robustness/failure:** Honest failures (attributed), no silent fallback, cleanup retry, process supervision.
- **Strengths vs RE:** A well-structured provider/evidence model that RE's perception/reconstruction provider interfaces (IDepthBackend, ISegmentationBackend, IReconstructionBackend) can learn from.
- **Weaknesses/cost:** Commercial tools BYOS; not installable/testable here; out of RE's 3D domain.
- **License:** MIT for REA; provider tools have their own commercial licenses.
- **Decision:** ADAPT the provider-registry/evidence-ledger design patterns into RE's provider contracts (documented, no code copy). Status: research only.
---

## B. COLMAP (colmap/colmap, fa08906, New BSD)

COLMAP is the SfM/MVS baseline already wrapped by RE (`reconstruction/backend/colmap_backend.py`, `ReconstructionOrchestrator`).

### Capability B-1: Structure from Motion (SfM)
- **Problem:** Recover camera poses and sparse 3D structure from unordered overlapping images.
- **Algorithmic approach:** Feature extraction (SIFT), exhaustive or vocabulary-tree matching, geometric verification (essential/fundamental matrix), incremental or sparse bundle adjustment (sparse Cholesky), output COLMAP-text cameras.txt/images.txt/points3D.txt.
- **Data structures/interfaces:** COLMAP-text format; `ReconstructionResult {points, camera_poses, registration_status, merge_report, dense_points, dense_report}`.
- **Inputs/outputs:** Input = image directory + (optionally) camera prior; Output = sparse reconstruction + camera poses.
- **Compute/hardware:** C++ binary; GPU optional (CUDA build for MVS); feature extraction + matching are the costlier steps.
- **Accuracy/robustness/failure:** Industry-standard for photogrammetry; fails honestly when overlap insufficient (RE raises `ReconstructionBackendUnavailableError`/`ReconstructionStepError` with real stderr).
- **Strengths vs RE:** RE already wraps COLMAP; no gap to fill. COLMAP is the baseline until a controlled benchmark justifies change.
- **Weaknesses/cost:** Separate binary dependency; no Python library; GPU build needed for dense MVS at scale.
- **License:** New BSD (permissive) — safe to wrap as a subprocess; no linking.
- **Decision:** RETAIN as baseline. Status: implemented (colmap_backend + orchestrator + fake backend for tests). `tests/test_colmap_backend.py` + calibration/registration tests pass (94).

### Capability B-2: Multi-View Stereo (dense MVS) + meshing
- **Problem:** Dense depth from posed images; mesh + texture.
- **Algorithmic approach:** Stereo depth map fusion (fused.ply), Poisson meshing, texture mapping.
- **Data structures/interfaces:** `dense_points` (fused.ply parsed), `dense_report` (DenseMVSRun.to_dict + parse_facts), artifact-store ingestion via `ingest_fused_ply`.
- **Inputs/outputs:** Input = sparse model + images; Output = fused point cloud + mesh.
- **Compute/hardware:** GPU strongly recommended; memory-heavy.
- **Accuracy/robustness/failure:** Dense failure RAISES (never degrades to sparse-only success); artifact persisted through content-addressed store before temp workspace is deleted.
- **Strengths vs RE:** Already wired (dense opt-in continuation, provenance, artifact store).
- **Weaknesses/cost:** GPU/memory; not run in CI (slow marker, manual only).
- **License:** New BSD.
- **Decision:** RETAIN. Status: implemented (dense opt-in, artifact store). Not CI-tested due to GPU/binary absence in CI.

---

## C. Open3D (isl-org/Open3D, b6c5e19, MIT)

Open3D is a 3D data-processing library (point clouds, meshes, RGB-D images, voxel grids, registration, ICP, meshing).

### Capability C-1: Point-cloud registration (ICP / global registration)
- **Problem:** Align one point cloud to another (cross-source alignment).
- **Algorithmic approach:** Point-to-point ICP, point-to-plane ICP (caller-supplied normals), colored ICP, RANSAC-based global registration (FPFH features), multi-scale ICP, voxel downsampling, statistical outlier removal, cKDTree nearest-neighbor.
- **Data structures/interfaces:** `open3d.geometry.PointCloud`, `registration.registration_icp`, `registration_ccolor`, `global_registration` (RANSAC/FPFH), `voxel_down_sample`, `statistical_outlier_removal`.
- **Inputs/outputs:** Input = source/target point clouds + optional initial transform + normals; Output = refined transform + fitness/RMSE/inlier RMSE + correspondence set.
- **Compute/hardware:** C++ backend with Python bindings; CPU; GPU optional for some ops.
- **Accuracy/robustness/failure:** ICP needs good initialization (local convergence); global registration (RANSAC+FPFH) handles poor initialization at higher cost. Failure modes: local minima, degenerate geometry, normal estimation noise.
- **Strengths vs RE:** RE's `registration.registration.py` already implements `register_icp` (point-to-point via scipy.spatial.cKDTree) and `register_icp_point_to_plane` with `ResidualStats` + `RegistrationCovariance` derived from measured residuals, and `RegistrationEngine` with honest AttemptRecord/status. Open3D's added value over RE = colored ICP + global feature-based registration (for poor initialization).
- **Weaknesses/cost:** Open3D python wheel NOT installable here (PyPI unreachable); cannot be a runtime dependency in this environment.
- **License:** MIT.
- **Decision:** DEFER — Open3D python package unavailable in this env; RE's ICP already satisfies the registration baseline. Revisit if global registration (poor-initialization alignment) becomes a proven gap. Status: research only.

### Capability C-2: Point-cloud processing (downsampling, outlier removal, meshing)
- **Problem:** Clean, downsample, and mesh point clouds.
- **Algorithmic approach:** Voxel grid downsampling, statistical/radius outlier removal, Poisson ball meshing, surface normal estimation (PCA over k-nearest neighbors), oriented bounding boxes.
- **Data structures/interfaces:** PointCloud, TriangleMesh, KDTreeFlann.
- **Inputs/outputs:** Point cloud -> downsampled/cleaned point cloud / mesh.
- **Compute/hardware:** CPU; k-NN searches dominate.
- **Accuracy/robustness/failure:** Deterministic given fixed voxel size/k; outlier removal thresholds are parameters.
- **Strengths vs RE:** RE has point-cloud processing scattered across `reconstruction/fusion`, `reconstruction/mvs`, `depth_to_points.py`. Open3D would consolidate, but isn't installable here.
- **Weaknesses/cost:** Not installable (PyPI blocked).
- **License:** MIT.
- **Decision:** DEFER (open3d wheel unavailable). RE's numpy/scipy-based processing suffices for now.

---

## D. Depth & Segmentation (MiDaS / SAM / Depth Anything V2 / segment-anything)

RE already has `IDepthBackend` (`MiDaSDepthBackend`) and `ISegmentationBackend` (`SAMSegmentationBackend`) plus `perception/availability.py` (probe-only, no imports/downloads). All are **blocked at runtime**: model checkpoints are not cached and cannot be downloaded (PyPI unreachable). torch/torchvision/segment-anything are installed; only weights are missing.

### Capability D-1: Monocular relative/metric depth
- **Problem:** Per-pixel depth from a single RGB image.
- **Algorithmic approach:** MiDaS (DPT/Hybrid/Small) via torch.hub; Depth Anything V2 (ViT-S/B/L/G) — relative depth, V2 adds metric-depth option; better fine detail on transparent/reflective surfaces.
- **Data structures/interfaces:** `DepthMap {evidence_id, width, height, values[row][col], unit ("relative"|"meters"), uncertainty}`.
- **Inputs/outputs:** Input = PHOTO/VIDEO evidence items; Output = list of DepthMap (skip, never fabricate, items it can't process).
- **Compute/hardware:** torch + checkpoint; CPU slow, GPU recommended; ViT-S small (~few GB VRAM), Large/Giant 8+ GB.
- **Accuracy/robustness/failure:** Relative depth by default (needs scale reference to align with metric SfM); generalization across in-the-wild photos is the strength; failure = skip item honestly.
- **Strengths vs RE:** RE's MiDaS backend matches this design; Depth Anything V2 offers better raw quality but per-checkpoint licensing is uncertain.
- **Weaknesses/cost:** Checkpoint not downloadable here; relative depth needs metric anchoring; no GPU in this env.
- **License:** Apache 2.0 (code); checkpoint licenses vary — verify per-checkpoint (TECHNOLOGY_REGISTRY.md flags Low-to-Medium confidence).
- **Decision:** DEFER — checkpoint unavailable, no GPU, per-checkpoint licensing unresolved. Status: backend code exists, blocked at runtime.

### Capability D-2: Promptable/automatic segmentation (SAM)
- **Problem:** Instance/semantic masks from images.
- **Algorithmic approach:** SAM (ViT-B/L/H) automatic mask generation via `SamAutomaticMaskGenerator`; masks carry predicted_iou/stability_score; class-agnostic (label mapping is downstream).
- **Data structures/interfaces:** `SegmentedRegion {region_id, evidence_id, label, mask[row][col], confidence, uncertainty}`; `SegmentationResult {evidence_id, regions, model_name, model_version}`.
- **Inputs/outputs:** Input = PHOTO/VIDEO evidence; Output = SegmentationResult per item (skip, never fabricate).
- **Compute/hardware:** torch + torchvision + segment-anything (installed) + SAM checkpoint (missing); CPU slow, GPU recommended.
- **Accuracy/robustness/failure:** Strong open-vocabulary segmentation; class-agnostic labels need downstream ontology mapping; SAM2 adds video/temporal tracking unused by photo-only pipeline (TECHNOLOGY_REGISTRY.md: keep SAM1 for now).
- **Strengths vs RE:** RE's SAM backend already matches this; blocked only by checkpoint.
- **Weaknesses/cost:** Checkpoint unavailable; no GPU; class-agnostic.
- **License:** Apache 2.0 (code); weights separate terms — verify.
- **Decision:** DEFER — checkpoint unavailable, no GPU. Status: backend code exists, blocked at runtime.

---

## E. SLAM / VIO / pose-graph (ORB_SLAM3, VINS-Fusion, GTSAM, OpenVINS)

### Capability E-1: Visual-inertial odometry / SLAM
- **Problem:** Trajectory estimation, loop closure, relocalization from camera (+IMU).
- **Algorithmic approach:** ORB_SLAM3 (ORB features, PnP, bundle adjustment, loop closure, monocular/stereo/VI modes); VINS-Fusion (VINS, graph optimization); OpenVINS.
- **Data structures/interfaces:** Keyframe/pose graph, map points, loop-closure candidates.
- **Inputs/outputs:** Images (+IMU) -> trajectory (TUM format) + map.
- **Compute/hardware:** Heavy C++ builds (Pangolin, ROS/catkin); GPU/CPU.
- **Accuracy/robustness/failure:** State of the art for SLAM; GPLv3 for ORB_SLAM3/VINS-Fusion/OpenVINS.
- **Strengths vs RE:** RE has `trajectories/` (TUM, VIO federation, sync) + ICP + GNSS alignment, but no visual odometry or loop closure.
- **Weaknesses/cost:** GPLv3 contamination risk; heavy build; no robotics use case defined yet in RE.
- **License:** ORB_SLAM3 GPLv3; VINS-Fusion GPLv3; OpenVINS GPLv3; GTSAM BSD.
- **Decision:** REJECT ORB_SLAM3/VINS-Fusion/OpenVINS (GPLv3 + no defined slice); DEFER GTSAM (BSD pose-graph could support Layer 4 multi-session alignment later). Status: not implemented.

---

## F. Geospatial (CesiumJS, 3d-tiles, MapLibre, GDAL, PROJ)

Not cloned. Deferred until a geospatial visualization slice is prioritized. RE currently exports CityJSON/CityGML/glTF/USDA and has a frame graph with ENU/UTM/WGS84/ECEF frames. Capabilities to evaluate later: coordinate transformation (PROJ), tiled 3D streaming (3D Tiles/CesiumJS), large-scene LOD (MapLibre/Cesium), GDAL translation.

---

## G. BIM / CAD / semantic buildings (IfcOpenShell, OpenUSD, Blender, CityJSON/CityGML)

Not cloned. Deferred. RE exports CityJSON/CityGML but lacks IFC/USD adapters. Capabilities to evaluate later: IFC import/export (IfcOpenShell), USD composition (OpenUSD), loss-aware format translation. License mix (IfcOpenShell LGPL/MIT; OpenUSD Apache 2.0) must be verified before combining with RE.

---

## Cross-cutting findings (REA-informed)

1. **Provider + evidence honesty:** REA's `AnalysisProviderRegistry` (deterministic selection, no fallback), `SessionProviderRouter`, `EvidenceLedger`, and `Unknown` ownership mirror RE's existing conventions (`IReconstructionBackend` raises rather than fabricates; `RegistrationResult` status=blocked with reason; `StageStatus` complete/partial/unavailable/failed; `availability.py` probe-only). These patterns are aligned — no change needed, documented as RETAIN/ADAPT patterns.
2. **Immutable domain models:** REA uses domain value types; RE uses `@dataclass(frozen=True)` for WorldIR/math/config (ADR #1). Aligned — RETAIN.
3. **Runtime blocking must be loud:** REA's deep providers are BYOS and setup is opt-in/approval-gated; RE's unavailable backends raise explicit `*UnavailableError` with remediation. Aligned — RETAIN.

**Bottom line of decomposition:** RE's core capabilities (SfM via COLMAP, ICP/GNSS registration, metric scale, depth consistency, WorldIR, WorldStore, exporters) are already implemented and honest. The genuine gaps are all **environment-blocked** (neural checkpoints unavailable, no GPU, open3d wheel unavailable) or **scope-deferred** (SLAM, BIM, geospatial, multi-session pose-graph). No capability requires a rewrite of a working RE component.

