# Reality Engine - Technology Integration Audit

Audit date 2026-11-05. Branch `claude/integration-2026-11-05`. Replaces the earlier garbled stub, whose "installed" claims for Open3D, PCL, CityGML, USD and GDAL were wrong (see section 1).

Codes: **A** operational, **B** implemented but not integrated, **C** partial, **D** simulated or heuristic, **E** docs-only, **F** broken, **G** unverified.
`*` after a code means reach and availability were verified, but that path was not executed in this audit.
"App path" means reachable from `apps/api/jobs.py`, `engine/pipeline/vertical_slice.py`, `engine/orchestrator/universal_pipeline.py` or `apps/cli/main.py` without going through tests.

## 1. Observed environment

| Item | Observation |
|---|---|
| venv | `reality-engine/.venv`, package list from `pip list` (command run). |
| Installed and relevant | torch 2.14.1+cu130, torchvision 0.29.1, kornia 0.9.0, opencv-python-headless 5.0.0.93 (`FFMPEG: YES (prebuilt binaries)`, `GStreamer: NO`), numpy 2.5.3, scipy 1.18.1, pyproj 3.8.0, trimesh 5.2.0, shapely 2.2.0, networkx 3.7, ifcopenshell 0.9.0.post1, OpenEXR 3.5.2, pillow 12.3, fastapi/sqlalchemy/aiosqlite |
| NOT installed (`importlib.util.find_spec` = None) | pycolmap, lightglue, transformers, timm, ultralytics, open3d, pymeshlab, gtsam, ceres, av, ffmpeg-python, imagehash, scenedetect, exiftool, osgeo (GDAL), nerfstudio, gsplat, segment_anything, sam2, easyocr, paddleocr, pytesseract, onnxruntime, pcl, faiss, sklearn, skimage, matplotlib, pyvista |
| GPU | `torch.cuda.is_available()` True, NVIDIA GeForce RTX 4050 Laptop GPU, 6140 MiB total |
| COLMAP | `C:\Users\shres\tools\colmap-extracted\bin\colmap.exe`, `COLMAP 4.2.0 (Commit be5e291 on 2026-08-31 with CUDA)`. On PATH in this shell. |
| Not on PATH | ffmpeg, ffprobe, exiftool, glomap (standalone), OpenMVS, ORB-SLAM3 binaries |
| Source only | `~/tools/open_vins` holds the OpenVINS source tree (Dockerfiles, ReadMe). No built binary found. |
| Torch hub cache `~/.cache/torch/hub/checkpoints` | aliked-n16.pth, aliked_lightglue_v0-1_arxiv-pth, disk_lightglue_v0-1_arxiv-pth, depth-save.pth (DISK), dpt_hybrid_384.pt (469.9M), midas_v21_small_256.pt, maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth, sam_vit_b_01ec64.pth, resnet18, efficientnet_b0, tf_efficientnet_lite3. Repo dirs: intel-isl_MiDaS_master, facebookresearch_segment-anything_main. |
| HF cache `~/.cache/huggingface/hub` | depth-anything/Depth-Anything-V2-Small-hf, google/owlvit-base-patch32, faster-whisper-medium, ModernBERT, MiniLM, others |
| Missing checkpoint | `fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth` (needed by `torchvision-coco` detector). No XFeat, DeDoDe, Metric3D, OWLv2, Grounding DINO, SAM2, OneFormer weights. |

Cached weights are useless without their runtime package. Two cases are confirmed: Depth Anything V2 Small and OWL-ViT weights are cached but `transformers` is absent, and `sam_vit_b` is cached but `segment_anything` is absent.

## 2. Capability table

| Area | Upstream | Reality Engine file(s) | App-path reach | Class | Evidence |
|---|---|---|---|---|---|
| Video decode | OpenCV (bundled FFmpeg) | `evidence/video.py:252,354`, `evidence/importers.py:396,453`, `reconstruction/video_frames.py`, `reconstruction/capture_quality.py:241` | Yes: importers and the evidence path | A* | `cv2.getBuildInformation()` shows FFMPEG YES. No video decoded in this audit. |
| Video decode | FFmpeg binary / PyAV | none | no | absent | `where ffmpeg` finds nothing; `av` not installed; no `subprocess` call to ffmpeg anywhere (grep). |
| EXIF | Pillow `getexif` | `evidence/geo_anchor.py:44-54`, `evidence/image_check.py` | Yes (`jobs.py:43` imports `inspect_image`) | A* | ExifTool not installed; Pillow only. |
| Near-duplicate hashing | imagehash | `evidence/dedup.py` (own dHash/pHash) | only re-exported from `evidence/__init__.py`; no caller in `apps/`, `engine/`, `reconstruction/` | B | grep for `evidence.dedup`/`compute_phash` outside the module returns only `__init__`. imagehash not installed, own implementation. |
| Scene/shot detection | PySceneDetect | none | no | absent | Not installed, no code. |
| SfM (sparse) | COLMAP 4.2.0 CLI | `reconstruction/backend/colmap_backend.py` (feature_extractor, exhaustive_matcher, mapper; `image_registrator` via `colmap_session.py`) | Yes: `vertical_slice.py:337`, `jobs.py:642` | A* | Binary runs (`colmap help`). SIFT with affine shape + domain-size pooling forces CPU (`colmap_backend.py:222-231,404-406`). Prior doc claims a 22/32-camera South Building run; not re-run here. |
| SfM (global) | GLOMAP / `colmap global_mapper` | none | no | absent (capability present in binary) | `colmap help` lists `global_mapper`, `hierarchical_mapper`, `pose_prior_mapper`; grep for `glomap\|global_mapper` in source = 0 hits. No standalone `glomap`. |
| SfM python API | pycolmap | none | no | absent | Not installed, no code. |
| Dense MVS | COLMAP `image_undistorter` + `patch_match_stereo` + `stereo_fusion` | `reconstruction/dense_pipeline.py`, `reconstruction/backend/dense_output.py` | Yes (`vertical_slice.py:1380-1414`, gated by `engine/pipeline/dense_gate.py`, `jobs.py:657`) | A* | `dense_mvs_available()` probes `colmap help`; build has CUDA. Prior doc cites 246,816 fused points; not re-run. |
| Surface meshing | COLMAP `poisson_mesher` | `reconstruction/meshing/surface.py:66-73`, `vertical_slice.py:1625` | Yes | A* | Probed at runtime, not run here. |
| Dense MVS (CPU) | own plane sweep on OpenCV | `reconstruction/backend/plane_sweep.py`, `dense_registry.py` | `vertical_slice.py:1156`, off by default | B | Real algorithm, internal, default off. |
| Dense MVS | OpenMVS | `reconstruction/backend/registry.py:58` (descriptor only) | no | E | No binary, descriptor lists names only. |
| Feature: SIFT | COLMAP SIFT (primary), OpenCV SIFT | colmap_backend.py; `reconstruction/matching/sift.py` | Yes | A* | See SfM row. |
| Feature: ALIKED + LightGlue | kornia 0.9.0 `ALIKED`, `LightGlue` | `reconstruction/matching/lightglue.py` (`ALIKEDLightGlueMatcher`), `matching/__init__.py:get_matcher` | Selectable by name only (benchmarks/tests). Not used for SfM. | B | Ran on a synthetic 640x480 pair on CPU: 1041 matches, 1040 inliers, 49.8 s including model load. Weights cached (`aliked-n16.pth`, `aliked_lightglue_v0-1_arxiv-pth`). |
| Feature: DISK + LightGlue | kornia `DISK`, `LightGlue` | `matching/lightglue.py` (`LightGlueMatcher`) | **Yes**: `make_pair_verifier("auto")` in `scene_identity.py:96-105` picks `lightglue` when kornia imports; called by `jobs.py:_scene_gate` (L388-403). Scene grouping only, not SfM. | A | Same synthetic pair, CPU: 2068 matches, 2032 inliers, 12.7 s including load. Weights cached (`disk_lightglue_v0-1_arxiv-pth`, `depth-save.pth`). |
| Feature: XFeat | verlab/accelerated_features | `reconstruction/matching/xfeat.py` | via `get_matcher("xfeat")` only | D | `from modules.xfeat import XFeat` is not installed; the bare `except Exception` silently falls back to OpenCV AKAZE (xfeat.py L70-72) while `name = "xfeat@accelerated_features"`. `available_matchers()` returns `"xfeat": "available"` unconditionally (`matching/__init__.py:49`). Result is mislabelled. No weights. |
| Feature: DeDoDe | Parskatt/DeDoDe | `reconstruction/matching/dedode.py` | via `get_matcher("dedode")` only | D | `from dedode import DeDoDe` is not a package we have; silent fallback to OpenCV BRISK (dedode.py L71-73) under the name `dedode@v2`. `available_matchers()` says "available". kornia 0.9.0 does ship `kornia.feature.DeDoDe` (signature `from_pretrained(detector_weights='L-C4-v2', descriptor_weights='G-upright')`), which is not used. |
| Feature: LoFTR, MASt3R | kornia LoFTR, naver/mast3r | `matching/__init__.py` `_NOT_IMPLEMENTED` | no | E | Explicitly marked not implemented. kornia has `kf.LoFTR(pretrained='outdoor')` but weights are not cached. |
| COLMAP built-in ALIKED / LightGlue | COLMAP 4.2 ONNX models | none | no | absent (capability present) | Binary contains the enum strings `ALIKED_N16ROT`, `ALIKED_N32`, `SIFT_LIGHTGLUE`, `ALIKED_BRUTEFORCE`, `ALIKED_LIGHTGLUE` (found with `grep -a -o` on colmap.exe). `colmap feature_extractor -h` shows `--FeatureExtraction.type`, `--AlikedExtraction.*`; `exhaustive_matcher -h` shows `--FeatureMatching.type`, `--AlikedMatching.lightglue_model_path`. ONNX files download from GitHub on first use; none are cached. |
| SLAM / VIO | ORB-SLAM3, OpenVINS, Basalt | `trajectories/backend/{orb_slam3,openvins,basalt}_backend.py`, `selection.py` | Only `engine/core/verification.py:200` (diagnostics) | B | Subprocess adapters. No binary on PATH. `synthetic/vio.py` generates synthetic trajectories and states it is not ORB-SLAM3. OpenVINS source is cloned, not built. |
| Depth: MiDaS | intel-isl/MiDaS via torch.hub | `perception/depth/midAS_backend.py` | Default: `VerticalSliceOptions.depth_model="DPT_Hybrid"` (`vertical_slice.py:95`), `jobs.py:665`; run at `vertical_slice.py:875-882` | **F** in this venv | Reproduced: `torch.hub.load(<cache>/intel-isl_MiDaS_master, "DPT_Hybrid", source="local")` raises `ModuleNotFoundError: No module named 'timm'`. `perception/availability.py` reports "available" because it only checks `torch` and the cached `dpt_hybrid_384.pt`. The failure is caught at `vertical_slice.py:883-889` and recorded as `status: skipped`, so the pipeline continues without depth. `pyproject.toml` `perception` extra lists no `timm`. |
| Depth: Depth Anything V2 | HF `transformers` | `perception/depth/depth_anything_backend.py` | Selected only if `depth_model` contains `depth_anything` (`vertical_slice.py:876`); not the default | F | `DepthBackendUnavailableError: Missing dependencies ... transformers`. Weights are cached. |
| Depth: Metric3D | Metric3D | none | no | absent | grep for `metric3d` outside the heuristic file's disclaimers = 0. |
| Depth: heuristic prior | none (not a model) | `perception/depth/metric_depth_backend.py` (`HeuristicDepthPriorBackend`) | no (tests and `perception/model_registry.py:243` only) | D | See section 3. |
| Detection: Mask R-CNN | torchvision `maskrcnn_resnet50_fpn` | `perception/detection/maskrcnn_backend.py` | `vertical_slice.py:952-975`, but only after metric depth maps exist (`vertical_slice.py:941-949` returns "no metricized depth maps survived") | C | `report_real_models()` says available (weights `maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth` cached). The step is unreachable while depth is F. Inference not run here. |
| Detection: OWL-ViT | HF `google/owlvit-base-patch32` | `perception/detection/backends.py` `OwlVitDetector` | via `get_detector()`; used by `perception/detail/crop_pass.py`, which is imported by no non-test module | B (and unavailable) | `available()` returns False: `No module named 'transformers'`. Code is real (loads processor/model with `local_files_only=True`). |
| Detection: torchvision COCO | torchvision FasterRCNN MobileNetV3 | `backends.py` `TorchvisionCocoDetector` | via `get_detector()` only | B (unavailable) | `available()` False: weights `fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth` not found. |
| Detection: YOLOX, MaskDINO | Megvii YOLOX, IDEA MaskDINO | `backends.py` `YoloxDetector`, `MaskDinoDetector` | via registry only | E | Hard-coded boxes are gone. Both now raise `DetectorUnavailable("... real inference not implemented ...")`. Neither package is installed. See section 3. |
| Detection: OWLv2, Grounding DINO, YOLO(v8+) | n/a | none | no | absent | |
| Segmentation: SAM | segment-anything | `perception/segmentation/sam_backend.py` | not called from the vertical slice | F | `report_real_models()`: `status='blocked'`, "segment_anything package is not installed". Checkpoint `sam_vit_b_01ec64.pth` is cached. |
| Segmentation: SAM2 | facebookresearch/sam2 | none | no | absent | |
| Segmentation: OneFormer | HF transformers | `perception/segmentation/oneformer_backend.py` | `engine/technologies/registry.py` only (no app caller) | **D** | `_ensure_model()` swallows the load error and sets `"fallback"`; `segment()` then returns two fixed regions, "wall" 0.92 and "floor" 0.95 with 2x2 boolean masks (L99-118). Both the model load (L44-46) and the real inference (L96) swallow every exception. This is the one remaining hard-coded perception output. |
| Tracking | own implementation | `perception/tracking/temporal.py`, `perception/instances/track_backend.py` | Not in the vertical slice | B | No ByteTrack/DeepSORT/re-id upstream anywhere. |
| OCR | none | none | no | absent | grep for ocr/tesseract/easyocr/paddleocr = 0. |
| Point cloud | Open3D | none | no | absent | `open3d` not installed; the stub's "Installed" claim was false. |
| Point cloud | PCL | `perception/pointcloud/pcl_core.py` (numpy rewrite) | no | D (self-written) | See section 4. `pcl` not installed. |
| Mesh | trimesh 5.2.0 | `reconstruction/meshing/repair.py:23` (guarded import) | `repair` is not imported by the pipeline | B | Installed. Not called by the vertical slice or the technology adapters. |
| Mesh | PyMeshLab | none | no | absent | Not installed. |
| Mesh simplification | meshoptimizer | `reconstruction/meshing/meshopt_simplifier.py` (own QEM) | no | D (self-written) | Not the upstream library. |
| Optimisation | scipy 1.18.1 | 17 non-test modules (e.g. `reconstruction/consistency.py`, `reconstruction/mvs/point_cloud_ops.py`, `engine/pipeline/dense_judge.py`) | Yes | A* | Imported in app modules. Not exercised here. |
| Optimisation | GTSAM, Ceres | `trajectories/optimization/factor_graph.py` (own LM) | no | D (self-written) | Neither library installed. COLMAP's own bundle adjustment is what the pipeline uses. |
| BIM / IFC | ifcopenshell 0.9.0.post1 | `exporters/ifc_bridge.py`, `apps/api/routes_export.py:89` | Yes: `format=ifc` export route, 501 when missing | A* | `import exporters.ifc_bridge` and `import ifcopenshell` succeed. Header says it targets the 0.8.x API; export not run here. |
| GIS | pyproj 3.8.0 | `world_ir/geo.py:20-79`, `geo/osm_context.py:21` | Yes | A* | Real `Transformer.from_crs` calls. |
| GIS | GDAL / osgeo | none (`geo/terrain_analysis.py` is numpy) | no | absent | `osgeo` not installed. |
| GIS | `colmap model_aligner` georegistration | none | no | absent (capability present) | `model_aligner` listed in `colmap help`. |
| Neural recon | nerfstudio, gsplat | none | no | absent | Not installed; grep = 0. |
| Rendering / viewers | three 0.186, maplibre-gl 6.10 | `frontend/package.json:16,21` | Yes (frontend) | A* | Dependency declared; frontend not built in this audit. |
| Storage / jobs | FastAPI, SQLAlchemy 2 + aiosqlite, `world_ir.artifact_store`, `worldstore` | `apps/api/jobs.py`, `apps/api/models.py` | Yes | A* | Packages installed; job runner uses `asyncio.to_thread` + COLMAP subprocess ownership (`reconstruction/proc`). |
| ONNX Runtime | onnxruntime | `perception/acceleration/onnx_quantizer.py` (VRAM estimator) | no | D | Package absent. Output is arithmetic, not a quantised model. |

### Net result for the application path

Operational with a real upstream: COLMAP (SIFT, mapper, registrator, dense, Poisson), kornia LightGlue@DISK (scene grouping only), OpenCV, scipy, pyproj, ifcopenshell, Pillow EXIF.
Broken in this venv: the default monocular depth (MiDaS) and every stage that depends on its output (Mask R-CNN lifting). The cause is missing `timm`/`transformers` and a probe in `perception/availability.py` that reports "available" without importing the model's dependencies.

## 3. Specific readings requested

### `heuristic_depth_prior` (`perception/depth/metric_depth_backend.py`)
Per pixel, it does the following (lines 100-121):
1. `gray` = BT.601 luminance of the image.
2. `base_depth = 1.8 + 2.4 * (1 - clip(|y - cy| / cy, 0, 0.85))`. Rows near the image centre (the horizon) get about 4.2 m. The top and bottom edges get 1.8 + 2.4*0.15 = 2.16 m.
3. `depth = base_depth + 0.4 * (1 - clip(3 * |grad(gray)|, 0, 1))`. Smooth regions are pushed 0.4 m farther.
4. Multiplied by `f_px / max(w, h) / 1.2`, which is exactly 1.0 when `f_px` defaults to `1.2 * max(w, h)`, and a pure rescale otherwise.
5. Clipped to [0.3, 20] m.

It ignores scene content, camera pose and COLMAP geometry. The output is a fixed radial pattern perturbed by edges. The unit label is "meters" but nothing is metric. The `_checkpoint` constructor argument is stored and never used. The reported confidence is a constant 0.50 and `sigma = max(0.05, 0.02 * mean_depth)` is a fixed fraction, not an estimate. The file's own honesty contract (name, `scale_state="estimated"`) is accurate. It is not reachable from the application path and should stay out of it.

### Detection backends (`perception/detection/backends.py`)
| Backend | What it really does |
|---|---|
| `FakeDetector` | Returns a caller-supplied script. Test fixture. |
| `OwlVitDetector` | Real HF OWL-ViT forward pass with `local_files_only=True`; unavailable here (no `transformers`). |
| `TorchvisionCocoDetector` | Real FasterRCNN-MobileNetV3 forward pass; unavailable here (weights file missing). |
| `YoloxDetector` / `MaskDinoDetector` | `available()` only does `find_spec`. `detect()` always raises `DetectorUnavailable`. No hard-coded output remains in this file (checked by reading the whole file, 199 lines). |

Still hard-coded: `OneFormerSegmentationBackend` fallback (section 2) and the `engine/technologies/adapters.py` simulated outputs (section 4).

## 4. `engine/technologies` - 50 "technologies"

Reachability: `engine/technologies/*` is imported only by `apps/api/routes_technologies.py` (list/inspect/benchmark endpoints) and `apps/cli/cmd_techs.py`. Nothing in the reconstruction pipeline calls it. Of the 50 catalog entries, **0** call their real upstream library: `adapters.py` imports no upstream package, only Reality Engine modules (L16-30, plus lazy imports at L92, 183, 208, 237, 326, 406, 496, 524, 558, 590, 686-687, 768).

Registry scan (`get_technology_registry().scan_all()`, run): 32 `UNAVAILABLE`, 16 `INTERNAL_IMPLEMENTATION`, 2 `WRAPPED_EXECUTED` (kornia, trimesh). `WRAPPED_EXECUTED` only means `find_spec` succeeded. The adapter never calls trimesh, and its kornia branch calls a numpy module instead of kornia (see below).

Adapter dispatch is by subsystem (7 classes). Each class has a small number of `if self.tech_id == ...` branches. Every other tech id hits a final branch that returns fixed constants labelled `ExecutionClassification.SIMULATED`.

### 4a. The 13 ids with a "native" branch (all self-written numpy re-derivations)
| id | Branch calls | Real upstream called? | Class |
|---|---|---|---|
| kornia | `reconstruction/features/kornia_primitives.py` | No: imports numpy only (`import kornia` appears nowhere in that file). Hard-coded detection `{"label":"warped_surface","score":0.99}` (adapters.py L108). | D |
| teaser_plus_plus | `reconstruction/scale/teaser_solver.py` | No (`teaserpp_python` absent). Self-contained TLS. Certifiability not verified. | D (self-written) |
| gtsam | `trajectories/optimization/factor_graph.py` | No. 3-D position-only Levenberg-Marquardt. Not SE(3) iSAM2. | D (self-written) |
| sophus | `reconstruction/geometry/lie_algebra.py` | No. SO(3)/SE(3) in numpy. | D (self-written) |
| kiss_icp | `reconstruction/registration/kiss_icp.py` | No (`kiss_icp` absent). Returns `registered_cameras: 2`, `reprojection_error_px: 0.0` constants (L339-340). | D (self-written) |
| pcl | `perception/pointcloud/pcl_core.py` | No. SOR + RANSAC plane. Returns `mean_iou: 0.91` constant (L425). | D |
| manifold | `geometry/kernels/manifold_kernel.py` | No (`manifold3d` absent). Also used by `engine/compiler/geometry_compiler.py`. | D (self-written) |
| meshoptimizer | `reconstruction/meshing/meshopt_simplifier.py` | No. Reports `is_manifold: True, is_watertight: True, self_intersections: 0` as constants (L542-544). | D |
| openvdb | `geometry/openvdb_sdf.py` | No. Reports manifold/watertight constants (L575-577). | D |
| openmesh | `geometry/halfedge_mesh.py` | No. `is_manifold: True` constant (L604). | D |
| whitebox_tools, gdal | `geo/terrain_analysis.py` | No. Same function for both; reports `epsg_code: 4326` constant (L706). | D |
| onnx_runtime | `perception/acceleration/onnx_quantizer.py` | No (`onnxruntime` absent). VRAM arithmetic only; `speedup_factor` hard-coded 3.4/2.1 (L786). | D |

None of these self-written modules is imported by the application path except lie_algebra (only via kiss_icp), manifold_kernel (`geometry_compiler.py`) and `onnx_quantizer` (`perception/acceleration/__init__.py`). I checked each module for importers outside tests and `engine/technologies`: teaser_solver, factor_graph, kiss_icp, pcl_core, meshopt_simplifier, openvdb_sdf, halfedge_mesh, terrain_analysis and kornia_primitives have none.

### 4b. The 37 ids with no native branch
Every one returns the generic `SIMULATED` output with `confidence=0.85` and invented numbers. Examples: detections chair 0.92 / table 0.88 / wall 0.95 (L115-119), `rmse=0.012`, `inlier_ratio=0.985` (L262-279), 25 registered cameras, 15,000 points, `reprojection_error_px=0.65` (L353-368), floor/wall/ceiling segment counts (L431-436), 4,500 vertices (L629-636), 500,000 points streamed (L716), `speedup_factor=2.8`, `vram_mb=650` (L798-799).

| Subsystem | ids | Upstream state |
|---|---|---|
| visual_perception (11) | yolox, mmdetection, detectron2, paddledetection, maskdino, oneformer, mmsegmentation, paddleseg, segmentation_models_pytorch, xfeat, dedode | none installed. yolox, oneformer, xfeat, dedode are flagged `INTERNAL_IMPLEMENTATION` by the registry (`_INTERNAL_MODULE_MAP`) although yolox's detector raises unavailable, and xfeat/dedode are mislabelled OpenCV fallbacks. |
| geometric_estimation (4) | mrcal, kalibr, opengv, ceres_solver | none installed |
| reconstruction_mapping (7) | alicevision, openmvg, micmac, opendronemap, rtabmap, cartographer, basalt | none installed |
| pointcloud_3d (7) | pointcept, pointnext, minkowski_engine, torchsparse, spconv, kpconv, pytorch3d | none installed |
| geometry_cad_mesh (4) | libigl, trimesh, vtk, opencascade | trimesh installed but never called by the adapter |
| geospatial (2) | pdal, entwine | none installed |
| inference (2) | openvino, apache_tvm | none installed |

(The registry lists mmdetection/detectron2/paddledetection/maskdino/mmsegmentation/paddleseg/smp as plain `UNAVAILABLE`; the adapter would nevertheless return a SIMULATED success if invoked.)

Final status across the 50: 13 numpy re-derivations (one of them, kornia, ignores an installed upstream); 37 fixed-constant simulations (36 with the upstream not installed, plus trimesh, which is installed but never called).

## 5. `engine/technologies/benchmark.py` - numbers that are not measurements

| Line(s) | Instance |
|---|---|
| 243 | `measured_mem_mb = max(exec_result.memory_peak_mb, tracemalloc peak)`. The first operand is a constant set in the adapter (adapters.py L104,127,193,222,247,273,337,362,424,444,508,540,573,602,628,702,725,807). `tracemalloc` does not see CUDA or native allocations. |
| 290 | `vram_mb = min(host_vram_mb, 1250.0)` (visual perception). |
| 315 | `min(host_vram_mb, 180.0)` (geometric estimation). |
| 340 | `min(host_vram_mb, 2400.0)` (reconstruction/mapping). |
| 354 | `min(host_vram_mb, 1850.0)` (point cloud). |
| 379 | `min(host_vram_mb, 220.0)` (geometry/mesh). |
| 404 | `min(host_vram_mb, 350.0)` (geospatial). |
| 429 | `min(host_vram_mb, 580.0)` (inference). These seven are reported as `vram_peak_mb` with no `torch.cuda.max_memory_allocated` call anywhere. The value is the constant, capped by total VRAM, or 0.0 on CPU hosts. |
| 280-289 | For perception ids without `mean_gradient_magnitude`, `detection_confidence = exec_result.confidence` (the hard-coded 0.85 or 0.99) is stored as a metric with `passes_threshold = conf >= 0.80`. A constant passes its own threshold. |
| 290 / 306-314 | `inlier_ratio` 0.985 (simulated) and `registration_rmse_m` 0.012 (simulated) are compared against thresholds as measured values. Native branches such as kiss_icp also return `reprojection_error_px: 0.0`. |
| 431 | `all_passed = ... if metrics else True`: an id with zero metrics is "all passed". |
| 432-436 | Verdict `RECOMMENDED` for any `evaluation_priority == 1` id that "passes", with no check of `classification`. A simulated maskdino (priority 1, constant 0.85 > 0.80) is RECOMMENDED. |
| 450 | `model_checkpoint=None` for every result; no weights are ever identified. |
| 451 | `technology_version = desc.license_notes or "v1.0.0"`: the version string is invented, not read from a package. |
| 453-457 | `measurement_methodology` is one of two strings keyed on classification only. |
| 257, 461 | `failure_rate_pct = 0.0` constant; the adapter call at L238 has no try/except, so failures raise instead of being counted. |
| 248-253, 464 | `world_ir_promotion_verified` is true whenever `world.metadata` is non-empty, and every `promote_to_world_ir` writes metadata, so it is always true. |
| 506 | The markdown report prints `VRAM Safety Guarantee: Verified < 6,144 MB (RTX 4050 6GB Enforced)` as a literal; no check precedes it. |
| 47-55 | The only real hardware reading: `torch.cuda.get_device_name` and total VRAM. |
| 41-43 (classification) | Positive: the harness does carry `classification` from the adapter (SIMULATED vs INTERNAL), so downstream consumers can filter. The verdicts above ignore it. |

## 6. Findings, ranked

1. **Default depth is broken and silently skipped** (F). `timm` missing for MiDaS; `transformers` missing for Depth Anything V2. This blocks metric depth, object lifting and everything downstream, yet `perception/availability.py` reports MiDaS "available". The `perception` extra in `pyproject.toml` also omits `timm` and `transformers`. Fix: install them with the user's approval, add them to the extra, and make the probe import-test the model graph.
2. **COLMAP 4.2.0 already ships ALIKED, SIFT-LightGlue, ALIKED-LightGlue and `global_mapper`**, none used. The pipeline uses CPU SIFT because affine-shape SIFT is CPU-only.
3. **XFeat and DeDoDe matchers misreport.** They run AKAZE/BRISK under the names `xfeat@accelerated_features` and `dedode@v2`, and `available_matchers()` says "available". This contradicts the module's own stated rule that a benchmark must name the model that produced the matches.
4. **OneFormer fallback returns fabricated regions** and swallows errors.
5. **`engine/technologies` overstates**: 0 of 50 upstream libraries are called, 13 are numpy re-derivations, 37 are fixed-constant simulations; benchmark numbers in section 5 are constants. The `/api/technologies/benchmark` route and `reality techs benchmark` therefore return fabricated-looking measurements with `QUALIFIED`/`RECOMMENDED` verdicts.
6. `HeuristicDepthPriorBackend` is honest in its labels but must not feed lifting or scale.
7. Cached weights without runtimes: `sam_vit_b` (needs `segment_anything`), Depth Anything V2 Small and OWL-ViT (need `transformers`).

## 7. Not verified

- Any end-to-end run of the vertical slice, dense MVS or Poisson meshing in this audit (the pipeline is being run by another agent).
- Mask R-CNN inference, IFC export output, video decode, frontend build.
- The prior-doc figures (246,816 fused points; 22/32 cameras).
- Whether the API process sees the same PATH as this shell (COLMAP is on the bash PATH here).
- COLMAP ALIKED/LightGlue ONNX downloads and quality (not downloaded, per audit rules).
