# Technology Selection Matrix

Companion to `docs/audits/TECHNOLOGY_INTEGRATION_AUDIT.md` (same date, 2026-11-05). Hardware target: RTX 4050 Laptop, 6140 MiB (observed), CPU fallback.

How to read this:
- **Decision**: adopt (use as is), extend (keep and widen), replace, defer, reject. A decision is only given when the audit has evidence for it; otherwise the cell says **needs benchmark**.
- **License** comes from repo docstrings/registries where one exists (cited), otherwise from general knowledge of the upstream project. **No upstream LICENSE file was re-read in this pass.** `docs/TECHNOLOGY_REGISTRY.md` already warns that model-weight licences can differ from code licences; verify per checkpoint before commercial use.
- **6 GB cost**: "measured" means observed in this audit. "est." is an order-of-magnitude estimate from model size and general knowledge, not measured here.

| Capability | Candidates | License | 6 GB RTX 4050 cost | Existing RE equivalent | Decision | Justification (evidence) |
|---|---|---|---|---|---|---|
| Sparse SfM | COLMAP 4.2.0 | BSD-3-Clause (`dense_registry.py`) | CUDA build present; SIFT extraction forced to CPU by affine shape + DSP (`colmap_backend.py:222-231`) | `reconstruction/backend/colmap_backend.py` | **adopt** | Only SfM engine on the app path; binary runs (`colmap help`); incremental `image_registrator` already wired. |
| Sparse SfM, global | COLMAP `global_mapper` (GLOMAP merged), standalone GLOMAP | BSD-3 (est.) | CPU; est. low | none | **needs benchmark** | `global_mapper` exists in the installed binary's command list; RE does not call it. No data on its quality on this repo's captures. |
| Sparse SfM, Python API | pycolmap | BSD-3 (est.) | n/a | none | **defer** | Not installed; CLI already covers everything used. |
| Dense MVS | COLMAP `patch_match_stereo` + `stereo_fusion` | BSD-3-Clause | CUDA; VRAM not measured. Gated by `dense_gate.py` | `reconstruction/dense_pipeline.py` | **adopt** | Wired and gated; probe via `colmap help`. |
| Dense MVS | OpenMVS | AGPL-3.0 (`reconstruction/backend/registry.py:58`) | est. GPU optional | descriptor only | **reject** for now | AGPL, no binary, redundant with COLMAP MVS. |
| Dense MVS (CPU) | own OpenCV plane sweep | Apache-2.0 (OpenCV) | CPU | `reconstruction/backend/plane_sweep.py` | **keep, default off** | Real algorithm but no quality data; off by default in `VerticalSliceOptions`. |
| Surface mesh | COLMAP `poisson_mesher`; Open3D Poisson/TSDF; PyMeshLab | BSD-3; MIT (est.); GPL-3 (est.) | CPU | `reconstruction/meshing/surface.py` | COLMAP Poisson **adopt**; Open3D **defer**; PyMeshLab **reject** (GPL, est.) | Poisson path is wired. Open3D not installed; no comparison data. |
| Mesh utilities | trimesh 5.2.0 | MIT (est.) | CPU | `reconstruction/meshing/repair.py` (guarded import, not called by pipeline) | **adopt** (wire in) | Installed; `repair.py` already targets it. |
| Local features | COLMAP SIFT; OpenCV SIFT | BSD-3; Apache-2 | CPU in current config | colmap_backend.py; `matching/sift.py` | **adopt** | Production path. |
| Local features | ALIKED (kornia) + LightGlue | Apache-2.0 (kornia, est.); weights per upstream | CPU measured: 49.8 s for one 640x480 pair including model load. GPU est. low | `matching/lightglue.py` `ALIKEDLightGlueMatcher` | **extend**, quality **needs benchmark** | Runs for real (1041 matches, 1040 inliers on a synthetic pair). Weights cached. Not wired into SfM; no SfM comparison exists. |
| Local features | DISK (kornia) + LightGlue | Apache-2.0 (est.) | CPU measured: 12.7 s for one pair including load | `matching/lightglue.py` `LightGlueMatcher` | **adopt** for pair verification | Already used by `jobs.py:_scene_gate`; 2068 matches / 2032 inliers on a synthetic pair. |
| Local features | COLMAP-native ALIKED_N16ROT / ALIKED_N32 / SIFT_LIGHTGLUE / ALIKED_LIGHTGLUE | BSD-3 (COLMAP); ONNX models from COLMAP releases | GPU via COLMAP; est. | none | **needs benchmark** | Enum strings verified in `colmap.exe`; flags verified in `-h`. ONNX files not cached (not downloaded). This is the cheapest way to get learned matching into SfM. |
| Local features | XFeat | Apache-2.0 (per repo docstring) | est. very low | `matching/xfeat.py` (AKAZE fallback, mislabelled) | **reject** current code; real XFeat **defer** | Package/weights absent; current class reports a different algorithm under XFeat's name. |
| Local features | DeDoDe | MIT (per repo docstring) | est. moderate | `matching/dedode.py` (BRISK fallback, mislabelled) | **replace** with kornia `kf.DeDoDe`, **needs benchmark** | kornia 0.9.0 ships `DeDoDe.from_pretrained(detector_weights='L-C4-v2', descriptor_weights='G-upright')`. Weights not cached. |
| Dense matcher | LoFTR (kornia); MASt3R | Apache-2 (est.); CC-BY-NC (`matching/__init__.py`) | est. high | declared not implemented | LoFTR **defer**; MASt3R **reject** | NC licence; no weights. |
| Monocular depth | Depth Anything V2 Small | Apache-2.0 for Small per repo docstring; other tiers reported non-commercial, verify | ~25M params per repo docstring; est. well under 1 GB | `perception/depth/depth_anything_backend.py` | **adopt** (needs `transformers`), quality **needs benchmark** | Weights cached; backend written; blocked only by the missing package. |
| Monocular depth | MiDaS DPT_Hybrid | MIT (est.) | weights 469.9M; est. 1-2 GB | `perception/depth/midAS_backend.py` | **replace** | Currently the default and broken (`timm` missing). If kept, MiDaS_small (81.8M, cached) is the cheaper fallback; still needs `timm`. |
| Metric depth | Metric3D v2, ZoeDepth | per upstream; unverified | est. high | none | **defer** | No weights, no code. Metric scale already comes from COLMAP + operator measurements (`jobs.py:612`). |
| Metric depth | `heuristic_depth_prior` | Apache-2.0 (own) | CPU | `metric_depth_backend.py` | **reject** for pipeline use | Fixed radial pattern; not a measurement (audit section 3). |
| Detection | Mask R-CNN R50-FPN (torchvision) | BSD-3 (est.) | weights 169.8M; est. 2-3 GB at 800 px; not measured | `perception/detection/maskrcnn_backend.py` | **adopt** | Weights cached, probe reports available, wired in `vertical_slice.py:952`. Blocked only by upstream depth failure. |
| Detection | OWL-ViT B/32 | Apache-2.0 (est.) | est. ~1 GB | `backends.py` `OwlVitDetector` | **adopt** once `transformers` present | Weights cached; code real. |
| Detection | OWLv2, Grounding DINO | per upstream | est. high | none | **defer** | No weights; no data. |
| Detection | YOLOX, MaskDINO | Apache-2.0 | est. moderate | stubs that raise | **reject** current registry flags | Registry marks yolox `INTERNAL_IMPLEMENTATION`; no inference exists. |
| Detection | Ultralytics YOLO | AGPL-3.0 (est.) | est. low | none | **reject** | AGPL, no demonstrated need. |
| Segmentation | SAM ViT-B | Apache-2.0 (est.) | weights 357.7M; est. 2-3 GB | `perception/segmentation/sam_backend.py` | **adopt** once `segment-anything` installed | In the `perception` extra already; checkpoint cached; only the package is missing. |
| Segmentation | SAM2 | Apache-2.0 (est.) | est. moderate | none | **defer** | |
| Segmentation | OneFormer | MIT (per repo docstring) | est. high (swin-large) | `oneformer_backend.py` | **replace fallback with "unavailable"** | The fallback returns fabricated wall/floor masks. |
| Tracking / re-id | ByteTrack, DeepSORT etc. | n/a | n/a | `perception/tracking/temporal.py` (own) | **defer** | No video-tracking requirement demonstrated on the app path. |
| OCR | Tesseract, EasyOCR, PaddleOCR | n/a | n/a | none | **defer** | No code or use case found. |
| Point cloud processing | Open3D | MIT (est.) | CPU/GPU optional | numpy code in `reconstruction/mvs/point_cloud_ops.py` | **needs benchmark** | Not installed. |
| Point cloud | PCL | BSD (est.) | CPU | `pcl_core.py` (numpy rewrite) | **reject** as a claim | No Python binding installed; the rewrite should not be labelled PCL. |
| Bundle adjustment / pose-graph | COLMAP `bundle_adjuster`, scipy, GTSAM, Ceres | BSD-3; BSD-3; BSD (est.); BSD-3 (est.) | CPU | scipy in 17 modules; `factor_graph.py` (own) | COLMAP BA **adopt**; scipy **adopt**; GTSAM/Ceres **defer** | GTSAM/Ceres not installed and no pipeline stage needs them yet. |
| SLAM / VIO | ORB-SLAM3, OpenVINS, Basalt | GPL-3.0 (ORB-SLAM3, per `orb_slam3_backend.py`); GPL-3.0 OpenVINS (est.); BSD-3 Basalt (est.) | CPU | subprocess adapters under `trajectories/backend/` | **defer** | No binary built; no IMU/VIO capture path demonstrated. GPL forces subprocess isolation (already done). |
| Video decode | OpenCV (bundled FFmpeg); PyAV; FFmpeg CLI | Apache-2; BSD; LGPL/GPL by build | CPU | `evidence/video.py` | OpenCV **adopt**; PyAV/FFmpeg **defer** | OpenCV reports `FFMPEG: YES`; no code path needs seek-accurate or hardware decode. |
| EXIF / metadata | Pillow; ExifTool | HPND; Perl (Artistic/GPL, est.) | CPU | `evidence/geo_anchor.py`, `image_check.py` | Pillow **adopt**; ExifTool **defer** | ExifTool not installed; no failing case evidenced. |
| Near-duplicate | imagehash; own hashes | BSD (est.) | CPU | `evidence/dedup.py` | **keep own, wire in** | Own dHash/pHash exist; not on the app path. |
| BIM / IFC | ifcopenshell | LGPL-3.0 (est.) | CPU | `exporters/ifc_bridge.py` | **adopt** | Installed (0.9.0.post1) and exposed by the export route. |
| GIS | pyproj; GDAL | MIT; MIT/X (est.) | CPU | `world_ir/geo.py`; numpy terrain | pyproj **adopt**; GDAL **defer** | GDAL not installed; no raster I/O requirement shown. |
| Georegistration | COLMAP `model_aligner` | BSD-3 | CPU | `reconstruction/scale/georeferencing.py` (own) | **needs benchmark** | Present in binary; own implementation exists. |
| Neural recon | gsplat, nerfstudio | Apache-2.0 (est.) | likely tight on 6 GB; Windows build not verified | none | **defer** | Not installed; no consumer in the pipeline. |
| Rendering / viewers | three.js 0.186, maplibre-gl 6.10 | MIT; BSD-3 (est.) | browser GPU | `frontend/` | **adopt** | Declared in `frontend/package.json`. |
| Storage / jobs | FastAPI, SQLAlchemy+aiosqlite | MIT; MIT | CPU | `apps/api` | **adopt** | Installed and used. |
| Inference runtime | onnxruntime, OpenVINO, TVM | MIT; Apache-2 | n/a | `onnx_quantizer.py` (estimator) | **defer** | Not installed; no ONNX model is run by RE code. |

## Constraints that bind every GPU row
- Total VRAM 6140 MiB. COLMAP dense, depth and detection should not overlap on the same GPU. Whether the pipeline serialises them was not checked.
- Anything marked "needs benchmark" must be run with real `torch.cuda.max_memory_allocated()` readings, not with `engine/technologies/benchmark.py` (see audit section 5).
