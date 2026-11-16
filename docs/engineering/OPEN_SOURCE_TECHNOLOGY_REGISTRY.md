# Open-source technology registry

Facts (licence, last push, release, HEAD commit) were read on **2026-11-15** from the GitHub and Hugging Face APIs and from the LICENSE files of repositories whose API licence was `NOASSERTION`. **Classification and maturity are my assessments.** No accuracy has been measured for any candidate (no annotated data exists). Source of truth: `engine/technologies/open_source_registry.json`; regenerate this file with `python scripts/build_open_source_registry.py`.

Maturity ladder: Implemented < Installed < Integrated < API reachable < Functionally tested < Real-data validated < Production ready.

## Counts

| Classification | Count |
|---|---|
| ALREADY_INTEGRATED | 7 |
| APPROVED | 17 |
| APPROVED_WITH_RESTRICTIONS | 10 |
| LICENCE_REVIEW_REQUIRED | 12 |
| RESEARCH_ONLY | 8 |
| REJECTED | 7 |

## Ten highest-value opportunities, in integration order

1. **SAM 2 instance masks (Grounding DINO boxes as prompts)** (`sam2`)  
   Replaces the depth-band mask that leaked floor/wall into objects (the measured cause of a 7 m sofa). transformers already ships Sam2Model; Apache-2.0 code and weights; no new dependency.  
   Measured by: instance mIoU and boundary F1 once annotated; until then cross-view extent plausibility on the 10-photo room.

2. **Human-verified ground truth: CVAT or Label Studio + FiftyOne feeding the Reality Benchmark** (`cvat`)  
   Every accuracy metric is blocked by the absence of annotations. Review tooling, not a new benchmark.  
   Measured by: enables precision/recall/mAP; no number exists before it.

3. **DINOv2 appearance embeddings in the association resolver** (`dinov2`)  
   Association is geometry-only, so views of one object split (two rugs) or merge. Code exists and is unreachable; weights are cached.  
   Measured by: duplicate-entity rate, identity precision/recall.

4. **Small-object tiled pass wired into perception** (`grounding_dino`)  
   perception/entities/small_objects.py is unreachable from the API; sockets, switches and handles are not found at full-frame scale.  
   Measured by: small-object recall by group.

5. **Open3D outlier removal, robust registration and TSDF/Poisson** (`open3d`)  
   MIT, wheel available. Cleans the sparse and dense clouds before plane fitting; TSDF/Poisson are candidates for the mesh stage.  
   Measured by: Chamfer / surface distance vs surveyed room; plane-fit residual.

6. **hloc + LightGlue matching for SfM reproducibility** (`hloc`)  
   Run-to-run geometry varies (995 vs 1,423 points). Extractor weights licence (SuperPoint) must be resolved first.  
   Measured by: registration rate, reprojection error, run-to-run spread.

7. **Cross-view depth consistency (metric depth, one global scale)** (`depth_anything_v2_metric_indoor`)  
   Best measured fit to SfM (0.117 vs 0.199) but the model card declares no licence; blocked on a licence read.  
   Measured by: scale consistency, dimensional error.

8. **Florence-2 as a second-opinion detector (native in transformers)** (`florence2`)  
   MIT weights; no remote code now. Only replaces Grounding DINO if it wins on annotated data.  
   Measured by: detection precision/recall vs Grounding DINO.

9. **manifold3d for guaranteed-valid mesh booleans / watertight checks** (`manifold`)  
   Apache-2.0, wheel available. Protects export validity.  
   Measured by: watertightness, topological validity of exported meshes.

10. **IfcOpenShell: validate IFC against real captures** (`ifcopenshell`)  
   Already integrated (LGPL-3.0); the gap is real-data validation of the semantics, not a new library.  
   Measured by: IFC element completeness vs measured room.

## Records

| id | classification | repo licence | weights licence | last push | maturity | integration |
|---|---|---|---|---|---|---|
| colmap | ALREADY_INTEGRATED | BSD-3-Clause (LICENSE read; its dependencies are licensed separately) | n/a | 2026-11-12 | Functionally tested | stage: reconstruction (apps/api/jobs.py) |
| grounding_dino | ALREADY_INTEGRATED | Apache-2.0 | IDEA-Research/grounding-dino-tiny: apache-2.0 | 2024-08-12 | Functionally tested | stage: perception (perception/detection/grounded_scene.py) |
| dpt_hybrid | ALREADY_INTEGRATED | MIT (repo archived) | Intel/dpt-hybrid-midas: apache-2.0 | 2024-08-23 (archived) | Functionally tested | stage: depth (vertical_slice) |
| trimesh | ALREADY_INTEGRATED | MIT | n/a | 2026-11-13 | Functionally tested | optional, guarded import (meshing extra) |
| opencv | ALREADY_INTEGRATED | Apache-2.0 | n/a | 2026-11-15 | Functionally tested | core dependency |
| ifcopenshell | ALREADY_INTEGRATED | LGPL-3.0 | n/a | 2026-11-15 | Functionally tested | exporters/ifc_bridge.py via the SDK/API (export extra) |
| blender | ALREADY_INTEGRATED | GPL-3.0 (COPYING read) | n/a | 2026-11-15 | Functionally tested | exporters/blender generates a Python script; Blender is never linked |
| pycolmap | APPROVED | BSD-3-Clause | n/a | 2024-07-09 (archived) | Implemented | not integrated; the standalone repo is archived, development moved into colmap/colmap |
| sam2 | APPROVED | Apache-2.0 | facebook/sam2.1-hiera-small, -large: apache-2.0 | 2026-05-30 | API reachable | perception/segmentation/sam2_box_prompter.py -> grounded_scene -> vertical_slice perception stage (default mask source) |
| grounded_sam2 | APPROVED | Apache-2.0 | inherits Grounding DINO + SAM 2 | 2025-11-11 | Implemented | pattern only: reproduce with our own adapters, not by vendoring |
| dinov2 | APPROVED | Apache-2.0 | facebook/dinov2-small, -base: apache-2.0 | 2026-11-05 | Integrated | perception/instances/dinov2_hf.py -> vertical_slice perception_appearance option (off by default) |
| florence2 | APPROVED | no GitHub repo (microsoft/Florence-2 returns 404); code ships inside transformers | microsoft/Florence-2-base: mit | - | Implemented | not integrated |
| depth_anything_v2_small | APPROVED | Apache-2.0 | depth-anything/Depth-Anything-V2-Small-hf: apache-2.0 | 2026-03-24 | Functionally tested | not the live depth model |
| open3d | APPROVED | MIT (LICENSE read) | n/a | 2026-11-14 | Implemented | not integrated; one reference in code |
| pcl | APPROVED | BSD (LICENSE.txt read) | n/a | 2026-11-14 | Implemented | an internal module named pcl_core exists; the C++ library is not used |
| manifold | APPROVED | Apache-2.0 | n/a | 2026-11-15 | Implemented | not integrated; an internal 'manifold_kernel' name exists in the catalog map, unverified |
| bytetrack | APPROVED | MIT | n/a | 2024-06-19 | Implemented | perception/tracking exists but is not reachable from the API |
| gsplat | APPROVED | Apache-2.0 | n/a | 2026-09-19 | Implemented | splat ingest/render exists separately (PR #299) |
| nerfstudio | APPROVED | Apache-2.0 | n/a | 2025-07-29 | Implemented | not integrated |
| compas | APPROVED | MIT | n/a | 2026-11-11 | Implemented | not integrated |
| cvat | APPROVED | MIT | n/a | 2026-11-14 | Implemented | not integrated |
| label_studio | APPROVED | Apache-2.0 | n/a | 2026-11-14 | Implemented | not integrated |
| fiftyone | APPROVED | Apache-2.0 | n/a | 2026-11-14 | Implemented | not integrated |
| limap | APPROVED | BSD-3-Clause | n/a | 2026-09-30 | Implemented | not integrated |
| hloc | APPROVED_WITH_RESTRICTIONS | Apache-2.0 | depends on the chosen extractor: SuperPoint non-commercial; ALIKED/DISK weights not verified | 2025-12-10 | Implemented | not integrated |
| alicevision | APPROVED_WITH_RESTRICTIONS | MPL-2.0 (COPYING.md read; includes third-party components) | n/a | 2026-11-13 | Implemented | not integrated |
| meshroom | APPROVED_WITH_RESTRICTIONS | MPL-2.0 (COPYING.md read) | n/a | 2026-11-13 | Implemented | not integrated |
| grounded_sam | APPROVED_WITH_RESTRICTIONS | Apache-2.0 | SAM 1 weights not verified | 2024-09-05 | Implemented | superseded by Grounded-SAM-2 |
| detectron2 | APPROVED_WITH_RESTRICTIONS | Apache-2.0 | model-zoo weights vary; not audited | 2026-10-27 | Implemented | not integrated |
| sam_hq | APPROVED_WITH_RESTRICTIONS | Apache-2.0 | not verified | 2025-09-12 | Implemented | not integrated |
| marigold | APPROVED_WITH_RESTRICTIONS | Apache-2.0 | marigold-depth-v1-1: openrail++ (use restrictions) | 2026-09-06 | Implemented | not integrated |
| mmtracking | APPROVED_WITH_RESTRICTIONS | Apache-2.0 | varies | 2023-09-19 | Implemented | not integrated |
| openusd | APPROVED_WITH_RESTRICTIONS | Tomorrow Open Source Technology License 1.0 (modified Apache 2.0; LICENSE.txt read) | n/a | 2026-11-14 | Functionally tested | usda text export exists without the library |
| web_ifc | APPROVED_WITH_RESTRICTIONS | MPL-2.0 | n/a | 2026-11-13 | Implemented | not integrated |
| lightglue | LICENCE_REVIEW_REQUIRED | Apache-2.0 | HF lightglue_superpoint: 'other' (SuperPoint terms, unresolved) | 2026-10-25 | Implemented | adapter exists in repo, not reachable from the API |
| openmvg | LICENCE_REVIEW_REQUIRED | MPL-2.0 (API); sub-components not audited | n/a | 2026-11-07 | Implemented | not integrated |
| mask2former | LICENCE_REVIEW_REQUIRED | MIT (repo archived) | HF COCO weights: 'other' (unresolved) | 2024-07-29 (archived) | Implemented | not integrated |
| depth_anything_v2_metric_indoor | LICENCE_REVIEW_REQUIRED | Apache-2.0 (code) | model card declares NO licence (verified: null) | 2026-03-24 | Functionally tested | not integrated |
| metric3d | LICENCE_REVIEW_REQUIRED | BSD-2-Clause | weights host returned HTTP 401 (not verified) | 2025-03-13 | Implemented | not integrated |
| pointcept | LICENCE_REVIEW_REQUIRED | MIT (code) | pretrained weights are trained on ScanNet etc. whose terms are non-commercial | 2026-11-03 | Implemented | not integrated |
| cgal | LICENCE_REVIEW_REQUIRED | dual: GPL-3.0/LGPL-3.0 by package, or a commercial licence (LICENSE.md read) | n/a | 2026-11-14 | Implemented | not integrated |
| scannet | LICENCE_REVIEW_REQUIRED | code MIT-style; DATA terms are non-commercial (not verified here) | n/a | 2025-11-03 | Implemented | not integrated |
| arkitscenes | LICENCE_REVIEW_REQUIRED | Apple sample-code licence (LICENSE read); data terms not verified | n/a | 2026-09-11 | Implemented | not integrated |
| scannetpp | LICENCE_REVIEW_REQUIRED | no licence declared in the repo | n/a | 2026-07-10 | Implemented | not integrated |
| hypersim | LICENCE_REVIEW_REQUIRED | Apple sample-code licence (LICENSE read); data terms not verified | n/a | 2026-09-11 | Implemented | not integrated |
| structured3d | LICENCE_REVIEW_REQUIRED | MIT (code); data terms not verified | n/a | 2025-02-24 | Implemented | not integrated |
| depth_anything_v2_base_large | RESEARCH_ONLY | Apache-2.0 (code) | Base: cc-by-nc-4.0 (verified on the model card) | 2026-03-24 | Implemented | not integrated |
| depth_pro | RESEARCH_ONLY | Apple sample-code licence (LICENSE read) | apple/DepthPro: apple-amlr (research) | 2026-09-11 | Implemented | not integrated |
| cotracker | RESEARCH_ONLY | CC BY-NC 4.0 (LICENSE.md read) | facebook/cotracker3: cc-by-nc-4.0 | 2026-03-03 | Implemented | not integrated |
| graphdeco_3dgs | RESEARCH_ONLY | Inria/MPII custom, research use (LICENSE.md read) | n/a | 2025-10-17 | Implemented | not integrated |
| qgis | RESEARCH_ONLY | GPL-2.0 | n/a | 2026-11-15 | Implemented | not integrated |
| vggt | RESEARCH_ONLY | VGGT License v1 (custom; LICENSE.txt read: research materials with an acceptable-use policy) | facebook/VGGT-1B: cc-by-nc-4.0 | 2026-05-19 | Implemented | not integrated |
| dust3r | RESEARCH_ONLY | CC BY-NC-SA 4.0 (LICENSE read) | same | 2025-09-24 | Implemented | not integrated |
| mast3r | RESEARCH_ONLY | CC BY-NC-SA 4.0 (LICENSE read) | same | 2025-06-30 | Implemented | not integrated |
| openmvs | REJECTED | AGPL-3.0 | n/a | 2026-11-14 | Implemented | listed as a backend descriptor only (reconstruction/backend/registry.py); never executed |
| ultralytics | REJECTED | AGPL-3.0 | AGPL-3.0 | 2026-11-15 | Implemented | not integrated |
| libigl | REJECTED | GPL-3.0 | n/a | 2026-10-29 | Implemented | catalog mention only |
| opensplat | REJECTED | AGPL-3.0 | n/a | 2026-10-29 | Implemented | not integrated |
| xeokit | REJECTED | AGPL-3.0 | n/a | 2026-11-02 | Implemented | not integrated |
| odm | REJECTED | AGPL-3.0 | n/a | 2026-11-13 | Implemented | not integrated |
| pyslam | REJECTED | GPL-3.0 | n/a | 2026-10-25 | Implemented | not integrated |

## Notes per record

* **colmap** (COLMAP): Geometry differs run to run on the same photos (995 vs 1,423 points observed).
* **pycolmap** (pycolmap): Would remove the long-workspace-path failure class only if SfM moved in-process; no measured need yet.
* **hloc** (hloc): Candidate for SfM reproducibility; extractor weights licence must be checked before use.
* **lightglue** (LightGlue): Code is Apache-2.0; the pretrained weights' licence is what blocks.
* **openmvg** (openMVG): Duplicates COLMAP; no measured gap it would close.
* **alicevision** (AliceVision): Full alternative pipeline: would be a second orchestrator, so not adopted.
* **meshroom** (Meshroom): GUI over AliceVision; no role in a headless pipeline.
* **openmvs** (OpenMVS): AGPL-3.0 is incompatible with a closed commercial product; usable only as a separate research tool.
* **grounding_dino** (Grounding DINO (tiny, via transformers)): Prompt wording changes recall sharply ('coffee table' 0/6 views vs 'wooden coffee table' 6/6); fp16 .half() crashed inside the API.
* **sam2** (SAM 2 (via transformers Sam2Model)): Top opportunity: today's masks are a depth-band cut of a box and leak floor/wall (measured cause of a 7 m 'sofa').
* **grounded_sam2** (Grounded-SAM-2): A glue repo; our adapters should follow its box-prompt pattern.
* **grounded_sam** (Grounded-Segment-Anything): Last push 2024-09; use the SAM 2 route instead.
* **dinov2** (DINOv2 (via transformers Dinov2Model)): Wired but unproven: no measured gain, so it stays opt-in. dinov2_reid.py silently substitutes a colour-statistic vector when torch.hub fails; do not use it for association.
* **florence2** (Florence-2 (via transformers Florence2ForConditionalGeneration)): Possible second-opinion detector; must beat Grounding DINO on annotated data before replacing anything.
* **detectron2** (Detectron2): Overlaps torchvision Mask R-CNN already integrated; Windows build is painful.
* **mask2former** (Mask2Former): Archived upstream; weights licence unresolved.
* **ultralytics** (Ultralytics YOLO): AGPL blocks a closed commercial product unless a commercial licence is bought.
* **sam_hq** (SAM-HQ): Only worth testing if SAM 2 boundaries fail on thin structures.
* **depth_anything_v2_small** (Depth Anything V2 Small): Worse fit to SfM than DPT_Hybrid in the one scene tested.
* **depth_anything_v2_base_large** (Depth Anything V2 Base / Large): Non-commercial weights.
* **depth_anything_v2_metric_indoor** (Depth Anything V2 Metric-Indoor Small): Best measured fit to SfM, but undeclared licence and no effect on floor layering in an A/B; no default change.
* **dpt_hybrid** (DPT_Hybrid / MiDaS): Per-view scale drifts about 2x between views; objects land up to ~2.5 m apart across views.
* **metric3d** (Metric3D): Needs the weights licence read before any test.
* **marigold** (Marigold): Slow diffusion model; no evidence it would beat the current depth on our metric.
* **depth_pro** (Apple Depth Pro): Research-only weights.
* **open3d** (Open3D): Fifth opportunity: outlier removal and robust registration before plane fitting.
* **pcl** (PCL): C++ build burden for no gain over Open3D.
* **pointcept** (Pointcept): Code is MIT, but the dataset terms behind the weights are the blocker.
* **trimesh** (trimesh): Present in the environment.
* **cgal** (CGAL): Per-package licensing; core packages are not permissive.
* **libigl** (libigl): GPL-3.0.
* **manifold** (Manifold (manifold3d)): Candidate for guaranteed-valid mesh booleans.
* **cotracker** (CoTracker3): Non-commercial.
* **bytetrack** (ByteTrack): Repo moved to FoundationVision/ByteTrack; last push 2024-06; the algorithm is small enough to reimplement.
* **mmtracking** (MMTracking): Unmaintained since 2023-09; heavy mm* dependency stack.
* **opencv** (OpenCV): In use.
* **gsplat** (gsplat): Appearance only: not geometric accuracy.
* **nerfstudio** (nerfstudio): No evidence it is needed.
* **opensplat** (OpenSplat): AGPL-3.0.
* **graphdeco_3dgs** (Original 3D Gaussian Splatting): Non-commercial terms.
* **openusd** (OpenUSD): The licence differs from Apache-2.0; read it before linking the library.
* **ifcopenshell** (IfcOpenShell): LGPL-3.0: dynamic use is fine, modifying/bundling carries obligations.
* **web_ifc** (web-ifc): File-level copyleft; frontend only.
* **xeokit** (xeokit): AGPL-3.0.
* **qgis** (QGIS): GPL: use as an external tool only, never as a linked dependency.
* **compas** (COMPAS): No identified gap.
* **blender** (Blender): Safe only as a separate process / generated script.
* **cvat** (CVAT): Second opportunity: blocks every accuracy metric.
* **label_studio** (Label Studio): Alternative to CVAT; pick one.
* **fiftyone** (FiftyOne): Complements the existing benchmark; does not replace it.
* **scannet** (ScanNet): Datasets need a separate licence read.
* **arkitscenes** (ARKitScenes): Repo moved to apple-aiml-research.
* **scannetpp** (ScanNet++): No declared licence.
* **hypersim** (Hypersim): Synthetic; cannot validate real-world accuracy.
* **structured3d** (Structured3D): Synthetic.
* **vggt** (VGGT): Non-commercial weights.
* **dust3r** (DUSt3R): Non-commercial.
* **mast3r** (MASt3R): Non-commercial.
* **odm** (OpenDroneMap / WebODM): AGPL-3.0.
* **pyslam** (pySLAM): GPL-3.0.
* **limap** (LIMAP): Line features could help wall edges; no measured need yet.
