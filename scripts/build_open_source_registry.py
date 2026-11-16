"""Build the open-source technology registry (machine-readable JSON + the markdown view) from curated assessments.

Facts (licence, last push, latest release, HEAD sha) are READ from engine/technologies/open_source_probe.json, which
scripts/probe_open_source_repos.py produced from the GitHub and Hugging Face APIs. Judgements (classification, integration
status, maturity) are curated below and marked as assessments. Re-run after re-probing:

    python scripts/build_open_source_registry.py

Classifications: APPROVED, APPROVED_WITH_RESTRICTIONS, RESEARCH_ONLY, LICENCE_REVIEW_REQUIRED, REJECTED, ALREADY_INTEGRATED.
A classification is never changed silently: tests/test_open_source_registry.py pins the licence rules, and a change to an entry
shows up as a diff of the JSON.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROBE = json.loads((ROOT / "engine" / "technologies" / "open_source_probe.json").read_text(encoding="utf-8"))
PROBED_ON = "2026-11-15"

# Maturity ladder (the directive's seven levels, in order). An entry sits at the HIGHEST level with evidence.
MATURITY = ("Implemented", "Installed", "Integrated", "API reachable", "Functionally tested", "Real-data validated", "Production ready")

# id | probe key | name | repo licence (as read; override when the API says NOASSERTION) | weights licence | classification |
# installation | hardware | capabilities | integration status | maturity | benchmark | fallback | note
E = [
    # ---- A. photogrammetry
    ("colmap", "colmap/colmap", "COLMAP", "BSD-3-Clause (LICENSE read; its dependencies are licensed separately)", "n/a", "ALREADY_INTEGRATED",
     "binary on PATH (4.x)", "CPU; dense needs CUDA build", ["sfm", "mvs", "camera_pose"], "stage: reconstruction (apps/api/jobs.py)", "Functionally tested",
     "none measured against ground truth", "level-0 single-image fallback", "Geometry differs run to run on the same photos (995 vs 1,423 points observed)."),
    ("pycolmap", "colmap/pycolmap", "pycolmap", "BSD-3-Clause", "n/a", "APPROVED", "not installed (wheel 4.2.1 exists)", "CPU", ["sfm"],
     "not integrated; the standalone repo is archived, development moved into colmap/colmap", "Implemented", "not benchmarked", "COLMAP binary",
     "Would remove the long-workspace-path failure class only if SfM moved in-process; no measured need yet."),
    ("hloc", "cvg/Hierarchical-Localization", "hloc", "Apache-2.0", "depends on the chosen extractor: SuperPoint non-commercial; ALIKED/DISK weights not verified", "APPROVED_WITH_RESTRICTIONS",
     "not installed", "GPU recommended", ["feature_matching", "localization"], "not integrated", "Implemented", "not benchmarked", "COLMAP's own SIFT matching",
     "Candidate for SfM reproducibility; extractor weights licence must be checked before use."),
    ("lightglue", "cvg/LightGlue", "LightGlue", "Apache-2.0", "HF lightglue_superpoint: 'other' (SuperPoint terms, unresolved)", "LICENCE_REVIEW_REQUIRED",
     "weights cached; python package not installed", "GPU/CPU", ["feature_matching"], "adapter exists in repo, not reachable from the API", "Implemented",
     "not benchmarked", "COLMAP matching", "Code is Apache-2.0; the pretrained weights' licence is what blocks."),
    ("openmvg", "openMVG/openMVG", "openMVG", "MPL-2.0 (API); sub-components not audited", "n/a", "LICENCE_REVIEW_REQUIRED", "not installed", "CPU", ["sfm"], "not integrated",
     "Implemented", "not benchmarked", "COLMAP", "Duplicates COLMAP; no measured gap it would close."),
    ("alicevision", "alicevision/AliceVision", "AliceVision", "MPL-2.0 (COPYING.md read; includes third-party components)", "n/a", "APPROVED_WITH_RESTRICTIONS", "not installed",
     "CUDA for dense", ["sfm", "mvs", "meshing"], "not integrated", "Implemented", "not benchmarked", "COLMAP", "Full alternative pipeline: would be a second orchestrator, so not adopted."),
    ("meshroom", "alicevision/Meshroom", "Meshroom", "MPL-2.0 (COPYING.md read)", "n/a", "APPROVED_WITH_RESTRICTIONS", "not installed", "CUDA", ["gui_pipeline"], "not integrated",
     "Implemented", "n/a", "n/a", "GUI over AliceVision; no role in a headless pipeline."),
    ("openmvs", "cdcseacave/openMVS", "OpenMVS", "AGPL-3.0", "n/a", "REJECTED", "not installed", "CUDA optional", ["mvs", "meshing", "texturing"],
     "listed as a backend descriptor only (reconstruction/backend/registry.py); never executed", "Implemented", "n/a", "COLMAP patch-match",
     "AGPL-3.0 is incompatible with a closed commercial product; usable only as a separate research tool."),
    # ---- B. detection / segmentation / identity
    ("grounding_dino", "IDEA-Research/GroundingDINO", "Grounding DINO (tiny, via transformers)", "Apache-2.0", "IDEA-Research/grounding-dino-tiny: apache-2.0", "ALREADY_INTEGRATED",
     "weights cached; transformers AutoModelForZeroShotObjectDetection", "fp32 batch 2: ~0.4 s/prompt, 3.4 GB peak (measured RTX 4050)", ["open_vocab_detection"],
     "stage: perception (perception/detection/grounded_scene.py)", "Functionally tested", "no precision/recall: no annotations exist",
     "Mask R-CNN (COCO classes)", "Prompt wording changes recall sharply ('coffee table' 0/6 views vs 'wooden coffee table' 6/6); fp16 .half() crashed inside the API."),
    ("sam2", "facebookresearch/sam2", "SAM 2 (via transformers Sam2Model)", "Apache-2.0", "facebook/sam2.1-hiera-small, -large: apache-2.0", "APPROVED",
     "NOT downloaded; transformers 5.16.1 already provides Sam2Model/Sam2Processor/Sam2VideoModel, so no new dependency", "not measured here", ["instance_masks", "video_propagation"],
     "perception/segmentation/sam2_box_prompter.py -> grounded_scene -> vertical_slice perception stage (default mask source)", "API reachable",
     "10-photo room via the API: 110/110 masks from SAM 2, no fallback; sofa no longer 7 m; consistency check, not accuracy", "depth-band masks (automatic fallback, recorded)",
     "Top opportunity: today's masks are a depth-band cut of a box and leak floor/wall (measured cause of a 7 m 'sofa')."),
    ("grounded_sam2", "IDEA-Research/Grounded-SAM-2", "Grounded-SAM-2", "Apache-2.0", "inherits Grounding DINO + SAM 2", "APPROVED", "not installed", "not measured", ["detection+masks", "video_tracking"],
     "pattern only: reproduce with our own adapters, not by vendoring", "Implemented", "not benchmarked", "current stack", "A glue repo; our adapters should follow its box-prompt pattern."),
    ("grounded_sam", "IDEA-Research/Grounded-Segment-Anything", "Grounded-Segment-Anything", "Apache-2.0", "SAM 1 weights not verified", "APPROVED_WITH_RESTRICTIONS", "not installed", "n/a", ["detection+masks"],
     "superseded by Grounded-SAM-2", "Implemented", "n/a", "n/a", "Last push 2024-09; use the SAM 2 route instead."),
    ("dinov2", "facebookresearch/dinov2", "DINOv2 (via transformers Dinov2Model)", "Apache-2.0", "facebook/dinov2-small, -base: apache-2.0", "APPROVED",
     "facebook/dinov2-small cached (revision ed25f3a3)", "small", ["appearance_embedding", "reid"],
     "perception/instances/dinov2_hf.py -> vertical_slice perception_appearance (ON by default; falls back to geometry only, on the record)", "API reachable",
     "10-photo room: separates same- vs different-label crops (cosine 0.63 vs 0.16) but changed no association decision (16 objects either way); consistency check, not accuracy",
     "geometry-only association (automatic fallback)", "On by default but unproven: no measured gain on the 10-photo room (16 objects either way). dinov2_reid.py silently substitutes a colour-statistic vector when torch.hub fails; do not use it for association."),
    ("florence2", None, "Florence-2 (via transformers Florence2ForConditionalGeneration)", "no GitHub repo (microsoft/Florence-2 returns 404); code ships inside transformers", "microsoft/Florence-2-base: mit",
     "APPROVED", "not downloaded; native in transformers 5.16.1 (no trust_remote_code needed)", "base ~0.23B params; not measured", ["captioning", "grounding", "detection"], "not integrated",
     "Implemented", "not benchmarked", "Grounding DINO", "Possible second-opinion detector; must beat Grounding DINO on annotated data before replacing anything."),
    ("detectron2", "facebookresearch/detectron2", "Detectron2", "Apache-2.0", "model-zoo weights vary; not audited", "APPROVED_WITH_RESTRICTIONS", "not installed", "GPU", ["instance_segmentation", "panoptic"], "not integrated",
     "Implemented", "not benchmarked", "Mask R-CNN via torchvision (in use)", "Overlaps torchvision Mask R-CNN already integrated; Windows build is painful."),
    ("mask2former", "facebookresearch/Mask2Former", "Mask2Former", "MIT (repo archived)", "HF COCO weights: 'other' (unresolved)", "LICENCE_REVIEW_REQUIRED", "not installed", "GPU", ["panoptic"], "not integrated",
     "Implemented", "not benchmarked", "n/a", "Archived upstream; weights licence unresolved."),
    ("ultralytics", "ultralytics/ultralytics", "Ultralytics YOLO", "AGPL-3.0", "AGPL-3.0", "REJECTED", "not installed", "GPU", ["closed_set_detection"], "not integrated", "Implemented", "n/a", "n/a",
     "AGPL blocks a closed commercial product unless a commercial licence is bought."),
    ("sam_hq", "SysCV/sam-hq", "SAM-HQ", "Apache-2.0", "not verified", "APPROVED_WITH_RESTRICTIONS", "not installed", "n/a", ["high_quality_masks"], "not integrated", "Implemented", "not benchmarked", "SAM 2",
     "Only worth testing if SAM 2 boundaries fail on thin structures."),
    # ---- C. depth
    ("depth_anything_v2_small", "DepthAnything/Depth-Anything-V2", "Depth Anything V2 Small", "Apache-2.0", "depth-anything/Depth-Anything-V2-Small-hf: apache-2.0", "APPROVED", "weights cached", "0.044 s, 0.42 GB (measured)",
     ["relative_depth"], "not the live depth model", "Functionally tested", "residual vs SfM 0.291 (consistency, not accuracy)", "DPT_Hybrid", "Worse fit to SfM than DPT_Hybrid in the one scene tested."),
    ("depth_anything_v2_base_large", "DepthAnything/Depth-Anything-V2", "Depth Anything V2 Base / Large", "Apache-2.0 (code)", "Base: cc-by-nc-4.0 (verified on the model card)", "RESEARCH_ONLY", "not cached", "n/a", ["relative_depth"],
     "not integrated", "Implemented", "n/a", "Small", "Non-commercial weights."),
    ("depth_anything_v2_metric_indoor", "DepthAnything/Depth-Anything-V2", "Depth Anything V2 Metric-Indoor Small", "Apache-2.0 (code)", "model card declares NO licence (verified: null)", "LICENCE_REVIEW_REQUIRED",
     "weights cached", "0.040 s, 0.24 GB (measured)", ["metric_depth"], "not integrated", "Functionally tested", "residual vs SfM 0.117 with ONE global scale (consistency, not accuracy)", "DPT_Hybrid",
     "Best measured fit to SfM, but undeclared licence and no effect on floor layering in an A/B; no default change."),
    ("dpt_hybrid", "isl-org/MiDaS", "DPT_Hybrid / MiDaS", "MIT (repo archived)", "Intel/dpt-hybrid-midas: apache-2.0", "ALREADY_INTEGRATED", "torch hub cache", "0.16 s, 0.68 GB (measured)", ["relative_depth"],
     "stage: depth (vertical_slice)", "Functionally tested", "residual vs SfM 0.199", "n/a", "Per-view scale drifts about 2x between views; objects land up to ~2.5 m apart across views."),
    ("metric3d", "YvanYin/Metric3D", "Metric3D", "BSD-2-Clause", "weights host returned HTTP 401 (not verified)", "LICENCE_REVIEW_REQUIRED", "not installed", "n/a", ["metric_depth"], "not integrated", "Implemented",
     "not benchmarked", "n/a", "Needs the weights licence read before any test."),
    ("marigold", "prs-eth/Marigold", "Marigold", "Apache-2.0", "marigold-depth-v1-1: openrail++ (use restrictions)", "APPROVED_WITH_RESTRICTIONS", "not installed", "diffusion; heavy for 6 GB (not measured)", ["relative_depth"],
     "not integrated", "Implemented", "not benchmarked", "DPT_Hybrid", "Slow diffusion model; no evidence it would beat the current depth on our metric."),
    ("depth_pro", "apple/ml-depth-pro", "Apple Depth Pro", "Apple sample-code licence (LICENSE read)", "apple/DepthPro: apple-amlr (research)", "RESEARCH_ONLY", "not installed", "n/a", ["metric_depth"], "not integrated", "Implemented", "n/a", "n/a",
     "Research-only weights."),
    # ---- D. point clouds / meshes
    ("open3d", "isl-org/Open3D", "Open3D", "MIT (LICENSE read)", "n/a", "APPROVED", "not installed (0.20.0 wheel exists for this Python)", "CPU; CUDA optional", ["icp", "outlier_removal", "tsdf", "poisson", "normals"],
     "not integrated; one reference in code", "Implemented", "not benchmarked", "numpy/scipy + trimesh (current)", "Fifth opportunity: outlier removal and robust registration before plane fitting."),
    ("pcl", "PointCloudLibrary/pcl", "PCL", "BSD (LICENSE.txt read)", "n/a", "APPROVED", "not installed", "CPU", ["point_cloud_ops"], "an internal module named pcl_core exists; the C++ library is not used", "Implemented", "n/a", "Open3D",
     "C++ build burden for no gain over Open3D."),
    ("pointcept", "Pointcept/Pointcept", "Pointcept", "MIT (code)", "pretrained weights are trained on ScanNet etc. whose terms are non-commercial", "LICENCE_REVIEW_REQUIRED", "not installed", "GPU, heavy", ["semantic_point_segmentation"], "not integrated",
     "Implemented", "not benchmarked", "plane-role heuristics", "Code is MIT, but the dataset terms behind the weights are the blocker."),
    ("trimesh", "mikedh/trimesh", "trimesh", "MIT", "n/a", "ALREADY_INTEGRATED", "installed", "CPU", ["mesh_io", "mesh_repair"], "optional, guarded import (meshing extra)", "Functionally tested", "none", "numpy", "Present in the environment."),
    ("cgal", "CGAL/cgal", "CGAL", "dual: GPL-3.0/LGPL-3.0 by package, or a commercial licence (LICENSE.md read)", "n/a", "LICENCE_REVIEW_REQUIRED", "not installed", "CPU", ["geometry_kernels"], "not integrated", "Implemented", "n/a", "n/a",
     "Per-package licensing; core packages are not permissive."),
    ("libigl", "libigl/libigl", "libigl", "GPL-3.0", "n/a", "REJECTED", "not installed", "CPU", ["mesh_processing"], "catalog mention only", "Implemented", "n/a", "trimesh", "GPL-3.0."),
    ("manifold", "elalish/manifold", "Manifold (manifold3d)", "Apache-2.0", "n/a", "APPROVED", "not installed (wheel 3.5.5 exists)", "CPU", ["mesh_booleans", "watertight_validation"],
     "not integrated; an internal 'manifold_kernel' name exists in the catalog map, unverified", "Implemented", "not benchmarked", "trimesh", "Candidate for guaranteed-valid mesh booleans."),
    # ---- E. video / tracking
    ("cotracker", "facebookresearch/co-tracker", "CoTracker3", "CC BY-NC 4.0 (LICENSE.md read)", "facebook/cotracker3: cc-by-nc-4.0", "RESEARCH_ONLY", "not cached", "n/a", ["point_tracking"], "not integrated", "Implemented", "n/a", "OpenCV optical flow",
     "Non-commercial."),
    ("bytetrack", "ifzhang/ByteTrack", "ByteTrack", "MIT", "n/a", "APPROVED", "not installed", "CPU", ["object_tracking"], "perception/tracking exists but is not reachable from the API", "Implemented", "not benchmarked", "n/a",
     "Repo moved to FoundationVision/ByteTrack; last push 2024-06; the algorithm is small enough to reimplement."),
    ("mmtracking", "open-mmlab/mmtracking", "MMTracking", "Apache-2.0", "varies", "APPROVED_WITH_RESTRICTIONS", "not installed", "GPU", ["object_tracking"], "not integrated", "Implemented", "n/a", "n/a",
     "Unmaintained since 2023-09; heavy mm* dependency stack."),
    ("opencv", "opencv/opencv", "OpenCV", "Apache-2.0", "n/a", "ALREADY_INTEGRATED", "installed (opencv-python-headless)", "CPU", ["image_ops", "video_decode"], "core dependency", "Functionally tested", "n/a", "n/a", "In use."),
    # ---- F. splats / rendering
    ("gsplat", "nerfstudio-project/gsplat", "gsplat", "Apache-2.0", "n/a", "APPROVED", "not installed", "CUDA compile; hard on Windows (not tried)", ["gaussian_splatting"], "splat ingest/render exists separately (PR #299)", "Implemented", "not benchmarked", "n/a",
     "Appearance only: not geometric accuracy."),
    ("nerfstudio", "nerfstudio-project/nerfstudio", "nerfstudio", "Apache-2.0", "n/a", "APPROVED", "not installed", "GPU", ["nerf", "splat_training"], "not integrated", "Implemented", "n/a", "n/a", "No evidence it is needed."),
    ("opensplat", "pierotofy/OpenSplat", "OpenSplat", "AGPL-3.0", "n/a", "REJECTED", "not installed", "n/a", ["gaussian_splatting"], "not integrated", "Implemented", "n/a", "n/a", "AGPL-3.0."),
    ("graphdeco_3dgs", "graphdeco-inria/gaussian-splatting", "Original 3D Gaussian Splatting", "Inria/MPII custom, research use (LICENSE.md read)", "n/a", "RESEARCH_ONLY", "not installed", "CUDA", ["gaussian_splatting"], "not integrated", "Implemented", "n/a", "gsplat",
     "Non-commercial terms."),
    ("openusd", "PixarAnimationStudios/OpenUSD", "OpenUSD", "Tomorrow Open Source Technology License 1.0 (modified Apache 2.0; LICENSE.txt read)", "n/a", "APPROVED_WITH_RESTRICTIONS", "not installed", "CPU", ["usd_export"], "usda text export exists without the library", "Functionally tested", "n/a", "text export",
     "The licence differs from Apache-2.0; read it before linking the library."),
    # ---- G. BIM / architecture
    ("ifcopenshell", "IfcOpenShell/IfcOpenShell", "IfcOpenShell", "LGPL-3.0", "n/a", "ALREADY_INTEGRATED", "installed", "CPU", ["ifc_export", "bim_semantics"], "exporters/ifc_bridge.py via the SDK/API (export extra)", "Functionally tested", "no real-data validation",
     "501 when absent", "LGPL-3.0: dynamic use is fine, modifying/bundling carries obligations."),
    ("web_ifc", "ThatOpen/engine_web-ifc", "web-ifc", "MPL-2.0", "n/a", "APPROVED_WITH_RESTRICTIONS", "not installed", "browser", ["ifc_viewer"], "not integrated", "Implemented", "n/a", "n/a", "File-level copyleft; frontend only."),
    ("xeokit", "xeokit/xeokit-sdk", "xeokit", "AGPL-3.0", "n/a", "REJECTED", "not installed", "browser", ["bim_viewer"], "not integrated", "Implemented", "n/a", "three.js (in use)", "AGPL-3.0."),
    ("qgis", "qgis/QGIS", "QGIS", "GPL-2.0", "n/a", "RESEARCH_ONLY", "not installed", "desktop", ["gis"], "not integrated", "Implemented", "n/a", "n/a", "GPL: use as an external tool only, never as a linked dependency."),
    ("compas", "compas-dev/compas", "COMPAS", "MIT", "n/a", "APPROVED", "not installed", "CPU", ["geometry_toolkit"], "not integrated", "Implemented", "n/a", "n/a", "No identified gap."),
    ("blender", "blender/blender", "Blender", "GPL-3.0 (COPYING read)", "n/a", "ALREADY_INTEGRATED", "not installed here", "GPU", ["export_script_target"], "exporters/blender generates a Python script; Blender is never linked", "Functionally tested", "n/a", "n/a",
     "Safe only as a separate process / generated script."),
    # ---- H. annotation / evaluation / datasets
    ("cvat", "cvat-ai/cvat", "CVAT", "MIT", "n/a", "APPROVED", "not installed", "server (Docker)", ["annotation"], "not integrated", "Implemented", "n/a", "n/a", "Second opportunity: blocks every accuracy metric."),
    ("label_studio", "HumanSignal/label-studio", "Label Studio", "Apache-2.0", "n/a", "APPROVED", "not installed", "server", ["annotation"], "not integrated", "Implemented", "n/a", "n/a", "Alternative to CVAT; pick one."),
    ("fiftyone", "voxel51/fiftyone", "FiftyOne", "Apache-2.0", "n/a", "APPROVED", "not installed (wheel 1.22.1 exists)", "CPU + MongoDB", ["dataset_inspection", "error_analysis"], "not integrated", "Implemented", "n/a", "benchmarks/reality_benchmark",
     "Complements the existing benchmark; does not replace it."),
    ("scannet", "ScanNet/ScanNet", "ScanNet", "code MIT-style; DATA terms are non-commercial (not verified here)", "n/a", "LICENCE_REVIEW_REQUIRED", "not downloaded", "large", ["dataset"], "not integrated", "Implemented", "n/a", "n/a", "Datasets need a separate licence read."),
    ("arkitscenes", "apple/ARKitScenes", "ARKitScenes", "Apple sample-code licence (LICENSE read); data terms not verified", "n/a", "LICENCE_REVIEW_REQUIRED", "not downloaded", "large", ["dataset"], "not integrated", "Implemented", "n/a", "n/a", "Repo moved to apple-aiml-research."),
    ("scannetpp", "scannetpp/scannetpp", "ScanNet++", "no licence declared in the repo", "n/a", "LICENCE_REVIEW_REQUIRED", "not downloaded", "large", ["dataset"], "not integrated", "Implemented", "n/a", "n/a", "No declared licence."),
    ("hypersim", "apple/ml-hypersim", "Hypersim", "Apple sample-code licence (LICENSE read); data terms not verified", "n/a", "LICENCE_REVIEW_REQUIRED", "not downloaded", "large", ["dataset"], "not integrated", "Implemented", "n/a", "n/a", "Synthetic; cannot validate real-world accuracy."),
    ("structured3d", "bertjiazheng/Structured3D", "Structured3D", "MIT (code); data terms not verified", "n/a", "LICENCE_REVIEW_REQUIRED", "not downloaded", "large", ["dataset"], "not integrated", "Implemented", "n/a", "n/a", "Synthetic."),
    # ---- non-commercial geometry foundation models and copyleft tools
    ("vggt", "facebookresearch/vggt", "VGGT", "VGGT License v1 (custom; LICENSE.txt read: research materials with an acceptable-use policy)", "facebook/VGGT-1B: cc-by-nc-4.0", "RESEARCH_ONLY", "not cached", "1B params; untested on 6 GB", ["pose", "depth", "point_maps"], "not integrated",
     "Implemented", "not benchmarked", "COLMAP", "Non-commercial weights."),
    ("dust3r", "naver/dust3r", "DUSt3R", "CC BY-NC-SA 4.0 (LICENSE read)", "same", "RESEARCH_ONLY", "not cached", "n/a", ["two_view_geometry"], "not integrated", "Implemented", "n/a", "n/a", "Non-commercial."),
    ("mast3r", "naver/mast3r", "MASt3R", "CC BY-NC-SA 4.0 (LICENSE read)", "same", "RESEARCH_ONLY", "not cached", "n/a", ["matching", "two_view_geometry"], "not integrated", "Implemented", "n/a", "n/a", "Non-commercial."),
    ("odm", "OpenDroneMap/ODM", "OpenDroneMap / WebODM", "AGPL-3.0", "n/a", "REJECTED", "not installed", "n/a", ["aerial_photogrammetry"], "not integrated", "Implemented", "n/a", "n/a", "AGPL-3.0."),
    ("pyslam", "luigifreda/pyslam", "pySLAM", "GPL-3.0", "n/a", "REJECTED", "not installed", "n/a", ["slam"], "not integrated", "Implemented", "n/a", "n/a", "GPL-3.0."),
    ("limap", "cvg/limap", "LIMAP", "BSD-3-Clause", "n/a", "APPROVED", "not installed", "CPU", ["line_mapping"], "not integrated", "Implemented", "not benchmarked", "n/a", "Line features could help wall edges; no measured need yet."),
]

FIELDS = ("id", "probe_key", "name", "repo_licence", "weights_licence", "classification", "installation", "hardware", "capabilities", "integration", "maturity", "benchmark", "fallback", "note")

OPPORTUNITIES = [
    ("SAM 2 instance masks (Grounding DINO boxes as prompts)", "sam2",
     "Replaces the depth-band mask that leaked floor/wall into objects (the measured cause of a 7 m sofa). transformers already ships Sam2Model; Apache-2.0 code and weights; no new dependency.",
     "instance mIoU and boundary F1 once annotated; until then cross-view extent plausibility on the 10-photo room"),
    ("Human-verified ground truth: CVAT or Label Studio + FiftyOne feeding the Reality Benchmark", "cvat",
     "Every accuracy metric is blocked by the absence of annotations. Review tooling, not a new benchmark.", "enables precision/recall/mAP; no number exists before it"),
    ("DINOv2 appearance embeddings in the association resolver", "dinov2",
     "Association is geometry-only, so views of one object split (two rugs) or merge. Code exists and is unreachable; weights are cached.",
     "duplicate-entity rate, identity precision/recall"),
    ("Small-object tiled pass wired into perception", "grounding_dino",
     "perception/entities/small_objects.py is unreachable from the API; sockets, switches and handles are not found at full-frame scale.", "small-object recall by group"),
    ("Open3D outlier removal, robust registration and TSDF/Poisson", "open3d",
     "MIT, wheel available. Cleans the sparse and dense clouds before plane fitting; TSDF/Poisson are candidates for the mesh stage.", "Chamfer / surface distance vs surveyed room; plane-fit residual"),
    ("hloc + LightGlue matching for SfM reproducibility", "hloc",
     "Run-to-run geometry varies (995 vs 1,423 points). Extractor weights licence (SuperPoint) must be resolved first.", "registration rate, reprojection error, run-to-run spread"),
    ("Cross-view depth consistency (metric depth, one global scale)", "depth_anything_v2_metric_indoor",
     "Best measured fit to SfM (0.117 vs 0.199) but the model card declares no licence; blocked on a licence read.", "scale consistency, dimensional error"),
    ("Florence-2 as a second-opinion detector (native in transformers)", "florence2",
     "MIT weights; no remote code now. Only replaces Grounding DINO if it wins on annotated data.", "detection precision/recall vs Grounding DINO"),
    ("manifold3d for guaranteed-valid mesh booleans / watertight checks", "manifold",
     "Apache-2.0, wheel available. Protects export validity.", "watertightness, topological validity of exported meshes"),
    ("IfcOpenShell: validate IFC against real captures", "ifcopenshell",
     "Already integrated (LGPL-3.0); the gap is real-data validation of the semantics, not a new library.", "IFC element completeness vs measured room"),
]


def build() -> dict:
    repos = PROBE["repos"]
    records = []
    for row in E:
        rec = dict(zip(FIELDS, row))
        assert rec["maturity"] in MATURITY, rec["id"]
        key = rec.pop("probe_key")
        meta = repos.get(key, {}) if key else {}
        rec["repository"] = f"https://github.com/{meta.get('full_name') or key}" if key else None
        rec["commit_sha"] = meta.get("head_sha")
        rec["latest_release"] = meta.get("latest_release")
        rec["last_push"] = (meta.get("pushed_at") or "")[:10] or None
        rec["archived"] = meta.get("archived")
        rec["github_spdx_reported"] = meta.get("license")
        rec["provenance"] = {"probed_on": PROBED_ON, "source": "GitHub API + LICENSE files + Hugging Face API; classification and maturity are assessments",
                             "probe_file": "engine/technologies/open_source_probe.json"}
        records.append(rec)
    ids = [r["id"] for r in records]
    assert len(ids) == len(set(ids)), "duplicate id"
    known = set(ids)
    assert all(o[1] in known for o in OPPORTUNITIES), "opportunity points at an unknown technology"
    return {"schema": 1, "probed_on": PROBED_ON, "maturity_ladder": list(MATURITY), "records": records,
            "top_opportunities": [{"rank": i + 1, "title": t, "technology_id": tid, "why": why, "measure": m} for i, (t, tid, why, m) in enumerate(OPPORTUNITIES)]}


ORDER = ("ALREADY_INTEGRATED", "APPROVED", "APPROVED_WITH_RESTRICTIONS", "LICENCE_REVIEW_REQUIRED", "RESEARCH_ONLY", "REJECTED")


def render_md(reg: dict) -> str:
    recs = reg["records"]
    by: dict = {}
    for r in recs:
        by.setdefault(r["classification"], []).append(r)
    cell = lambda s: str(s).replace("|", "/")  # noqa: E731
    out = ["# Open-source technology registry", "",
           f"Facts (licence, last push, release, HEAD commit) were read on **{reg['probed_on']}** from the GitHub and Hugging Face APIs and from the "
           "LICENSE files of repositories whose API licence was `NOASSERTION`. **Classification and maturity are my assessments.** "
           "No accuracy has been measured for any candidate (no annotated data exists). Source of truth: `engine/technologies/open_source_registry.json`; "
           "regenerate this file with `python scripts/build_open_source_registry.py`.", "",
           "Maturity ladder: " + " < ".join(reg["maturity_ladder"]) + ".", "", "## Counts", "", "| Classification | Count |", "|---|---|"]
    out += [f"| {k} | {len(by.get(k, []))} |" for k in ORDER]
    out += ["", "## Ten highest-value opportunities, in integration order", ""]
    for o in reg["top_opportunities"]:
        out += [f"{o['rank']}. **{o['title']}** (`{o['technology_id']}`)  ", f"   {o['why']}  ", f"   Measured by: {o['measure']}.", ""]
    out += ["## Records", "", "| id | classification | repo licence | weights licence | last push | maturity | integration |", "|---|---|---|---|---|---|---|"]
    for k in ORDER:
        for r in by.get(k, []):
            out.append(f"| {r['id']} | {r['classification']} | {cell(r['repo_licence'])} | {cell(r['weights_licence'])} | {r['last_push'] or '-'}{' (archived)' if r['archived'] else ''} | {r['maturity']} | {cell(r['integration'])} |")
    out += ["", "## Notes per record", ""]
    out += [f"* **{r['id']}** ({r['name']}): {r['note']}" for r in recs]
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    reg = build()
    (ROOT / "engine" / "technologies" / "open_source_registry.json").write_text(json.dumps(reg, indent=1) + "\n", encoding="utf-8")
    (ROOT / "docs" / "engineering" / "OPEN_SOURCE_TECHNOLOGY_REGISTRY.md").write_text(render_md(reg), encoding="utf-8")
    print(len(reg["records"]), "records written")
