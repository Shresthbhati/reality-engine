"""Mandatory Sequential Maximum-Fidelity Stages Definition.

Defines the 13 configurable stages of the Maximum Fidelity reconstruction policy:
Stage 1: Image preparation and COLMAP reconstruction
Stage 2: MapAnything
Stage 3: VGGT and VGGT-Omega
Stage 4: DUSt3R and MASt3R
Stage 5: Depth Anything 3 and depth alternatives
Stage 6: Grounding DINO and Florence-2 (with OWL-ViT)
Stage 7: SAM2 and SAM3/SAM3.1 (with SAM ViT-B)
Stage 8: DINOv2 and tracking systems
Stage 9: Geometry processing, registration and surface reconstruction
Stage 10: Multi-engine consensus
Stage 11: WorldIR compilation
Stage 12: Quality verification
Stage 13: Iterative refinement
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from engine.multiengine.contract import Task


class StageId(str, Enum):
    STAGE_01_COLMAP = "stage_01_colmap"
    STAGE_02_MAPANYTHING = "stage_02_mapanything"
    STAGE_03_VGGT = "stage_03_vggt"
    STAGE_04_DUST3R = "stage_04_dust3r"
    STAGE_05_DEPTH = "stage_05_depth"
    STAGE_06_DETECTION = "stage_06_detection"
    STAGE_07_SEGMENTATION = "stage_07_segmentation"
    STAGE_08_TRACKING = "stage_08_tracking"
    STAGE_09_SURFACE = "stage_09_surface"
    STAGE_10_CONSENSUS = "stage_10_consensus"
    STAGE_11_WORLDIR = "stage_11_worldir"
    STAGE_12_QUALITY = "stage_12_quality"
    STAGE_13_REFINEMENT = "stage_13_refinement"


@dataclass(frozen=True)
class PipelineStage:
    stage_id: StageId
    name: str
    task: Task | None
    required_engines: tuple[str, ...]
    dependencies: tuple[StageId, ...]
    description: str


STAGE_DEFINITIONS: tuple[PipelineStage, ...] = (
    PipelineStage(
        stage_id=StageId.STAGE_01_COLMAP,
        name="COLMAP Reconstruction",
        task=Task.SFM,
        required_engines=("colmap",),
        dependencies=(),
        description="Feature extraction, stereoscopic matching, sparse SfM, and patch-match dense MVS.",
    ),
    PipelineStage(
        stage_id=StageId.STAGE_02_MAPANYTHING,
        name="MapAnything Dense Mapping",
        task=Task.SFM,
        required_engines=("mapanything",),
        dependencies=(),
        description="Unstructured multi-view dense scene mapping and camera pose estimation.",
    ),
    PipelineStage(
        stage_id=StageId.STAGE_03_VGGT,
        name="VGGT & VGGT-Omega Feedforward Geometry",
        task=Task.SFM,
        required_engines=("vggt", "vggt_omega"),
        dependencies=(),
        description="Visual Geometry Grounded Transformer feedforward multi-view camera poses and point tokens.",
    ),
    PipelineStage(
        stage_id=StageId.STAGE_04_DUST3R,
        name="DUSt3R & MASt3R Stereoscopic Pointmaps",
        task=Task.SFM,
        required_engines=("dust3r", "mast3r"),
        dependencies=(),
        description="Pairwise unconstrained 3D pointmap regression and dense matching (subject to CC-BY-NC-4.0).",
    ),
    PipelineStage(
        stage_id=StageId.STAGE_05_DEPTH,
        name="Depth Anything 3 & Monocular Depth",
        task=Task.DEPTH,
        required_engines=("depth_anything_v3", "depth_anything", "midas"),
        dependencies=(),
        description="High-resolution monocular depth estimation across multiple independent architectures.",
    ),
    PipelineStage(
        stage_id=StageId.STAGE_06_DETECTION,
        name="Grounding DINO & Florence-2 Detection",
        task=Task.DETECTION,
        required_engines=("grounding_dino", "florence2", "owlvit"),
        dependencies=(),
        description="Open-vocabulary text-phrase grounded object detection and dense captioning across scale tiers.",
    ),
    PipelineStage(
        stage_id=StageId.STAGE_07_SEGMENTATION,
        name="SAM2 & SAM3/3.1 Instance Segmentation",
        task=Task.SEGMENTATION,
        required_engines=("sam2", "sam3", "sam"),
        dependencies=(StageId.STAGE_06_DETECTION,),
        description="Promptable instance mask extraction with interior hole topology (cv2.RETR_CCOMP).",
    ),
    PipelineStage(
        stage_id=StageId.STAGE_08_TRACKING,
        name="DINOv2 & Multi-View Tracking",
        task=Task.SCENE_UNDERSTANDING,
        required_engines=("dinov2", "tracking"),
        dependencies=(),
        description="Self-supervised patch representations for entity re-ID and temporal correspondence tracking.",
    ),
    PipelineStage(
        stage_id=StageId.STAGE_09_SURFACE,
        name="Geometry Processing & Surface Mesh",
        task=Task.MVS,
        required_engines=("open3d_tsdf",),
        dependencies=(StageId.STAGE_01_COLMAP, StageId.STAGE_05_DEPTH),
        description="Volumetric TSDF fusion, voxel hashing, and watertight Marching Cubes surface extraction.",
    ),
    PipelineStage(
        stage_id=StageId.STAGE_10_CONSENSUS,
        name="Multi-Engine Evidence Consensus",
        task=None,
        required_engines=(),
        dependencies=(
            StageId.STAGE_01_COLMAP,
            StageId.STAGE_05_DEPTH,
            StageId.STAGE_06_DETECTION,
            StageId.STAGE_07_SEGMENTATION,
        ),
        description="Robust Sim(3) frame alignment, correlation group filtering, consensus sets, and image-evidence override.",
    ),
    PipelineStage(
        stage_id=StageId.STAGE_11_WORLDIR,
        name="WorldIR Compilation",
        task=None,
        required_engines=(),
        dependencies=(StageId.STAGE_10_CONSENSUS,),
        description="Assembly of verified geometry, architectural boundaries, and semantic entities into WorldIR.",
    ),
    PipelineStage(
        stage_id=StageId.STAGE_12_QUALITY,
        name="Quality Verification Gate",
        task=None,
        required_engines=(),
        dependencies=(StageId.STAGE_11_WORLDIR,),
        description="Evaluation against benchmark-established quality gates and regression checks.",
    ),
    PipelineStage(
        stage_id=StageId.STAGE_13_REFINEMENT,
        name="Iterative Refinement & Capture Requests",
        task=None,
        required_engines=(),
        dependencies=(StageId.STAGE_12_QUALITY,),
        description="Diagnose missing/uncertain features, reprocess targeted models, or emit physical capture requests.",
    ),
)
