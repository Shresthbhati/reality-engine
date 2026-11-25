"""Machine-readable Perception Improvement Ledger (``docs/PERCEPTION_IMPROVEMENT_LEDGER.json``).

Strictly separates ``implemented`` improvements (backed by real code changes in
execution paths) from ``proposed`` and ``blocked`` backlog items.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_improvement_ledger() -> dict[str, Any]:
    implemented: list[dict[str, Any]] = [
        {
            "id": "IMP-0001",
            "subsystem": "detection_contracts",
            "current_limitation": "Scattered .box / .bbox / .box_xyxy / .score / .confidence access across backends",
            "proposed_improvement": "Single normalization boundary with strict XYXY validation, None confidence preservation, and EMPTY vs FAILED separation",
            "expected_impact": "Eliminates interface AttributeErrors and silent invalid-box leaks",
            "dependencies": ["perception/detection/policy.py", "provenance"],
            "priority": "P0",
            "status": "implemented",
            "changed_files": ["perception/detection/normalized.py", "perception/scene_intelligence.py"],
            "product_behavior_enabled": "All detector backends normalize into NormalizedDetectionBatch before segmentation, depth, or WorldIR bridging.",
        },
        {
            "id": "IMP-0002",
            "subsystem": "object_and_arch_detection",
            "current_limitation": "COCO Mask R-CNN alone returns 0 detections for walls, floors, ceilings, windows, doors, and built-ins",
            "proposed_improvement": "Integrate local-cache Grounding DINO + OWL-ViT + Florence-2 with same-model/same-label NMS",
            "expected_impact": "Full architectural + furniture open-vocabulary coverage while suppressing same-class duplicates",
            "dependencies": ["transformers", "IDEA-Research/grounding-dino-tiny"],
            "priority": "P0",
            "status": "implemented",
            "changed_files": ["perception/detection/grounded_arch.py", "perception/detection/backends.py", "perception/detection/normalized.py"],
            "product_behavior_enabled": "Detects structural elements, openings, architectural details, and furniture on real COLMAP indoor photos.",
        },
        {
            "id": "IMP-0003",
            "subsystem": "segmentation",
            "current_limitation": "SAM 2.1 unavailable via standalone sam2 package; raw masks unvalidated and unrefined",
            "proposed_improvement": "Integrate HF transformers Sam2BoxPrompter with morphological refinement, boundary sharpness, box-fill rejection, and depth occlusion",
            "expected_impact": "Usable architectural & object masks with explicit quality/uncertainty metadata",
            "dependencies": ["transformers", "facebook/sam2.1-hiera-small", "opencv-python"],
            "priority": "P0",
            "status": "implemented",
            "changed_files": ["perception/segmentation/arch_segmenter.py"],
            "product_behavior_enabled": "Produces 104/104 (balanced) and 181/182 (forensic) valid SAM2 masks on real indoor photos.",
        },
        {
            "id": "IMP-0004",
            "subsystem": "depth_and_geometry",
            "current_limitation": "Depth maps disconnected from masks; no surface orientation or discontinuity checks",
            "proposed_improvement": "Compute mask-scoped median depth, IQR dispersion, boundary discontinuity, and qualitative surface orientation",
            "expected_impact": "Separates foreground/background boundaries and preserves relative vs metric scale honesty",
            "dependencies": ["perception/depth/depth_anything_backend.py"],
            "priority": "P0",
            "status": "implemented",
            "changed_files": ["perception/evidence_enrichment.py"],
            "product_behavior_enabled": "Observations carry depth_evidence and never fabricate metric coordinates from relative depth.",
        },
        {
            "id": "IMP-0005",
            "subsystem": "cross_view_association",
            "current_limitation": "Appearance-only matching risks merging repeated doors/windows",
            "proposed_improvement": "Fuse DINOv2 embeddings with epipolar_residual_px, disambiguate repeated structures, and map failures to FailureKind",
            "expected_impact": "Prevents false merges on repeated architectural elements and preserves contradictions",
            "dependencies": ["facebook/dinov2-small", "perception/instances/epipolar.py", "perception/evaluation/failure_detector.py"],
            "priority": "P0",
            "status": "implemented",
            "changed_files": ["perception/evidence_enrichment.py"],
            "product_behavior_enabled": "Cross-view candidates carry geometric support, ambiguity status, and precision/recall diagnostics.",
        },
        {
            "id": "IMP-0006",
            "subsystem": "video_intelligence",
            "current_limitation": "Video frames not decoded into job pipeline or tracked across occlusions",
            "proposed_improvement": "Decode video keyframes with duplicate suppression, optical-flow camera motion, and occlusion-gap track recovery wired into evidence_world_stage",
            "expected_impact": "First-class MP4/MOV/WEBM ingestion in API jobs with timestamp/frame provenance",
            "dependencies": ["opencv-python"],
            "priority": "P0",
            "status": "implemented",
            "changed_files": ["perception/evidence_enrichment.py", "apps/api/evidence_world_stage.py"],
            "product_behavior_enabled": "Video files submitted to evidence_world_stage.run_for_job emit keyframes, motion metadata, and spatial_inputs_v1.json.",
        },
        {
            "id": "IMP-0007",
            "subsystem": "orchestration_and_hardware",
            "current_limitation": "7-model execution ran unconditionally without VRAM/content-aware plan selection",
            "proposed_improvement": "Add FAST_STRUCTURAL_PLAN, BALANCED_INTERIOR_PLAN, FULL_FORENSIC_PLAN, and select_inference_plan_for_inputs",
            "expected_impact": "Cuts 3-photo RTX 4050 runtime from 33.2s to 19.5s in balanced mode while retaining 100% valid SAM2 masks",
            "dependencies": ["perception/availability.py", "perception/scene_intelligence.py"],
            "priority": "P1",
            "status": "implemented",
            "changed_files": ["perception/orchestrator.py", "perception/model_registry.py", "perception/availability.py"],
            "product_behavior_enabled": "Content- and VRAM-aware plan execution with explicit InferencePlanError on missing required models.",
        },
        {
            "id": "IMP-0008",
            "subsystem": "spatial_inputs_and_api",
            "current_limitation": "No spatial_inputs.v1 producer or API endpoint for reconstruction consumers",
            "proposed_improvement": "Implement SpatialInputsV1, register_inputs_provider, evidence_world_stage emission, and GET .../spatial-inputs",
            "expected_impact": "Deterministic, validated spatial_inputs_v1.json emitted and inspectable per API run",
            "dependencies": ["apps/api/evidence_world_stage.py", "apps/api/routes_evidence_world.py"],
            "priority": "P0",
            "status": "implemented",
            "changed_files": ["perception/spatial_inputs.py", "apps/api/evidence_world_stage.py", "apps/api/routes_evidence_world.py"],
            "product_behavior_enabled": "Real API runs produce and serve spatial_inputs.v1 alongside scene_intelligence telemetry.",
        },
        {
            "id": "IMP-0009",
            "subsystem": "image_quality_and_materials",
            "current_limitation": "Image quality lacked motion-blur/noise metrics; materials used box crops",
            "proposed_improvement": "Add directional motion blur & MAD noise estimation; compute material hypotheses from valid segmentation mask crops",
            "expected_impact": "Cleaner appearance statistics and grounded capture recommendations tied to missing_evidence",
            "dependencies": ["perception/materials/material_hypotheses.py"],
            "priority": "P1",
            "status": "implemented",
            "changed_files": ["perception/world_bridge.py", "perception/scene_intelligence.py"],
            "product_behavior_enabled": "Material hypotheses stay unverified (ESTIMATED) and avoid background wall color bleed.",
        },
        {
            "id": "IMP-0010",
            "subsystem": "evaluation_and_inventory",
            "current_limitation": "No human-reviewed annotation manifest or machine-readable engine inventory",
            "proposed_improvement": "Build AnnotationManifest (separating pseudo-labels from reviewed GT), dataset adapters, and ENGINE_INVENTORY.json",
            "expected_impact": "Prevents self-evaluation leaks and exposes reproducible metric runners",
            "dependencies": ["benchmarks/dataset_adapters.py"],
            "priority": "P1",
            "status": "implemented",
            "changed_files": ["benchmarks/annotation_manifest.py", "benchmarks/dataset_adapters.py", "perception/inventory.py"],
            "product_behavior_enabled": "Reproducible detection/segmentation/depth metric evaluation whenever GT manifests are supplied.",
        },
        {
            "id": "IMP-0014",
            "subsystem": "frontend_evidence_inspector",
            "current_limitation": "Desktop EvidenceRunPanel did not surface spatial_inputs.v1 mask/association/scale summaries",
            "proposed_improvement": "Wire GET /api/worlds/{world_id}/evidence-world/{run_id}/spatial-inputs into EvidenceRunPanel via getSpatialInputsV1",
            "expected_impact": "Interactive UI inspection of observation counts, masked observations, cross-view matches, and scale basis",
            "dependencies": ["apps/api/routes_evidence_world.py"],
            "priority": "P2",
            "status": "implemented",
            "changed_files": ["frontend/src/lib/api/evidence-world.ts", "frontend/src/components/workspace/EvidenceRunPanel.tsx"],
            "product_behavior_enabled": "Reality Studio EvidenceRunPanel displays spatial_inputs.v1 observation, mask, match, and scale summaries.",
        },
    ]

    blocked: list[dict[str, Any]] = [
        {
            "id": "IMP-0011",
            "subsystem": "ground_truth_benchmarking",
            "current_limitation": "ScanNet, Matterport3D, Structured3D, Hypersim, and Replica archives are not on disk",
            "proposed_improvement": "Run compare_detectors_on_manifest and depth/segmentation benchmarks on ScanNet & Hypersim splits",
            "expected_impact": "Calibrated precision, recall, mIoU, and AbsRel baselines",
            "dependencies": ["RE_SCANNET_DIR or RE_HYPERSIM_DIR"],
            "priority": "P0",
            "status": "blocked",
            "changed_files": [],
            "product_behavior_enabled": "Blocked on external dataset download/license access.",
        },
        {
            "id": "IMP-0012",
            "subsystem": "claude_spatial_global_integration",
            "current_limitation": "Claude spatial_global consumer is not present on branch feat/reconstruction-sprint10",
            "proposed_improvement": "Execute end-to-end API job -> spatial_inputs.v1 -> spatial_global -> WorldIR persistence",
            "expected_impact": "Full closed-loop spatial reconstruction from perception evidence",
            "dependencies": ["Claude spatial_global merge"],
            "priority": "P0",
            "status": "blocked",
            "changed_files": [],
            "product_behavior_enabled": "Producer side ready; awaiting consumer module on integration branch.",
        },
    ]

    proposed: list[dict[str, Any]] = [
        {
            "id": "IMP-0013",
            "subsystem": "calibration_and_enablement",
            "current_limitation": "RE_EVIDENCE_WORLD remains opt-in (False by default)",
            "proposed_improvement": "Calibrate confidence thresholds and quality gates against ScanNet/Hypersim and satisfy ENABLEMENT_CRITERIA",
            "expected_impact": "Safe default-on enablement of evidence-to-world in production jobs",
            "dependencies": ["IMP-0011", "IMP-0012"],
            "priority": "P1",
            "status": "proposed",
            "changed_files": [],
            "product_behavior_enabled": "Pending labelled calibration dataset.",
        },
    ]

    all_entries = implemented + blocked + proposed
    return {
        "schema_version": "perception_improvement_ledger.v1",
        "counts": {
            "total": len(all_entries),
            "implemented": len(implemented),
            "blocked": len(blocked),
            "proposed": len(proposed),
        },
        "entries": all_entries,
    }


def write_improvement_ledger(path: Path = Path("docs/PERCEPTION_IMPROVEMENT_LEDGER.json")) -> Path:
    data = build_improvement_ledger()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path
