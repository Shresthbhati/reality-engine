"""Unified A5 Production Perception Pipeline (A5 Objective 3).

Implements the full end-to-end production flow:
  Image ingestion
  -> Image quality assessment
  -> Adaptive image processing (tiles/crops)
  -> Engine selection (via ModelLifecycleManager)
  -> Detection (external/live Grounding DINO or supplied observations)
  -> Segmentation (SAM 2 / SAM 3 / SAM-HQ with zero fabrication)
  -> Duplicate resolution (A5.2)
  -> Cross-view association (A5.3 with appearance + Claude camera poses)
  -> Architectural understanding (A5.4 + A4.3/A4.4/elevation)
  -> Furniture understanding (A5.5 + A4.5)
  -> Evidence fusion (A4.9)
  -> Normalized observations
  -> Claude handoff (perception_handoff.v1)
  -> WorldIR / EvidenceMatrix persistence & reload verification.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from perception.contracts.claude_handoff_v1 import HandoffBundle
from perception.contracts.claude_integration_v2 import (
    ClaudeIntegrationVerification,
    ClaudeReconstructionFeedback,
    execute_claude_handoff_and_persist,
)
from perception.intelligence.active_v2 import ActivePerceptionReport, run_active_perception_v2
from perception.intelligence.architectural import ArchitecturalAnalysisResult, analyze_architecture
from perception.intelligence.architectural_elevation import (
    ArchitecturalElevationReport,
    analyze_architectural_elevations,
)
from perception.intelligence.architectural_relations import (
    ArchitecturalRelationGraph,
    reason_architectural_relationships,
)
from perception.intelligence.architectural_v2 import (
    ArchitecturalGeometry2Result,
    analyze_architectural_geometry_v2,
)
from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    CoordinateSystemSpec,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    PerceptionBundle,
    ScoreSemantics,
)
from perception.intelligence.deduplication_v2 import (
    DuplicateResolutionResult,
    resolve_duplicate_detections_v2,
)
from perception.intelligence.depth_aware import (
    DepthAwarePerceptionResult,
    DepthMapContract,
    associate_observations_with_depth,
)
from perception.intelligence.diagnostics import DiagnosticReport
from perception.intelligence.diagnostics_v2 import run_production_diagnostics_v2
from perception.intelligence.furniture import FurnitureAnalysisResult, analyze_furniture
from perception.intelligence.furniture_v2 import Furniture2AnalysisResult, analyze_furniture_v2
from perception.intelligence.fusion_v2 import ExplainableFusionDecision, fuse_observations_v2
from perception.intelligence.gpu_lifecycle import (
    ModelLifecycleManager,
    ModelStageExecutionRecord,
)
from perception.intelligence.identity_v2 import (
    IdentityResolutionV2Report,
    resolve_cross_view_identities_v2,
)
from perception.intelligence.image_adaptive import (
    ImageQualityProfile,
    analyze_image_quality,
    plan_adaptive_crops,
)
from perception.intelligence.open_vocab import Domain, generate_survey_prompts
from perception.intelligence.scene_audit_v2 import (
    ScenePerceptionAuditReport,
    run_scene_perception_audit,
)
from perception.intelligence.scene_refinement_v2 import (
    SceneRefinementReport,
    refine_scene_observations_v2,
)
from perception.intelligence.segmentation_v2 import (
    SegmentationExecutionReport,
    attach_masks_to_observations,
)


@dataclass(frozen=True)
class A5ProductionResult:
    run_id: str
    bundle: PerceptionBundle
    handoff: HandoffBundle
    claude_verification: ClaudeIntegrationVerification | None
    image_qualities: dict[str, ImageQualityProfile]
    crop_counts_by_image: dict[str, int]
    segmentation_report: SegmentationExecutionReport
    dedup_result: DuplicateResolutionResult
    refinement_report: SceneRefinementReport
    identity_v2: IdentityResolutionV2Report
    arch_legacy: ArchitecturalAnalysisResult
    arch_geometry: dict[str, ArchitecturalGeometry2Result]
    arch_relations: ArchitecturalRelationGraph
    arch_elevations: ArchitecturalElevationReport
    furniture_legacy: FurnitureAnalysisResult
    furniture_v2: Furniture2AnalysisResult
    depth_results: tuple[DepthAwarePerceptionResult, ...]
    fusion_decisions: tuple[ExplainableFusionDecision, ...]
    active_refinement: ActivePerceptionReport
    scene_audit: ScenePerceptionAuditReport
    diagnostics: DiagnosticReport
    stage_telemetry: tuple[ModelStageExecutionRecord, ...]
    wall_time_s: float

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "wall_time_s": round(self.wall_time_s, 3),
            "images_ingested": len(self.image_qualities),
            "raw_observations": self.dedup_result.summary.get("input_count", 0),
            "dedup_canonical_count": self.dedup_result.summary.get("canonical_count", 0),
            "dedup_suppressed_count": self.dedup_result.summary.get("suppressed_count", 0),
            "dedup_reduction_rate": self.dedup_result.summary.get("duplicate_reduction_rate", 0.0),
            "competing_hypotheses": self.dedup_result.summary.get("competing_hypotheses_count", 0),
            "refinement_accepted": self.refinement_report.summary.get("accepted_count", 0),
            "refinement_rejected_arch": self.refinement_report.summary.get("rejected_architectural_count", 0),
            "refinement_rejected_furn": self.refinement_report.summary.get("rejected_furniture_count", 0),
            "segmentation_status": self.segmentation_report.status,
            "masks_produced": self.segmentation_report.masks_produced,
            "empty_masks": self.segmentation_report.empty_masks,
            "mean_mask_quality": self.segmentation_report.mean_mask_quality,
            "mean_mask_coverage": self.segmentation_report.mean_mask_coverage,
            "identity_summary": self.identity_v2.summary,
            "arch_relations": self.arch_relations.summary,
            "arch_elevations": self.arch_elevations.summary,
            "furniture_v2": self.furniture_v2.summary,
            "fusion_clusters": len(self.fusion_decisions),
            "audit_score": self.scene_audit.overall_readiness_score,
            "audit_ready": self.scene_audit.is_survey_ready,
            "handoff_accepted": len(self.handoff.observations),
            "handoff_rejected": len(self.handoff.rejected),
            "evidence_matrix_persisted": (
                self.claude_verification.evidence_matrix_persisted
                if self.claude_verification
                else False
            ),
            "diagnostics": self.diagnostics.to_dict(),
        }


def _stable_obs_id(run_id: str, image_id: str, idx: int, label: str, bbox: tuple[float, ...]) -> str:
    payload = f"{run_id}|{image_id}|{idx}|{label}|{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}"
    return "obs_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:14]


def ingest_images_rgb(
    image_paths: Sequence[Path | str],
    *,
    max_dimension_px: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    """Ingest RGB photographs with optional memory-safe downscaling while tracking paths."""
    rgb_map: dict[str, np.ndarray] = {}
    path_map: dict[str, Path] = {}
    for p in image_paths:
        path = Path(p)
        if not path.exists():
            continue
        with Image.open(path) as im:
            im_rgb = im.convert("RGB")
            if max_dimension_px and max(im_rgb.size) > max_dimension_px:
                im_rgb.thumbnail((max_dimension_px, max_dimension_px))
            rgb_map[path.name] = np.asarray(im_rgb, dtype=np.uint8)
            path_map[path.name] = path
    return rgb_map, path_map


def run_live_detection_stage(
    image_paths: Sequence[Path | str],
    images_rgb: Mapping[str, np.ndarray],
    *,
    run_id: str,
    domain: Domain = Domain.INTERIOR,
    threshold: float = 0.28,
    lifecycle: ModelLifecycleManager | None = None,
) -> tuple[list[NormalizedPerceptionObservation], ModelStageExecutionRecord | None]:
    """Run Grounding DINO via ModelLifecycleManager and normalize into ``NormalizedPerceptionObservation``."""
    from perception.detection_adapter import DEFAULT_SURVEY_QUERIES, _grounding_dino_run

    mgr = lifecycle or ModelLifecycleManager()
    prompts = tuple(generate_survey_prompts(domain=domain, max_prompts=30)) or DEFAULT_SURVEY_QUERIES
    str_paths = [str(Path(p)) for p in image_paths if Path(p).exists()]
    if not str_paths:
        return [], None

    def _run_det(_device: str, _batch: int, _res: int) -> Any:
        return _grounding_dino_run(str_paths, threshold=threshold, queries=prompts, run_id=run_id)

    out, rec = mgr.run_stage(
        "detection_grounding_dino",
        "IDEA-Research/grounding-dino-tiny",
        _run_det,
        num_images=len(str_paths),
        estimated_vram_mb=1800.0,
        timeout_s=120.0,
    )
    if out is None:
        return [], rec

    arch_set = frozenset({
        "wall", "floor", "ceiling", "door", "door_frame", "window",
        "window_frame", "window_sill", "opening", "column", "beam", "stairs",
    })
    observations: list[NormalizedPerceptionObservation] = []
    for idx, orec in enumerate(out.observations):
        if orec.bbox_px is None:
            continue
        img_name = Path(orec.source_image_id).name
        img_arr = images_rgb.get(img_name)
        if img_arr is None:
            continue
        ih, iw = img_arr.shape[:2]
        lbl = str(orec.label or "object").strip().lower().replace(" ", "_")
        cat = "architecture" if lbl in arch_set else "furniture"
        bbox = tuple(float(v) for v in orec.bbox_px[:4])
        oid = _stable_obs_id(run_id, img_name, idx, lbl, bbox)
        observations.append(NormalizedPerceptionObservation(
            observation_id=oid,
            schema_version=SCHEMA_VERSION,
            raw_label=str(orec.label),
            canonical_label=lbl,
            taxonomy_id=f"re:{lbl}",
            category=cat,
            bbox_px=bbox,
            coords=CoordinateSystemSpec(
                frame_id="image_px",
                units="pixels",
                image_width=iw,
                image_height=ih,
            ),
            scores=ScoreSemantics(
                raw_score=orec.confidence,
                score_emitted=orec.confidence is not None,
            ),
            provenance=DetectionProvenance(
                source_image_id=img_name,
                execution=ModelExecutionProvenance(
                    engine_id=orec.model_id.engine_id,
                    architecture_family=orec.model_id.architecture_id,
                    checkpoint_id=orec.model_id.checkpoint_id,
                ),
            ),
        ))
    return observations, rec


def run_a5_production_pipeline(
    run_id: str,
    *,
    image_paths: Sequence[Path | str] = (),
    images_rgb: Mapping[str, np.ndarray] | None = None,
    observations: Sequence[NormalizedPerceptionObservation] | None = None,
    precomputed_masks: Mapping[str, np.ndarray] | None = None,
    run_live_detector: bool = False,
    run_live_segmenter: bool = True,
    domain: Domain = Domain.INTERIOR,
    detection_threshold: float = 0.28,
    claude_feedback: ClaudeReconstructionFeedback | None = None,
    depth_contracts: Sequence[DepthMapContract] = (),
    depth_refs_by_image: Mapping[str, str] | None = None,
    work_dir: Path | str | None = None,
    persist_to_world: bool = True,
) -> A5ProductionResult:
    """Execute the full unified A5 perception pipeline with provenance, lifecycle, and handoff."""
    t0 = time.perf_counter()
    lifecycle = ModelLifecycleManager()
    telemetry: list[ModelStageExecutionRecord] = []

    # 1. Image ingestion
    ingested_rgb, ingested_paths = ingest_images_rgb(image_paths) if image_paths else ({}, {})
    rgb_all: dict[str, np.ndarray] = dict(ingested_rgb)
    if images_rgb:
        rgb_all.update(images_rgb)

    # 2. Image quality assessment & 3. Adaptive crops
    img_qualities: dict[str, ImageQualityProfile] = {}
    crop_counts: dict[str, int] = {}
    for img_id, arr in sorted(rgb_all.items()):
        q = analyze_image_quality(arr)
        img_qualities[img_id] = q
        tiles, regions = plan_adaptive_crops(q, ())
        crop_counts[img_id] = len(tiles) + len(regions)

    # 4-5. Engine selection & Detection
    raw_obs: list[NormalizedPerceptionObservation] = list(observations or ())
    if run_live_detector and image_paths:
        live_obs, det_rec = run_live_detection_stage(
            image_paths,
            rgb_all,
            run_id=run_id,
            domain=domain,
            threshold=detection_threshold,
            lifecycle=lifecycle,
        )
        if det_rec is not None:
            telemetry.append(det_rec)
        raw_obs.extend(live_obs)

    # 6. Segmentation (SAM2 / SAM3 / SAM-HQ or precomputed, never fabricated)
    if precomputed_masks is not None or (run_live_segmenter and raw_obs and (rgb_all or ingested_paths)):
        segmented_obs, seg_report = attach_masks_to_observations(
            raw_obs,
            images_rgb=rgb_all,
            image_paths=ingested_paths,
            precomputed_masks=precomputed_masks,
            preferred_backend="sam2",
            local_files_only=True,
        )
    else:
        segmented_obs = raw_obs
        seg_report = SegmentationExecutionReport(
            segmenter_id="sam2",
            checkpoint_id="none",
            status="skipped",
            total_prompts=len(raw_obs),
            masks_produced=0,
            empty_masks=0,
            rejected_masks=0,
            mean_mask_quality=0.0,
            mean_mask_coverage=0.0,
            wall_time_s=0.0,
            peak_vram_mb=None,
            diagnostics=("segmentation_skipped",),
        )

    # 7. Duplicate resolution (A5.2)
    dedup_res = resolve_duplicate_detections_v2(segmented_obs)

    # 8. Architectural & furniture false-positive refinement (A5.4 / A5.5)
    ref_res = refine_scene_observations_v2(dedup_res.canonical_observations, images_rgb=rgb_all)
    clean_obs = list(ref_res.accepted_observations)

    # 9. Cross-view association (A5.3 with appearance + Claude camera poses)
    cameras = claude_feedback.cameras if claude_feedback else None
    id_v2 = resolve_cross_view_identities_v2(clean_obs, images_rgb=rgb_all, cameras=cameras)

    # 10. Architectural understanding
    arch_legacy = analyze_architecture(clean_obs)
    arch_geom: dict[str, ArchitecturalGeometry2Result] = {}
    for img_id, arr in sorted(rgb_all.items()):
        img_obs = [o for o in clean_obs if o.provenance.source_image_id == img_id]
        arch_geom[img_id] = analyze_architectural_geometry_v2(img_id, arr, img_obs)
    arch_rels = reason_architectural_relationships(clean_obs)
    arch_elev = analyze_architectural_elevations(clean_obs)

    # 11. Furniture understanding
    builtin_ids = tuple(bid for g in arch_geom.values() for bid in g.builtin_cabinet_ids)
    furn_legacy = analyze_furniture(clean_obs)
    furn_v2 = analyze_furniture_v2(clean_obs, builtin_ids=builtin_ids)

    # 12. Depth association & Evidence fusion (A4.9)
    depth_results = tuple(associate_observations_with_depth(clean_obs, dc) for dc in depth_contracts)
    overall_grade = (
        "unusable_blank"
        if img_qualities and all(q.is_blank for q in img_qualities.values())
        else ("degraded" if any(q.quality_grade == "degraded" for q in img_qualities.values()) else "good")
    )
    fusion_decisions = tuple(fuse_observations_v2(clean_obs, image_quality_grade=overall_grade))

    # 13. Bundle + Active Refinement + Audit + Diagnostics
    bundle = PerceptionBundle(
        run_id=run_id,
        observations=clean_obs,
        architectural=list(arch_legacy.observations),
        furniture=list(furn_legacy.items),
        relationships=list(arch_rels.relationships),
        identities=list(id_v2.base_report.identities),
    )
    active_rep = run_active_perception_v2(
        clean_obs,
        architectural=arch_legacy.observations,
        identities=id_v2.base_report.identities,
        image_qualities={k: v.to_dict() for k, v in img_qualities.items()},
    )
    audit = run_scene_perception_audit(
        run_id,
        clean_obs,
        elevation_report=arch_elev,
        furniture_report=furn_v2,
        multiview_report=None,
        has_depth=len(depth_results) > 0,
    )
    peak_vram = max(
        [v for v in [seg_report.peak_vram_mb, *(r.peak_vram_mb for r in telemetry)] if v is not None],
        default=None,
    )
    diag = run_production_diagnostics_v2(
        bundle,
        image_qualities={k: v.to_dict() for k, v in img_qualities.items()},
        arch_result=arch_legacy,
        identity_report=id_v2.base_report,
        depth_results=depth_results,
        peak_vram_mb=peak_vram,
    )

    # 14. Claude handoff & EvidenceMatrix persistence
    if persist_to_world and work_dir is not None:
        handoff, claude_ver = execute_claude_handoff_and_persist(
            bundle,
            matrix_root=Path(work_dir) / "evidence_matrix",
            claude_feedback=claude_feedback,
            depth_refs_by_image=depth_refs_by_image,
        )
    else:
        from perception.contracts.claude_handoff_v1 import export_claude_handoff_bundle

        handoff = export_claude_handoff_bundle(
            bundle,
            camera_ids_by_image={k: k for k in (cameras or {}).keys()} if cameras else None,
            depth_refs_by_image=dict(depth_refs_by_image) if depth_refs_by_image else None,
        )
        claude_ver = None

    # Optional resumable stage checkpoint
    if work_dir is not None:
        wdir = Path(work_dir)
        wdir.mkdir(parents=True, exist_ok=True)
        (wdir / "a5_pipeline_checkpoint.json").write_text(
            json.dumps({
                "run_id": run_id,
                "completed_stages": [
                    "ingest", "quality", "detect", "segment", "dedup",
                    "refine", "identity_v2", "architecture", "furniture",
                    "fusion", "handoff", "diagnostics",
                ],
                "observation_count": len(clean_obs),
            }, indent=2),
            encoding="utf-8",
        )

    return A5ProductionResult(
        run_id=run_id,
        bundle=bundle,
        handoff=handoff,
        claude_verification=claude_ver,
        image_qualities=img_qualities,
        crop_counts_by_image=crop_counts,
        segmentation_report=seg_report,
        dedup_result=dedup_res,
        refinement_report=ref_res,
        identity_v2=id_v2,
        arch_legacy=arch_legacy,
        arch_geometry=arch_geom,
        arch_relations=arch_rels,
        arch_elevations=arch_elev,
        furniture_legacy=furn_legacy,
        furniture_v2=furn_v2,
        depth_results=depth_results,
        fusion_decisions=fusion_decisions,
        active_refinement=active_rep,
        scene_audit=audit,
        diagnostics=diag,
        stage_telemetry=tuple(telemetry),
        wall_time_s=round(time.perf_counter() - t0, 4),
    )


__all__ = [
    "A5ProductionResult",
    "ingest_images_rgb",
    "run_a5_production_pipeline",
    "run_live_detection_stage",
]
