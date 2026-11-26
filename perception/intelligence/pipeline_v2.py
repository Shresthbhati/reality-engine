"""A4 Perception 2.0 End-to-End Orchestrator (A4.3 - A4.13).

Composes all A4 subsystems into a single clean, traceable entry point:
  1. Image quality & adaptive analysis
  2. Architectural geometry 2.0 & relationship reasoning
  3. Furniture 2.0 (mobility, arrangements, support, occlusions)
  4. Multi-view identity & continuity (with optional Claude camera poses)
  5. Depth-aware association (with strict scale-state honesty)
  6. Explainable Fusion 2.0
  7. Active Perception 2.0
  8. Claude Handoff Contract (``perception_handoff.v1``)
  9. Production Diagnostics 2.0
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from perception.contracts.claude_handoff_v1 import HandoffBundle, export_claude_handoff_bundle
from perception.intelligence.active_v2 import ActivePerceptionReport, run_active_perception_v2
from perception.intelligence.architectural import analyze_architecture
from perception.intelligence.architectural_relations import (
    ArchitecturalRelationGraph,
    reason_architectural_relationships,
)
from perception.intelligence.architectural_v2 import (
    ArchitecturalGeometry2Result,
    analyze_architectural_geometry_v2,
)
from perception.intelligence.data_model import (
    NormalizedPerceptionObservation,
    PerceptionBundle,
)
from perception.intelligence.depth_aware import (
    DepthAwarePerceptionResult,
    DepthMapContract,
    associate_observations_with_depth,
)
from perception.intelligence.diagnostics import DiagnosticReport
from perception.intelligence.diagnostics_v2 import run_production_diagnostics_v2
from perception.intelligence.furniture import analyze_furniture
from perception.intelligence.furniture_v2 import Furniture2AnalysisResult, analyze_furniture_v2
from perception.intelligence.fusion_v2 import ExplainableFusionDecision, fuse_observations_v2
from perception.intelligence.identity import CameraPoseConstraint
from perception.intelligence.image_adaptive import ImageQualityProfile, analyze_image_quality
from perception.intelligence.multiview import MultiViewPerceptionReport, run_multiview_perception
from perception.intelligence.relationships import infer_scene_relationships


@dataclass(frozen=True)
class A4PerceptionPipelineResult:
    run_id: str
    bundle: PerceptionBundle
    handoff: HandoffBundle
    image_qualities: dict[str, ImageQualityProfile]
    arch_geometry: dict[str, ArchitecturalGeometry2Result]
    arch_relations: ArchitecturalRelationGraph
    furniture_v2: Furniture2AnalysisResult
    multiview: MultiViewPerceptionReport
    depth_results: tuple[DepthAwarePerceptionResult, ...]
    fusion_decisions: tuple[ExplainableFusionDecision, ...]
    active_refinement: ActivePerceptionReport
    diagnostics: DiagnosticReport

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "observation_count": len(self.bundle.observations),
            "handoff_accepted": len(self.handoff.observations),
            "handoff_rejected": len(self.handoff.rejected),
            "image_qualities": {k: v.to_dict() for k, v in self.image_qualities.items()},
            "arch_geometry": {k: v.to_dict() for k, v in self.arch_geometry.items()},
            "arch_relations": self.arch_relations.summary,
            "furniture_v2": self.furniture_v2.summary,
            "multiview": self.multiview.summary,
            "depth_images_processed": len(self.depth_results),
            "fusion_clusters": len(self.fusion_decisions),
            "active_next_step": self.active_refinement.recommended_next_step,
            "diagnostics": self.diagnostics.to_dict(),
        }


def run_a4_perception_pipeline(
    run_id: str,
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    images_rgb: Mapping[str, np.ndarray] | None = None,
    cameras: Mapping[str, CameraPoseConstraint] | None = None,
    depth_contracts: Sequence[DepthMapContract] = (),
    camera_ids_by_image: dict[str, str] | None = None,
    depth_refs_by_image: dict[str, str] | None = None,
    failed_engines: Mapping[str, str] | None = None,
    timed_out_engines: Sequence[str] = (),
    peak_vram_mb: float | None = None,
) -> A4PerceptionPipelineResult:
    """Execute the complete A4 perception intelligence pipeline."""
    imgs = images_rgb or {}
    qualities: dict[str, ImageQualityProfile] = {
        img_id: analyze_image_quality(arr) for img_id, arr in sorted(imgs.items())
    }
    quality_dicts = {k: v.to_dict() for k, v in qualities.items()}

    arch = analyze_architecture(observations)
    arch_rels = reason_architectural_relationships(observations)
    arch_geom: dict[str, ArchitecturalGeometry2Result] = {}
    builtin_ids: list[str] = []
    for img_id, arr in sorted(imgs.items()):
        img_obs = [o for o in observations if o.provenance.source_image_id == img_id]
        ag = analyze_architectural_geometry_v2(img_id, arr, img_obs)
        arch_geom[img_id] = ag
        builtin_ids.extend(ag.builtin_cabinet_ids)

    furn = analyze_furniture(observations)
    furn_v2 = analyze_furniture_v2(observations, builtin_ids=builtin_ids)
    mv = run_multiview_perception(observations, cameras=cameras)
    rels = infer_scene_relationships(observations)

    depth_res: list[DepthAwarePerceptionResult] = []
    for dc in depth_contracts:
        depth_res.append(associate_observations_with_depth(observations, dc))

    # Fusion 2.0 per image quality grade
    worst_grade = "good"
    if any(q.quality_grade == "unusable_blank" for q in qualities.values()):
        worst_grade = "unusable_blank"
    elif any(q.quality_grade == "degraded" for q in qualities.values()):
        worst_grade = "degraded"
    fusion_v2 = tuple(fuse_observations_v2(observations, image_quality_grade=worst_grade))

    active = run_active_perception_v2(
        observations,
        architectural=arch.observations,
        identities=mv.identity_report.identities,
        image_qualities=quality_dicts,
    )

    bundle = PerceptionBundle(
        run_id=run_id,
        observations=list(observations),
        architectural=list(arch.observations),
        furniture=list(furn.items),
        relationships=list(rels.relationships) + list(arch_rels.relationships),
        identities=list(mv.identity_report.identities),
    )

    handoff = export_claude_handoff_bundle(
        bundle,
        camera_ids_by_image=camera_ids_by_image,
        depth_refs_by_image=depth_refs_by_image,
    )

    diag = run_production_diagnostics_v2(
        bundle,
        image_qualities=quality_dicts,
        arch_result=arch,
        identity_report=mv.identity_report,
        depth_results=depth_res,
        failed_engines=failed_engines,
        timed_out_engines=timed_out_engines,
        peak_vram_mb=peak_vram_mb,
    )

    return A4PerceptionPipelineResult(
        run_id=run_id,
        bundle=bundle,
        handoff=handoff,
        image_qualities=qualities,
        arch_geometry=arch_geom,
        arch_relations=arch_rels,
        furniture_v2=furn_v2,
        multiview=mv,
        depth_results=tuple(depth_res),
        fusion_decisions=fusion_v2,
        active_refinement=active,
        diagnostics=diag,
    )


__all__ = [
    "A4PerceptionPipelineResult",
    "run_a4_perception_pipeline",
]
