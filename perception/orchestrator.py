"""Unified Perception Inference Orchestrator (Programme A3).

Executes an explicit ``InferencePlan`` across depth, detection, segmentation,
embeddings, image quality, scene intelligence, material hypotheses, and
``spatial_inputs.v1`` generation.

Invariants:
  - Never silently skips or substitutes a REQUIRED model.
  - Records exact model availability, plan configuration, and per-stage telemetry.
  - Emits a validated ``SpatialInputsV1`` contract ready for Claude's consumer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

from perception.availability import real_models_by_name
from perception.detection.grounded_arch import FULL_ARCHITECTURAL_VOCABULARY
from perception.scene_intelligence import (
    ARCHITECTURAL_QUERIES,
    VisualSceneGraph,
    run_sequential_perception_pipeline,
)
from perception.spatial_inputs import SpatialInputsV1, get_inputs_provider


class InferencePlanError(RuntimeError):
    """Raised when a required model in an InferencePlan is unavailable or fails."""


@dataclass(frozen=True)
class InferencePlan:
    plan_name: str
    required_models: Tuple[str, ...]
    optional_models: Tuple[str, ...] = ()
    queries: Tuple[str, ...] = FULL_ARCHITECTURAL_VOCABULARY[:20]
    max_edge: int | None = 640
    use_owlvit: bool = True
    use_grounding_dino: bool = True
    use_florence2: bool = False
    use_sam2: bool = True
    use_dinov2: bool = True
    extract_materials: bool = True
    reason: str = "Architectural & object perception for spatial reconstruction"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_ARCHITECTURAL_PLAN = InferencePlan(
    plan_name="architectural_standard_v1",
    required_models=("depth_anything", "mask_rcnn"),
    optional_models=("owlvit", "grounding_dino", "sam2_hf", "dinov2"),
    queries=FULL_ARCHITECTURAL_VOCABULARY[:20],
    max_edge=640,
    use_owlvit=True,
    use_grounding_dino=True,
    use_florence2=False,
    use_sam2=True,
    use_dinov2=True,
    extract_materials=True,
)

FAST_STRUCTURAL_PLAN = InferencePlan(
    plan_name="fast_structural_v1",
    required_models=("depth_anything", "mask_rcnn"),
    optional_models=("grounding_dino",),
    queries=FULL_ARCHITECTURAL_VOCABULARY[:13],
    max_edge=640,
    use_owlvit=False,
    use_grounding_dino=True,
    use_florence2=False,
    use_sam2=False,
    use_dinov2=False,
    extract_materials=False,
    reason="Fast 2/3-model structural pass for rapid scene triage",
)

BALANCED_INTERIOR_PLAN = InferencePlan(
    plan_name="balanced_interior_v1",
    required_models=("depth_anything", "mask_rcnn"),
    optional_models=("grounding_dino", "sam2_hf", "dinov2"),
    queries=FULL_ARCHITECTURAL_VOCABULARY[:20],
    max_edge=640,
    use_owlvit=False,
    use_grounding_dino=True,
    use_florence2=False,
    use_sam2=True,
    use_dinov2=True,
    extract_materials=True,
    reason="Balanced 4/5-model interior reconstruction evidence pass",
)

FULL_FORENSIC_PLAN = InferencePlan(
    plan_name="full_forensic_v1",
    required_models=("depth_anything", "mask_rcnn"),
    optional_models=("owlvit", "grounding_dino", "florence2", "sam2_hf", "dinov2"),
    queries=FULL_ARCHITECTURAL_VOCABULARY[:20],
    max_edge=1024,
    use_owlvit=True,
    use_grounding_dino=True,
    use_florence2=True,
    use_sam2=True,
    use_dinov2=True,
    extract_materials=True,
    reason="Maximum-coverage 7-model forensic pass at 1024px",
)


@dataclass(frozen=True)
class OrchestratedPerceptionResult:
    run_id: str
    plan: InferencePlan
    scene_graph: VisualSceneGraph
    spatial_inputs: SpatialInputsV1
    cross_view_associations: Tuple[Dict[str, Any], ...]
    segmentation_summary: Dict[str, Any]
    telemetry: Dict[str, Any]
    skipped_optional_models: Tuple[str, ...] = ()


def run_inference_plan(
    image_paths: Sequence[str],
    *,
    run_id: str = "perc_run",
    plan: InferencePlan = DEFAULT_ARCHITECTURAL_PLAN,
    cameras: Mapping[str, Any] | None = None,
    storey_id: str | None = None,
) -> OrchestratedPerceptionResult:
    """Validate required models in `plan`, run sequential perception, and emit `SpatialInputsV1`."""
    avail = real_models_by_name()
    missing_req = [
        m for m in plan.required_models
        if m not in avail or avail[m].status != "available"
    ]
    if missing_req:
        details = {m: (avail[m].reason if m in avail else "unknown model") for m in missing_req}
        raise InferencePlanError(f"Required model(s) unavailable for plan {plan.plan_name!r}: {details}")

    skipped_opt = tuple(
        m for m in plan.optional_models
        if m not in avail or avail[m].status != "available"
    )

    out = run_sequential_perception_pipeline(
        image_paths,
        use_owlvit=plan.use_owlvit and ("owlvit" not in skipped_opt),
        use_grounding_dino=plan.use_grounding_dino and ("grounding_dino" not in skipped_opt),
        use_florence2=plan.use_florence2 and ("florence2" not in skipped_opt),
        use_sam2=plan.use_sam2 and ("sam2_hf" not in skipped_opt),
        use_dinov2=plan.use_dinov2 and ("dinov2" not in skipped_opt),
        queries=plan.queries or ARCHITECTURAL_QUERIES,
        max_edge=plan.max_edge,
        extract_materials=plan.extract_materials,
    )
    sg: VisualSceneGraph = out["scene_graph"]
    assocs = tuple(out.get("cross_view_associations") or ())
    provider = get_inputs_provider("perception_scene_intelligence")
    si_v1 = provider(
        run_id,
        sg,
        cameras=cameras,
        cross_view_associations=assocs,
        storey_id=storey_id,
    )
    return OrchestratedPerceptionResult(
        run_id=run_id,
        plan=plan,
        scene_graph=sg,
        spatial_inputs=si_v1,
        cross_view_associations=assocs,
        segmentation_summary=dict(out.get("segmentation_summary") or {}),
        telemetry={**out["telemetry"], "plan_name": plan.plan_name, "skipped_optional_models": list(skipped_opt)},
        skipped_optional_models=skipped_opt,
    )


def select_inference_plan_for_inputs(
    image_paths: Sequence[str],
    *,
    has_video: bool = False,
    vram_mb: Optional[float] = None,
    forensic: bool = False,
) -> InferencePlan:
    """Select an appropriate InferencePlan based on batch size, video presence, and GPU VRAM."""
    if forensic:
        return FULL_FORENSIC_PLAN
    if vram_mb is None:
        try:
            import torch

            if torch.cuda.is_available():
                _, total_b = torch.cuda.mem_get_info()
                vram_mb = total_b / (1024.0 * 1024.0)
        except Exception:  # noqa: BLE001
            vram_mb = None
    if (vram_mb is not None and vram_mb < 4500.0) or len(image_paths) > 12 or has_video and len(image_paths) > 8:
        return FAST_STRUCTURAL_PLAN
    return BALANCED_INTERIOR_PLAN
