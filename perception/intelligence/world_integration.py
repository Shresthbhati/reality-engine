"""Canonical World Pipeline Integration (A3.12).

Connects ``PerceptionBundle`` (normalized architectural, furniture, mask,
material, relationship, and cross-view identity observations) to the canonical
``EvidenceMatrix`` / ``ObservationDraft`` / ``SpatialInputsV1`` pathway without
creating a competing WorldIR or modifying Claude's reconstruction internals.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from engine.multiengine.observation import ObservationDraft
from engine.multiengine.spatial_intelligence.taxonomy import resolve_taxonomy_class
from perception.intelligence.architectural import analyze_architecture
from perception.intelligence.data_model import (
    SCHEMA_VERSION,
    CoordinateSystemSpec,
    DetectionProvenance,
    ModelExecutionProvenance,
    NormalizedPerceptionObservation,
    PerceptionBundle,
    ScoreSemantics,
    stable_observation_id,
)
from perception.intelligence.diagnostics import run_perception_diagnostics
from perception.intelligence.furniture import analyze_furniture
from perception.intelligence.identity import resolve_cross_view_identities
from perception.intelligence.refinement import plan_active_refinement
from perception.intelligence.relationships import infer_scene_relationships


def observation_from_bbox_draft(
    draft: ObservationDraft,
    *,
    image_width: int,
    image_height: int,
    architecture_family: str = "grounding-dino",
    checkpoint_id: str = "unknown",
) -> NormalizedPerceptionObservation:
    """Convert an ``ObservationDraft(quantity='bbox')`` into ``NormalizedPerceptionObservation``."""
    pred = dict(draft.predicted or {})
    raw_lbl = str(pred.get("raw_label") or pred.get("label") or "object")
    tax = resolve_taxonomy_class(str(pred.get("label") or raw_lbl))
    box = tuple(float(v) for v in (draft.image_region or (0.0, 0.0, 1.0, 1.0)))
    emitted = bool(pred.get("confidence_emitted", draft.confidence is not None))
    raw_score = float(pred["raw_score"]) if pred.get("raw_score") is not None else (float(draft.confidence) if (emitted and draft.confidence is not None) else None)
    cal_prob = float(pred["calibrated_confidence"]) if pred.get("calibrated_confidence") is not None else None
    is_cal = cal_prob is not None
    engine_id = str(pred.get("inner_engine") or "detection_pipeline")
    oid = stable_observation_id("perc", draft.source_id, engine_id, tax.name, box)

    coords = CoordinateSystemSpec(
        frame_id=draft.coordinate_system or f"image:{draft.source_id}",
        units="pixels",
        image_width=image_width,
        image_height=image_height,
    )
    scores = ScoreSemantics(
        raw_score=raw_score,
        score_emitted=emitted,
        calibrated_probability=cal_prob,
        is_calibrated=is_cal,
        uncertainty=pred.get("detection_uncertainty"),
        calibration_note=str(pred.get("calibration_status") or ("calibrated" if is_cal else "uncalibrated")),
    )
    prov = DetectionProvenance(
        source_image_id=draft.source_id,
        execution=ModelExecutionProvenance(
            engine_id=engine_id,
            architecture_family=architecture_family,
            checkpoint_id=checkpoint_id,
        ),
        inference_mode="tiled" if "tiled" in engine_id else ("host_crop" if "subpart" in engine_id else "full_frame"),
    )
    return NormalizedPerceptionObservation(
        observation_id=oid,
        schema_version=SCHEMA_VERSION,
        raw_label=raw_lbl,
        canonical_label=tax.name,
        taxonomy_id=tax.taxonomy_id,
        category=tax.category,
        bbox_px=box,
        coords=coords,
        scores=scores,
        provenance=prov,
        parent_observation_id=pred.get("parent_pipeline_box_id"),
        child_observation_ids=tuple(pred.get("child_box_ids", ())),
        is_partial=bool(pred.get("is_truncated", False)) or str(pred.get("occlusion_level", "none")) != "none",
        is_truncated=bool(pred.get("is_truncated", False)),
        occlusion_level=str(pred.get("occlusion_level", "none")),
    )


def build_intelligence_bundle(
    run_id: str,
    observations: Sequence[NormalizedPerceptionObservation],
    *,
    image_qualities: dict[str, dict[str, Any]] | None = None,
    failed_engines: dict[str, str] | None = None,
) -> PerceptionBundle:
    """Run architectural, furniture, identity, relationship, refinement, and diagnostic passes."""
    arch = analyze_architecture(observations)
    furn = analyze_furniture(observations)
    ident = resolve_cross_view_identities(observations)
    rels = infer_scene_relationships(observations)
    ref = plan_active_refinement(
        observations,
        architectural=arch.observations,
        identities=ident.identities,
    )

    bundle = PerceptionBundle(
        run_id=run_id,
        observations=list(observations),
        architectural=list(arch.observations),
        furniture=list(furn.items),
        relationships=list(rels.relationships),
        identities=list(ident.identities),
    )
    diag = run_perception_diagnostics(
        bundle,
        image_qualities=image_qualities,
        arch_result=arch,
        identity_report=ident,
        failed_engines=failed_engines,
    )
    bundle.diagnostics = {
        "diagnostic_report": diag.to_dict(),
        "architectural_summary": arch.summary,
        "furniture_summary": furn.summary,
        "identity_summary": ident.summary,
        "relationship_summary": rels.summary,
        "refinement_plan": ref.summary,
    }
    return bundle


def to_evidence_matrix_drafts(bundle: PerceptionBundle) -> list[ObservationDraft]:
    """Emit canonical ``ObservationDraft`` records from a ``PerceptionBundle`` for EvidenceMatrix."""
    drafts: list[ObservationDraft] = []
    for o in bundle.observations:
        drafts.append(ObservationDraft(
            task="detection",
            quantity="bbox",
            source_id=o.provenance.source_image_id,
            coordinate_system=o.coords.frame_id,
            image_region=o.bbox_px,
            predicted={
                "observation_id": o.observation_id,
                "schema_version": o.schema_version,
                "label": o.canonical_label,
                "raw_label": o.raw_label,
                "taxonomy_id": o.taxonomy_id,
                "category": o.category,
                "raw_score": o.scores.raw_score,
                "calibrated_confidence": o.scores.calibrated_probability,
                "is_calibrated": o.scores.is_calibrated,
                "inner_engine": o.provenance.execution.engine_id,
                "is_partial": o.is_partial,
                "is_truncated": o.is_truncated,
                "occlusion_level": o.occlusion_level,
                "competing_labels": list(o.competing_labels),
                "material_hypotheses": [m.to_dict() for m in o.material_hypotheses],
            },
            original_coordinates={
                "box_xyxy_px": list(o.bbox_px),
                "image_size_px": [o.coords.image_width, o.coords.image_height],
            },
            confidence=o.scores.raw_score,
        ))
    return drafts


__all__ = [
    "build_intelligence_bundle",
    "observation_from_bbox_draft",
    "to_evidence_matrix_drafts",
]
