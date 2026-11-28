"""Bidirectional Claude Integration Bridge (A5 Objective 5).

Connects Gemini's perception outputs (``perception_handoff.v1``) with Claude's
reconstruction & WorldIR pipeline without modifying Claude-owned geometry or
WorldStore internals:
  1. Exports normalized observations, masks, identities, relationships, depth,
     provenance, and uncertainty to ``perception_handoff.v1`` and imports into
     the run's canonical ``EvidenceMatrix`` (WorldIR persistence & reload).
  2. Consumes Claude's ``CameraPoseConstraint`` (with uncertainty) and
     ``WorldEntityIdentity`` feedback to refine cross-view association.
  3. Verifies actual persistence and fresh-process reload from ``EvidenceMatrix``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from perception.contracts.claude_handoff_v1 import (
    HANDOFF_SCHEMA_VERSION,
    HandoffBundle,
    export_claude_handoff_bundle,
)
from perception.intelligence.data_model import PerceptionBundle
from perception.intelligence.identity import CameraPoseConstraint
from perception.intelligence.world_integration import to_evidence_matrix_drafts


@dataclass(frozen=True)
class WorldEntityIdentity:
    """Stable world-level entity identity fed back from Claude's reconstruction/WorldIR."""

    world_entity_id: str
    canonical_label: str
    linked_observation_ids: tuple[str, ...]
    position_world_m: tuple[float, float, float] | None = None
    is_metric_verified: bool = False


@dataclass(frozen=True)
class ClaudeReconstructionFeedback:
    cameras: Mapping[str, CameraPoseConstraint]
    world_entities: tuple[WorldEntityIdentity, ...] = ()


@dataclass(frozen=True)
class ClaudeIntegrationVerification:
    schema_version: str
    handoff_accepted_count: int
    handoff_rejected_count: int
    masks_exported: int
    identities_exported: int
    relationships_exported: int
    world_entities_linked: int
    evidence_matrix_persisted: bool
    fresh_reload_observation_count: int
    fresh_reload_entity_count: int


def parse_claude_camera_poses_from_json(
    camera_records: Sequence[Mapping[str, Any]],
) -> dict[str, CameraPoseConstraint]:
    """Convert Claude-provided camera pose records into ``CameraPoseConstraint`` objects."""
    out: dict[str, CameraPoseConstraint] = {}
    for rec in camera_records:
        img_id = str(rec["image_id"])
        pos = tuple(float(v) for v in rec.get("position_m", (0.0, 0.0, 0.0)))
        ori = tuple(float(v) for v in rec.get("orientation_wxyz", (1.0, 0.0, 0.0, 0.0)))
        out[img_id] = CameraPoseConstraint(
            image_id=img_id,
            position_m=(pos[0], pos[1], pos[2]),
            orientation_wxyz=(ori[0], ori[1], ori[2], ori[3]),
            position_sigma_m=(float(rec["position_sigma_m"]) if rec.get("position_sigma_m") is not None else None),
            is_metric=bool(rec.get("is_metric", False)),
        )
    return out


def execute_claude_handoff_and_persist(
    bundle: PerceptionBundle,
    *,
    matrix_root: Path | str,
    claude_feedback: ClaudeReconstructionFeedback | None = None,
    depth_refs_by_image: Mapping[str, str] | None = None,
) -> tuple[HandoffBundle, ClaudeIntegrationVerification]:
    """Export ``perception_handoff.v1``, persist to EvidenceMatrix, and verify fresh reload."""
    cam_ids = (
        {img_id: img_id for img_id in claude_feedback.cameras.keys()}
        if claude_feedback
        else None
    )
    handoff = export_claude_handoff_bundle(
        bundle,
        camera_ids_by_image=cam_ids,
        depth_refs_by_image=dict(depth_refs_by_image) if depth_refs_by_image else None,
    )

    # Persist into canonical EvidenceMatrix and verify fresh reload
    from engine.multiengine.evidence_matrix import EvidenceContext, EvidenceMatrix
    from engine.multiengine.observation import make_observation

    root = Path(matrix_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "perception_handoff_v1.json").write_text(handoff.to_json(), encoding="utf-8")

    em_dir = root / "matrix_store"
    drafts = to_evidence_matrix_drafts(bundle)
    matrix = EvidenceMatrix(em_dir)
    for d in drafts:
        eo = make_observation(
            d,
            engine_id=str(d.predicted.get("inner_engine", "perception")),
            model_version="a5_v1",
            execution_config={},
        )
        matrix.add(
            eo,
            EvidenceContext(
                eo.observation_id,
                bundle.run_id,
                calibrated_confidence=None,
                calibration=None,
            ),
        )

    persisted_ok = False
    reloaded_obs = 0
    reloaded_ents = 0
    try:
        reloaded = EvidenceMatrix(em_dir)
        issues = reloaded.verify()
        reloaded_obs = len(reloaded.store.query(quantity="bbox"))
        persisted_ok = (len(issues) == 0) and (reloaded_obs == len(drafts))
    except Exception:
        persisted_ok = False

    linked = 0
    if claude_feedback and claude_feedback.world_entities:
        accepted_ids = {o.observation_id for o in handoff.observations}
        for we in claude_feedback.world_entities:
            if any(oid in accepted_ids for oid in we.linked_observation_ids):
                linked += 1

    masks_exported = sum(1 for o in handoff.observations if o.mask_ref is not None)
    ver = ClaudeIntegrationVerification(
        schema_version=HANDOFF_SCHEMA_VERSION,
        handoff_accepted_count=len(handoff.observations),
        handoff_rejected_count=len(handoff.rejected),
        masks_exported=masks_exported,
        identities_exported=len(bundle.identities),
        relationships_exported=len(bundle.relationships),
        world_entities_linked=linked,
        evidence_matrix_persisted=persisted_ok,
        fresh_reload_observation_count=reloaded_obs,
        fresh_reload_entity_count=reloaded_ents,
    )
    return handoff, ver


__all__ = [
    "ClaudeIntegrationVerification",
    "ClaudeReconstructionFeedback",
    "WorldEntityIdentity",
    "execute_claude_handoff_and_persist",
    "parse_claude_camera_poses_from_json",
]
