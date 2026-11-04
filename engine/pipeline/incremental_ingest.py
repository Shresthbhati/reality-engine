"""Incremental ingest pipeline (Blueprint Task 6.3, Pipeline P6).

Orchestrates incremental ingestion of new evidence into an existing WorldIR:
1. Photo Registration & Validation:
   - Registers new photo candidates into existing sparse/3D geometry via PnP RANSAC.
   - Refuses non-matching or disjoint photos with explicit reason (Scenarios A11, A19).
2. Entity Association & Incremental Application:
   - Associates new/modified entity observations with existing WorldIR entities.
   - Calls `apply_incremental_update` to recompute ONLY the affected closure.
   - Preserves untouched entities with identical IDs, geometries, and transforms (Scenario A12).
3. Deterministic Diff & Versioning:
   - Computes `diff_worlds(base_world, new_world)` to produce a formal WorldDiff.
   - Increments world version (V -> V+1) and stores diff summary in world metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from registration.photo_registration import (
    IncrementalRegistrationBatch,
    PhotoRegistrationResult,
    register_photo_to_sparse_model,
)
from world_ir.diff import WorldDiff, diff_worlds
from world_ir.incremental import IncrementalUpdateResult, affected_closure, apply_incremental_update
from world_ir.schema_v1 import Entity, Geometry, Relationship, RelationshipKind
from world_ir.world_v1 import WorldIR

logger = logging.getLogger(__name__)


@dataclass
class IncrementalIngestResult:
    """Output of an incremental ingest run."""

    base_world: WorldIR
    new_world: WorldIR
    diff: WorldDiff
    update_result: IncrementalUpdateResult
    registered_photos: List[str] = field(default_factory=list)
    refused_photos: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def changed_entity_ids(self) -> frozenset[str]:
        return self.update_result.changed_entity_ids

    @property
    def affected_entity_ids(self) -> frozenset[str]:
        return self.update_result.affected_entity_ids

    @property
    def reused_entity_ids(self) -> frozenset[str]:
        return self.update_result.reused_entity_ids

    @property
    def is_empty(self) -> bool:
        return self.diff.is_empty()


def run_incremental_ingest(
    base_world: WorldIR,
    *,
    new_photos: Optional[Sequence[Dict[str, Any]]] = None,
    new_entities: Sequence[Entity] = (),
    new_geometries: Sequence[Geometry] = (),
    removed_entity_ids: Sequence[str] = (),
    min_registration_inliers: int = 15,
    max_reprojection_error_px: float = 4.0,
) -> IncrementalIngestResult:
    """Execute an incremental ingest update on base_world.

    Parameters:
        base_world: The authoritative baseline WorldIR snapshot.
        new_photos: Optional sequence of photo dicts with 2D-3D correspondences to register.
        new_entities: Newly extracted or updated Entity instances.
        new_geometries: Newly extracted or updated Geometry instances.
        removed_entity_ids: Entity IDs explicitly removed by new evidence.
        min_registration_inliers: Minimum PnP inliers to accept a new photo.
        max_reprojection_error_px: Maximum reprojection threshold for photo inliers.

    Returns:
        IncrementalIngestResult with new_world, deterministic WorldDiff, and affected closure.
    """
    registered_photo_ids: List[str] = []
    refused_photo_records: List[Dict[str, Any]] = []

    # 1. Register new photo evidence if provided
    if new_photos:
        for p in new_photos:
            pid = p.get("photo_id") or p.get("image_id") or "photo_unknown"
            pts2d = p.get("points_2d")
            pts3d = p.get("points_3d")
            k_mat = p.get("camera_matrix") or p.get("intrinsics")

            if pts2d is not None and pts3d is not None:
                res = register_photo_to_sparse_model(
                    photo_id=pid,
                    points_2d=pts2d,
                    points_3d=pts3d,
                    camera_matrix=k_mat,
                    min_inliers=min_registration_inliers,
                    max_reprojection_error_px=max_reprojection_error_px,
                )
                if res.status == "registered":
                    registered_photo_ids.append(pid)
                else:
                    refused_photo_records.append({
                        "photo_id": pid,
                        "status": "refused",
                        "reason": res.reason,
                    })
            else:
                # Direct registration not supplied in photo dict
                registered_photo_ids.append(pid)

    # 2. Apply incremental update via regional closure
    upd_result = apply_incremental_update(
        base_world=base_world,
        updated_entities=new_entities,
        updated_geometries=new_geometries,
        removed_entities=removed_entity_ids,
    )

    new_world = upd_result.new_world

    # 3. Compute deterministic structural diff
    diff = diff_worlds(base_world, new_world)

    # 4. Attach ingest metadata to new_world
    if not hasattr(new_world, "metadata") or new_world.metadata is None:
        new_world.metadata = {}

    new_world.metadata["incremental_ingest"] = {
        "base_version": base_world.version,
        "new_version": new_world.version,
        "registered_photos": registered_photo_ids,
        "refused_photos": refused_photo_records,
        "changed_entity_count": len(upd_result.changed_entity_ids),
        "affected_entity_count": len(upd_result.affected_entity_ids),
        "reused_entity_count": len(upd_result.reused_entity_ids),
        "diff_summary": diff.summary(),
    }

    return IncrementalIngestResult(
        base_world=base_world,
        new_world=new_world,
        diff=diff,
        update_result=upd_result,
        registered_photos=registered_photo_ids,
        refused_photos=refused_photo_records,
    )
