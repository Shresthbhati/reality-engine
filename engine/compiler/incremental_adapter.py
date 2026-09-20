"""Reconstruction to WorldOS Incremental Update Adapter (Goal Phase 2).

Bridges FreeBuff canonical reconstruction outputs into Claude's
`apply_incremental_update` primitive without hiding semantic incompatibilities.

Contracts bridged:
- FreeBuff: `ReconstructionResult` (points with per-point Uncertainty, camera poses, registration status)
            + optional `SessionAlignment` / `RigidTransform`
- Claude WorldOS: `apply_incremental_update(base_world, updated_entities, updated_geometries)`

Preserves:
- Entity identity: Updates target entity or establishes deterministic new entity
- Geometry identity: Creates/updates content-addressed PointCloud geometry with valid bounds
- Evidence lineage: Attaches observations tracing to raw evidence IDs
- Session identity: Records `source_session_id` on entity custom properties and observations
- Provenance: RECONSTRUCTED (never falsely upgraded to OBSERVED)
- Uncertainty: Propagates point cloud confidences without inventing certainty
- Coordinate frame: Explicitly applies cross-session rigid registration transform into WORLD frame
- Refusals: Loudly rejects failed reconstructions, unaligned sessions, and malformed inputs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from engine.math import Vec3
from provenance import Provenance, Uncertainty
from reconstruction.backend.interface import (
    ReconstructedCameraPose,
    ReconstructedPoint,
    ReconstructionResult,
)
from reconstruction.calibration.transforms import RigidTransform
from registration.cross_session import SessionAlignment
from world_ir.artifact_store import ArtifactStore
from world_ir.coordinates import Frame, Transform
from world_ir.geometry_data import PointCloudData
from world_ir.incremental import IncrementalUpdateResult, apply_incremental_update
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Observation,
    Vector3,
)
from world_ir.world_v1 import WorldIR


class ReconstructionAdapterError(ValueError):
    """Raised when reconstruction result or registration cannot be adapted."""
    pass


@dataclass(frozen=True)
class IncrementalUpdatePackage:
    """The normalized, validated output of the adapter ready for WorldOS."""
    updated_entities: List[Entity]
    updated_geometries: List[Geometry]
    target_entity_id: str
    target_geometry_id: str
    source_session_id: str
    source_evidence_ids: List[str]
    point_count: int
    mean_confidence: float
    bounds_min: Vector3
    bounds_max: Vector3


def adapt_reconstruction_to_incremental_update(
    base_world: WorldIR,
    reconstruction: Union[ReconstructionResult, Any],
    *,
    session_id: str,
    target_entity_id: str,
    alignment: Optional[Union[SessionAlignment, RigidTransform, Transform]] = None,
    target_entity_type: Optional[EntityType] = None,
    target_entity_name: Optional[str] = None,
    artifact_store: Optional[ArtifactStore] = None,
    timestamp: Optional[float] = None,
) -> IncrementalUpdatePackage:
    """Adapts a FreeBuff `ReconstructionResult` or `ReconstructionContract` into updated entities/geometries.

    Args:
        base_world: The current canonical world state (World V1).
        reconstruction: The output of a reconstruction pass (ReconstructionResult or ReconstructionContract).
        session_id: Identity of the capture session.
        target_entity_id: ID of entity being updated or created.
        alignment: Optional rigid registration transform from session to world.
        target_entity_type: Entity classification (defaults to existing entity type or UNKNOWN).
        target_entity_name: Human-readable entity name.
        artifact_store: Content-addressed store for storing point cloud binary payloads.
        timestamp: Monotonic acquisition timestamp.

    Returns:
        IncrementalUpdatePackage with entities and geometries ready for `apply_incremental_update`.
    """
    # 0. Canonical ReconstructionContract Support (P1)
    contract_meta: Optional[Dict[str, Any]] = None
    if hasattr(reconstruction, "contract_version") and hasattr(reconstruction, "status"):
        if reconstruction.status == "failed" or reconstruction.result is None:
            detail = getattr(reconstruction, "detail", "status is failed")
            raise ReconstructionAdapterError(
                f"Cannot adapt failed ReconstructionContract for session '{session_id}': {detail}"
            )
        contract_meta = reconstruction.to_dict() if hasattr(reconstruction, "to_dict") else {
            "contract_version": reconstruction.contract_version,
            "status": reconstruction.status,
            "diagnostics": getattr(reconstruction, "diagnostics", {}),
        }
        reconstruction = reconstruction.result

    # 1. Input Gate: Refuse failed or empty reconstructions
    if reconstruction.registration_status == "failed":
        raise ReconstructionAdapterError(
            f"Cannot adapt failed reconstruction for session '{session_id}'"
        )
    if not reconstruction.points:
        raise ReconstructionAdapterError(
            f"Empty reconstruction points for session '{session_id}'"
        )
    if not reconstruction.camera_poses:
        raise ReconstructionAdapterError(
            f"Cannot adapt reconstruction without camera poses: no valid sensor trace"
        )

    # 2. Registration / Coordinate Frame Handling
    if isinstance(alignment, SessionAlignment):
        if alignment.status == "refused":
            raise ReconstructionAdapterError(
                f"Registration refused between sessions '{alignment.from_session}' "
                f"and '{alignment.to_session}': {alignment.reason}"
            )
        transform = alignment.transform
    elif isinstance(alignment, (RigidTransform, Transform)):
        transform = alignment
    elif alignment is None:
        transform = None
    else:
        raise TypeError(f"Unsupported alignment type: {type(alignment)}")

    expected_frame = base_world.coordinate_frame.value if hasattr(base_world.coordinate_frame, "value") else str(base_world.coordinate_frame)
    valid_target_frames = {expected_frame, "world"}
    if isinstance(alignment, SessionAlignment):
        if alignment.from_session and alignment.from_session != session_id:
            raise ReconstructionAdapterError(
                f"Coordinate frame mismatch: alignment from '{alignment.from_session}' does not match session '{session_id}'"
            )
        valid_target_frames.add(alignment.to_session)

    if isinstance(transform, RigidTransform) and transform.to_frame:
        target_frame = transform.to_frame.value if hasattr(transform.to_frame, "value") else str(transform.to_frame)
        if target_frame not in valid_target_frames:
            raise ReconstructionAdapterError(
                f"Coordinate frame mismatch: alignment targets '{target_frame}', but world is in '{expected_frame}'"
            )

    # 3. Transform Points into Base World Frame
    transformed_points: List[Tuple[float, float, float]] = []
    confidences: List[float] = []
    source_evidence_ids_set = set()

    for pt in reconstruction.points:
        px, py, pz = pt.position
        if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
            raise ReconstructionAdapterError(
                f"Malformed point coordinates (non-finite): {(px, py, pz)}"
            )
        if transform is not None:
            if isinstance(transform, RigidTransform):
                v_out = transform.apply(Vec3(px, py, pz))
                px, py, pz = v_out.x, v_out.y, v_out.z
            elif isinstance(transform, Transform):
                px, py, pz = transform.apply((px, py, pz))
        transformed_points.append((px, py, pz))

        conf = pt.uncertainty.confidence
        if not (0.0 <= conf <= 1.0):
            raise ReconstructionAdapterError(
                f"Point confidence out of bounds [0, 1]: {conf}"
            )
        confidences.append(conf)
        source_evidence_ids_set.update(pt.source_evidence_ids)

    for cam in reconstruction.camera_poses:
        source_evidence_ids_set.add(cam.evidence_id)

    source_evidence_ids = sorted(source_evidence_ids_set)
    mean_confidence = sum(confidences) / len(confidences) if confidences else 1.0

    # 4. Compute Bounding Box
    min_x = min(p[0] for p in transformed_points)
    min_y = min(p[1] for p in transformed_points)
    min_z = min(p[2] for p in transformed_points)
    max_x = max(p[0] for p in transformed_points)
    max_y = max(p[1] for p in transformed_points)
    max_z = max(p[2] for p in transformed_points)

    bounds_min = Vector3(min_x, min_y, min_z)
    bounds_max = Vector3(max_x, max_y, max_z)

    # 5. Persist Geometry Artifact if ArtifactStore Provided
    geom_id = f"geom-{target_entity_id}"
    data_uri = ""
    data_hash = ""
    if artifact_store is not None:
        raw_bytes = PointCloudData.from_positions(transformed_points).to_bytes()
        data_uri, data_hash = artifact_store.put(raw_bytes)

    geometry = Geometry(
        id=geom_id,
        type=GeometryType.POINTCLOUD,
        vertex_count=len(transformed_points),
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        data_uri=data_uri,
        data_hash=data_hash,
        provenance=Provenance.RECONSTRUCTED,
        confidence=mean_confidence,
    )

    # 6. Entity Assembly (Preserving Existing Identity & Relationships)
    existing_entity = base_world.entities.get(target_entity_id)

    resolved_type = target_entity_type
    if resolved_type is None:
        resolved_type = existing_entity.type if existing_entity else EntityType.UNKNOWN

    resolved_name = target_entity_name
    if resolved_name is None:
        resolved_name = existing_entity.name if existing_entity else target_entity_id

    relationships = list(existing_entity.relationships) if existing_entity else []

    # Preserve and extend observations with source evidence links
    observations = list(existing_entity.observations) if existing_entity else []
    for eid in source_evidence_ids:
        obs = Observation(
            id=f"obs-{session_id}-{eid}",
            sensor_type="camera",
            timestamp=timestamp if timestamp is not None else 0.0,
            frame_id=session_id,
            data_uri=f"evidence://{eid}",
            confidence=mean_confidence,
        )
        observations.append(obs)

    # Custom properties record session traceability
    custom_props = dict(existing_entity.custom_properties) if existing_entity else {}
    custom_props["session_id"] = session_id
    custom_props["source_evidence_ids"] = source_evidence_ids
    custom_props["point_count"] = len(transformed_points)
    custom_props["registration_status"] = reconstruction.registration_status
    if contract_meta:
        custom_props["reconstruction_contract"] = contract_meta

    # Centroid for transform position
    centroid_x = (min_x + max_x) / 2.0
    centroid_y = (min_y + max_y) / 2.0
    centroid_z = (min_z + max_z) / 2.0
    transform_dict = {"position": {"x": centroid_x, "y": centroid_y, "z": centroid_z}}

    entity = Entity(
        id=target_entity_id,
        name=resolved_name,
        type=resolved_type,
        geometry_ids=[geom_id],
        relationships=relationships,
        observations=observations,
        transform=transform_dict,
        provenance=Provenance.RECONSTRUCTED,
        confidence=mean_confidence,
        custom_properties=custom_props,
    )

    return IncrementalUpdatePackage(
        updated_entities=[entity],
        updated_geometries=[geometry],
        target_entity_id=target_entity_id,
        target_geometry_id=geom_id,
        source_session_id=session_id,
        source_evidence_ids=source_evidence_ids,
        point_count=len(transformed_points),
        mean_confidence=mean_confidence,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def apply_reconstruction_update(
    base_world: WorldIR,
    reconstruction: Union[ReconstructionResult, Any],
    *,
    session_id: str,
    target_entity_id: str,
    alignment: Optional[Union[SessionAlignment, RigidTransform, Transform]] = None,
    target_entity_type: Optional[EntityType] = None,
    target_entity_name: Optional[str] = None,
    artifact_store: Optional[ArtifactStore] = None,
    tile_size: float = 10.0,
    timestamp: Optional[float] = None,
) -> IncrementalUpdateResult:
    """Adapts a FreeBuff `ReconstructionResult` and executes Claude's `apply_incremental_update`.

    Returns the `IncrementalUpdateResult` containing the updated WorldIR (V2)
    with strict reference reuse of all untouched entities/geometries.
    """
    pkg = adapt_reconstruction_to_incremental_update(
        base_world=base_world,
        reconstruction=reconstruction,
        session_id=session_id,
        target_entity_id=target_entity_id,
        alignment=alignment,
        target_entity_type=target_entity_type,
        target_entity_name=target_entity_name,
        artifact_store=artifact_store,
        timestamp=timestamp,
    )

    return apply_incremental_update(
        base_world=base_world,
        updated_entities=pkg.updated_entities,
        updated_geometries=pkg.updated_geometries,
        tile_size=tile_size,
    )
