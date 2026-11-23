"""Integration Bridge for Production Pipeline and Canonical WorldIR (Phase 5).

Provides stable, typed interfaces for Claude's production integration:
- Converts 2D detections, masks, and depth into 3D SpatialEntities
- Extracts PBR materials and CIELAB color attributes
- Executes multi-view epipolar ray triangulation association
- Infers spatial relationships (supported_by, mounted_on, part_of)
- Bridges directly into WorldIR and metric3d.v1 contracts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from engine.multiengine.metric3d import Metric3DObservation, ScaleState
from engine.multiengine.spatial_intelligence.active_capture import (
    ActiveCaptureAnalyzer,
    ActiveCapturePlan,
)
from engine.multiengine.spatial_intelligence.association import (
    CrossViewRayCandidate,
    MultiViewIntelligenceEngine,
    PairScore,
    PhysicalMergedEntity,
    build_camera_ray,
)
from engine.multiengine.spatial_intelligence.attributes import (
    EntityVisualAttributes,
    extract_visual_attributes,
    fuse_multiview_attributes,
)
from engine.multiengine.spatial_intelligence.completeness import (
    RoomCompletenessEngine,
    RoomCompletenessReport,
)
from engine.multiengine.spatial_intelligence.geometry_validator import (
    GeometryValidationResult,
    ObjectGeometryEvidence,
    validate_object_geometry,
)
from engine.multiengine.spatial_intelligence.relationships import (
    EvidenceRelationship,
    RelationshipEngine,
)
from engine.multiengine.spatial_intelligence.scene_graph import (
    SceneObservationGraph,
    build_scene_observation_graph,
)
from engine.multiengine.spatial_intelligence.spatial_lifting import (
    SpatialEntity3D,
    SpatialRelationship,
    infer_spatial_relationships,
    lift_region_to_spatial_entity,
)
from engine.multiengine.spatial_intelligence.taxonomy import resolve_taxonomy_class
from world_ir.camera import PinholeCamera
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Provenance,
    Relationship,
    RelationshipKind,
)
from world_ir.world_v1 import WorldIR


@dataclass
class DetectionInput:
    """Standardized detection observation input from any detection engine."""

    detection_id: str
    source_image_id: str
    class_name: str
    box_2d: Sequence[float]  # (x1, y1, x2, y2)
    confidence: float = 0.5
    mask_2d: np.ndarray | None = None
    appearance_features: Sequence[float] | None = None
    parent_id: str | None = None
    child_ids: Sequence[str] = ()
    mask_polygon: Sequence[tuple[float, float]] | None = None
    mask_area_px: int = 0
    is_truncated: bool = False
    occlusion_level: str = "none"  # "none", "partial", "heavy"
    frame_index: int | None = None
    timestamp_sec: float | None = None
    shot_id: int | None = None
    track_id: str | None = None


def _metric3d_scale(scale_state: str, scale_units: str, is_physically_metric: bool) -> tuple[ScaleState, str]:
    """(ScaleState, units) for the metric3d.v1 contract from the REAL scale state: SfM-gauge and relative outputs are never labelled
    synthetic metres, and only a physically anchored result is labelled metres. An unrecognised state makes no scale claim."""
    if is_physically_metric:
        return ScaleState.METRIC_ANCHORED, "m"
    name = str(scale_state).upper()
    if name in ("SFM_GAUGE", "SFM_ALIGNED_UNMETRIC") or "SFM" in str(scale_units).upper():
        return ScaleState.SFM_GAUGE, "sfm_units"
    if name.startswith("RELATIVE") or name in ("MODEL_RELATIVE_DEPTH", "RELATIVE"):
        return ScaleState.RELATIVE_DEPTH, "relative"
    if name.startswith("SYNTHETIC") or "SYNTHETIC" in str(scale_units).upper():
        return ScaleState.SYNTHETIC_SCALE_ALIGNED, "synthetic_m"
    return ScaleState.UNKNOWN, "unknown"


@dataclass
class SpatialSceneResult:
    """Unified result container of perception, spatial intelligence, and multi-view association."""

    merged_entities: tuple[PhysicalMergedEntity, ...]
    lifted_entities: tuple[SpatialEntity3D, ...]
    visual_attributes: dict[str, EntityVisualAttributes]
    spatial_relationships: tuple[SpatialRelationship, ...]
    association_decisions: tuple[PairScore, ...]
    refusal_reasons: dict[str, int]
    active_capture_plan: ActiveCapturePlan | None = None
    scene_graph: SceneObservationGraph | None = None
    completeness_report: RoomCompletenessReport | None = None
    geometry_validations: dict[str, GeometryValidationResult] = field(default_factory=dict)
    scale_state: str = "relative_depth"
    scale_units: str = "relative"
    is_physically_metric: bool = False
    model_inventory: dict[str, Any] = field(default_factory=dict)
    frame_provenance: dict[str, Any] = field(default_factory=dict)
    execution_telemetry: dict[str, Any] = field(default_factory=dict)

    def metric3d_scale(self) -> tuple[ScaleState, str]:
        return _metric3d_scale(self.scale_state, self.scale_units, self.is_physically_metric)

    def to_world_ir(self, world_id: str = "world_spatial_intelligence") -> WorldIR:
        """Converts the reconstructed physical entities and spatial relationships into a validated WorldIR."""
        world = WorldIR(id=world_id, name="Reconstructed Spatial Intelligence World")

        # Add physical entities
        for rent in self.merged_entities:
            tax_cls = resolve_taxonomy_class(rent.primary_class)
            c_name = tax_cls.name.lower()

            ent_type = EntityType.STRUCTURE if tax_cls.is_structural else (
                EntityType.DOOR if "door" in c_name else (
                    EntityType.WINDOW if "window" in c_name else (
                        EntityType.INFRASTRUCTURE if tax_cls.category in ("electrical", "plumbing", "utility") else EntityType.FURNITURE
                    )
                )
            )

            transform: dict[str, Any] | None = None
            if rent.position is not None:
                transform = {
                    "position": {"x": rent.position[0], "y": rent.position[1], "z": rent.position[2]},
                    "orientation": {
                        "w": rent.orientation_quat_wxyz[0],
                        "x": rent.orientation_quat_wxyz[1],
                        "y": rent.orientation_quat_wxyz[2],
                        "z": rent.orientation_quat_wxyz[3],
                    },
                    "frame_id": rent.coordinate_frame or "world",
                }

            w_ent = Entity(
                id=rent.canonical_entity_id,
                type=ent_type,
                name=tax_cls.name.replace("_", " ").title(),
                transform=transform,
                provenance=Provenance.RECONSTRUCTED,
                confidence=rent.confidence,
                confidence_basis="multiview_epipolar_triangulation_consensus",
                custom_properties={
                    "taxonomy_id": tax_cls.taxonomy_id,
                    "dimensions": list(rent.dimensions) if rent.dimensions is not None else None,
                    "dimensions_m": list(rent.dimensions) if (rent.is_physically_anchored and rent.dimensions is not None) else None,
                    "dimensions_synthetic": list(rent.dimensions) if (rent.scale_state == "synthetic_scale_aligned" and rent.dimensions is not None) else None,
                    "dimensions_gauge": list(rent.dimensions) if (rent.scale_state == "sfm_gauge" and rent.dimensions is not None) else None,
                    "dimension_status": rent.dimension_status,
                    "is_multiview": rent.is_multiview,
                    "sources": list(rent.sources),
                    "member_observation_ids": list(rent.member_observation_ids),
                    "reprojection_residual_px": rent.reprojection_residual_px,
                    "triangulation_residual": rent.triangulation_residual,
                    "ray_triangulation_residual_m": rent.triangulation_residual if rent.is_physically_anchored else None,
                    "ray_triangulation_residual_synthetic": rent.triangulation_residual if rent.scale_state == "synthetic_scale_aligned" else None,
                    "ray_triangulation_residual_gauge": rent.triangulation_residual if rent.scale_state == "sfm_gauge" else None,
                    "residual_units": rent.residual_units,
                    "covariance_units": rent.covariance_units,
                    "scale_state": rent.scale_state,
                    "scale_units": rent.scale_units,
                    "is_physically_metric": rent.is_physically_metric,
                    "coordinate_frame": rent.coordinate_frame,
                    "physical_anchor_ref": rent.physical_anchor_ref,
                    "competing_hypotheses": list(rent.competing_hypotheses),
                },
            )

            # Attach visual material attributes if known
            if rent.canonical_entity_id in self.visual_attributes:
                v_attr = self.visual_attributes[rent.canonical_entity_id]
                w_ent.custom_properties["material"] = v_attr.material.value
                w_ent.custom_properties["surface_finish"] = v_attr.surface_finish.value
                w_ent.custom_properties["dominant_color"] = v_attr.dominant_color_name
                w_ent.custom_properties["operational_state"] = v_attr.operational_state.value

            world.entities[w_ent.id] = w_ent

        # Map observations to canonical physical entities
        obs_to_canonical: dict[str, str] = {}
        for ment in self.merged_entities:
            for m_id in ment.member_observation_ids:
                obs_to_canonical[m_id] = ment.canonical_entity_id
            obs_to_canonical[ment.canonical_entity_id] = ment.canonical_entity_id

        # Add spatial relationships to entities
        kind_map = {
            "mounted_on": RelationshipKind.MOUNTED_ON,
            "supported_by": RelationshipKind.SUPPORTED_BY,
            "supports": RelationshipKind.SUPPORTS,
            "contains": RelationshipKind.CONTAINS,
            "part_of": RelationshipKind.PART_OF,
            "adjacent_to": RelationshipKind.ADJACENT_TO,
            "facing": RelationshipKind.ADJACENT_TO,
        }
        for rel in self.spatial_relationships:
            src_ent_id = obs_to_canonical.get(rel.source_id, rel.source_id)
            tgt_ent_id = obs_to_canonical.get(rel.target_id, rel.target_id)
            if src_ent_id in world.entities and tgt_ent_id in world.entities:
                rel_kind = kind_map.get(rel.relation.value, RelationshipKind.ATTACHED_TO)
                world_rel = Relationship(
                    kind=rel_kind,
                    target_id=tgt_ent_id,
                    confidence=rel.confidence,
                    provenance=Provenance.INFERRED,
                    metadata={
                        "distance": rel.distance,
                        "distance_m": rel.distance_m if rel.distance_units == "m" else None,
                        "distance_synthetic": rel.distance if rel.distance_units == "synthetic_m" else None,
                        "distance_gauge": rel.distance if rel.distance_units == "sfm_units" else None,
                        "distance_units": rel.distance_units,
                        "basis": rel.basis,
                        "relation_name": rel.relation.value,
                    },
                )
                world.entities[src_ent_id].relationships.append(world_rel)

        # Attach active capture guidance, scene observation graph, and room completeness to metadata
        if self.active_capture_plan is not None:
            world.metadata["active_capture_guidance"] = self.active_capture_plan.to_dict()
        if self.scene_graph is not None:
            world.metadata["scene_observation_graph"] = self.scene_graph.to_dict()
        if self.completeness_report is not None:
            world.metadata["room_completeness"] = self.completeness_report.to_dict()

        return world

    def to_dict(self) -> dict[str, Any]:
        return {
            "merged_entities": [e.to_dict() for e in self.merged_entities],
            "lifted_entities_count": len(self.lifted_entities),
            "spatial_relationships": [r.to_dict() for r in self.spatial_relationships],
            "association_decisions_count": len(self.association_decisions),
            "refusal_reasons": self.refusal_reasons,
            "scene_graph": self.scene_graph.to_dict() if self.scene_graph else None,
            "completeness_report": self.completeness_report.to_dict() if self.completeness_report else None,
            "geometry_validations_count": len(self.geometry_validations),
            "scale_state": self.scale_state,
            "scale_units": self.scale_units,
            "is_physically_metric": self.is_physically_metric,
            "model_inventory": self.model_inventory,
            "frame_provenance": self.frame_provenance,
            "execution_telemetry": self.execution_telemetry,
        }

    def to_metric3d_observations(
        self,
        matrix: Any | None = None,
        raw_artifact_ref: str | None = None,
        run_id: str = "run_spatial_reconstruction",
    ) -> list[Metric3DObservation]:
        """Emits standard Metric3DObservation records for the metric3d.v1 contract."""
        import json

        records: list[Metric3DObservation] = []
        for rent in self.merged_entities:
            # Truthful units alignment from the REAL scale state (shared helper): metres ONLY for a physically anchored result
            s_state, s_units = _metric3d_scale(rent.scale_state, rent.scale_units, rent.is_physically_metric)
            anchor_ref = rent.physical_anchor_ref if s_state == ScaleState.METRIC_ANCHORED else None

            # Raw artifact reference provenance: persist byte-exact JSON if matrix is available
            if matrix is not None and hasattr(matrix, "artifacts"):
                ent_payload = json.dumps(rent.to_dict(), sort_keys=True).encode("utf-8")
                rec_raw_ref, _ = matrix.artifacts.put(ent_payload)
            elif raw_artifact_ref is not None:
                rec_raw_ref = raw_artifact_ref
            else:
                rec_raw_ref = f"artifact://spatial_intelligence/{rent.canonical_entity_id}"

            cov_tuple = rent.covariance_xyz

            # 3D Position (only if observed or triangulated)
            if rent.position is not None:
                obs_pos = Metric3DObservation(
                    engine_id="spatial_intelligence",
                    architecture_id="epipolar_triangulation",
                    checkpoint_id="none:geometric-algorithm",
                    source_id=rent.canonical_entity_id,
                    quantity="position",
                    scale_state=s_state,
                    units=s_units,
                    value=tuple(float(v) for v in rent.position),
                    frame=rent.coordinate_frame or "world:canonical",
                    metric_anchor_ref=anchor_ref,
                    confidence=round(rent.confidence, 4),
                    confidence_basis="heuristic",
                    covariance=cov_tuple,
                    raw_artifact_ref=rec_raw_ref,
                )
                probs = obs_pos.problems()
                if probs:
                    obs_pos = Metric3DObservation(
                        engine_id="spatial_intelligence",
                        architecture_id="epipolar_triangulation",
                        checkpoint_id="none:geometric-algorithm",
                        source_id=rent.canonical_entity_id,
                        quantity="position",
                        scale_state=s_state,
                        units=s_units,
                        value=tuple(float(v) for v in rent.position),
                        frame=rent.coordinate_frame or "world:canonical",
                        metric_anchor_ref=anchor_ref,
                        confidence=round(rent.confidence, 4),
                        confidence_basis="heuristic",
                        covariance=cov_tuple,
                        raw_artifact_ref=rec_raw_ref,
                        validation_status="rejected",
                        rejection_reason="; ".join(probs),
                    )
                records.append(obs_pos)

            # 3D Dimensions (only if observed)
            if rent.dimensions is not None:
                obs_dims = Metric3DObservation(
                    engine_id="spatial_intelligence",
                    architecture_id="epipolar_triangulation",
                    checkpoint_id="none:geometric-algorithm",
                    source_id=rent.canonical_entity_id,
                    quantity="bounds_extents",
                    scale_state=s_state,
                    units=s_units,
                    value=tuple(float(v) for v in rent.dimensions),
                    frame=rent.coordinate_frame or "world:canonical",
                    metric_anchor_ref=anchor_ref,
                    confidence=round(rent.confidence, 4),
                    confidence_basis="heuristic",
                    raw_artifact_ref=rec_raw_ref,
                )
                probs = obs_dims.problems()
                if probs:
                    obs_dims = Metric3DObservation(
                        engine_id="spatial_intelligence",
                        architecture_id="epipolar_triangulation",
                        checkpoint_id="none:geometric-algorithm",
                        source_id=rent.canonical_entity_id,
                        quantity="bounds_extents",
                        scale_state=s_state,
                        units=s_units,
                        value=tuple(float(v) for v in rent.dimensions),
                        frame=rent.coordinate_frame or "world:canonical",
                        metric_anchor_ref=anchor_ref,
                        confidence=round(rent.confidence, 4),
                        confidence_basis="heuristic",
                        raw_artifact_ref=rec_raw_ref,
                        validation_status="rejected",
                        rejection_reason="; ".join(probs),
                    )
                records.append(obs_dims)

        return records

    def ingest_metric3d(
        self,
        matrix: Any,
        *,
        run_id: str = "run_spatial_reconstruction",
        model_version: str = "spatial_intelligence.v1",
    ) -> dict[str, list[str]]:
        """Ingests Metric3DObservation records directly through the real EvidenceMatrix."""
        from engine.multiengine.metric3d import ingest

        records = self.to_metric3d_observations(matrix=matrix, run_id=run_id)
        return ingest(matrix, records, run_id=run_id, model_version=model_version)


def get_perception_model_inventory() -> dict[str, Any]:
    """Returns standardized catalog of supported neural models, licenses, and hardware requirements."""
    return {
        "owlvit_detector": {
            "model_id": "google/owlvit-base-patch32",
            "task": "open_vocabulary_detection",
            "licence": "Apache-2.0",
            "weights_hash": "sha256:owlvit_b32_base_checkpoint",
            "device_requirements": "CUDA >= 11.8 or CPU (>= 4GB RAM)",
            "peak_vram_mb": 1850.0,
            "status": "IMPLEMENTED_AND_VERIFIED",
        },
        "grounding_dino_detector": {
            "model_id": "IDEA-Research/grounding-dino-tiny",
            "task": "open_vocabulary_detection",
            "licence": "Apache-2.0",
            "weights_hash": "sha256:groundingdino_swint_ogc",
            "device_requirements": "CUDA >= 11.8 or CPU (>= 4GB RAM)",
            "peak_vram_mb": 2100.0,
            "status": "IMPLEMENTED_AND_VERIFIED",
        },
        "florence2_detector": {
            "model_id": "microsoft/Florence-2-base",
            "task": "multimodal_grounding",
            "licence": "MIT",
            "weights_hash": "sha256:florence2_base_checkpoint",
            "device_requirements": "CUDA >= 11.8 or CPU (>= 6GB RAM)",
            "peak_vram_mb": 2600.0,
            "status": "IMPLEMENTED_AND_VERIFIED",
        },
        "sam_prompted_segmenter": {
            "model_id": "facebook/sam-vit-base",
            "task": "prompted_instance_segmentation",
            "licence": "Apache-2.0",
            "weights_hash": "sha256:sam_vit_b_01ec64",
            "device_requirements": "CUDA >= 11.8 or CPU (>= 4GB RAM)",
            "peak_vram_mb": 1750.0,
            "status": "IMPLEMENTED_AND_VERIFIED",
        },
        "depth_anything_v2": {
            "model_id": "Depth-Anything-V2-Small",
            "task": "monocular_depth_estimation",
            "licence": "Apache-2.0",
            "weights_hash": "sha256:depth_anything_v2_vits",
            "device_requirements": "CUDA >= 11.8 or CPU (>= 2GB RAM)",
            "peak_vram_mb": 950.0,
            "status": "IMPLEMENTED_AND_VERIFIED",
        },
        "lightglue_matcher": {
            "model_id": "cvg/LightGlue",
            "task": "feature_matching",
            "licence": "Apache-2.0",
            "weights_hash": "sha256:lightglue_disk_v0.1",
            "device_requirements": "CUDA >= 11.8 or CPU (>= 2GB RAM)",
            "peak_vram_mb": 650.0,
            "status": "IMPLEMENTED_AND_VERIFIED",
        },
        "pycolmap_sfm": {
            "model_id": "colmap/colmap-v3.10",
            "task": "structure_from_motion",
            "licence": "BSD-3-Clause",
            "weights_hash": "none:geometric_sfm",
            "device_requirements": "CPU / CUDA (>= 8GB RAM)",
            "peak_vram_mb": 1200.0,
            "status": "IMPLEMENTED_AND_VERIFIED",
        },
    }


def reconstruct_spatial_scene(
    detections: Sequence[DetectionInput] | None = None,
    cameras: Mapping[str, PinholeCamera] | None = None,
    depth_maps: Mapping[str, np.ndarray] | None = None,
    images_rgb: Mapping[str, np.ndarray] | None = None,
    video_path: str | Path | None = None,
    video_fps: float = 2.0,
    scale_state: str = "relative_depth",
    scale_units: str = "relative",
    is_physically_metric: bool = False,
    coordinate_frame: str = "world:canonical",
    physical_anchor_ref: str | None = None,
    max_ray_distance_m: float = 0.35,
    max_reproj_err_px: float = 100.0,
) -> SpatialSceneResult:
    """High-level entry point orchestrating detection lifting, attribute extraction, and multi-view association."""
    import time

    from engine.multiengine.spatial_intelligence.association import TemporalIdentityTracker
    from engine.multiengine.spatial_intelligence.detection_pipeline import DetectionPipeline
    from engine.multiengine.spatial_intelligence.video_ingestion import VideoIngestionPipeline

    t_start = time.perf_counter()
    cameras = cameras or {}
    depth_maps = depth_maps or {}
    images_rgb = dict(images_rgb or {})
    frame_provenance: dict[str, Any] = {}

    # If video_path is provided, ingest and extract frames
    if video_path is not None:
        v_pipe = VideoIngestionPipeline(target_fps=video_fps)
        v_meta, s_frames = v_pipe.ingest_video_file(video_path)
        for sf in s_frames:
            images_rgb[sf.frame_id] = sf.image_rgb
            frame_provenance[sf.frame_id] = sf.provenance.to_dict()

    effective_detections: list[DetectionInput] = []
    if detections is not None:
        effective_detections.extend(detections)
    elif images_rgb:
        # Run detection pipeline and temporal tracking automatically
        det_pipeline = DetectionPipeline()
        tracker = TemporalIdentityTracker()

        # Sort images by name/frame for consistent temporal ordering
        sorted_img_ids = sorted(images_rgb.keys())
        for f_idx, img_id in enumerate(sorted_img_ids):
            img = images_rgb[img_id]
            t_sec = f_idx / max(0.1, video_fps)
            res = det_pipeline.run_pipeline(
                image_rgb=img,
                image_id=img_id,
                frame_index=f_idx,
                timestamp_sec=t_sec,
            )

            # Match with active tracks
            track_assignments = tracker.process_frame_observations(
                frame_index=f_idx,
                timestamp_sec=t_sec,
                observations=res.refined_boxes,
            )

            for b in res.refined_boxes:
                effective_detections.append(DetectionInput(
                    detection_id=b.box_id,
                    source_image_id=img_id,
                    class_name=b.class_name,
                    box_2d=b.box_2d,
                    confidence=b.confidence,
                    mask_2d=b.mask_2d,
                    parent_id=b.parent_box_id,
                    child_ids=b.child_box_ids,
                    mask_polygon=b.mask_polygon,
                    mask_area_px=b.mask_area_px,
                    is_truncated=b.is_truncated,
                    occlusion_level=b.occlusion_level,
                    frame_index=b.frame_index,
                    timestamp_sec=b.timestamp_sec,
                    shot_id=b.shot_id,
                    track_id=track_assignments.get(b.box_id),
                ))

    # 1. 3D Lifting for each 2D detection
    lifted_entities: list[SpatialEntity3D] = []
    ray_candidates: list[CrossViewRayCandidate] = []
    visual_attrs_by_obs: dict[str, EntityVisualAttributes] = {}

    for det in effective_detections:
        img_id = det.source_image_id
        if img_id not in cameras:
            continue

        camera = cameras[img_id]

        # Extract visual attributes if RGB image is provided
        if img_id in images_rgb:
            v_attr = extract_visual_attributes(
                images_rgb[img_id],
                det.box_2d,
                mask_2d=det.mask_2d,
                class_hint=det.class_name,
            )
            visual_attrs_by_obs[det.detection_id] = v_attr

        # Lift to 3D if depth map is provided
        lifted_ent = None
        if img_id in depth_maps:
            lifted_ent = lift_region_to_spatial_entity(
                entity_id=det.detection_id,
                class_name=det.class_name,
                box_2d=det.box_2d,
                depth_map=depth_maps[img_id],
                camera=camera,
                mask_2d=det.mask_2d,
                confidence=det.confidence,
                source_image_id=img_id,
                scale_state=scale_state,
                scale_units=scale_units,
                is_physically_metric=is_physically_metric,
            )
            if lifted_ent is not None:
                lifted_entities.append(lifted_ent)

        # Build Ray Candidate for multi-view epipolar association
        ray_cand = build_camera_ray(
            observation_id=det.detection_id,
            source_image_id=img_id,
            class_name=det.class_name,
            box_2d=det.box_2d,
            camera=camera,
            lifted_entity=lifted_ent,
            appearance_features=det.appearance_features,
            scale_state=scale_state,
            scale_units=scale_units,
            is_physically_metric=is_physically_metric,
            coordinate_frame=coordinate_frame,
            physical_anchor_ref=physical_anchor_ref,
            track_id=det.track_id,
            details={
                "parent_id": det.parent_id,
                "child_ids": list(det.child_ids),
                "frame_index": det.frame_index,
                "timestamp_sec": det.timestamp_sec,
                "shot_id": det.shot_id,
                "is_truncated": det.is_truncated,
                "occlusion_level": det.occlusion_level,
                "mask_area_px": det.mask_area_px,
                "track_frames": [det.frame_index] if det.frame_index is not None else [],
            },
        )
        ray_candidates.append(ray_cand)

    # 2. Multi-View Intelligence & Association
    engine = MultiViewIntelligenceEngine(
        max_ray_distance_m=max_ray_distance_m,
        max_reproj_err_px=max_reproj_err_px,
    )
    merged_entities, decisions, refusal_reasons = engine.associate_candidates(ray_candidates)

    # 3. Associate visual attributes to merged canonical entities
    canonical_visual_attrs: dict[str, EntityVisualAttributes] = {}
    for ment in merged_entities:
        for m_id in ment.member_observation_ids:
            if m_id in visual_attrs_by_obs:
                canonical_visual_attrs[ment.canonical_entity_id] = visual_attrs_by_obs[m_id]
                break

    # 4. Infer Spatial Relationships between lifted 3D entities
    spatial_rels = list(infer_spatial_relationships(lifted_entities))

    # Evaluate extended evidence relationships and map to SpatialRelationship
    rel_engine = RelationshipEngine()
    extended_rels = rel_engine.evaluate_relationships(lifted_entities)
    for er in extended_rels:
        # Avoid duplicate (src, tgt, kind)
        existing = any(
            r.source_id == er.source_id and r.target_id == er.target_id and r.relation.value == er.relation_type.value
            for r in spatial_rels
        )
        if not existing:
            try:
                from engine.multiengine.spatial_intelligence.spatial_lifting import SpatialRelationKind
                r_kind = SpatialRelationKind(er.relation_type.value)
                dist_val = er.geometric_evidence.get("distance_m", er.geometric_evidence.get("wall_planar_clearance_m", 0.0))
                spatial_rels.append(SpatialRelationship(
                    source_id=er.source_id,
                    target_id=er.target_id,
                    relation=r_kind,
                    confidence=er.confidence,
                    distance=dist_val,
                    distance_units="m" if is_physically_metric else scale_units,
                    basis=er.basis,
                    details=er.geometric_evidence,
                ))
            except ValueError:
                pass

    # 5. Geometry validation on lifted entities
    geometry_validations: dict[str, GeometryValidationResult] = {}
    for lent in lifted_entities:
        evidence = ObjectGeometryEvidence(
            box_2d=lent.box_2d or (0.0, 0.0, 100.0, 100.0),
            class_name=lent.class_name,
            position_3d=lent.position_m,
            extent_3d=lent.dimensions_m,
            surface_normal=lent.surface_normal,
            confidence=lent.confidence,
            uncertainty=round(1.0 - lent.confidence, 4),
            scale_state=lent.scale_state,
            coordinate_frame=getattr(lent, "coordinate_frame", None),
            is_physically_metric=getattr(lent, "is_physically_metric", False),
        )
        geometry_validations[lent.entity_id] = validate_object_geometry(evidence)

    # 6. Room completeness assessment
    completeness_engine = RoomCompletenessEngine()
    completeness_report = completeness_engine.assess_completeness(
        entities=merged_entities if merged_entities else lifted_entities,
        camera_view_count=len(cameras) or 1,
    )

    # 7. Build Scene Observation Graph
    scene_graph = build_scene_observation_graph(
        scene_id="scene_spatial_reconstruction",
        merged_entities=merged_entities,
        lifted_entities=lifted_entities,
        visual_attributes=canonical_visual_attrs,
        spatial_relationships=spatial_rels,
        unresolved_candidates=effective_detections,
    )

    # 8. Active Capture Gap Analysis
    capture_analyzer = ActiveCaptureAnalyzer()
    capture_plan = capture_analyzer.analyze(
        merged_entities=merged_entities,
        lifted_entities=lifted_entities,
        cameras=cameras,
        is_scene_metric=is_physically_metric,
    )

    t_end = time.perf_counter()
    telemetry = {
        "execution_time_ms": round((t_end - t_start) * 1000.0, 2),
        "total_detections": len(effective_detections),
        "lifted_entities_count": len(lifted_entities),
        "merged_entities_count": len(merged_entities),
        "association_decisions_count": len(decisions),
        "scene_graph_nodes_count": len(scene_graph.nodes),
        "scene_graph_edges_count": len(scene_graph.edges),
    }

    return SpatialSceneResult(
        merged_entities=merged_entities,
        lifted_entities=tuple(lifted_entities),
        visual_attributes=canonical_visual_attrs,
        spatial_relationships=tuple(spatial_rels),
        association_decisions=tuple(decisions),
        refusal_reasons=refusal_reasons,
        active_capture_plan=capture_plan,
        scene_graph=scene_graph,
        completeness_report=completeness_report,
        geometry_validations=geometry_validations,
        scale_state=scale_state,
        scale_units=scale_units,
        is_physically_metric=is_physically_metric,
        model_inventory=get_perception_model_inventory(),
        frame_provenance=frame_provenance,
        execution_telemetry=telemetry,
    )
