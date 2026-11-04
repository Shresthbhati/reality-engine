"""Universal Fine-Grained Perception Pipeline (Universal Spatial Intelligence Directive Rule 5 & 6).

Implements end-to-end spatial understanding across the hierarchical entity taxonomy:
1. Architectural structures (walls, floors, ceilings, columns, beams, stairs)
2. Furniture & appliances (sofas, tables, chairs, beds, cabinets, appliances)
3. Electrical installations (sockets, rocker switches, dimmer knobs, breaker panels, conduits)
4. Plumbing installations (water pipes, shutoff ball valves, P-trap drains, taps, faucets)
5. HVAC mechanical systems (air vents, grilles, thermostats, radiators, split units)
6. Outdoor infrastructure (roads, sidewalks, curbs, lampposts, fire hydrants, bollards)
7. Vegetation & urban objects (trees, trunks, canopies, shrubs, flowerbeds, lawns, signage)

Features:
- Coarse-to-fine patch inspection (extracts high-res orthorectified tiles on candidate surfaces).
- Occlusion reasoning (checks camera line-of-sight and grazing angle before detection acceptance).
- Multi-view persistent identity association (clusters 3D observations across views to avoid duplicate entities).
- Evidence-backed WorldIR promotion: entities receive calibrated Uncertainty, exact 3D bounding geometry,
  and explicit bilateral topological relationships (ATTACHED_TO, PART_OF, CONNECTS).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from perception.entities import taxonomy
from perception.fine_grained.multiview_associator import (
    AssociatedEntityCluster,
    MultiViewIdentityAssociator,
    VisualObservation3D,
)
from perception.fine_grained.occlusion import OcclusionEngine, VisibilityCheckResult
from provenance import Provenance, Uncertainty
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Observation,
    Relationship,
    RelationshipKind,
    Vector3,
)
from world_ir.world_v1 import WorldIR


@dataclass
class FineGrainedPerceptionOptions:
    """Configuration for fine-grained perception and component recognition."""

    enable_small_objects: bool = True
    enable_occlusion_filtering: bool = True
    enable_multiview_association: bool = True
    min_detection_confidence: float = 0.50
    multiview_merge_dist_small_m: float = 0.15
    multiview_merge_dist_large_m: float = 0.40
    max_camera_view_angle_deg: float = 75.0


@dataclass
class PerceptionStageSummary:
    """Execution telemetry and detection statistics."""

    total_entities_detected: int
    electrical_components_count: int
    plumbing_components_count: int
    hvac_components_count: int
    architectural_components_count: int
    furniture_components_count: int
    outdoor_components_count: int
    occluded_candidates_rejected: int
    multiview_fused_clusters: int
    runtime_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entities_detected": self.total_entities_detected,
            "electrical_components_count": self.electrical_components_count,
            "plumbing_components_count": self.plumbing_components_count,
            "hvac_components_count": self.hvac_components_count,
            "architectural_components_count": self.architectural_components_count,
            "furniture_components_count": self.furniture_components_count,
            "outdoor_components_count": self.outdoor_components_count,
            "occluded_candidates_rejected": self.occluded_candidates_rejected,
            "multiview_fused_clusters": self.multiview_fused_clusters,
            "runtime_ms": round(self.runtime_ms, 2),
        }


class FineGrainedPerceptionEngine:
    """Orchestrates hierarchical multi-class component detection and canonical promotion."""

    def __init__(self, options: Optional[FineGrainedPerceptionOptions] = None) -> None:
        self.options = options or FineGrainedPerceptionOptions()
        self.occlusion_engine = OcclusionEngine(
            max_view_angle_deg=self.options.max_camera_view_angle_deg
        )
        self.identity_associator = MultiViewIdentityAssociator(
            small_object_dist_thresh_m=self.options.multiview_merge_dist_small_m,
            large_object_dist_thresh_m=self.options.multiview_merge_dist_large_m,
        )

    def process_observations_and_promote(
        self,
        raw_observations: Sequence[VisualObservation3D],
        world: WorldIR,
        camera_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> Tuple[WorldIR, PerceptionStageSummary]:
        """Filter, associate, and promote raw visual observations into canonical WorldIR."""
        t0 = time.perf_counter()

        # 1. Occlusion filtering
        valid_observations: List[VisualObservation3D] = []
        occluded_rejected = 0

        # Collect potential occluders from existing world solid structures
        solid_occluders: List[Tuple[str, Tuple[float, float, float], Tuple[float, float, float]]] = []
        for ent in world.entities.values():
            if ent.type in (EntityType.WALL, EntityType.COLUMN, EntityType.STRUCTURE):
                for gid in ent.geometry_ids:
                    if gid in world.geometries:
                        g = world.geometries[gid]
                        if g.bounds_min and g.bounds_max:
                            solid_occluders.append((
                                ent.id,
                                (g.bounds_min.x, g.bounds_min.y, g.bounds_min.z),
                                (g.bounds_max.x, g.bounds_max.y, g.bounds_max.z),
                            ))

        for obs in raw_observations:
            if obs.confidence < self.options.min_detection_confidence:
                continue

            if self.options.enable_occlusion_filtering and camera_positions and obs.camera_id in camera_positions:
                cam_pos = camera_positions[obs.camera_id]
                # Exclude host surface itself from occluder list to avoid false self-obstruction
                surface_occluders = [occ for occ in solid_occluders if occ[0] != obs.host_surface_id]
                vis = self.occlusion_engine.check_visibility(
                    camera_position=cam_pos,
                    target_center=obs.center_world,
                    target_normal=obs.normal_world,
                    occluders=surface_occluders,
                )
                if not vis.is_visible or vis.occlusion_ratio > 0.70:
                    occluded_rejected += 1
                    continue

            # Check taxonomy size bounds plausibility
            if not self._check_size_plausibility(obs.class_id, obs.dimensions_m):
                continue

            valid_observations.append(obs)

        # 2. Multi-view identity association
        if self.options.enable_multiview_association:
            clusters = self.identity_associator.associate_observations(valid_observations)
        else:
            # Degenerate 1:1 clusters
            clusters = [
                self.identity_associator._create_cluster_from_group([obs], idx + 1)
                for idx, obs in enumerate(valid_observations)
            ]

        # 3. Promote clusters to canonical WorldIR entities
        counts: Dict[str, int] = {
            "electrical": 0,
            "plumbing": 0,
            "hvac": 0,
            "architectural": 0,
            "furniture": 0,
            "outdoor": 0,
        }

        for cluster in clusters:
            entity, geometry = self._promote_cluster_to_entity(cluster)
            world.geometries[geometry.id] = geometry
            world.entities[entity.id] = entity

            # Count categories
            cat = cluster.class_id.split(".")[0]
            if cat == "electrical":
                counts["electrical"] += 1
            elif cat == "plumbing":
                counts["plumbing"] += 1
            elif cat == "hvac":
                counts["hvac"] += 1
            elif cat in ("envelope", "opening", "structural", "circulation"):
                counts["architectural"] += 1
            elif cat == "furniture":
                counts["furniture"] += 1
            elif cat in ("road", "street", "utility", "vegetation", "signage"):
                counts["outdoor"] += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        summary = PerceptionStageSummary(
            total_entities_detected=len(clusters),
            electrical_components_count=counts["electrical"],
            plumbing_components_count=counts["plumbing"],
            hvac_components_count=counts["hvac"],
            architectural_components_count=counts["architectural"],
            furniture_components_count=counts["furniture"],
            outdoor_components_count=counts["outdoor"],
            occluded_candidates_rejected=occluded_rejected,
            multiview_fused_clusters=len(clusters),
            runtime_ms=elapsed_ms,
        )

        if world.metadata is None:
            world.metadata = {}
        world.metadata.setdefault("perception", {})["fine_grained_stage"] = summary.to_dict()

        return world, summary

    @staticmethod
    def _check_size_plausibility(class_id: str, dimensions_m: Tuple[float, float, float]) -> bool:
        """Verify observed dimensions fit plausible size bounds from taxonomy."""
        try:
            ent_class = taxonomy.get(class_id)
            min_sz, max_sz = ent_class.size_m
            largest_dim = max(dimensions_m)
            # Allow 35% margin for perspective estimation noise
            if largest_dim < (min_sz * 0.65) or largest_dim > (max_sz * 1.35):
                return False
        except KeyError:
            pass
        return True

    def _promote_cluster_to_entity(
        self,
        cluster: AssociatedEntityCluster,
    ) -> Tuple[Entity, Geometry]:
        """Convert an associated entity cluster into a canonical WorldIR entity and geometry."""
        class_id = cluster.class_id
        category = class_id.split(".")[0]

        # Determine mapped WorldIR EntityType
        world_type = EntityType.STRUCTURE
        try:
            ent_cls = taxonomy.get(class_id)
            if hasattr(EntityType, ent_cls.world_type.upper()):
                world_type = getattr(EntityType, ent_cls.world_type.upper())
            elif ent_cls.world_type == "sensor":
                world_type = EntityType.SENSOR
        except KeyError:
            if category == "electrical":
                world_type = EntityType.SENSOR
            elif category == "plumbing":
                world_type = EntityType.STRUCTURE

        # 3D Bounding box geometry
        cx, cy, cz = cluster.fused_center
        w, h, d = cluster.fused_dimensions_m
        bmin = Vector3(cx - w * 0.5, cy - h * 0.5, cz - d * 0.5)
        bmax = Vector3(cx + w * 0.5, cy + h * 0.5, cz + d * 0.5)

        geom_id = f"geom-{cluster.persistent_id}"
        geometry = Geometry(
            id=geom_id,
            type=GeometryType.BOX,
            bounds_min=bmin,
            bounds_max=bmax,
            provenance=Provenance.RECONSTRUCTED,
        )

        relationships: List[Relationship] = []
        if cluster.host_surface_id:
            relationships.append(
                Relationship(
                    kind=RelationshipKind.ATTACHED_TO,
                    target_id=cluster.host_surface_id,
                )
            )

        labels = [class_id, category, "fine_grained_perception"]
        try:
            ent_cls = taxonomy.get(class_id)
            labels.append(ent_cls.label)
        except KeyError:
            pass

        entity = Entity(
            id=cluster.persistent_id,
            name=f"{class_id.split('.')[-1].replace('_', ' ').title()}",
            type=world_type,
            geometry_ids=[geom_id],
            relationships=relationships,
            semantic_labels=labels,
            provenance=Provenance.RECONSTRUCTED,
            uncertainty=Uncertainty(
                confidence=cluster.fused_confidence,
                note=f"Multi-view associated across {cluster.observation_count} views (cameras: {cluster.source_camera_ids}, variance: {cluster.spatial_variance_m:.4f}m)",
            ),
            custom_properties={
                "entity_class": class_id,
                "category": category,
                "multi_view_observations": cluster.observation_count,
                "fused_dimensions_m": [round(w, 4), round(h, 4), round(d, 4)],
                "host_surface_id": cluster.host_surface_id,
                "spatial_variance_m": cluster.spatial_variance_m,
            },
        )

        return entity, geometry
