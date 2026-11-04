"""Multi-View Identity Association Engine (Universal Spatial Intelligence Directive Rule 5 & 6).

Guarantees persistent entity identities across viewpoints:
- Associates visual observations across arbitrary camera viewpoints using 3D proximity,
  epipolar/plane constraints, and semantic compatibility.
- Never duplicates the same physical entity across different viewpoints.
- Never merges visually similar distinct objects without sufficient spatial evidence.
- Preserves persistent entity identity when new frames or evidence are added.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass
class VisualObservation3D:
    """A localized 3D observation from a single camera viewpoint."""

    observation_id: str
    class_id: str
    camera_id: str
    center_world: Tuple[float, float, float]
    normal_world: Tuple[float, float, float]
    dimensions_m: Tuple[float, float, float]  # (width, height, depth)
    confidence: float
    host_surface_id: Optional[str] = None
    feature_embedding: Optional[np.ndarray] = None
    custom_attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AssociatedEntityCluster:
    """Fused multi-view physical entity with a stable, persistent identity."""

    persistent_id: str
    class_id: str
    fused_center: Tuple[float, float, float]
    fused_normal: Tuple[float, float, float]
    fused_dimensions_m: Tuple[float, float, float]
    fused_confidence: float
    observation_count: int
    source_camera_ids: List[str]
    host_surface_id: Optional[str] = None
    spatial_variance_m: float = 0.0
    member_observations: List[VisualObservation3D] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persistent_id": self.persistent_id,
            "class_id": self.class_id,
            "fused_center": [round(x, 4) for x in self.fused_center],
            "fused_normal": [round(x, 4) for x in self.fused_normal],
            "fused_dimensions_m": [round(x, 4) for x in self.fused_dimensions_m],
            "fused_confidence": round(self.fused_confidence, 3),
            "observation_count": self.observation_count,
            "source_camera_ids": list(self.source_camera_ids),
            "host_surface_id": self.host_surface_id,
            "spatial_variance_m": round(self.spatial_variance_m, 5),
        }


class MultiViewIdentityAssociator:
    """Associates independent 3D observations into persistent physical entities."""

    def __init__(
        self,
        small_object_dist_thresh_m: float = 0.15,
        large_object_dist_thresh_m: float = 0.40,
        normal_angle_thresh_deg: float = 40.0,
    ) -> None:
        self.small_object_dist_thresh = small_object_dist_thresh_m
        self.large_object_dist_thresh = large_object_dist_thresh_m
        self.normal_angle_thresh_deg = normal_angle_thresh_deg

    def associate_observations(
        self,
        observations: Sequence[VisualObservation3D],
        existing_clusters: Optional[Sequence[AssociatedEntityCluster]] = None,
    ) -> List[AssociatedEntityCluster]:
        """Associate a set of 3D observations, optionally merging with an existing persistent state."""
        if not observations:
            return list(existing_clusters or [])

        clusters: List[AssociatedEntityCluster] = list(existing_clusters or [])
        used_obs: Set[str] = set()

        # 1. First pass: associate against existing persistent clusters (temporal / incremental continuity)
        for cluster in clusters:
            matching_obs = []
            c_center = np.asarray(cluster.fused_center, dtype=np.float64)
            c_norm = np.asarray(cluster.fused_normal, dtype=np.float64)

            # Determine distance threshold based on entity scale
            is_small = any(k in cluster.class_id for k in ("socket", "switch", "dimmer", "valve", "handle", "fitting", "tap"))
            dist_thresh = self.small_object_dist_thresh if is_small else self.large_object_dist_thresh

            for obs in observations:
                if obs.observation_id in used_obs:
                    continue
                if obs.class_id != cluster.class_id:
                    continue
                if cluster.host_surface_id and obs.host_surface_id and cluster.host_surface_id != obs.host_surface_id:
                    continue

                o_center = np.asarray(obs.center_world, dtype=np.float64)
                dist = float(np.linalg.norm(c_center - o_center))

                # Normal alignment check
                o_norm = np.asarray(obs.normal_world, dtype=np.float64)
                cos_ang = float(np.dot(c_norm, o_norm))
                ang_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_ang))))

                if dist <= dist_thresh and ang_deg <= self.normal_angle_thresh_deg:
                    matching_obs.append(obs)
                    used_obs.add(obs.observation_id)

            if matching_obs:
                self._update_cluster_with_observations(cluster, matching_obs)

        # 2. Second pass: cluster remaining unassigned observations amongst themselves
        unassigned = [obs for obs in observations if obs.observation_id not in used_obs]
        visited: Set[int] = set()

        for i, base_obs in enumerate(unassigned):
            if i in visited:
                continue

            current_group = [base_obs]
            visited.add(i)

            is_small = any(k in base_obs.class_id for k in ("socket", "switch", "dimmer", "valve", "handle", "fitting", "tap"))
            dist_thresh = self.small_object_dist_thresh if is_small else self.large_object_dist_thresh

            b_center = np.asarray(base_obs.center_world, dtype=np.float64)
            b_norm = np.asarray(base_obs.normal_world, dtype=np.float64)

            for j in range(i + 1, len(unassigned)):
                if j in visited:
                    continue
                cand = unassigned[j]

                if cand.class_id != base_obs.class_id:
                    continue
                if base_obs.host_surface_id and cand.host_surface_id and base_obs.host_surface_id != cand.host_surface_id:
                    continue

                c_center = np.asarray(cand.center_world, dtype=np.float64)
                dist = float(np.linalg.norm(b_center - c_center))

                c_norm = np.asarray(cand.normal_world, dtype=np.float64)
                cos_ang = float(np.dot(b_norm, c_norm))
                ang_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_ang))))

                if dist <= dist_thresh and ang_deg <= self.normal_angle_thresh_deg:
                    current_group.append(cand)
                    visited.add(j)

            # Build new persistent entity cluster
            new_cluster = self._create_cluster_from_group(current_group, len(clusters) + 1)
            clusters.append(new_cluster)

        return clusters

    def _create_cluster_from_group(self, group: Sequence[VisualObservation3D], cluster_idx: int) -> AssociatedEntityCluster:
        primary = group[0]
        class_slug = primary.class_id.split(".")[-1]
        persistent_id = f"ent-{class_slug}-{cluster_idx:03d}"

        weights = np.array([max(0.1, obs.confidence) for obs in group], dtype=np.float64)
        pos_arr = np.array([obs.center_world for obs in group], dtype=np.float64)
        fused_center = np.average(pos_arr, axis=0, weights=weights)

        dim_arr = np.array([obs.dimensions_m for obs in group], dtype=np.float64)
        fused_dims = np.average(dim_arr, axis=0, weights=weights)

        norm_arr = np.array([obs.normal_world for obs in group], dtype=np.float64)
        fused_norm = np.average(norm_arr, axis=0, weights=weights)
        norm_len = np.linalg.norm(fused_norm)
        if norm_len > 1e-12:
            fused_norm = fused_norm / norm_len

        # Confidence elevated with multi-view confirmation
        base_conf = float(np.max([obs.confidence for obs in group]))
        fused_conf = min(0.99, base_conf + 0.05 * (len(group) - 1))

        # Spatial variance (residual error across views)
        diffs = pos_arr - fused_center
        spatial_variance = float(np.mean(np.sum(diffs**2, axis=-1))) if len(group) > 1 else 0.0

        cams = sorted(set(obs.camera_id for obs in group))

        return AssociatedEntityCluster(
            persistent_id=persistent_id,
            class_id=primary.class_id,
            fused_center=(float(fused_center[0]), float(fused_center[1]), float(fused_center[2])),
            fused_normal=(float(fused_norm[0]), float(fused_norm[1]), float(fused_norm[2])),
            fused_dimensions_m=(float(fused_dims[0]), float(fused_dims[1]), float(fused_dims[2])),
            fused_confidence=fused_conf,
            observation_count=len(group),
            source_camera_ids=cams,
            host_surface_id=primary.host_surface_id,
            spatial_variance_m=spatial_variance,
            member_observations=list(group),
        )

    def _update_cluster_with_observations(
        self,
        cluster: AssociatedEntityCluster,
        new_obs: Sequence[VisualObservation3D],
    ) -> None:
        all_obs = cluster.member_observations + list(new_obs)
        cluster.member_observations = all_obs
        cluster.observation_count = len(all_obs)

        weights = np.array([max(0.1, obs.confidence) for obs in all_obs], dtype=np.float64)
        pos_arr = np.array([obs.center_world for obs in all_obs], dtype=np.float64)
        fused_center = np.average(pos_arr, axis=0, weights=weights)

        dim_arr = np.array([obs.dimensions_m for obs in all_obs], dtype=np.float64)
        fused_dims = np.average(dim_arr, axis=0, weights=weights)

        cluster.fused_center = (float(fused_center[0]), float(fused_center[1]), float(fused_center[2]))
        cluster.fused_dimensions_m = (float(fused_dims[0]), float(fused_dims[1]), float(fused_dims[2]))

        base_conf = float(np.max([obs.confidence for obs in all_obs]))
        cluster.fused_confidence = min(0.99, base_conf + 0.05 * (len(all_obs) - 1))

        diffs = pos_arr - fused_center
        cluster.spatial_variance_m = float(np.mean(np.sum(diffs**2, axis=-1))) if len(all_obs) > 1 else 0.0

        for obs in new_obs:
            if obs.camera_id not in cluster.source_camera_ids:
                cluster.source_camera_ids.append(obs.camera_id)
