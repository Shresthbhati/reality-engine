"""Metric 3D Lifting from Depth, Camera Poses, Masks, and Detections.

Lifts 2D observations (bounding boxes and segmentation masks) into metric 3D entities
using calibrated pinhole camera geometry and metric depth maps.

Produces:
- Position (x, y, z) in metric world frame
- Dimensions (w, d, h) in metres from oriented bounding box
- 3D Orientation quaternion (qx, qy, qz, qw)
- Coordinate frame identity (e.g. "world:canonical", "world:colmap")
- 3x3 Covariance uncertainty matrix for position and dimensions
- Object-part geometry decomposition and parent association (e.g. chair_leg -> chair)
- Metric observation drafts ready for EvidenceMatrix and CanonicalEntity resolution
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from engine.math import Quat
from engine.multiengine.observation import ObservationDraft
from world.ontology.universal_object_ontology import (
    UniversalObjectOntology,
    get_universal_ontology,
)
from world_ir.camera import PinholeCamera


@dataclass(frozen=True)
class LiftedEntity3D:
    """A metric 3D entity reconstructed from 2D perception + metric depth."""

    entity_id: str
    class_name: str
    position: tuple[float, float, float]
    dimensions: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    coordinate_frame: str
    covariance_position: tuple[tuple[float, ...], ...] | None
    covariance_dimensions: tuple[tuple[float, ...], ...] | None
    point_count: int
    parent_entity_id: str | None = None
    is_part: bool = False
    relative_transform_to_parent: tuple[tuple[float, ...], ...] | None = None
    source_observation_ids: tuple[str, ...] = ()
    reprojection_error_px: float | None = None
    confidence: float | None = None
    confidence_basis: str | None = None


class Metric3DLifter:
    """Performs metric 3D unprojection, oriented bounding box fitting, and part-parent linking."""

    def __init__(self, ontology: UniversalObjectOntology | None = None) -> None:
        self.ontology = ontology or get_universal_ontology()

    def lift_detection(
        self,
        entity_id: str,
        class_name: str,
        box_2d: Sequence[float],  # (x1, y1, x2, y2)
        depth_map: np.ndarray,
        camera: PinholeCamera,
        target_frame: str = "world:canonical",
        mask_2d: np.ndarray | None = None,
        min_points: int = 5,
        sensor_depth_sigma_m: float = 0.02,
        confidence: float | None = None,
        source_id: str = "camera0",
    ) -> LiftedEntity3D | None:
        """Lifts a single 2D detection or mask into a metric 3D entity.

        Returns None if points are unobserved, behind camera, or depth is invalid.
        Zero geometry is fabricated.
        """
        x1, y1, x2, y2 = int(round(box_2d[0])), int(round(box_2d[1])), int(round(box_2d[2])), int(round(box_2d[3]))
        dh, dw = depth_map.shape[:2]

        x1 = max(0, min(dw - 1, x1))
        x2 = max(0, min(dw, x2))
        y1 = max(0, min(dh - 1, y1))
        y2 = max(0, min(dh, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        # Sample pixel coordinates and depths
        us: list[int] = []
        vs: list[int] = []
        depths: list[float] = []

        for v in range(y1, y2):
            for u in range(x1, x2):
                if mask_2d is not None and not mask_2d[v, u]:
                    continue
                d = float(depth_map[v, u])
                if math.isfinite(d) and d > 0.05:  # Valid depth > 5 cm
                    us.append(u)
                    vs.append(v)
                    depths.append(d)

        if len(depths) < min_points:
            return None

        depth_arr = np.array(depths, dtype=float)
        # Statistical outlier rejection along depth
        med_d = float(np.median(depth_arr))
        mad_d = float(np.median(np.abs(depth_arr - med_d)))
        tol = max(0.05, 3.0 * mad_d) if mad_d > 0.001 else 0.20 * med_d

        valid_pts_world: list[np.ndarray] = []
        for u, v, d in zip(us, vs, depths):
            if abs(d - med_d) <= tol:
                p_world_vec = camera.unproject(float(u), float(v), float(d))
                valid_pts_world.append(np.array([p_world_vec.x, p_world_vec.y, p_world_vec.z]))

        if len(valid_pts_world) < min_points:
            return None

        pts = np.vstack(valid_pts_world)  # Shape (N, 3)

        # 1. Position: robust median centroid in metres
        pos_x = float(np.median(pts[:, 0]))
        pos_y = float(np.median(pts[:, 1]))
        pos_z = float(np.median(pts[:, 2]))
        position = (pos_x, pos_y, pos_z)

        # 2. Dimensions & Orientation: PCA on X-Y plane
        centered = pts - np.array([pos_x, pos_y, pos_z])
        cov_xy = np.cov(centered[:, :2], rowvar=False)

        try:
            eigvals, eigvecs = np.linalg.eigh(cov_xy)
            # Principal direction along largest variance
            idx = np.argmax(eigvals)
            dir_x, dir_y = float(eigvecs[0, idx]), float(eigvecs[1, idx])
            yaw = math.atan2(dir_y, dir_x)
        except Exception:
            yaw = 0.0

        # Rotate points to object local horizontal frame
        c_yaw = math.cos(-yaw)
        s_yaw = math.sin(-yaw)
        rot_pts_x = centered[:, 0] * c_yaw - centered[:, 1] * s_yaw
        rot_pts_y = centered[:, 0] * s_yaw + centered[:, 1] * c_yaw
        rot_pts_z = centered[:, 2]

        # Use 5th to 95th percentiles to reject noise spikes
        dim_w = float(max(0.01, np.percentile(rot_pts_x, 95) - np.percentile(rot_pts_x, 5)))
        dim_d = float(max(0.01, np.percentile(rot_pts_y, 95) - np.percentile(rot_pts_y, 5)))
        dim_h = float(max(0.01, np.percentile(rot_pts_z, 95) - np.percentile(rot_pts_z, 5)))
        dimensions = (dim_w, dim_d, dim_h)

        # Quaternion (w, x, y, z) representing yaw rotation around Z axis
        half_yaw = yaw / 2.0
        quat = Quat(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
        orientation = (quat.x, quat.y, quat.z, quat.w)

        # 3. Position Covariance
        sample_cov = np.cov(pts, rowvar=False)
        # Covariance of centroid: sample_cov / N + depth_sensor_variance * I
        pos_cov_arr = (sample_cov / max(1, len(pts))) + (sensor_depth_sigma_m ** 2) * np.eye(3)
        covariance_pos = tuple(tuple(float(v) for v in row) for row in pos_cov_arr)

        # Dimension covariance
        dim_cov_arr = (sample_cov / max(1, len(pts))) * 2.0
        covariance_dim = tuple(tuple(float(v) for v in row) for row in dim_cov_arr)

        # Clean resolved class
        resolved_class = self.ontology.resolve_class_name(class_name) or class_name.lower().strip()

        return LiftedEntity3D(
            entity_id=entity_id,
            class_name=resolved_class,
            position=position,
            dimensions=dimensions,
            orientation=orientation,
            coordinate_frame=target_frame,
            covariance_position=covariance_pos,
            covariance_dimensions=covariance_dim,
            point_count=len(pts),
            source_observation_ids=(source_id,),
            confidence=confidence,
            confidence_basis="depth_unprojection_pca" if confidence is not None else None,
        )

    def associate_parts_and_parents(
        self,
        entities: Sequence[LiftedEntity3D],
        containment_tolerance_m: float = 0.15,
    ) -> list[LiftedEntity3D]:
        """Associates subparts to parent objects based on ontology and 3D bounding box proximity.

        For each entity that is a recognized part (e.g. chair_leg, faucet, door_handle),
        finds candidate parents (e.g. chair, sink, door) whose 3D volume contains or contacts
        the part.
        """
        result: list[LiftedEntity3D] = []

        # Index potential parents by class
        potential_parents = [e for e in entities if not self._is_subpart_name(e.class_name)]

        for e in entities:
            if not self._is_subpart_name(e.class_name):
                result.append(e)
                continue

            # This is a subpart; find matching parent
            best_parent: LiftedEntity3D | None = None
            best_dist: float = float("inf")

            for p in potential_parents:
                if self.ontology.is_valid_part(p.class_name, e.class_name):
                    # Check 3D distance between part and parent
                    dist = float(np.linalg.norm(np.array(e.position) - np.array(p.position)))
                    # Check if part is within parent bounding sphere + tolerance
                    parent_max_radius = 0.5 * float(np.linalg.norm(np.array(p.dimensions))) + containment_tolerance_m
                    if dist <= parent_max_radius and dist < best_dist:
                        best_dist = dist
                        best_parent = p

            if best_parent is not None:
                # Compute relative translation
                dx = e.position[0] - best_parent.position[0]
                dy = e.position[1] - best_parent.position[1]
                dz = e.position[2] - best_parent.position[2]
                rel_transform = (
                    (1.0, 0.0, 0.0, dx),
                    (0.0, 1.0, 0.0, dy),
                    (0.0, 0.0, 1.0, dz),
                    (0.0, 0.0, 0.0, 1.0),
                )
                updated = LiftedEntity3D(
                    entity_id=e.entity_id,
                    class_name=e.class_name,
                    position=e.position,
                    dimensions=e.dimensions,
                    orientation=e.orientation,
                    coordinate_frame=e.coordinate_frame,
                    covariance_position=e.covariance_position,
                    covariance_dimensions=e.covariance_dimensions,
                    point_count=e.point_count,
                    parent_entity_id=best_parent.entity_id,
                    is_part=True,
                    relative_transform_to_parent=rel_transform,
                    source_observation_ids=e.source_observation_ids,
                    reprojection_error_px=e.reprojection_error_px,
                    confidence=e.confidence,
                    confidence_basis=e.confidence_basis,
                )
                result.append(updated)
            else:
                # Subpart observed without candidate parent: remains standalone part
                updated = LiftedEntity3D(
                    entity_id=e.entity_id,
                    class_name=e.class_name,
                    position=e.position,
                    dimensions=e.dimensions,
                    orientation=e.orientation,
                    coordinate_frame=e.coordinate_frame,
                    covariance_position=e.covariance_position,
                    covariance_dimensions=e.covariance_dimensions,
                    point_count=e.point_count,
                    parent_entity_id=None,
                    is_part=True,
                    source_observation_ids=e.source_observation_ids,
                    confidence=e.confidence,
                )
                result.append(updated)

        return result

    def to_observation_drafts(self, entity: LiftedEntity3D) -> list[ObservationDraft]:
        """Converts a lifted 3D entity into canonical ObservationDraft records."""
        drafts: list[ObservationDraft] = []
        src = entity.source_observation_ids[0] if entity.source_observation_ids else "reconstruction"

        # 1. Position observation (compatible with canonical.determine(POSITION))
        drafts.append(
            ObservationDraft(
                task="reconstruction_3d",
                quantity="position",
                source_id=src,
                coordinate_system=entity.coordinate_frame,
                predicted={
                    "value": list(entity.position),
                    "entity_id": entity.entity_id,
                    "class_name": entity.class_name,
                    "parent_entity_id": entity.parent_entity_id,
                    "is_part": entity.is_part,
                },
                uncertainty={
                    "covariance": [list(row) for row in entity.covariance_position]
                    if entity.covariance_position else None
                },
                confidence=entity.confidence,
            )
        )

        # 2. Dimensions observation (compatible with canonical.determine(DIMENSIONS))
        drafts.append(
            ObservationDraft(
                task="reconstruction_3d",
                quantity="dimensions",
                source_id=src,
                coordinate_system=entity.coordinate_frame,
                predicted={
                    "value": list(entity.dimensions),
                    "entity_id": entity.entity_id,
                    "class_name": entity.class_name,
                },
                uncertainty={
                    "covariance": [list(row) for row in entity.covariance_dimensions]
                    if entity.covariance_dimensions else None
                },
                confidence=entity.confidence,
            )
        )

        # 3. Full 3D Bounding Box observation
        drafts.append(
            ObservationDraft(
                task="reconstruction_3d",
                quantity="bbox_3d",
                source_id=src,
                coordinate_system=entity.coordinate_frame,
                predicted={
                    "center": list(entity.position),
                    "size": list(entity.dimensions),
                    "orientation": list(entity.orientation),
                    "class_name": entity.class_name,
                    "entity_id": entity.entity_id,
                    "point_count": entity.point_count,
                    "parent_entity_id": entity.parent_entity_id,
                    "is_part": entity.is_part,
                },
                confidence=entity.confidence,
            )
        )

        return drafts

    def _is_subpart_name(self, class_name: str) -> bool:
        """Determines if a class name represents a subpart in the ontology."""
        clean = class_name.lower().strip()
        for parent_canonical in self.ontology._classes:
            if self.ontology.is_valid_part(parent_canonical, clean):
                return True
        return False
