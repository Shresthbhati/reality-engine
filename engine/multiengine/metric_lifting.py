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
from typing import Any

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
    # Unit quaternion (w, x, y, z) following WorldIR convention, or None if degenerate/symmetric
    orientation: tuple[float, float, float, float] | None
    coordinate_frame: str
    covariance_position: tuple[tuple[float, ...], ...] | None
    covariance_dimensions: tuple[tuple[float, ...], ...] | None
    point_count: int
    rejected_point_count: int = 0
    depth_source: str = "calibrated_metric_depth"
    calibration_source: str = "camera_metadata"
    fitting_method: str = "pca_horizontal_quantile"
    is_orientation_known: bool = True
    orientation_reason: str | None = None
    dimension_uncertainty_explanation: str | None = None
    parent_entity_id: str | None = None
    is_part: bool = False
    relative_transform_to_parent: tuple[tuple[float, ...], ...] | None = None
    source_observation_ids: tuple[str, ...] = ()
    reprojection_error_px: float | None = None
    confidence: float | None = None
    confidence_basis: str | None = None

    @property
    def orientation_quat(self) -> Quat | None:
        """Returns orientation as engine.math.Quat(w, x, y, z) or None."""
        if self.orientation is None:
            return None
        return Quat(self.orientation[0], self.orientation[1], self.orientation[2], self.orientation[3])

    def to_dict(self) -> dict[str, Any]:
        """Explicit dictionary representation with explicit quaternion keys."""
        return {
            "entity_id": self.entity_id,
            "class_name": self.class_name,
            "position": {"x": self.position[0], "y": self.position[1], "z": self.position[2]},
            "dimensions": {"width": self.dimensions[0], "depth": self.dimensions[1], "height": self.dimensions[2]},
            "orientation": (
                {"w": self.orientation[0], "x": self.orientation[1], "y": self.orientation[2], "z": self.orientation[3]}
                if self.orientation is not None
                else None
            ),
            "is_orientation_known": self.is_orientation_known,
            "orientation_reason": self.orientation_reason,
            "coordinate_frame": self.coordinate_frame,
            "point_count": self.point_count,
            "rejected_point_count": self.rejected_point_count,
            "depth_source": self.depth_source,
            "calibration_source": self.calibration_source,
            "fitting_method": self.fitting_method,
            "covariance_position": self.covariance_position,
            "covariance_dimensions": self.covariance_dimensions,
            "dimension_uncertainty_explanation": self.dimension_uncertainty_explanation,
            "parent_entity_id": self.parent_entity_id,
            "is_part": self.is_part,
            "confidence": self.confidence,
            "confidence_basis": self.confidence_basis,
        }


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
        is_metric_depth: bool = True,
        depth_source: str = "calibrated_metric_depth",
        calibration_source: str = "camera_metadata",
        confidence: float | None = None,
        source_id: str = "camera0",
    ) -> LiftedEntity3D | None:
        """Lifts a single 2D detection or mask into a metric 3D entity.

        Returns None if depth is relative/unscaled, masks are incompatible,
        points are unobserved, or geometry is degenerate. Zero geometry is fabricated.
        """
        # 1. Enforce verified metric depth contract
        if not is_metric_depth:
            # Relative monocular depth cannot be unprojected into metric metres without scale alignment
            return None

        # 2. Validate mask dimensions and integrity against depth map
        dh, dw = depth_map.shape[:2]
        if mask_2d is not None:
            if mask_2d.ndim != 2:
                return None
            mh, mw = mask_2d.shape[:2]
            if (mh, mw) != (dh, dw):
                return None
            if not np.any(mask_2d):
                return None

        # 3. Validate bounding box coordinates
        if len(box_2d) < 4:
            return None
        x1_raw, y1_raw, x2_raw, y2_raw = float(box_2d[0]), float(box_2d[1]), float(box_2d[2]), float(box_2d[3])
        if any(not math.isfinite(v) for v in (x1_raw, y1_raw, x2_raw, y2_raw)):
            return None

        x1 = max(0, min(dw - 1, int(round(x1_raw))))
        x2 = max(0, min(dw, int(round(x2_raw))))
        y1 = max(0, min(dh - 1, int(round(y1_raw))))
        y2 = max(0, min(dh, int(round(y2_raw))))

        if x2 <= x1 or y2 <= y1:
            return None

        # 4. Sample pixel coordinates and depths
        us: list[int] = []
        vs: list[int] = []
        depths: list[float] = []
        rejected_count = 0

        for v in range(y1, y2):
            for u in range(x1, x2):
                if mask_2d is not None and not mask_2d[v, u]:
                    rejected_count += 1
                    continue
                d = float(depth_map[v, u])
                if math.isfinite(d) and d > 0.05:  # Valid positive metric depth > 5 cm
                    us.append(u)
                    vs.append(v)
                    depths.append(d)
                else:
                    rejected_count += 1

        if len(depths) < min_points:
            return None

        # 5. Statistical outlier rejection along depth (MAD)
        depth_arr = np.array(depths, dtype=float)
        med_d = float(np.median(depth_arr))
        mad_d = float(np.median(np.abs(depth_arr - med_d)))
        tol = max(0.05, 3.0 * mad_d) if mad_d > 0.001 else 0.20 * med_d

        valid_pts_world: list[np.ndarray] = []
        for u, v, d in zip(us, vs, depths):
            if abs(d - med_d) <= tol:
                p_world_vec = camera.unproject(float(u), float(v), float(d))
                valid_pts_world.append(np.array([p_world_vec.x, p_world_vec.y, p_world_vec.z]))
            else:
                rejected_count += 1

        if len(valid_pts_world) < min_points:
            return None

        pts = np.vstack(valid_pts_world)  # Shape (N, 3)

        # 6. Position: robust median centroid in metres
        pos_x = float(np.median(pts[:, 0]))
        pos_y = float(np.median(pts[:, 1]))
        pos_z = float(np.median(pts[:, 2]))
        position = (pos_x, pos_y, pos_z)

        # 7. Planar analysis on horizontal plane (X-Y)
        centered = pts - np.array([pos_x, pos_y, pos_z])
        cov_xy = np.cov(centered[:, :2], rowvar=False)

        is_orientation_known = True
        orientation_reason: str | None = None
        orientation_tuple: tuple[float, float, float, float] | None = None

        try:
            eigvals, eigvecs = np.linalg.eigh(cov_xy)
            sorted_indices = np.argsort(eigvals)
            lam0 = float(eigvals[sorted_indices[0]])
            lam1 = float(eigvals[sorted_indices[1]])

            # Check degeneracy: if max planar variance < 1e-6 m^2, points are degenerate/collinear
            if lam1 < 1e-6:
                is_orientation_known = False
                orientation_reason = "degenerate_collinear_planar_points"
                yaw = 0.0
            # Check rotational symmetry: if eigenvalues are nearly equal (|l1-l0|/(l1+l0) < 0.15)
            elif (lam1 - lam0) / max(1e-9, lam1 + lam0) < 0.15:
                is_orientation_known = False
                orientation_reason = "planar_symmetry_circular_eigenvalues"
                yaw = 0.0
            else:
                idx = sorted_indices[1]  # Dominant axis
                dir_x, dir_y = float(eigvecs[0, idx]), float(eigvecs[1, idx])
                yaw = math.atan2(dir_y, dir_x)
                half_yaw = yaw / 2.0
                # Canonical Quat(w, x, y, z) following WorldIR
                quat = Quat(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))
                orientation_tuple = (float(quat.w), float(quat.x), float(quat.y), float(quat.z))
                orientation_reason = "pca_principal_eigenvector"
        except Exception as e:
            is_orientation_known = False
            orientation_reason = f"pca_eigenvalue_solver_failure: {e}"
            yaw = 0.0

        # Rotate points to object local horizontal frame
        c_yaw = math.cos(-yaw)
        s_yaw = math.sin(-yaw)
        rot_pts_x = centered[:, 0] * c_yaw - centered[:, 1] * s_yaw
        rot_pts_y = centered[:, 0] * s_yaw + centered[:, 1] * c_yaw
        rot_pts_z = centered[:, 2]

        # 8. Dimensions: 5th to 95th percentiles to reject edge noise
        raw_w = float(np.percentile(rot_pts_x, 95) - np.percentile(rot_pts_x, 5))
        raw_d = float(np.percentile(rot_pts_y, 95) - np.percentile(rot_pts_y, 5))
        raw_h = float(np.percentile(rot_pts_z, 95) - np.percentile(rot_pts_z, 5))

        # Check for degeneracy: primary observed dimensions must be physically positive (> 5 mm)
        if raw_w < 0.005 or raw_d < 0.005:
            return None

        # Depth thickness: report actual observed extent honestly (0.0 for fronto-parallel planar surface)
        # without faking a clamp to 0.01 m
        dim_h = round(max(0.0, raw_h), 4)
        dimensions = (round(raw_w, 4), round(raw_d, 4), dim_h)

        # 9. Statistically justified covariance matrices
        n_pts = len(pts)
        sample_cov = np.cov(pts, rowvar=False)
        pos_cov_arr = (sample_cov / max(1, n_pts)) + (sensor_depth_sigma_m ** 2) * np.eye(3)
        covariance_pos = tuple(tuple(float(v) for v in row) for row in pos_cov_arr)

        covariance_dim: tuple[tuple[float, ...], ...] | None = None
        dim_uncer_explanation: str | None = None

        if n_pts >= 20:
            # Statistical order-statistic quantile variance: Var(Q_p) ≈ p(1-p) / (N * f(Q_p)^2)
            # Bandwidth approximation for quantile density f(q)
            try:
                iqr_x = max(1e-4, float(np.percentile(rot_pts_x, 75) - np.percentile(rot_pts_x, 25)))
                iqr_y = max(1e-4, float(np.percentile(rot_pts_y, 75) - np.percentile(rot_pts_y, 25)))
                iqr_z = max(1e-4, float(np.percentile(rot_pts_z, 75) - np.percentile(rot_pts_z, 25)))

                var_w = (0.05 * 0.95) / (n_pts * (1.0 / iqr_x) ** 2) + (sensor_depth_sigma_m ** 2)
                var_d = (0.05 * 0.95) / (n_pts * (1.0 / iqr_y) ** 2) + (sensor_depth_sigma_m ** 2)
                var_h = (0.05 * 0.95) / (n_pts * (1.0 / iqr_z) ** 2) + (sensor_depth_sigma_m ** 2)

                dim_cov_mat = np.diag([var_w, var_d, var_h])
                covariance_dim = tuple(tuple(float(v) for v in row) for row in dim_cov_mat)
                dim_uncer_explanation = "order_statistic_quantile_variance_plus_sensor_noise"
            except Exception:
                covariance_dim = None
                dim_uncer_explanation = "quantile_density_estimation_error"
        else:
            covariance_dim = None
            dim_uncer_explanation = f"insufficient_point_density_for_quantile_variance (n={n_pts} < 20)"

        resolved_class = self.ontology.resolve_class_name(class_name) or class_name.lower().strip()

        return LiftedEntity3D(
            entity_id=entity_id,
            class_name=resolved_class,
            position=position,
            dimensions=dimensions,
            orientation=orientation_tuple,
            coordinate_frame=target_frame,
            covariance_position=covariance_pos,
            covariance_dimensions=covariance_dim,
            point_count=n_pts,
            rejected_point_count=rejected_count,
            depth_source=depth_source,
            calibration_source=calibration_source,
            fitting_method="pca_horizontal_quantile",
            is_orientation_known=is_orientation_known,
            orientation_reason=orientation_reason,
            dimension_uncertainty_explanation=dim_uncer_explanation,
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
                    "orientation": list(entity.orientation) if entity.orientation is not None else None,
                    "is_orientation_known": entity.is_orientation_known,
                    "orientation_reason": entity.orientation_reason,
                    "class_name": entity.class_name,
                    "entity_id": entity.entity_id,
                    "point_count": entity.point_count,
                    "rejected_point_count": entity.rejected_point_count,
                    "depth_source": entity.depth_source,
                    "calibration_source": entity.calibration_source,
                    "fitting_method": entity.fitting_method,
                    "dimension_uncertainty_explanation": entity.dimension_uncertainty_explanation,
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
