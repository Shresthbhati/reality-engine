"""Spatial Understanding, Robust Depth Aggregation, and 3D Lifting (Phase 2).

Implements:
1. Pinhole ray unprojection with vectorized foreground depth aggregation.
2. Robust outlier rejection (MAD depth filtering + SAM mask isolation).
3. 3D centroid, oriented bounding box (PCA yaw alignment), and extent estimation.
4. Rigorous covariance propagation from pixel & depth variance to 3D spatial covariance.
5. Surface normal estimation via eigenvalue planar decomposition.
6. Spatial relationship reasoning (supported_by, mounted_on_wall, contained_in, adjacent_to).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from world_ir.camera import PinholeCamera, Quat


class SpatialRelationKind(str, Enum):
    SUPPORTED_BY = "supported_by"  # A rests on top of B
    SUPPORTS = "supports"          # B supports A
    MOUNTED_ON = "mounted_on"      # A is attached/mounted to surface B (e.g. wall)
    CONTAINS = "contains"          # B geometrically encloses or hosts A
    PART_OF = "part_of"            # A is a component of composite entity B
    ADJACENT_TO = "adjacent_to"    # A and B are adjacent within physical clearance
    FACING = "facing"              # A's front normal points towards B


@dataclass(frozen=True)
class SpatialRelationship:
    """Directed spatial relationship between two physical entities."""

    source_id: str
    target_id: str
    relation: SpatialRelationKind
    confidence: float
    distance_m: float
    basis: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation.value,
            "confidence": round(self.confidence, 3),
            "distance_m": round(self.distance_m, 3),
            "basis": self.basis,
            "details": self.details,
        }


@dataclass(frozen=True)
class SpatialEntity3D:
    """Evidence-backed 3D physical entity in the world coordinate frame."""

    entity_id: str
    class_name: str
    position_m: tuple[float, float, float]  # Centroid (X, Y, Z) in world frame
    dimensions_m: tuple[float, float, float]  # Width, Depth, Height
    orientation_quat_wxyz: tuple[float, float, float, float]  # (w, x, y, z)
    covariance_xyz: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    surface_normal: tuple[float, float, float] | None = None
    planarity_score: float | None = None
    primitive_type: str = "box"  # box, plane, cylinder, point_proxy
    confidence: float = 0.5
    scale_state: str = "relative_depth"
    scale_units: str = "relative"
    is_physically_metric: bool = False
    source_image_id: str = ""
    box_2d: tuple[float, float, float, float] | None = None
    mask_pixels_count: int = 0
    point_cloud_sample: tuple[tuple[float, float, float], ...] = ()
    is_truncated_by_border: bool = False
    viewing_angle_deg: float | None = None
    dimension_uncertainty_m: tuple[float, float, float] = (0.05, 0.05, 0.05)

    @property
    def bounds_min(self) -> tuple[float, float, float]:
        cx, cy, cz = self.position_m
        w, d, h = self.dimensions_m
        return (cx - 0.5 * w, cy - 0.5 * d, cz - 0.5 * h)

    @property
    def bounds_max(self) -> tuple[float, float, float]:
        cx, cy, cz = self.position_m
        w, d, h = self.dimensions_m
        return (cx + 0.5 * w, cy + 0.5 * d, cz + 0.5 * h)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "class_name": self.class_name,
            "position_m": [round(c, 4) for c in self.position_m],
            "dimensions_m": [round(d, 4) for d in self.dimensions_m],
            "dimension_uncertainty_m": [round(u, 4) for u in self.dimension_uncertainty_m],
            "orientation_quat_wxyz": [round(q, 4) for q in self.orientation_quat_wxyz],
            "covariance_xyz": [[round(v, 6) for v in row] for row in self.covariance_xyz],
            "surface_normal": [round(n, 4) for n in self.surface_normal] if self.surface_normal else None,
            "planarity_score": round(self.planarity_score, 4) if self.planarity_score is not None else None,
            "primitive_type": self.primitive_type,
            "confidence": round(self.confidence, 4),
            "scale_state": self.scale_state,
            "scale_units": self.scale_units,
            "is_physically_metric": self.is_physically_metric,
            "source_image_id": self.source_image_id,
            "box_2d": list(self.box_2d) if self.box_2d else None,
            "mask_pixels_count": self.mask_pixels_count,
            "is_truncated_by_border": self.is_truncated_by_border,
            "viewing_angle_deg": round(self.viewing_angle_deg, 1) if self.viewing_angle_deg is not None else None,
        }


def propagate_unprojection_covariance(
    u: float,
    v: float,
    depth: float,
    camera: PinholeCamera,
    pixel_sigma_px: float = 2.0,
    depth_relative_sigma: float = 0.04,
) -> np.ndarray:
    """Computes 3x3 spatial covariance matrix Sigma_xyz in world coordinates via Jacobian propagation.

    Unprojection mapping:
      X_cam = (u - cx) * depth / fx
      Y_cam = (v - cy) * depth / fy
      Z_cam = depth
      P_world = C + R_wc @ P_cam
    """
    fx = camera.intrinsics.fx
    fy = camera.intrinsics.fy
    cx = camera.intrinsics.cx
    cy = camera.intrinsics.cy

    sigma_u = max(0.5, pixel_sigma_px)
    sigma_v = max(0.5, pixel_sigma_px)
    sigma_z = max(0.01, depth * depth_relative_sigma)

    # Jacobian of P_cam with respect to [u, v, depth]
    J_cam = np.array([
        [depth / fx, 0.0, (u - cx) / fx],
        [0.0, depth / fy, (v - cy) / fy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    Sigma_uvz = np.diag([sigma_u ** 2, sigma_v ** 2, sigma_z ** 2])
    Sigma_cam = J_cam @ Sigma_uvz @ J_cam.T

    # World rotation matrix R_wc
    q = camera.extrinsics.rotation
    R_wc = np.array([
        [1.0 - 2.0 * (q.y ** 2 + q.z ** 2), 2.0 * (q.x * q.y - q.z * q.w), 2.0 * (q.x * q.z + q.y * q.w)],
        [2.0 * (q.x * q.y + q.z * q.w), 1.0 - 2.0 * (q.x ** 2 + q.z ** 2), 2.0 * (q.y * q.z - q.x * q.w)],
        [2.0 * (q.x * q.z - q.y * q.w), 2.0 * (q.y * q.z + q.x * q.w), 1.0 - 2.0 * (q.x ** 2 + q.y ** 2)],
    ], dtype=np.float64)

    Sigma_world = R_wc @ Sigma_cam @ R_wc.T
    return Sigma_world


def lift_region_to_spatial_entity(
    entity_id: str,
    class_name: str,
    box_2d: Sequence[float],
    depth_map: np.ndarray,
    camera: PinholeCamera,
    mask_2d: np.ndarray | None = None,
    min_points: int = 5,
    confidence: float = 0.5,
    source_image_id: str = "",
    scale_state: str = "relative_depth",
    scale_units: str = "relative",
    is_physically_metric: bool = False,
) -> SpatialEntity3D | None:
    """Lifts a 2D bounding box and optional instance mask into a 3D SpatialEntity3D."""
    dh, dw = depth_map.shape[:2]
    x1 = max(0, min(dw - 1, int(round(box_2d[0]))))
    y1 = max(0, min(dh - 1, int(round(box_2d[1]))))
    x2 = max(0, min(dw, int(round(box_2d[2]))))
    y2 = max(0, min(dh, int(round(box_2d[3]))))

    if x2 <= x1 or y2 <= y1:
        return None

    depth_crop = depth_map[y1:y2, x1:x2]
    valid_mask = np.isfinite(depth_crop) & (depth_crop > 0.05)

    if mask_2d is not None and mask_2d.shape[:2] == (dh, dw):
        crop_mask = mask_2d[y1:y2, x1:x2].astype(bool)
        if np.any(crop_mask):
            valid_mask = valid_mask & crop_mask

    v_rel, u_rel = np.where(valid_mask)
    if len(u_rel) < min_points:
        return None

    u_all = u_rel + x1
    v_all = v_rel + y1
    d_all = depth_crop[valid_mask]

    # Robust depth filtering via Median Absolute Deviation (MAD)
    med_d = float(np.median(d_all))
    mad_d = float(np.median(np.abs(d_all - med_d)))
    tol = max(0.04, 2.5 * mad_d) if mad_d > 0.001 else 0.15 * med_d

    inliers = np.abs(d_all - med_d) <= tol
    if np.sum(inliers) < min_points:
        return None

    u_inliers = u_all[inliers]
    v_inliers = v_all[inliers]
    d_inliers = d_all[inliers]

    # Subsample if large
    if len(d_inliers) > 2000:
        step = max(1, len(d_inliers) // 2000)
        u_inliers = u_inliers[::step]
        v_inliers = v_inliers[::step]
        d_inliers = d_inliers[::step]

    pts_world: list[np.ndarray] = []
    for u, v, d in zip(u_inliers, v_inliers, d_inliers):
        p_vec = camera.unproject(float(u), float(v), float(d))
        pts_world.append(np.array([p_vec.x, p_vec.y, p_vec.z]))

    pts = np.vstack(pts_world)  # (N, 3)

    # Centroid
    cx, cy, cz = float(np.median(pts[:, 0])), float(np.median(pts[:, 1])), float(np.median(pts[:, 2]))
    centroid = (cx, cy, cz)

    # Planar PCA and surface normal estimation
    centered = pts - np.array([cx, cy, cz])
    cov_3d = np.cov(centered, rowvar=False)

    normal_vec = None
    planarity = None
    yaw = 0.0
    quat_tuple = (1.0, 0.0, 0.0, 0.0)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov_3d)
        order = np.argsort(eigvals)
        l0, l1, l2 = eigvals[order[0]], eigvals[order[1]], eigvals[order[2]]
        sum_l = max(1e-9, l0 + l1 + l2)
        planarity = float(1.0 - (3.0 * l0 / sum_l))

        # Surface normal is the eigenvector of smallest variance
        n_raw = eigvecs[:, order[0]]
        # Orient normal towards camera
        cam_pos = np.array([camera.extrinsics.position.x, camera.extrinsics.position.y, camera.extrinsics.position.z])
        view_dir = cam_pos - np.array([cx, cy, cz])
        if np.dot(n_raw, view_dir) < 0:
            n_raw = -n_raw
        normal_vec = (float(n_raw[0]), float(n_raw[1]), float(n_raw[2]))

        # Principal horizontal axis for yaw
        h_vec = eigvecs[:2, order[2]]
        yaw = float(math.atan2(h_vec[1], h_vec[0]))
        half_y = yaw / 2.0
        q = Quat(math.cos(half_y), 0.0, 0.0, math.sin(half_y)).normalized()
        quat_tuple = (float(q.w), float(q.x), float(q.y), float(q.z))
    except Exception:
        pass

    # Dimensions in object local frame
    c_y = math.cos(-yaw)
    s_y = math.sin(-yaw)
    rot_x = centered[:, 0] * c_y - centered[:, 1] * s_y
    rot_y = centered[:, 0] * s_y + centered[:, 1] * c_y
    rot_z = centered[:, 2]

    w_dim = max(0.04, float(np.percentile(rot_x, 98) - np.percentile(rot_x, 2)))
    d_dim = max(0.04, float(np.percentile(rot_y, 98) - np.percentile(rot_y, 2)))
    h_dim = max(0.04, float(np.percentile(rot_z, 98) - np.percentile(rot_z, 2)))
    dimensions = (w_dim, d_dim, h_dim)

    # Uncertainty covariance propagation at central point
    u_mid = 0.5 * (x1 + x2)
    v_mid = 0.5 * (y1 + y2)
    sigma_world = propagate_unprojection_covariance(u_mid, v_mid, med_d, camera)
    cov_tuple = (
        (float(sigma_world[0, 0]), float(sigma_world[0, 1]), float(sigma_world[0, 2])),
        (float(sigma_world[1, 0]), float(sigma_world[1, 1]), float(sigma_world[1, 2])),
        (float(sigma_world[2, 0]), float(sigma_world[2, 1]), float(sigma_world[2, 2])),
    )
    dim_uncert = (
        float(max(0.01, math.sqrt(max(1e-6, sigma_world[0, 0])))),
        float(max(0.01, math.sqrt(max(1e-6, sigma_world[1, 1])))),
        float(max(0.01, math.sqrt(max(1e-6, sigma_world[2, 2])))),
    )

    # Primitive type selection
    cls_lower = class_name.lower()
    if planarity is not None and planarity > 0.85 and any(k in cls_lower for k in ("wall", "floor", "ceiling", "window", "door", "glazing", "mirror", "artwork", "carpet", "road", "footpath")):
        prim_type = "plane"
    elif any(k in cls_lower for k in ("column", "pole", "pipe", "trunk", "leg", "lamppost", "bollard", "hydrant", "fan", "lamp")):
        prim_type = "cylinder"
    elif min(w_dim, d_dim, h_dim) <= 0.05 and planarity is not None and planarity > 0.80:
        prim_type = "plane"
    else:
        prim_type = "box"

    # Border truncation check
    is_truncated = (x1 <= 2 or y1 <= 2 or x2 >= dw - 2 or y2 >= dh - 2)

    # Viewing angle
    view_angle_deg = None
    if normal_vec is not None:
        cam_p = np.array([camera.extrinsics.position.x, camera.extrinsics.position.y, camera.extrinsics.position.z])
        v_dir = cam_p - np.array([cx, cy, cz])
        v_dist = np.linalg.norm(v_dir)
        if v_dist > 1e-4:
            cos_a = float(np.dot(normal_vec, v_dir / v_dist))
            cos_a = max(-1.0, min(1.0, cos_a))
            view_angle_deg = float(math.acos(cos_a) * 180.0 / math.pi)

    # Sample representative point cloud
    sample_indices = np.linspace(0, len(pts) - 1, min(16, len(pts)), dtype=int)
    sample_tuple = tuple((float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])) for i in sample_indices)

    return SpatialEntity3D(
        entity_id=entity_id,
        class_name=class_name,
        position_m=centroid,
        dimensions_m=dimensions,
        orientation_quat_wxyz=quat_tuple,
        covariance_xyz=cov_tuple,
        surface_normal=normal_vec,
        planarity_score=planarity,
        primitive_type=prim_type,
        confidence=confidence,
        scale_state=scale_state,
        scale_units=scale_units,
        is_physically_metric=is_physically_metric,
        source_image_id=source_image_id,
        box_2d=(float(x1), float(y1), float(x2), float(y2)),
        mask_pixels_count=len(u_inliers),
        point_cloud_sample=sample_tuple,
        is_truncated_by_border=is_truncated,
        viewing_angle_deg=view_angle_deg,
        dimension_uncertainty_m=dim_uncert,
    )


def infer_spatial_relationships(
    entities: Sequence[SpatialEntity3D],
    wall_attachment_dist_m: float = 0.25,
    support_resting_dist_m: float = 0.15,
) -> list[SpatialRelationship]:
    """Infers evidence-grounded spatial relationships between reconstructed 3D entities."""
    relationships: list[SpatialRelationship] = []
    n = len(entities)

    for i in range(n):
        ent_a = entities[i]
        cls_a = ent_a.class_name.lower()
        pos_a = np.array(ent_a.position_m)
        bmin_a = np.array(ent_a.bounds_min)
        bmax_a = np.array(ent_a.bounds_max)

        for j in range(n):
            if i == j:
                continue
            ent_b = entities[j]
            cls_b = ent_b.class_name.lower()
            pos_b = np.array(ent_b.position_m)
            bmin_b = np.array(ent_b.bounds_min)
            bmax_b = np.array(ent_b.bounds_max)

            dist_centers = float(np.linalg.norm(pos_a - pos_b))

            # 1. Wall Mounting (e.g. socket, switch, sconce mounted on wall)
            if "wall" in cls_b and any(k in cls_a for k in ("socket", "switch", "lamp", "sconce", "mirror", "frame", "door", "window")):
                # Check distance from ent_a to wall surface
                horiz_dist = float(np.hypot(pos_a[0] - pos_b[0], pos_a[1] - pos_b[1]))
                if horiz_dist <= wall_attachment_dist_m:
                    relationships.append(SpatialRelationship(
                        source_id=ent_a.entity_id,
                        target_id=ent_b.entity_id,
                        relation=SpatialRelationKind.MOUNTED_ON,
                        confidence=0.85,
                        distance_m=horiz_dist,
                        basis="proximity_to_wall_plane",
                    ))

            # 2. Floor / Surface Resting & Support (e.g. chair resting on floor, lamp resting on table)
            if "floor" in cls_b or "table" in cls_b or "desk" in cls_b:
                # Vertical clearance between bottom of A and top of B
                vert_clearance = abs(bmin_a[2] - bmax_b[2])
                # Horizontal footprint overlap
                x_overlap = max(0.0, min(bmax_a[0], bmax_b[0]) - max(bmin_a[0], bmin_b[0]))
                y_overlap = max(0.0, min(bmax_a[1], bmax_b[1]) - max(bmin_a[1], bmin_b[1]))
                footprint_overlap = (x_overlap > 0.05 and y_overlap > 0.05) if "floor" not in cls_b else True

                if vert_clearance <= support_resting_dist_m and footprint_overlap:
                    relationships.append(SpatialRelationship(
                        source_id=ent_a.entity_id,
                        target_id=ent_b.entity_id,
                        relation=SpatialRelationKind.SUPPORTED_BY,
                        confidence=0.80,
                        distance_m=vert_clearance,
                        basis="vertical_contact_footprint_overlap",
                    ))
                    relationships.append(SpatialRelationship(
                        source_id=ent_b.entity_id,
                        target_id=ent_a.entity_id,
                        relation=SpatialRelationKind.SUPPORTS,
                        confidence=0.80,
                        distance_m=vert_clearance,
                        basis="vertical_contact_footprint_overlap",
                    ))

            # 3. Containment / Part-of (e.g. handle is part of door)
            if "door" in cls_b and "handle" in cls_a:
                if dist_centers <= 0.8:
                    relationships.append(SpatialRelationship(
                        source_id=ent_a.entity_id,
                        target_id=ent_b.entity_id,
                        relation=SpatialRelationKind.PART_OF,
                        confidence=0.90,
                        distance_m=dist_centers,
                        basis="semantic_part_hierarchy",
                    ))

            # 4. Adjacent objects (e.g. chair near desk)
            if any(k in cls_a for k in ("chair", "table", "desk", "sofa")) and any(k in cls_b for k in ("chair", "table", "desk", "sofa")):
                if 0.1 < dist_centers <= 1.2:
                    relationships.append(SpatialRelationship(
                        source_id=ent_a.entity_id,
                        target_id=ent_b.entity_id,
                        relation=SpatialRelationKind.ADJACENT_TO,
                        confidence=0.75,
                        distance_m=dist_centers,
                        basis="horizontal_physical_proximity",
                    ))

    return relationships
