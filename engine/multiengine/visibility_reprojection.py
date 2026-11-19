"""Image-Backed Reprojection and Visibility Validation using Camera Geometry.

Provides rigorous multi-view geometric validation for 3D reconstructed entities against
calibrated pinhole camera parameters, poses, image boundaries, and depth-based occlusion.

Principles:
1. Camera Geometry:
   Uses the neutral `PinholeCamera` contract (from `world_ir.camera`) supporting
   Brown-Conrady radial and tangential distortion (k1, k2, p1, p2, k3).
2. Visibility Verification:
   A 3D entity is visible in a camera view if and only if:
   a. It lies strictly in front of the camera plane (Z_cam > 0.001 m).
   b. Its projected coordinate falls strictly inside the image frame [0, width] x [0, height].
   c. It is not occluded by closer surfaces (verified against a metric depth map if present).
3. 2D-3D Consistency:
   Reprojection error is measured as the Euclidean pixel distance between projected 3D center
   and observed 2D center. In addition, the 2D bounding box / convex hull of the projected 3D
   box is evaluated against observed 2D detections via 2D Intersection-over-Union (IoU).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.math import Quat, Vec3
from world_ir.camera import CameraExtrinsics, CameraIntrinsics, PinholeCamera


@dataclass(frozen=True)
class VisibilityResult:
    """Visibility outcome for a single 3D point in a camera view."""

    is_visible: bool
    is_in_front: bool
    is_in_frustum: bool
    is_unoccluded: bool
    camera_depth_m: float
    pixel_coord: tuple[float, float] | None
    occlusion_depth_m: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class BoxVisibilityResult:
    """Visibility outcome for a 3D bounding box across camera views."""

    center_visible: bool
    visible_corner_count: int
    projected_bbox_2d: tuple[float, float, float, float] | None  # (u_min, v_min, u_max, v_max) in px
    projected_center_2d: tuple[float, float] | None
    projected_corners_2d: tuple[tuple[float, float], ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class ReprojectionEvaluation:
    """Reprojection and consistency evaluation between a 3D entity and a 2D observation."""

    is_visible: bool
    reprojection_error_px: float | None
    iou_2d: float | None
    is_consistent: bool
    observed_region: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    projected_region: tuple[float, float, float, float] | None
    details: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class ValidatedCameraMetadata:
    """Rigorous camera validation record preserving provenance, status, and uncertainty."""

    is_calibrated: bool
    camera: PinholeCamera | None
    source: str | None = None
    method: str | None = None
    version: str | None = None
    coordinate_frame: str = "canonical_world"
    uncertainty: dict[str, Any] | None = None
    rejection_reason: str | None = None


class VisibilityReprojectionValidator:
    """Evaluates multi-view camera visibility and geometric reprojection consistency."""

    @classmethod
    def validate_camera_metadata(cls, cam_meta: dict[str, Any]) -> ValidatedCameraMetadata:
        """Validates camera metadata without fabricating defaults.

        Returns ValidatedCameraMetadata with is_calibrated=True and PinholeCamera if valid,
        or is_calibrated=False and camera=None with explicit rejection_reason.
        """
        source = cam_meta.get("calibration_source") or cam_meta.get("source")
        method = cam_meta.get("calibration_method") or cam_meta.get("method")
        version = cam_meta.get("calibration_version") or cam_meta.get("version")
        coord_frame = str(cam_meta.get("coordinate_frame", "canonical_world"))
        uncertainty = cam_meta.get("uncertainty") if isinstance(cam_meta.get("uncertainty"), dict) else None

        intr_raw = cam_meta.get("intrinsics")
        if intr_raw is None:
            return ValidatedCameraMetadata(
                is_calibrated=False, camera=None, source=source, method=method,
                version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                rejection_reason="missing_intrinsics",
            )

        # 1. Parse and validate intrinsics parameters
        fx: float | None = None
        fy: float | None = None
        cx: float | None = None
        cy: float | None = None
        w: int | None = None
        h: int | None = None
        k1, k2, p1, p2, k3 = 0.0, 0.0, 0.0, 0.0, 0.0

        if isinstance(intr_raw, dict):
            fx_val = intr_raw.get("fx")
            fy_val = intr_raw.get("fy")
            cx_val = intr_raw.get("cx")
            cy_val = intr_raw.get("cy")
            if None in (fx_val, fy_val, cx_val, cy_val):
                return ValidatedCameraMetadata(
                    is_calibrated=False, camera=None, source=source, method=method,
                    version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                    rejection_reason="missing_focal_or_principal_point",
                )
            fx = float(fx_val)
            fy = float(fy_val)
            cx = float(cx_val)
            cy = float(cy_val)

            w_val = intr_raw.get("width") or cam_meta.get("width")
            h_val = intr_raw.get("height") or cam_meta.get("height")
            if (w_val is None or h_val is None) and isinstance(cam_meta.get("resolution"), (list, tuple)) and len(cam_meta["resolution"]) >= 2:
                w_val = cam_meta["resolution"][0]
                h_val = cam_meta["resolution"][1]

            if w_val is None or h_val is None:
                return ValidatedCameraMetadata(
                    is_calibrated=False, camera=None, source=source, method=method,
                    version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                    rejection_reason="missing_image_dimensions",
                )
            w = int(w_val)
            h = int(h_val)
            k1 = float(intr_raw.get("k1", 0.0))
            k2 = float(intr_raw.get("k2", 0.0))
            p1 = float(intr_raw.get("p1", 0.0))
            p2 = float(intr_raw.get("p2", 0.0))
            k3 = float(intr_raw.get("k3", 0.0))

        elif isinstance(intr_raw, (list, tuple)):
            if len(intr_raw) >= 4 and not isinstance(intr_raw[0], (list, tuple)):
                fx = float(intr_raw[0])
                fy = float(intr_raw[1])
                cx = float(intr_raw[2])
                cy = float(intr_raw[3])
            elif len(intr_raw) >= 3 and isinstance(intr_raw[0], (list, tuple)) and len(intr_raw[0]) >= 3:
                fx = float(intr_raw[0][0])
                fy = float(intr_raw[1][1])
                cx = float(intr_raw[0][2])
                cy = float(intr_raw[1][2])
            else:
                return ValidatedCameraMetadata(
                    is_calibrated=False, camera=None, source=source, method=method,
                    version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                    rejection_reason="malformed_intrinsics_sequence",
                )

            w_val = cam_meta.get("width")
            h_val = cam_meta.get("height")
            if (w_val is None or h_val is None) and isinstance(cam_meta.get("resolution"), (list, tuple)) and len(cam_meta["resolution"]) >= 2:
                w_val = cam_meta["resolution"][0]
                h_val = cam_meta["resolution"][1]

            if w_val is None or h_val is None:
                return ValidatedCameraMetadata(
                    is_calibrated=False, camera=None, source=source, method=method,
                    version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                    rejection_reason="missing_image_dimensions",
                )
            w = int(w_val)
            h = int(h_val)
        else:
            return ValidatedCameraMetadata(
                is_calibrated=False, camera=None, source=source, method=method,
                version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                rejection_reason="unsupported_intrinsics_type",
            )

        # Validate finite values and positivity
        intr_vals = (fx, fy, cx, cy, float(w), float(h), k1, k2, p1, p2, k3)
        if any(not math.isfinite(v) for v in intr_vals):
            return ValidatedCameraMetadata(
                is_calibrated=False, camera=None, source=source, method=method,
                version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                rejection_reason="non_finite_intrinsics_or_distortion",
            )
        if fx <= 0.0 or fy <= 0.0:
            return ValidatedCameraMetadata(
                is_calibrated=False, camera=None, source=source, method=method,
                version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                rejection_reason="non_positive_focal_length",
            )
        if w <= 0 or h <= 0:
            return ValidatedCameraMetadata(
                is_calibrated=False, camera=None, source=source, method=method,
                version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                rejection_reason="non_positive_image_dimensions",
            )

        try:
            intr = CameraIntrinsics(
                fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h,
                k1=k1, k2=k2, p1=p1, p2=p2, k3=k3,
            )
        except Exception as e:
            return ValidatedCameraMetadata(
                is_calibrated=False, camera=None, source=source, method=method,
                version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                rejection_reason=f"intrinsics_constructor_error: {e}",
            )

        # 2. Build and validate CameraExtrinsics
        extr_raw = cam_meta.get("extrinsics") or cam_meta.get("pose_4x4")
        if extr_raw is not None:
            if isinstance(extr_raw, dict) and "position" in extr_raw and "rotation" in extr_raw:
                try:
                    p = extr_raw["position"]
                    pos_x = float(p[0] if isinstance(p, (list, tuple)) else p["x"])
                    pos_y = float(p[1] if isinstance(p, (list, tuple)) else p["y"])
                    pos_z = float(p[2] if isinstance(p, (list, tuple)) else p["z"])

                    r = extr_raw["rotation"]
                    qw = float(r[0] if isinstance(r, (list, tuple)) else r["w"])
                    qx = float(r[1] if isinstance(r, (list, tuple)) else r["x"])
                    qy = float(r[2] if isinstance(r, (list, tuple)) else r["y"])
                    qz = float(r[3] if isinstance(r, (list, tuple)) else r["z"])

                    if not all(math.isfinite(v) for v in (pos_x, pos_y, pos_z, qw, qx, qy, qz)):
                        return ValidatedCameraMetadata(
                            is_calibrated=False, camera=None, source=source, method=method,
                            version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                            rejection_reason="non_finite_extrinsics_values",
                        )

                    q_norm_sq = qw * qw + qx * qx + qy * qy + qz * qz
                    if abs(q_norm_sq - 1.0) > 1e-3:
                        return ValidatedCameraMetadata(
                            is_calibrated=False, camera=None, source=source, method=method,
                            version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                            rejection_reason=f"non_unit_quaternion_norm_{math.sqrt(q_norm_sq):.4f}",
                        )

                    quat = Quat(qw, qx, qy, qz).normalized()
                    extr = CameraExtrinsics(position=Vec3(pos_x, pos_y, pos_z), rotation=quat)
                except Exception as exc:
                    return ValidatedCameraMetadata(
                        is_calibrated=False, camera=None, source=source, method=method,
                        version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                        rejection_reason=f"extrinsics_dict_parse_error_{exc}",
                    )
            else:
                try:
                    mat = np.array(extr_raw, dtype=float)
                except Exception:
                    return ValidatedCameraMetadata(
                        is_calibrated=False, camera=None, source=source, method=method,
                        version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                        rejection_reason="extrinsics_array_conversion_error",
                    )

                if mat.shape not in ((4, 4), (3, 4)):
                    return ValidatedCameraMetadata(
                        is_calibrated=False, camera=None, source=source, method=method,
                        version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                        rejection_reason=f"malformed_extrinsics_shape_{mat.shape}",
                    )

                if not np.all(np.isfinite(mat)):
                    return ValidatedCameraMetadata(
                        is_calibrated=False, camera=None, source=source, method=method,
                        version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                        rejection_reason="non_finite_extrinsics_matrix",
                    )

                # Check if it's world-to-cam [R|t] or cam-to-world [R_c2w | pos]
                if cam_meta.get("is_world_to_cam", False):
                    R_w2c = mat[:3, :3]
                    t_w2c = mat[:3, 3]
                    R_c2w = R_w2c.T
                    pos = -R_c2w @ t_w2c
                else:
                    R_c2w = mat[:3, :3]
                    pos = mat[:3, 3]

                # Validate rigidity of rotation matrix: det(R) == +1.0 and R^T R == I
                det_r = float(np.linalg.det(R_c2w))
                if abs(det_r - 1.0) > 1e-3:
                    return ValidatedCameraMetadata(
                        is_calibrated=False, camera=None, source=source, method=method,
                        version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                        rejection_reason=f"non_rigid_rotation_determinant_{det_r:.4f}",
                    )

                ortho_dev = float(np.max(np.abs(R_c2w.T @ R_c2w - np.eye(3))))
                if ortho_dev > 1e-3:
                    return ValidatedCameraMetadata(
                        is_calibrated=False, camera=None, source=source, method=method,
                        version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                        rejection_reason=f"non_orthogonal_rotation_matrix_dev_{ortho_dev:.4f}",
                    )

                rows = (
                    (float(R_c2w[0, 0]), float(R_c2w[0, 1]), float(R_c2w[0, 2])),
                    (float(R_c2w[1, 0]), float(R_c2w[1, 1]), float(R_c2w[1, 2])),
                    (float(R_c2w[2, 0]), float(R_c2w[2, 1]), float(R_c2w[2, 2])),
                )
                from reconstruction.calibration.transforms import _quat_from_matrix_rows
                quat = _quat_from_matrix_rows(rows)
                extr = CameraExtrinsics(position=Vec3(pos[0], pos[1], pos[2]), rotation=quat)

        elif cam_meta.get("is_canonical_frame", False):
            # Identity pose is permitted ONLY when the source explicitly establishes canonical frame
            extr = CameraExtrinsics(position=Vec3(0.0, 0.0, 0.0), rotation=Quat.identity())
        else:
            return ValidatedCameraMetadata(
                is_calibrated=False, camera=None, source=source, method=method,
                version=version, coordinate_frame=coord_frame, uncertainty=uncertainty,
                rejection_reason="missing_extrinsics_and_not_canonical_frame",
            )

        pinhole = PinholeCamera(intrinsics=intr, extrinsics=extr)
        return ValidatedCameraMetadata(
            is_calibrated=True,
            camera=pinhole,
            source=source,
            method=method,
            version=version,
            coordinate_frame=coord_frame,
            uncertainty=uncertainty,
            rejection_reason=None,
        )

    @classmethod
    def camera_from_metadata(cls, cam_meta: dict[str, Any]) -> PinholeCamera | None:
        """Constructs a validated PinholeCamera from raw camera metadata dictionaries.

        Returns None if missing calibration or malformed. Never invents defaults.
        """
        validated = cls.validate_camera_metadata(cam_meta)
        return validated.camera

    @classmethod
    def check_point_visibility(
        cls,
        camera: PinholeCamera,
        point_world: Sequence[float],
        depth_map: np.ndarray | None = None,
        depth_tolerance_m: float = 0.05,
    ) -> VisibilityResult:
        """Evaluates whether a 3D world point is visible in the given camera view."""
        p_world = Vec3(float(point_world[0]), float(point_world[1]), float(point_world[2]))
        p_cam = camera.extrinsics.world_to_camera(p_world)

        depth_m = float(p_cam.z)
        if depth_m <= 0.001:
            return VisibilityResult(
                is_visible=False,
                is_in_front=False,
                is_in_frustum=False,
                is_unoccluded=False,
                camera_depth_m=depth_m,
                pixel_coord=None,
                reason="Point is behind or on the camera focal plane (Z_cam <= 0.001 m)",
            )

        proj = camera.project(p_world)
        if proj is None:
            return VisibilityResult(
                is_visible=False,
                is_in_front=True,
                is_in_frustum=False,
                is_unoccluded=False,
                camera_depth_m=depth_m,
                pixel_coord=None,
                reason="Projection failed or point is outside projectable cone",
            )

        u, v = proj
        w = camera.intrinsics.width
        h = camera.intrinsics.height
        in_frustum = (0.0 <= u <= float(w)) and (0.0 <= v <= float(h))
        if not in_frustum:
            return VisibilityResult(
                is_visible=False,
                is_in_front=True,
                is_in_frustum=False,
                is_unoccluded=False,
                camera_depth_m=depth_m,
                pixel_coord=(u, v),
                reason=f"Point projects outside image bounds ({u:.1f}, {v:.1f}) not in [{w}x{h}]",
            )

        # Depth occlusion check
        if depth_map is not None:
            iu = int(round(u))
            iv = int(round(v))
            dh, dw = depth_map.shape[:2]
            if 0 <= iu < dw and 0 <= iv < dh:
                measured_d = float(depth_map[iv, iu])
                # If depth map contains valid positive depth and surface is closer than point by > tolerance
                effective_tol = max(depth_tolerance_m, 0.05 * depth_m)
                if measured_d > 0.01 and measured_d < (depth_m - effective_tol):
                    return VisibilityResult(
                        is_visible=False,
                        is_in_front=True,
                        is_in_frustum=True,
                        is_unoccluded=False,
                        camera_depth_m=depth_m,
                        pixel_coord=(u, v),
                        occlusion_depth_m=measured_d,
                        reason=f"Occluded: foreground depth ({measured_d:.2f} m) is closer than point ({depth_m:.2f} m)",
                    )

        return VisibilityResult(
            is_visible=True,
            is_in_front=True,
            is_in_frustum=True,
            is_unoccluded=True,
            camera_depth_m=depth_m,
            pixel_coord=(u, v),
            reason="Point is in front, inside frustum, and unoccluded",
        )

    @classmethod
    def check_box_visibility(
        cls,
        camera: PinholeCamera,
        center: Sequence[float],
        size: Sequence[float],
        depth_map: np.ndarray | None = None,
    ) -> BoxVisibilityResult:
        """Projects a 3D bounding box into 2D pixel coordinates and verifies corner visibility."""
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        sx, sy, sz = float(size[0]), float(size[1]), float(size[2])
        hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

        # 8 bounding box corners
        corners = [
            (cx - hx, cy - hy, cz - hz),
            (cx + hx, cy - hy, cz - hz),
            (cx + hx, cy + hy, cz - hz),
            (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz),
            (cx + hx, cy - hy, cz + hz),
            (cx + hx, cy + hy, cz + hz),
            (cx - hx, cy + hy, cz + hz),
        ]

        center_vis = cls.check_point_visibility(camera, (cx, cy, cz), depth_map=depth_map)
        proj_center = center_vis.pixel_coord if center_vis.is_visible else None

        valid_corners_2d = []
        for corner in corners:
            vis = cls.check_point_visibility(camera, corner, depth_map=depth_map)
            if vis.pixel_coord is not None:
                valid_corners_2d.append(vis.pixel_coord)

        if not valid_corners_2d:
            return BoxVisibilityResult(
                center_visible=center_vis.is_visible,
                visible_corner_count=0,
                projected_bbox_2d=None,
                projected_center_2d=proj_center,
                reason="Zero corners projected into camera frame",
            )

        xs = [p[0] for p in valid_corners_2d]
        ys = [p[1] for p in valid_corners_2d]
        bbox_2d = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

        return BoxVisibilityResult(
            center_visible=center_vis.is_visible,
            visible_corner_count=len(valid_corners_2d),
            projected_bbox_2d=bbox_2d,
            projected_center_2d=proj_center,
            projected_corners_2d=tuple(valid_corners_2d),
            reason=f"{len(valid_corners_2d)}/8 corners visible",
        )

    @classmethod
    def evaluate_reprojection(
        cls,
        camera: PinholeCamera,
        center_3d: Sequence[float],
        size_3d: Sequence[float],
        observed_box_2d: Sequence[float],
        depth_map: np.ndarray | None = None,
        max_error_px: float = 20.0,
        min_iou: float = 0.20,
    ) -> ReprojectionEvaluation:
        """Evaluates Euclidean reprojection error and 2D IoU for a 3D box against a 2D detection."""
        box_vis = cls.check_box_visibility(camera, center_3d, size_3d, depth_map=depth_map)

        obs_x1, obs_y1, obs_x2, obs_y2 = (
            float(observed_box_2d[0]),
            float(observed_box_2d[1]),
            float(observed_box_2d[2]),
            float(observed_box_2d[3]),
        )
        obs_region = (obs_x1, obs_y1, obs_x2, obs_y2)
        obs_cx = (obs_x1 + obs_x2) / 2.0
        obs_cy = (obs_y1 + obs_y2) / 2.0

        if box_vis.projected_bbox_2d is None or box_vis.projected_center_2d is None:
            return ReprojectionEvaluation(
                is_visible=False,
                reprojection_error_px=None,
                iou_2d=None,
                is_consistent=False,
                observed_region=obs_region,
                projected_region=None,
                reason=f"3D entity not visible: {box_vis.reason}",
            )

        proj_x1, proj_y1, proj_x2, proj_y2 = box_vis.projected_bbox_2d
        proj_cx, proj_cy = box_vis.projected_center_2d

        # 1. Reprojection Euclidean error in pixels
        err_px = float(math.hypot(proj_cx - obs_cx, proj_cy - obs_cy))

        # 2. 2D Bounding Box IoU
        ix1 = max(obs_x1, proj_x1)
        iy1 = max(obs_y1, proj_y1)
        ix2 = min(obs_x2, proj_x2)
        iy2 = min(obs_y2, proj_y2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        intersection = iw * ih

        area_obs = max(0.0, (obs_x2 - obs_x1) * (obs_y2 - obs_y1))
        area_proj = max(0.0, (proj_x2 - proj_x1) * (proj_y2 - proj_y1))
        union = area_obs + area_proj - intersection

        iou = float(intersection / union) if union > 0.0 else 0.0

        is_consistent = (err_px <= max_error_px) and (iou >= min_iou)
        reason = (
            f"Reprojection verified (error: {err_px:.2f} px, IoU: {iou:.3f})"
            if is_consistent
            else f"Reprojection mismatch: error {err_px:.2f} px (limit: {max_error_px}) or IoU {iou:.3f} (limit: {min_iou})"
        )

        return ReprojectionEvaluation(
            is_visible=True,
            reprojection_error_px=err_px,
            iou_2d=iou,
            is_consistent=is_consistent,
            observed_region=obs_region,
            projected_region=box_vis.projected_bbox_2d,
            details={
                "projected_center": (proj_cx, proj_cy),
                "observed_center": (obs_cx, obs_cy),
                "visible_corners": box_vis.visible_corner_count,
            },
            reason=reason,
        )
