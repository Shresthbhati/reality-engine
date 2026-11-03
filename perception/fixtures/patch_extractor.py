"""Orthorectified high-resolution surface patch extractor for planar walls and ceilings (Task 3.4, Epic E05-1).

Removes perspective and oblique distortion by re-projecting calibrated camera imagery
onto detected 3D planar surfaces (walls, ceilings) at a fixed metric scale (e.g. 500 px/m = 2mm/pixel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from reconstruction.calibration.camera import (
    CameraIntrinsics,
    PinholeCamera,
    quat_to_matrix,
)

if TYPE_CHECKING:
    from perception.geometry.planes import DetectedPlane


@dataclass(frozen=True)
class WallPatch:
    """An orthorectified metric image patch extracted from a 3D planar surface."""

    patch_id: str
    source_evidence_id: str
    image: np.ndarray  # (H, W, 3) uint8
    plane_id: str
    plane_normal: Tuple[float, float, float]  # Outward unit normal
    plane_d: float  # normal . p + d = 0
    origin_world: Tuple[float, float, float]  # 3D position corresponding to top-left pixel (0, 0)
    u_axis_world: Tuple[float, float, float]  # Unit vector along patch horizontal (+X / columns)
    v_axis_world: Tuple[float, float, float]  # Unit vector along patch vertical (+Y / rows)
    pixels_per_meter: float
    width_m: float
    height_m: float
    mask: Optional[np.ndarray] = None  # (H, W) bool, True where validly sampled from camera

    @property
    def width_px(self) -> int:
        return self.image.shape[1]

    @property
    def height_px(self) -> int:
        return self.image.shape[0]

    def patch_to_world(self, u_px: float, v_px: float) -> Tuple[float, float, float]:
        """Convert 2D patch pixel coordinate (u_px, v_px) to 3D world coordinate."""
        orig = np.array(self.origin_world, dtype=np.float64)
        u_ax = np.array(self.u_axis_world, dtype=np.float64)
        v_ax = np.array(self.v_axis_world, dtype=np.float64)

        x_m = u_px / float(self.pixels_per_meter)
        y_m = v_px / float(self.pixels_per_meter)

        p = orig + x_m * u_ax + y_m * v_ax
        return (float(p[0]), float(p[1]), float(p[2]))

    def world_to_patch(self, point_world: Sequence[float]) -> Optional[Tuple[float, float]]:
        """Project 3D world point onto patch pixel coordinates (u_px, v_px)."""
        orig = np.array(self.origin_world, dtype=np.float64)
        u_ax = np.array(self.u_axis_world, dtype=np.float64)
        v_ax = np.array(self.v_axis_world, dtype=np.float64)
        p = np.array(point_world, dtype=np.float64)

        rel = p - orig
        u_m = float(np.dot(rel, u_ax))
        v_m = float(np.dot(rel, v_ax))

        u_px = u_m * self.pixels_per_meter
        v_px = v_m * self.pixels_per_meter

        if 0.0 <= u_px <= float(self.width_px) and 0.0 <= v_px <= float(self.height_px):
            return (u_px, v_px)
        return None

    def box_to_world_bounds(
        self,
        box: Tuple[float, float, float, float],
        default_depth_m: float = 0.03,
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
        """Convert a 2D bounding box (x0, y0, x1, y1) in patch pixels to 3D center, dimensions, and normal."""
        x0, y0, x1, y1 = box
        center_u = (x0 + x1) * 0.5
        center_v = (y0 + y1) * 0.5

        center_3d = self.patch_to_world(center_u, center_v)
        width_m = abs(x1 - x0) / float(self.pixels_per_meter)
        height_m = abs(y1 - y0) / float(self.pixels_per_meter)

        return center_3d, (width_m, height_m, default_depth_m), self.plane_normal


class OrthorectifiedPatchExtractor:
    """Extracts metric, frontal, orthorectified image patches from planar surfaces."""

    def __init__(
        self,
        pixels_per_meter: float = 500.0,  # 500 px/m = 2mm/pixel
        default_patch_width_m: float = 1.5,
        default_patch_height_m: float = 1.5,
        min_sampling_ratio: float = 0.1,  # Minimum valid pixel ratio to keep a patch
    ):
        self.pixels_per_meter = max(50.0, float(pixels_per_meter))
        self.default_patch_width_m = default_patch_width_m
        self.default_patch_height_m = default_patch_height_m
        self.min_sampling_ratio = min_sampling_ratio

    @staticmethod
    def compute_plane_basis(normal: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute an orthonormal (u_axis, v_axis, n_axis) basis for a plane normal."""
        n = np.array(normal, dtype=np.float64)
        n_len = np.linalg.norm(n)
        if n_len < 1e-9:
            raise ValueError("Plane normal must be non-zero")
        n = n / n_len

        # If vertical wall (|n_z| < 0.95), horizontal axis u is cross(n, world_up)
        if abs(n[2]) < 0.95:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            u = np.cross(n, up)
            u = u / np.linalg.norm(u)
            v = np.cross(u, n)  # Vertical downwards along wall
            if v[2] > 0:
                v = -v  # Ensure v points downward
        else:
            # Floor or ceiling (|n_z| >= 0.95)
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            u = np.cross(n, ref)
            u = u / np.linalg.norm(u)
            v = np.cross(n, u)

        return u, v, n

    def extract_patch(
        self,
        camera: PinholeCamera,
        image_rgb: np.ndarray,
        plane_normal: Sequence[float],
        plane_d: float,
        evidence_id: str,
        plane_id: str = "plane_0",
        center_world: Optional[Sequence[float]] = None,
        width_m: Optional[float] = None,
        height_m: Optional[float] = None,
    ) -> Optional[WallPatch]:
        """Extract an orthorectified planar patch centered around center_world."""
        u_ax, v_ax, n_ax = self.compute_plane_basis(plane_normal)
        cam_pos = np.array([
            camera.extrinsics.position.x,
            camera.extrinsics.position.y,
            camera.extrinsics.position.z,
        ], dtype=np.float64)

        w_m = float(width_m or self.default_patch_width_m)
        h_m = float(height_m or self.default_patch_height_m)

        # If center_world is not provided, compute camera principal ray intersection with plane
        if center_world is None:
            # Ray through camera center
            R_c2w = np.array(quat_to_matrix(camera.extrinsics.rotation), dtype=np.float64)
            ray_dir = R_c2w[:, 2]  # Principal optical axis (+Z in camera space)

            denom = np.dot(n_ax, ray_dir)
            if abs(denom) < 1e-6:
                return None  # Ray parallel to plane

            # Intersection: n . (cam_pos + lambda * ray_dir) + d = 0
            lam = -(np.dot(n_ax, cam_pos) + plane_d) / denom
            if lam <= 0.1 or lam > 50.0:
                return None  # Behind camera or too far

            p_center = cam_pos + lam * ray_dir
        else:
            p_center = np.array(center_world, dtype=np.float64)
            # Project onto plane to enforce plane equation
            dist_to_plane = np.dot(n_ax, p_center) + plane_d
            p_center = p_center - dist_to_plane * n_ax

        # Check visibility: camera must be on the positive normal side (facing the surface)
        view_vec = p_center - cam_pos
        if np.dot(n_ax, view_vec) >= 0.0:
            # Surface normal points away from camera (backface)
            return None

        # Patch dimensions in pixels
        w_px = max(16, int(round(w_m * self.pixels_per_meter)))
        h_px = max(16, int(round(h_m * self.pixels_per_meter)))

        # Top-left corner in 3D world space
        p_top_left = p_center - (w_m * 0.5) * u_ax - (h_m * 0.5) * v_ax

        # Grid of pixel coordinates
        xs = (np.arange(w_px, dtype=np.float64) + 0.5) / self.pixels_per_meter
        ys = (np.arange(h_px, dtype=np.float64) + 0.5) / self.pixels_per_meter
        grid_x, grid_y = np.meshgrid(xs, ys)

        # 3D coordinates for every patch pixel
        pts_3d = (
            p_top_left[None, None, :]
            + grid_x[:, :, None] * u_ax[None, None, :]
            + grid_y[:, :, None] * v_ax[None, None, :]
        )  # (H, W, 3)

        # Vectorized projection into camera
        R_c2w = np.array(quat_to_matrix(camera.extrinsics.rotation), dtype=np.float64)
        pts_flat = pts_3d.reshape(-1, 3)

        # World to camera: P_cam = (P_world - C) @ R_c2w
        p_cam = (pts_flat - cam_pos) @ R_c2w
        z_cam = p_cam[:, 2]

        in_front = z_cam > 0.05
        map_x = np.full(z_cam.shape, -1.0, dtype=np.float32)
        map_y = np.full(z_cam.shape, -1.0, dtype=np.float32)

        if np.any(in_front):
            valid_p_cam = p_cam[in_front]
            valid_z = z_cam[in_front]

            xn = valid_p_cam[:, 0] / valid_z
            yn = valid_p_cam[:, 1] / valid_z

            intr = camera.intrinsics
            if intr.k1 != 0.0 or intr.k2 != 0.0 or intr.p1 != 0.0 or intr.p2 != 0.0 or intr.k3 != 0.0:
                r2 = xn * xn + yn * yn
                radial = 1.0 + intr.k1 * r2 + intr.k2 * (r2 ** 2) + intr.k3 * (r2 ** 3)
                xd = xn * radial + 2.0 * intr.p1 * xn * yn + intr.p2 * (r2 + 2.0 * xn * xn)
                yd = yn * radial + intr.p1 * (r2 + 2.0 * yn * yn) + 2.0 * intr.p2 * xn * yn
            else:
                xd = xn
                yd = yn

            u_proj = intr.fx * xd + intr.cx
            v_proj = intr.fy * yd + intr.cy

            map_x[in_front] = u_proj.astype(np.float32)
            map_y[in_front] = v_proj.astype(np.float32)

        map_x = map_x.reshape(h_px, w_px)
        map_y = map_y.reshape(h_px, w_px)

        img_h, img_w = image_rgb.shape[:2]
        valid_mask = (map_x >= 0.0) & (map_x <= float(img_w - 1.0)) & (map_y >= 0.0) & (map_y <= float(img_h - 1.0))
        valid_ratio = float(np.mean(valid_mask))

        if valid_ratio < self.min_sampling_ratio:
            return None

        # Sample image using cv2.remap
        rectified = cv2.remap(
            image_rgb,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(128, 128, 128),
        )

        patch_id = f"patch-{evidence_id}-{plane_id}-{w_px}x{h_px}"
        return WallPatch(
            patch_id=patch_id,
            source_evidence_id=evidence_id,
            image=rectified,
            plane_id=plane_id,
            plane_normal=(float(n_ax[0]), float(n_ax[1]), float(n_ax[2])),
            plane_d=float(plane_d),
            origin_world=(float(p_top_left[0]), float(p_top_left[1]), float(p_top_left[2])),
            u_axis_world=(float(u_ax[0]), float(u_ax[1]), float(u_ax[2])),
            v_axis_world=(float(v_ax[0]), float(v_ax[1]), float(v_ax[2])),
            pixels_per_meter=self.pixels_per_meter,
            width_m=w_m,
            height_m=h_m,
            mask=valid_mask,
        )

    def extract_tiled_patches(
        self,
        camera: PinholeCamera,
        image_rgb: np.ndarray,
        plane_normal: Sequence[float],
        plane_d: float,
        evidence_id: str,
        plane_id: str = "plane_0",
        plane_bounds_world: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None,
        tile_size_m: float = 1.5,
        overlap: float = 0.25,
    ) -> List[WallPatch]:
        """Extract a set of overlapping orthorectified patches covering the visible wall plane."""
        patches: List[WallPatch] = []

        if plane_bounds_world is None:
            # Single central patch
            p = self.extract_patch(
                camera=camera,
                image_rgb=image_rgb,
                plane_normal=plane_normal,
                plane_d=plane_d,
                evidence_id=evidence_id,
                plane_id=plane_id,
                width_m=tile_size_m,
                height_m=tile_size_m,
            )
            if p is not None:
                patches.append(p)
            return patches

        bmin, bmax = np.array(plane_bounds_world[0]), np.array(plane_bounds_world[1])
        u_ax, v_ax, _ = self.compute_plane_basis(plane_normal)

        # Project bounding box onto (u, v) axes
        corners = np.array([
            [bmin[0], bmin[1], bmin[2]],
            [bmax[0], bmin[1], bmin[2]],
            [bmin[0], bmax[1], bmin[2]],
            [bmax[0], bmax[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]],
            [bmax[0], bmin[1], bmax[2]],
            [bmin[0], bmax[1], bmax[2]],
            [bmax[0], bmax[1], bmax[2]],
        ])
        u_vals = corners @ u_ax
        v_vals = corners @ v_ax

        u_min, u_max = float(np.min(u_vals)), float(np.max(u_vals))
        v_min, v_max = float(np.min(v_vals)), float(np.max(v_vals))

        step = max(0.5, tile_size_m * (1.0 - overlap))
        u_steps = np.arange(u_min + tile_size_m * 0.5, u_max + step * 0.5, step)
        v_steps = np.arange(v_min + tile_size_m * 0.5, v_max + step * 0.5, step)

        # Center reference point on plane
        p_ref = -(plane_d) * np.array(plane_normal, dtype=np.float64)

        for v_c in v_steps:
            for u_c in u_steps:
                tile_center = p_ref + u_c * u_ax + v_c * v_ax
                tile_patch = self.extract_patch(
                    camera=camera,
                    image_rgb=image_rgb,
                    plane_normal=plane_normal,
                    plane_d=plane_d,
                    evidence_id=evidence_id,
                    plane_id=plane_id,
                    center_world=tile_center,
                    width_m=tile_size_m,
                    height_m=tile_size_m,
                )
                if tile_patch is not None:
                    patches.append(tile_patch)

        return patches
