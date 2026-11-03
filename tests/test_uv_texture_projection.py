"""Tests for Seam-Blended UV Texture Projection (Epic E06-2)."""

from __future__ import annotations

import math
import numpy as np
import pytest

from engine.render.textures.projection import (
    PinholeCamera,
    PlanarSurface,
    TextureAtlas,
    measure_seam_discontinuity,
    multi_band_seam_blend,
    project_and_blend_planar_texture,
)


def _make_camera(cam_id: str, pos_world: tuple[float, float, float], look_dir: tuple[float, float, float], img: np.ndarray) -> PinholeCamera:
    # Camera looking along look_dir
    pos = np.array(pos_world, dtype=np.float64)
    z_axis = np.array(look_dir, dtype=np.float64)
    z_axis /= np.linalg.norm(z_axis)

    up = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(z_axis, up)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    # R is rows [x_axis, y_axis, z_axis]
    R = np.vstack([x_axis, y_axis, z_axis])
    t = -R @ pos

    T_wc = np.eye(4)
    T_wc[:3, :3] = R
    T_wc[:3, 3] = t

    H, W = img.shape[:2]
    K = [
        [float(W), 0.0, float(W) * 0.5],
        [0.0, float(W), float(H) * 0.5],
        [0.0, 0.0, 1.0],
    ]
    return PinholeCamera(camera_id=cam_id, intrinsics=K, world_to_cam=T_wc.tolist(), image_rgb=img)


def test_pinhole_camera_projection_and_bilinear_sampling():
    # Synthetic gradient image
    img = np.zeros((100, 100, 3), dtype=np.float64)
    img[:, :, 0] = np.linspace(0.0, 1.0, 100)[:, np.newaxis]
    img[:, :, 1] = 0.5

    cam = _make_camera("cam-0", (0.0, -3.0, 1.5), (0.0, 1.0, 0.0), img)

    # 3D point straight in front of camera
    pts = np.array([[0.0, 0.0, 1.5], [1.0, 0.0, 1.5]])
    pixels, depth = cam.project(pts)

    assert math.isclose(depth[0], 3.0, abs_tol=1e-3)
    assert math.isclose(pixels[0, 0], 50.0, abs_tol=1e-2)  # center X
    assert math.isclose(pixels[0, 1], 50.0, abs_tol=1e-2)  # center Y

    colors, valid = cam.sample_bilinear(pixels)
    assert bool(valid[0]) is True
    assert math.isclose(colors[0, 1], 0.5, abs_tol=1e-2)


def test_multi_band_seam_blending_eliminates_discontinuity():
    """Verify that multi-band seam blending eliminates sharp exposure/lighting boundaries."""
    # Create two synthetic 128x128 images with a 30% exposure step across a vertical seam
    H, W = 128, 128
    # Image A is brighter: base color 0.70 with high-frequency brick texture
    img_a = np.ones((H, W, 3), dtype=np.float64) * 0.70
    # Add subtle high-frequency pattern
    x = np.arange(W)
    img_a[:, :, 0] += 0.05 * np.sin(x * 0.5)

    # Image B is darker: base color 0.45 with same high-frequency pattern
    img_b = np.ones((H, W, 3), dtype=np.float64) * 0.45
    img_b[:, :, 0] += 0.05 * np.sin(x * 0.5)

    # Seam down the middle at x = 64
    mask_a = np.zeros((H, W), dtype=np.float64)
    mask_a[:, :64] = 1.0

    # Unblended raw stitch has a large jump: 0.70 vs 0.45 = 0.25 jump
    raw_stitch = img_a * mask_a[:, :, np.newaxis] + img_b * (1.0 - mask_a[:, :, np.newaxis])
    raw_discontinuity = measure_seam_discontinuity(raw_stitch, mask_a)
    assert raw_discontinuity >= 0.20

    # Apply multi-band seam blending
    blended = multi_band_seam_blend(img_a, img_b, mask_a, levels=4)
    blended_discontinuity = measure_seam_discontinuity(blended, mask_a)

    print(f"\nSeam Discontinuity: Raw={raw_discontinuity:.4f} -> Blended={blended_discontinuity:.4f}")

    # Success criteria: smooth transition with zero visible seam boundary step
    assert blended_discontinuity < 0.05
    assert blended_discontinuity < raw_discontinuity * 0.25  # > 75% reduction in step gradient


def test_planar_surface_texture_atlas_generation():
    # 4x3m wall surface in X-Z plane at Y = 0
    surface = PlanarSurface(
        origin=(-2.0, 0.0, 0.0),
        u_axis=(1.0, 0.0, 0.0),
        v_axis=(0.0, 0.0, 1.0),
        normal=(0.0, -1.0, 0.0),  # normal facing -Y towards cameras
        width_m=4.0,
        height_m=3.0,
    )

    # Two cameras: Left camera at (-1.0, -3.0, 1.5), Right camera at (1.0, -3.0, 1.5)
    img_left = np.ones((120, 120, 3), dtype=np.float64) * 0.75
    img_right = np.ones((120, 120, 3), dtype=np.float64) * 0.60

    cam_left = _make_camera("cam-left", (-1.0, -3.0, 1.5), (0.0, 1.0, 0.0), img_left)
    cam_right = _make_camera("cam-right", (1.0, -3.0, 1.5), (0.0, 1.0, 0.0), img_right)

    atlas = project_and_blend_planar_texture(
        surface=surface,
        cameras=[cam_left, cam_right],
        texture_width=256,
        texture_height=256,
        enable_seam_blending=True,
    )

    assert atlas.width == 256
    assert atlas.height == 256
    assert atlas.pixels.shape == (256, 256, 3)
    assert atlas.seams_blended == 1
    assert atlas.max_seam_discontinuity < 0.05

    u8 = atlas.to_uint8()
    assert u8.shape == (256, 256, 3)
    assert u8.dtype == np.uint8
