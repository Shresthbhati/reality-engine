"""Kornia algorithmic extraction: Differentiable Image Warping and Geometric Processing.

Implements core differentiable vision primitives inspired by Kornia:
1. Bilinear perspective warping using 3x3 homography matrices.
2. Pinhole camera intrinsic projection and depth unprojection.
3. Spatial image gradient estimation (Sobel operators).
4. Separable Gaussian image filtering.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np


def gaussian_blur_2d(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.5,
) -> np.ndarray:
    """Apply separable 2D Gaussian blur to image (H, W) or (H, W, C)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2

    # 1D Gaussian kernel
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel_1d = np.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d /= np.sum(kernel_1d)

    img = np.asarray(image, dtype=np.float64)
    has_channels = img.ndim == 3
    if not has_channels:
        img = img[:, :, None]

    h, w, c = img.shape
    blurred = np.zeros_like(img)

    for ch in range(c):
        channel = img[:, :, ch]
        # Horizontal pass
        padded_h = np.pad(channel, ((0, 0), (radius, radius)), mode="edge")
        temp = np.zeros((h, w), dtype=np.float64)
        for i in range(kernel_size):
            temp += padded_h[:, i : i + w] * kernel_1d[i]

        # Vertical pass
        padded_v = np.pad(temp, ((radius, radius), (0, 0)), mode="edge")
        out_ch = np.zeros((h, w), dtype=np.float64)
        for i in range(kernel_size):
            out_ch += padded_v[i : i + h, :] * kernel_1d[i]

        blurred[:, :, ch] = out_ch

    if not has_channels:
        return blurred[:, :, 0]
    return blurred


def compute_image_gradients(
    image: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute horizontal (dx), vertical (dy), and magnitude gradients using Sobel kernels."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = np.mean(img, axis=2)

    h, w = img.shape
    padded = np.pad(img, 1, mode="edge")

    # Sobel kernels
    # dx: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] / 8.0
    dx = (
        (padded[:h, 2:w+2] + 2.0 * padded[1:h+1, 2:w+2] + padded[2:h+2, 2:w+2]) -
        (padded[:h, :w] + 2.0 * padded[1:h+1, :w] + padded[2:h+2, :w])
    ) / 8.0

    # dy: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] / 8.0
    dy = (
        (padded[2:h+2, :w] + 2.0 * padded[2:h+2, 1:w+1] + padded[2:h+2, 2:w+2]) -
        (padded[:h, :w] + 2.0 * padded[:h, 1:w+1] + padded[:h, 2:w+2])
    ) / 8.0

    magnitude = np.sqrt(dx ** 2 + dy ** 2)
    return dx, dy, magnitude


def warp_perspective(
    image: np.ndarray,
    homography: np.ndarray,
    output_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Warp image using 3x3 homography matrix with bilinear sampling: p_src ~ H^-1 p_dst."""
    img = np.asarray(image, dtype=np.float64)
    src_h, src_w = img.shape[:2]
    dst_h, dst_w = output_shape if output_shape is not None else (src_h, src_w)

    h_inv = np.linalg.inv(homography)

    # Destination pixel coordinate grid
    ys, xs = np.indices((dst_h, dst_w), dtype=np.float64)
    ones = np.ones_like(xs)
    dst_coords = np.stack([xs.flatten(), ys.flatten(), ones.flatten()], axis=0)  # (3, N)

    # Map to source coordinates
    src_homo = h_inv @ dst_coords
    w_src = src_homo[2, :]
    w_src = np.where(np.abs(w_src) < 1e-8, 1e-8, w_src)
    x_src = src_homo[0, :] / w_src
    y_src = src_homo[1, :] / w_src

    # Bilinear interpolation
    x0 = np.floor(x_src).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(y_src).astype(np.int64)
    y1 = y0 + 1

    wx = (x_src - x0).reshape((dst_h, dst_w))
    wy = (y_src - y0).reshape((dst_h, dst_w))

    valid = (x0 >= 0) & (x1 < src_w) & (y0 >= 0) & (y1 < src_h)
    valid_mask = valid.reshape((dst_h, dst_w))

    x0_c = np.clip(x0, 0, src_w - 1).reshape((dst_h, dst_w))
    x1_c = np.clip(x1, 0, src_w - 1).reshape((dst_h, dst_w))
    y0_c = np.clip(y0, 0, src_h - 1).reshape((dst_h, dst_w))
    y1_c = np.clip(y1, 0, src_h - 1).reshape((dst_h, dst_w))

    has_channels = img.ndim == 3
    num_channels = img.shape[2] if has_channels else 1
    if not has_channels:
        img = img[:, :, None]

    warped = np.zeros((dst_h, dst_w, num_channels), dtype=np.float64)

    for c in range(num_channels):
        c_data = img[:, :, c]
        ia = c_data[y0_c, x0_c]
        ib = c_data[y0_c, x1_c]
        ic = c_data[y1_c, x0_c]
        id_pt = c_data[y1_c, x1_c]

        top = ia * (1.0 - wx) + ib * wx
        bottom = ic * (1.0 - wx) + id_pt * wx
        interpolated = top * (1.0 - wy) + bottom * wy
        warped[:, :, c] = np.where(valid_mask, interpolated, 0.0)

    if not has_channels:
        return warped[:, :, 0]
    return warped


def unproject_depth_to_points(
    depth_map: np.ndarray,
    intrinsic_k: np.ndarray,
) -> np.ndarray:
    """Unproject a 2D depth map into a 3D metric point cloud (N, 3): P = Z * K^-1 [u, v, 1]^T."""
    depth = np.asarray(depth_map, dtype=np.float64)
    h, w = depth.shape
    fx, fy = intrinsic_k[0, 0], intrinsic_k[1, 1]
    cx, cy = intrinsic_k[0, 2], intrinsic_k[1, 2]

    ys, xs = np.indices((h, w), dtype=np.float64)
    valid_mask = (depth > 0.01) & np.isfinite(depth)

    z = depth[valid_mask]
    u = xs[valid_mask]
    v = ys[valid_mask]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_3d = np.stack([x, y, z], axis=-1)
    return points_3d
