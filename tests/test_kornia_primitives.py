"""Unit tests for Kornia differentiable vision primitives extraction."""

import numpy as np
import pytest

from reconstruction.features.kornia_primitives import (
    compute_image_gradients,
    gaussian_blur_2d,
    unproject_depth_to_points,
    warp_perspective,
)


def test_gaussian_blur_smoothing():
    """Verify Gaussian blur attenuates high-frequency delta impulse."""
    img = np.zeros((31, 31), dtype=np.float64)
    img[15, 15] = 100.0  # Impulse at center

    blurred = gaussian_blur_2d(img, kernel_size=7, sigma=1.5)

    assert blurred[15, 15] < 100.0
    assert blurred[15, 15] > blurred[15, 16] > blurred[15, 17]
    # Energy preservation
    assert np.isclose(np.sum(blurred), 100.0, rtol=0.05)


def test_sobel_gradient_estimation():
    """Verify Sobel operator computes exact gradient for horizontal linear ramp."""
    # Linear ramp in x: I(x, y) = 2.0 * x
    xs = np.arange(20, dtype=np.float64) * 2.0
    img = np.tile(xs, (20, 1))

    dx, dy, mag = compute_image_gradients(img)

    # In interior pixels, dx should be approximately 2.0 and dy should be 0.0
    assert np.allclose(dx[2:18, 2:18], 2.0, atol=0.05)
    assert np.allclose(dy[2:18, 2:18], 0.0, atol=0.05)
    assert np.allclose(mag[2:18, 2:18], 2.0, atol=0.05)


def test_perspective_warp_identity_and_translation():
    """Verify homography warp preserves image under identity and shifts under translation."""
    img = np.zeros((40, 40), dtype=np.float64)
    img[15:25, 15:25] = 1.0  # Center square

    # Identity warp
    h_eye = np.eye(3, dtype=np.float64)
    warped_eye = warp_perspective(img, h_eye)
    assert np.allclose(img[1:39, 1:39], warped_eye[1:39, 1:39], atol=0.01)

    # Translation by +5 in x, +5 in y
    h_trans = np.array([
        [1.0, 0.0, 5.0],
        [0.0, 1.0, 5.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    warped_trans = warp_perspective(img, h_trans)

    # Center of original square [20, 20] should now be at [25, 25]
    assert warped_trans[25, 25] > 0.9
    assert warped_trans[15, 15] < 0.1


def test_unproject_depth_to_metric_points():
    """Verify unprojecting constant 2.0m depth map matches pinhole geometry."""
    depth = np.full((10, 10), 2.0, dtype=np.float64)
    k = np.array([
        [500.0, 0.0, 5.0],
        [0.0, 500.0, 5.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    pts = unproject_depth_to_points(depth, k)

    assert len(pts) == 100
    # All points have Z = 2.0m
    assert np.allclose(pts[:, 2], 2.0)
    # Principal point [5, 5] maps to [0, 0, 2.0]
    center_pt = pts[5 * 10 + 5]
    assert np.allclose(center_pt, [0.0, 0.0, 2.0], atol=1e-5)
