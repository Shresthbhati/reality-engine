"""Unit tests for perceptual hash deduplication (Master Development Plan Task 2.6)."""

import numpy as np
import pytest
from PIL import Image

from evidence.dedup import (
    compute_dhash,
    compute_phash,
    deduplicate_images,
    hamming_distance,
    hash_to_hex,
    hex_to_hash,
)


def _create_synthetic_image(pattern: str, size=(128, 128), seed: int = 42) -> Image.Image:
    """Generate deterministic test images with distinct visual structures."""
    rng = np.random.default_rng(seed)
    w, h = size
    if pattern == "horizontal_gradient":
        arr = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
        return Image.fromarray(arr, mode="L")
    elif pattern == "vertical_gradient":
        arr = np.tile(np.linspace(0, 255, h, dtype=np.uint8)[:, None], (1, w))
        return Image.fromarray(arr, mode="L")
    elif pattern == "checkerboard":
        arr = np.zeros((h, w), dtype=np.uint8)
        arr[::16, ::16] = 255
        return Image.fromarray(arr, mode="L")
    elif pattern == "noise":
        arr = rng.integers(0, 256, (h, w), dtype=np.uint8)
        return Image.fromarray(arr, mode="L")
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def test_dhash_and_phash_identity():
    img1 = _create_synthetic_image("horizontal_gradient")
    img2 = _create_synthetic_image("horizontal_gradient")

    h1_d = compute_dhash(img1)
    h2_d = compute_dhash(img2)
    assert h1_d == h2_d
    assert hamming_distance(h1_d, h2_d) == 0

    h1_p = compute_phash(img1)
    h2_p = compute_phash(img2)
    assert h1_p == h2_p
    assert hamming_distance(h1_p, h2_p) == 0


def test_dhash_robustness_to_scaling_and_minor_noise():
    # Large 256x256 image vs downscaled 64x64 copy
    img_large = _create_synthetic_image("horizontal_gradient", size=(256, 256))
    img_small = img_large.resize((64, 64), Image.Resampling.BILINEAR)

    h_large = compute_dhash(img_large)
    h_small = compute_dhash(img_small)
    assert hamming_distance(h_large, h_small) <= 2


def test_phash_dissimilar_images_large_distance():
    img_grad = _create_synthetic_image("horizontal_gradient")
    img_check = _create_synthetic_image("checkerboard")

    h_grad = compute_phash(img_grad)
    h_check = compute_phash(img_check)
    dist = hamming_distance(h_grad, h_check)
    # Distinct structural frequencies should differ by at least 15 bits out of 64
    assert dist >= 15


def test_hex_conversion_roundtrip():
    hash_val = 0x123456789ABCDEF0
    hex_str = hash_to_hex(hash_val)
    assert len(hex_str) == 16
    assert hex_to_hash(hex_str) == hash_val


def test_deduplicate_images_with_quality_aware_selection():
    base_img = _create_synthetic_image("horizontal_gradient")
    # Duplicate with slight brightness shift
    arr_dup = np.clip(np.asarray(base_img, dtype=np.int16) + 5, 0, 255).astype(np.uint8)
    dup_img = Image.fromarray(arr_dup, mode="L")

    # Distinct second image
    other_img = _create_synthetic_image("checkerboard")

    items = [
        ("img-01", base_img),
        ("img-02-dup", dup_img),
        ("img-03-other", other_img),
    ]

    # Explicit quality scores: img-02-dup has higher sharpness than img-01
    qualities = {
        "img-01": 45.0,
        "img-02-dup": 110.0,
        "img-03-other": 80.0,
    }

    result = deduplicate_images(items, max_distance=6, hash_type="dhash", quality_scores=qualities)

    assert result.total_input == 3
    assert result.duplicate_count == 1

    # img-02-dup was higher quality, so it should be retained as canonical
    assert "img-02-dup" in result.retained_ids
    assert "img-01" in result.duplicate_ids
    assert "img-03-other" in result.retained_ids

    # Check duplicate records
    dup_rec = result.duplicates[0]
    assert dup_rec.canonical_id == "img-02-dup"
    assert dup_rec.duplicate_id == "img-01"
    assert dup_rec.distance <= 6
