"""Perceptual hash deduplication (Master Development Plan Task 2.6).

Identifies and filters near-duplicate photographic frames and video keyframes
using perceptual hashing (dHash, pHash) and Hamming distance metrics.

Key principles:
1. Deterministic & Evidence-grounded:
   - Same image bytes always produce the exact same 64-bit integer hash.
2. Quality-Aware Keeper Selection:
   - When near-duplicate images are identified (Hamming distance <= threshold),
     the candidate with the highest measured quality score (e.g. Laplacian sharpness)
     is retained as canonical, rather than arbitrarily picking the first-seen item.
3. Fast 64-bit Hash Representations:
   - Difference Hash (dHash): $9 \times 8$ gradient comparison, robust to exposure/scale changes.
   - Perceptual Hash (pHash): $32 \times 32$ 2D Discrete Cosine Transform (DCT) low-frequency median split.
   - Hamming distance via native bitwise XOR and `.bit_count()`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

# Default Hamming distance threshold for 64-bit perceptual hashes:
# <= 6 bits difference out of 64 typically indicates the same scene with minor camera jitter/exposure variance.
DEFAULT_DHASH_THRESHOLD = 6
DEFAULT_PHASH_THRESHOLD = 8


# -----------------------------------------------------------------------------
# Core Perceptual Hash Algorithms
# -----------------------------------------------------------------------------

def _load_as_grayscale_array(
    source: Union[Path, str, Image.Image, np.ndarray],
    target_size: Tuple[int, int],
) -> np.ndarray:
    """Load image from path, PIL Image, or array and resize to target_size (width, height) grayscale."""
    if isinstance(source, (str, Path)):
        with Image.open(source) as img:
            gray_img = img.convert("L").resize(target_size, Image.Resampling.BILINEAR)
            return np.asarray(gray_img, dtype=np.float32)
    elif isinstance(source, Image.Image):
        gray_img = source.convert("L").resize(target_size, Image.Resampling.BILINEAR)
        return np.asarray(gray_img, dtype=np.float32)
    elif isinstance(source, np.ndarray):
        if source.ndim == 3 and source.shape[2] >= 3:
            # RGB to Gray
            gray = 0.2989 * source[:, :, 0] + 0.5870 * source[:, :, 1] + 0.1140 * source[:, :, 2]
        else:
            gray = source.squeeze()
        pil_img = Image.fromarray(gray.astype(np.uint8)).resize(target_size, Image.Resampling.BILINEAR)
        return np.asarray(pil_img, dtype=np.float32)
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")


def compute_dhash(source: Union[Path, str, Image.Image, np.ndarray]) -> int:
    """Compute 64-bit difference hash (dHash).

    Downscales image to 9 columns x 8 rows grayscale, compares adjacent columns
    (I(x+1, y) > I(x, y)), yielding 64 comparison bits packed into an integer.
    """
    arr = _load_as_grayscale_array(source, (9, 8))  # width=9, height=8
    # Compare adjacent pixels horizontally: 8 rows of 8 comparisons = 64 bits
    diff = arr[:, 1:] > arr[:, :-1]
    bits = diff.flatten()

    hash_val = 0
    for b in bits:
        hash_val = (hash_val << 1) | int(b)
    return hash_val


def _build_dct_matrix(n: int = 32) -> np.ndarray:
    """Precompute NxN 1D DCT-II orthonormal transform matrix."""
    mat = np.zeros((n, n), dtype=np.float64)
    factor = math.pi / (2.0 * n)
    c0 = 1.0 / math.sqrt(n)
    c1 = math.sqrt(2.0 / n)
    for u in range(n):
        c = c0 if u == 0 else c1
        for x in range(n):
            mat[u, x] = c * math.cos((2 * x + 1) * u * factor)
    return mat


_DCT_32 = _build_dct_matrix(32)


def compute_phash(source: Union[Path, str, Image.Image, np.ndarray]) -> int:
    """Compute 64-bit Discrete Cosine Transform perceptual hash (pHash).

    Downscales image to 32x32 grayscale, applies 2D DCT, extracts top-left 8x8
    low-frequency coefficients (excluding DC component (0,0)), compares against median.
    """
    arr = _load_as_grayscale_array(source, (32, 32))  # width=32, height=32
    # 2D DCT: C * A * C^T
    dct_2d = _DCT_32 @ arr @ _DCT_32.T

    # Extract 8x8 low frequencies
    low_freq = dct_2d[:8, :8]
    # Median over all 64 coefficients
    med = float(np.median(low_freq))

    # 64 bits: 1 if coefficient > median else 0
    bits = (low_freq > med).flatten()

    hash_val = 0
    for b in bits:
        hash_val = (hash_val << 1) | int(b)
    return hash_val


def hamming_distance(hash1: int, hash2: int) -> int:
    """Compute Hamming distance (number of differing bits) between two 64-bit integer hashes."""
    return (hash1 ^ hash2).bit_count()


def hash_to_hex(hash_val: int) -> str:
    """Format 64-bit hash as 16-character hexadecimal string."""
    return f"{hash_val:016x}"


def hex_to_hash(hex_str: str) -> int:
    """Parse 16-character hexadecimal string back to 64-bit integer."""
    return int(hex_str, 16)


# -----------------------------------------------------------------------------
# Deduplication Engine
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class DuplicateRecord:
    """Details of an identified duplicate image pair."""

    canonical_id: str
    duplicate_id: str
    distance: int
    hash_type: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_id": self.canonical_id,
            "duplicate_id": self.duplicate_id,
            "distance": self.distance,
            "hash_type": self.hash_type,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DeduplicationResult:
    """Overall outcome of a dataset deduplication pass."""

    retained_ids: List[str]
    duplicate_ids: List[str]
    duplicates: List[DuplicateRecord]
    total_input: int
    duplicate_count: int
    hash_type: str
    threshold: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "retained_ids": list(self.retained_ids),
            "duplicate_ids": list(self.duplicate_ids),
            "duplicates": [d.to_dict() for d in self.duplicates],
            "total_input": self.total_input,
            "duplicate_count": self.duplicate_count,
            "hash_type": self.hash_type,
            "threshold": self.threshold,
        }


def deduplicate_images(
    items: Sequence[Tuple[str, Union[Path, str, Image.Image, np.ndarray]]],
    max_distance: Optional[int] = None,
    hash_type: str = "dhash",
    quality_scores: Optional[Dict[str, float]] = None,
) -> DeduplicationResult:
    """Deduplicate a sequence of (item_id, image_source) pairs.

    Args:
        items: Sequence of (id, image) where image is a file path, PIL Image, or numpy array.
        max_distance: Max Hamming distance to consider two images duplicates (default: 6 for dhash, 8 for phash).
        hash_type: "dhash" (fast, exposure-robust) or "phash" (frequency-based).
        quality_scores: Optional map of item_id -> quality_score (e.g. Laplacian sharpness).
                       When duplicates are found, the item with the highest quality score is
                       retained as canonical.
    """
    if hash_type == "dhash":
        hasher: Callable[[Any], int] = compute_dhash
        threshold = max_distance if max_distance is not None else DEFAULT_DHASH_THRESHOLD
    elif hash_type == "phash":
        hasher = compute_phash
        threshold = max_distance if max_distance is not None else DEFAULT_PHASH_THRESHOLD
    else:
        raise ValueError(f"Unsupported hash_type: {hash_type}; expected 'dhash' or 'phash'")

    n = len(items)
    if n == 0:
        return DeduplicationResult(
            retained_ids=[],
            duplicate_ids=[],
            duplicates=[],
            total_input=0,
            duplicate_count=0,
            hash_type=hash_type,
            threshold=threshold,
        )

    # 1. Compute hashes
    item_ids = [item[0] for item in items]
    hashes = [hasher(item[1]) for item in items]
    qualities = quality_scores or {}

    # 2. Greedy cluster matching
    # Map item_idx -> canonical_idx (disjoint set / cluster leader)
    cluster_leader: List[int] = list(range(n))
    duplicates_list: List[DuplicateRecord] = []

    for i in range(n):
        for j in range(i + 1, n):
            # Check if j is already clustered with i
            root_i = cluster_leader[i]
            root_j = cluster_leader[j]
            if root_i == root_j:
                continue

            dist = hamming_distance(hashes[i], hashes[j])
            if dist <= threshold:
                # Merge clusters: pick the keeper with higher quality
                q_i = qualities.get(item_ids[root_i], 0.0)
                q_j = qualities.get(item_ids[root_j], 0.0)

                if q_j > q_i:
                    # j has strictly higher quality, j becomes leader
                    keeper_idx = root_j
                    discard_idx = root_i
                else:
                    # i has equal or higher quality, i remains leader
                    keeper_idx = root_i
                    discard_idx = root_j

                # Update all elements pointing to discard_idx
                for k in range(n):
                    if cluster_leader[k] == discard_idx:
                        cluster_leader[k] = keeper_idx

                duplicates_list.append(
                    DuplicateRecord(
                        canonical_id=item_ids[keeper_idx],
                        duplicate_id=item_ids[discard_idx],
                        distance=dist,
                        hash_type=hash_type,
                        reason=f"Hamming distance {dist} <= {threshold} (quality {qualities.get(item_ids[keeper_idx], 0.0):.2f} >= {qualities.get(item_ids[discard_idx], 0.0):.2f})",
                    )
                )

    # 3. Partition into retained and duplicate ids
    retained_set = set()
    for idx in range(n):
        retained_set.add(cluster_leader[idx])

    retained_ids = [item_ids[idx] for idx in range(n) if idx in retained_set]
    duplicate_ids = [item_ids[idx] for idx in range(n) if idx not in retained_set]

    return DeduplicationResult(
        retained_ids=retained_ids,
        duplicate_ids=duplicate_ids,
        duplicates=duplicates_list,
        total_input=n,
        duplicate_count=len(duplicate_ids),
        hash_type=hash_type,
        threshold=threshold,
    )
