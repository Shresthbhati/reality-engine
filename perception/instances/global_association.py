"""Global identity: one physical object across many photos (BUILD_SPEC B08).

Replaces order-dependent greedy joining. Candidate pairs (already gated by label/Mahalanobis/epipolar/appearance
upstream) are merged strongest-first with union-find; a merge is refused when it would put two observations of
the SAME image in one entity (an object cannot appear twice in one photo), and a part may only join groups whose
wholes are in the same entity.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np


def cluster_observations(
    image_ids: Sequence[str],
    pair_scores: Iterable[tuple[int, int, float]],
    parent_obs: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Entity index (0..K-1) per observation row; deterministic and independent of input order."""
    n = len(image_ids)
    _, img = np.unique(np.asarray(image_ids, dtype=str), return_inverse=True)
    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)
    imgs = [{int(img[i])} for i in range(n)]
    par = np.full(n, -1, dtype=np.int64) if parent_obs is None else np.asarray(parent_obs, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    pairs = np.asarray(list(pair_scores), dtype=float).reshape(-1, 3)
    pairs = pairs[pairs[:, 2] > 0]
    if len(pairs):
        lo = np.minimum(pairs[:, 0], pairs[:, 1]); hi = np.maximum(pairs[:, 0], pairs[:, 1])
        order = np.lexsort((hi, lo, -pairs[:, 2]))       # strongest first; ties broken by row ids, not input order
        for a, b in zip(lo[order].astype(int), hi[order].astype(int)):
            ra, rb = find(a), find(b)
            if ra == rb or (imgs[ra] & imgs[rb]):
                continue
            pa, pb = par[a], par[b]
            if (pa >= 0) != (pb >= 0):                    # whole vs part never merge
                continue
            if pa >= 0 and find(int(pa)) != find(int(pb)):  # parts must share the same whole's entity
                continue
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra; size[ra] += size[rb]; imgs[ra] |= imgs[rb]
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, ent = np.unique(roots, return_inverse=True)
    return ent.astype(np.int32)
