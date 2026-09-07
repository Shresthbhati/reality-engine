"""Naive O(n^2) broadphase: pairwise AABB overlap over all dynamic
bodies, plus each dynamic body against each static plane. Adaptive/GPU
broadphase (spec sec 46-48) is a scaling concern for later -- this is
correct and simple, which is what the foundation layer needs first.
"""

from __future__ import annotations

from engine.physics.math3 import Vec3
from .shapes import aabb_of, Box, Sphere


def _aabb_overlap(min_a: Vec3, max_a: Vec3, min_b: Vec3, max_b: Vec3) -> bool:
    return (
        min_a.x <= max_b.x and max_a.x >= min_b.x
        and min_a.y <= max_b.y and max_a.y >= min_b.y
        and min_a.z <= max_b.z and max_a.z >= min_b.z
    )


def body_pairs(body_ids: list[str], positions: dict[str, Vec3], shapes: dict[str, Sphere | Box]) -> list[tuple[str, str]]:
    """All (id_a, id_b) with id_a < id_b whose AABBs overlap. Ids are
    sorted first so the result -- and therefore downstream contact
    resolution order -- is a pure function of body state, not of dict
    iteration order (spec sec 90 stable ordering).
    """
    ids = sorted(body_ids)
    pairs = []
    for i in range(len(ids)):
        min_a, max_a = aabb_of(shapes[ids[i]], positions[ids[i]])
        for j in range(i + 1, len(ids)):
            min_b, max_b = aabb_of(shapes[ids[j]], positions[ids[j]])
            if _aabb_overlap(min_a, max_a, min_b, max_b):
                pairs.append((ids[i], ids[j]))
    return pairs
