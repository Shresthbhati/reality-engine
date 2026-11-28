"""Geometric adjacency/overlap inference from entity bounding-cube geometry.

Pure query: derives candidate ADJACENT_TO / OVERLAPS relationships from
entity transform positions + radius. Does not mutate WorldIR or write
anything back onto entities -- wiring this into an actual mutation is the
caller's job (e.g. via engine/commands/).

Scope: axis-aligned bounding cube per entity (center = position, half-extent
= radius, default 0.0 => a point). Does NOT infer SUPPORTS/RESTS_ON (needs
real shape + gravity reasoning) or PART_OF/CONTAINS (needs semantic logic).
"""
from __future__ import annotations

from typing import List, Tuple

from world_ir import RelationshipKind, WorldIR

Vec3 = Tuple[float, float, float]


def compute_aabb(position: dict, radius: float) -> Tuple[Vec3, Vec3]:
    """Axis-aligned bounding box for a point + radius half-extent."""
    x, y, z = position["x"], position["y"], position["z"]
    min_corner = (x - radius, y - radius, z - radius)
    max_corner = (x + radius, y + radius, z + radius)
    return min_corner, max_corner


def aabb_overlap(a_min: Vec3, a_max: Vec3, b_min: Vec3, b_max: Vec3) -> bool:
    """Standard interval-overlap AABB test, independently on all 3 axes.

    Touching (equal) boundaries count as overlap.
    """
    for i in range(3):
        if a_min[i] > b_max[i] or b_min[i] > a_max[i]:
            return False
    return True


def infer_geometric_relationships(
    world: WorldIR, adjacency_margin: float = 0.0
) -> List[Tuple[str, str, RelationshipKind]]:
    """Derive ADJACENT_TO / OVERLAPS pairs from entity bounding-cube geometry.

    Entities without a transform+position are skipped (not defaulted to
    origin). Each unordered pair is reported at most once, in canonical
    (sorted id) order.
    """
    candidates = []
    for entity in world.entities.values():
        if not entity.transform:
            continue
        pos = entity.transform.get("position")
        if not pos:
            continue
        radius = entity.transform.get("radius", 0.0)
        aabb = compute_aabb(pos, radius)
        candidates.append((entity.id, aabb))

    candidates.sort(key=lambda c: c[0])

    results: List[Tuple[str, str, RelationshipKind]] = []
    for i in range(len(candidates)):
        id_a, (a_min, a_max) = candidates[i]
        for j in range(i + 1, len(candidates)):
            id_b, (b_min, b_max) = candidates[j]

            if aabb_overlap(a_min, a_max, b_min, b_max):
                results.append((id_a, id_b, RelationshipKind.OVERLAPS))
                continue

            exp_a_min = tuple(c - adjacency_margin for c in a_min)
            exp_a_max = tuple(c + adjacency_margin for c in a_max)
            exp_b_min = tuple(c - adjacency_margin for c in b_min)
            exp_b_max = tuple(c + adjacency_margin for c in b_max)
            if aabb_overlap(exp_a_min, exp_a_max, exp_b_min, exp_b_max):
                results.append((id_a, id_b, RelationshipKind.ADJACENT_TO))

    return results
