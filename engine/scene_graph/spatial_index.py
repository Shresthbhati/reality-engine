"""Geometric spatial index/query engine over WorldIR (city-scale campaign).

This module provides both a flat O(n) reference implementation and an
accelerated uniform-grid + BVH implementation for city-scale workloads.

The accelerated index (AcceleratedSpatialIndex) uses a uniform 3D grid
with per-cell BVH structures, providing sub-linear query performance
while maintaining exact query semantics, deterministic ordering, and
all existing behaviors.

Phase 32 of the scale campaign: "select based on actual workload" and
"benchmark the choice" -- the grid+BVH is chosen because:
- O(1) cell lookup + O(log n) BVH = sub-linear for typical distributions
- Simple, dependency-free, deterministic
- Handles both point and AABB queries naturally
- Scales to 100k+ entities with ~ms query times
"""

from __future__ import annotations

from engine.scene_graph.spatial_index_base import (
    Point,
    Bounds,
    IndexedEntity,
    UnlocalizedEntity,
    _distance_sq,
    _aabb_overlaps,
    _point_in_aabb,
    entity_position,
    _entity_bounds,
)
from engine.scene_graph.spatial_index_accel import (
    AcceleratedSpatialIndex,
    SpatialIndex,  # backward-compatible alias
)

__all__ = [
    "Point",
    "Bounds",
    "SpatialIndex",
    "AcceleratedSpatialIndex",
    "IndexedEntity",
    "UnlocalizedEntity",
    "entity_position",
]
