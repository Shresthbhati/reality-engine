"""Geometric segmentation for architectural perception (P7-03
expansion): cluster the points that PLANES did not claim into
spatially-coherent candidate segments -- the geometric substrate the
parametric fits (cylinder/sphere/circle) consume.

Method (deterministic, no RNG): voxelize the non-plane points at a
fixed cell size, take 26-connected components (three passes over
sorted keys -- ordering is by sorted voxel key, never dict iteration),
and emit segments above a minimum point count. Undersized clusters are
NOT silently dropped -- they are returned with `below_min=True` so the
benchmark can report them as failure regions (honest visibility).

Plane subtraction reuses perception.geometry.planes.detect_planes's
inlier ids; a point claimed by a plane is wall/floor/ceiling material,
not dome/column candidate material.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

from reconstruction.backend.interface import ReconstructionResult

#: Default voxel cell size (meters). A ~0.25 m cell bridges small gaps
#: in a column shell without merging distinct columns 3 m apart.
DEFAULT_VOXEL_SIZE_M = 0.25

#: Segments below this size are reported but not fitted.
DEFAULT_MIN_SEGMENT_POINTS = 40


@dataclass(frozen=True)
class Segment:
    segment_id: str
    member_ids: Tuple[str, ...]  # sorted reconstruction track ids
    positions: Tuple[Tuple[float, float, float], ...]
    evidence_ids: Tuple[str, ...]
    below_min: bool

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "point_count": len(self.member_ids),
            "evidence_ids": list(self.evidence_ids),
            "below_min": self.below_min,
            "bounds_min": [min(p[i] for p in self.positions) for i in range(3)],
            "bounds_max": [max(p[i] for p in self.positions) for i in range(3)],
        }


def segment_non_plane_points(
    result: ReconstructionResult,
    plane_inlier_ids: set,
    voxel_size_m: float = DEFAULT_VOXEL_SIZE_M,
    min_segment_points: int = DEFAULT_MIN_SEGMENT_POINTS,
) -> List[Segment]:
    """Cluster non-plane points into candidate segments.

    `plane_inlier_ids` is the union of detect_planes' inlier track ids.
    Deterministic: identical input -> identical segments.
    """
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    if min_segment_points < 1:
        raise ValueError("min_segment_points must be >= 1")

    candidates = [
        p for p in result.points
        if p.track_id not in plane_inlier_ids
    ]
    # Voxel assignment (sorted for determinism).
    voxels: Dict[Tuple[int, int, int], List[int]] = {}
    for idx, p in enumerate(sorted(candidates, key=lambda q: q.track_id)):
        key = (
            math.floor(p.position[0] / voxel_size_m),
            math.floor(p.position[1] / voxel_size_m),
            math.floor(p.position[2] / voxel_size_m),
        )
        voxels.setdefault(key, []).append(idx)

    # 26-connected components over voxel keys (BFS over sorted neighbor
    # keys; visited set keyed by voxel -- deterministic).
    remaining = set(voxels.keys())
    components: List[List[Tuple[int, int, int]]] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        remaining.discard(seed)
        comp = []
        while stack:
            vk = stack.pop()
            comp.append(vk)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nk = (vk[0] + dx, vk[1] + dy, vk[2] + dz)
                        if nk in remaining:
                            remaining.discard(nk)
                            stack.append(nk)
        comp.sort()
        components.append(comp)

    ordered_points = sorted(candidates, key=lambda q: q.track_id)
    segments: List[Segment] = []
    for ci, comp in enumerate(sorted(components, key=lambda c: c[0])):
        member_idx = sorted(i for vk in comp for i in voxels[vk])
        members = [ordered_points[i] for i in member_idx]
        evidence: List[str] = []
        for m in members:
            for e in m.source_evidence_ids:
                if e not in evidence:
                    evidence.append(e)
        segments.append(Segment(
            segment_id=f"seg-{ci:04d}",
            member_ids=tuple(m.track_id for m in members),
            positions=tuple(tuple(m.position) for m in members),
            evidence_ids=tuple(sorted(evidence)),
            below_min=len(members) < min_segment_points,
        ))
    segments.sort(key=lambda s: s.segment_id)
    return segments
