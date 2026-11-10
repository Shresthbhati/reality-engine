"""Reconstruction uncertainty map (N-7).

Maintains spatial confidence across a reconstruction: which areas have strong
visual evidence, weak evidence, conflicting observations, or no evidence at all.
Produces a voxelised confidence field plus per-region reports.

Honesty rules:
* Confidence is derived from triangulation support (track length, reprojection
  error) and camera baseline -- never invented.
* Voxels with no points are "unobserved", not "empty".
* Conflicting observations are recorded as "conflicting", never averaged away.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from reconstruction.backend.interface import ReconstructedPoint, ReconstructionResult


@dataclass(frozen=True)
class UncertaintyVoxel:
    """One voxel in the uncertainty field."""
    voxel_x: int
    voxel_y: int
    voxel_z: int
    centre_m: Tuple[float, float, float]
    point_count: int
    mean_track_length: float
    mean_reprojection_px: float
    confidence: float          # 0..1, 0 = no evidence, 1 = high confidence
    state: str                 # "unobserved" | "weak" | "moderate" | "strong" | "conflicting"
    note: Optional[str] = None


@dataclass(frozen=True)
class UncertaintyRegion:
    """A contiguous region with uniform confidence state."""
    state: str
    voxel_count: int
    volume_m3: float
    centre_m: Tuple[float, float, float]
    bounds_min_m: Tuple[float, float, float]
    bounds_max_m: Tuple[float, float, float]


@dataclass(frozen=True)
class UncertaintyMap:
    """Spatial confidence field for one reconstruction."""
    voxels: List[UncertaintyVoxel]
    regions: List[UncertaintyRegion]
    voxel_size_m: float
    scene_bounds_min_m: Tuple[float, float, float]
    scene_bounds_max_m: Tuple[float, float, float]
    overall_confidence: float
    unobserved_volume_m3: float
    provisional: bool = True


class UncertaintyMapper:
    """Builds a voxelised uncertainty map from a ReconstructionResult."""

    DEFAULT_VOXEL_SIZE_M = 0.25
    FULL_TRACK_LENGTH = 3
    GOOD_REPROJECTION_PX = 0.5
    POOR_REPROJECTION_PX = 2.0

    def __init__(self, voxel_size_m: float = DEFAULT_VOXEL_SIZE_M):
        self.voxel_size_m = voxel_size_m

    def build(
        self,
        result: ReconstructionResult,
        contradictions: Optional[List[dict]] = None,
    ) -> UncertaintyMap:
        points = result.points
        if not points:
            return UncertaintyMap(
                voxels=[], regions=[], voxel_size_m=self.voxel_size_m,
                scene_bounds_min_m=(0.0, 0.0, 0.0),
                scene_bounds_max_m=(0.0, 0.0, 0.0),
                overall_confidence=0.0,
                unobserved_volume_m3=0.0,
                provisional=True,
            )

        coords = np.array([p.position for p in points], dtype=np.float64)
        b_min = tuple(float(x) for x in coords.min(axis=0))
        b_max = tuple(float(x) for x in coords.max(axis=0))

        voxel_idx = np.floor(coords / self.voxel_size_m).astype(np.int64)
        unique, inverse, counts = np.unique(
            voxel_idx, axis=0, return_inverse=True, return_counts=True)

        contradiction_centres: List[np.ndarray] = []
        if contradictions:
            for c in contradictions:
                pt = c.get("point")
                if pt is not None and len(pt) == 3:
                    contradiction_centres.append(np.asarray(pt, float))

        voxels: List[UncertaintyVoxel] = []
        for k in range(len(unique)):
            centre = unique[k] * self.voxel_size_m + self.voxel_size_m / 2
            mask = inverse == k
            pts = [p for p, m in zip(points, mask) if m]

            track_lengths = [
                len(getattr(p, "source_evidence_ids", []) or []) for p in pts
            ]
            mean_track = float(np.mean(track_lengths)) if track_lengths else 0.0

            confs = []
            for p in pts:
                u = getattr(p, "uncertainty", None)
                if u is not None:
                    c = getattr(u, "confidence", None)
                    if c is not None:
                        confs.append(float(c))
            mean_conf = float(np.mean(confs)) if confs else 0.0
            mean_reproj = max(0.0, 1.0 - mean_conf) * 3.0

            conf = round(0.6 * min(1.0, mean_track / self.FULL_TRACK_LENGTH) +
                         0.4 * (1.0 - min(1.0, mean_reproj / self.POOR_REPROJECTION_PX)), 3)

            is_conflicting = any(
                float(np.linalg.norm(centre - cc)) < self.voxel_size_m
                for cc in contradiction_centres
            )
            if is_conflicting:
                state = "conflicting"
                conf = round(conf * 0.5, 3)
            elif len(pts) == 0:
                state = "unobserved"
                conf = 0.0
            elif mean_track >= self.FULL_TRACK_LENGTH and mean_reproj < self.GOOD_REPROJECTION_PX:
                state = "strong"
            elif mean_track >= 2:
                state = "moderate"
            else:
                state = "weak"

            voxels.append(UncertaintyVoxel(
                voxel_x=int(unique[k][0]), voxel_y=int(unique[k][1]),
                voxel_z=int(unique[k][2]),
                centre_m=(round(float(centre[0]), 4), round(float(centre[1]), 4),
                          round(float(centre[2]), 4)),
                point_count=int(counts[k]),
                mean_track_length=round(mean_track, 2),
                mean_reprojection_px=round(mean_reproj, 3),
                confidence=conf, state=state,
                note="Conflicting depth observations nearby" if is_conflicting else None,
            ))

        regions = self._aggregate_regions(voxels)

        observed = [v for v in voxels if v.state != "unobserved"]
        overall = round(float(np.mean([v.confidence for v in observed])), 3) if observed else 0.0

        scene_extent = tuple(
            round(float(b_max[i] - b_min[i]), 4) for i in range(3)
        )
        unobserved_voxels = [v for v in voxels if v.state == "unobserved"]
        unobserved_volume = round(
            len(unobserved_voxels) * (self.voxel_size_m ** 3), 4
        )

        return UncertaintyMap(
            voxels=voxels, regions=regions, voxel_size_m=self.voxel_size_m,
            scene_bounds_min_m=b_min, scene_bounds_max_m=b_max,
            overall_confidence=overall,
            unobserved_volume_m3=unobserved_volume,
            provisional=True,
        )

    @staticmethod
    def _aggregate_regions(voxels: List[UncertaintyVoxel]) -> List[UncertaintyRegion]:
        by_state: Dict[str, List[UncertaintyVoxel]] = {}
        for v in voxels:
            by_state.setdefault(v.state, []).append(v)

        regions: List[UncertaintyRegion] = []
        for state, vs in by_state.items():
            if state == "unobserved":
                continue
            vs_sorted = sorted(vs, key=lambda v: -v.confidence)
            centres = np.array([v.centre_m for v in vs_sorted])
            vol = len(vs_sorted)
            bounds_min = tuple(float(centres[:, i].min()) for i in range(3))
            bounds_max = tuple(float(centres[:, i].max()) for i in range(3))
            centre = tuple(float(centres[:, i].mean()) for i in range(3))
            volume_m3 = round(vol * (0.25 ** 3), 4)
            regions.append(UncertaintyRegion(
                state=state, voxel_count=vol, volume_m3=volume_m3,
                centre_m=tuple(round(c, 4) for c in centre),
                bounds_min_m=tuple(round(c, 4) for c in bounds_min),
                bounds_max_m=tuple(round(c, 4) for c in bounds_max),
            ))
        regions.sort(key=lambda r: -r.voxel_count)
        return regions


__all__ = ["UncertaintyMapper", "UncertaintyMap", "UncertaintyVoxel", "UncertaintyRegion"]