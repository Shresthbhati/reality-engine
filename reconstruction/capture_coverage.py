"""Live capture coverage map (N-2).

Shows which surfaces/areas have been captured, which have weak coverage,
and which have not been observed. Distinguishes actual observations from
estimated coverage.

Honesty rules:
* Coverage estimates are plan-view approximations; furniture occlusion and
  lens FOV are not modelled. Every result is labelled provisional.
* Coverage is inferred from camera poses and viewing directions, not
  directly measured.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from world_ir import Entity, EntityType


@dataclass(frozen=True)
class SurfaceCoverage:
    """Coverage status for one surface/wall."""
    surface_id: str
    surface_type: str
    name: Optional[str]
    status: str                  # "seen" | "weak" | "not_observed"
    covered_fraction: float      # 0..1
    multi_view_fraction: float   # 0..1
    hint: Optional[str]          # what to photograph next
    provisional: bool = True


@dataclass(frozen=True)
class CoverageMap:
    """Coverage map for one world."""
    surfaces: List[SurfaceCoverage]
    summary: Dict[str, int]      # count by status
    provisional: bool = True


class CoverageMapper:
    """Computes per-surface coverage from camera poses."""

    SEEN_COVERED_FRACTION = 0.80
    SEEN_MULTI_FRACTION = 0.50
    NOT_SEEN_COVERED_FRACTION = 0.20
    ASSUMED_HALF_FOV_DEG = 35.0
    MAX_INCIDENCE_DEG = 75.0
    WALL_SAMPLES = 12

    def compute_coverage(
        self,
        surfaces: List[Dict],
        cameras: Sequence[dict],
        basis: Optional[dict] = None,
    ) -> CoverageMap:
        """Compute per-surface coverage from camera poses.

        Parameters
        ----------
        surfaces : list of dict
            Each dict has "id", "type", "name", "bounds_min", "bounds_max".
        cameras : sequence of dict
            Each dict has "evidence_id", "position_m", "rotation_wxyz".
        basis : dict, optional
            Plan basis {e1, e2, up}.
        """
        surfaces_out: List[SurfaceCoverage] = []

        for surf in surfaces:
            sid = surf["id"]
            sname = surf.get("name")
            stype = surf.get("type", "wall")
            bmin = surf.get("bounds_min")
            bmax = surf.get("bounds_max")
            if bmin is None or bmax is None:
                surfaces_out.append(SurfaceCoverage(
                    surface_id=sid, surface_type=stype, name=sname,
                    status="not_observed", covered_fraction=0.0,
                    multi_view_fraction=0.0,
                    hint="No geometry bounds available for this surface.",
                    provisional=True,
                ))
                continue

            pmin = np.asarray(bmin, float)
            pmax = np.asarray(bmax, float)
            centre = (pmin + pmax) / 2

            # Sample points across the surface
            samples = self._sample_surface(pmin, pmax)

            covered = 0.0
            multi = 0.0
            total = 0.0

            for pt in samples:
                try:
                    for cam in cameras:
                        try:
                            pos = np.asarray(cam["position_m"], float)
                            fwd = np.asarray(cam.get("rotation_wxyz", [1, 0, 0, 0]), float)
                            fwd = np.array([
                                2 * (fwd[1] * fwd[2] + fwd[0] * fwd[3]),
                                2 * (fwd[2] * fwd[0] - fwd[1] * fwd[3]),
                                1 - 2 * (fwd[1] ** 2 + fwd[2] ** 2),
                            ], float)
                            fwd_norm = np.linalg.norm(fwd)
                            if fwd_norm < 1e-9:
                                continue
                            fwd = fwd / fwd_norm

                            ray = pt - pos
                            d = float(np.linalg.norm(ray))
                            if d < 1e-9:
                                continue
                            ray = ray / d

                            # FOV check
                            if float(np.dot(ray, fwd)) < math.cos(math.radians(self.ASSUMED_HALF_FOV_DEG)):
                                continue

                            # Incidence check
                            normal = np.array([0.0, 1.0, 0.0])  # surface up normal (approximate)
                            if float(np.dot(-ray, normal)) < math.cos(math.radians(self.MAX_INCIDENCE_DEG)):
                                continue

                            # Occlusion check
                            occluded = False
                            for other in surfaces:
                                if other["id"] == sid:
                                    continue
                                obmin = np.asarray(other.get("bounds_min", [0, 0, 0]), float)
                                obmax = np.asarray(other.get("bounds_max", [0, 0, 0]), float)
                                if self._ray_intersects_box(pos, ray, obmin, obmax):
                                    occluded = True
                                    break

                            if not occluded:
                                covered += 1.0
                                multi += 1.0
                        except Exception:
                            continue  # Skip this camera if any error occurs
                except Exception:
                    continue  # Skip this sample point if any error occurs

                total += 1.0

            covered_fraction = covered / total if total > 0 else 0.0
            multi_view_fraction = multi / total if total > 0 else 0.0

            if covered_fraction >= self.SEEN_COVERED_FRACTION and multi_view_fraction >= self.SEEN_MULTI_FRACTION:
                status = "seen"
            elif covered_fraction < self.NOT_SEEN_COVERED_FRACTION:
                status = "not_observed"
            else:
                status = "weak"

            hint = None
            if status != "seen":
                hint = self._coverage_hint(surf, status)

            surfaces_out.append(SurfaceCoverage(
                surface_id=sid, surface_type=stype, name=sname,
                status=status, covered_fraction=round(covered_fraction, 3),
                multi_view_fraction=round(multi_view_fraction, 3),
                hint=hint, provisional=True,
            ))

        summary = {
            "seen": sum(1 for s in surfaces_out if s.status == "seen"),
            "weak": sum(1 for s in surfaces_out if s.status == "weak"),
            "not_observed": sum(1 for s in surfaces_out if s.status == "not_observed"),
        }

        return CoverageMap(
            surfaces=surfaces_out,
            summary=summary,
            provisional=True,
        )

    def _sample_surface(
        self,
        pmin: np.ndarray,
        pmax: np.ndarray,
        samples_per_dim: int = 8,
    ) -> List[np.ndarray]:
        """Sample points across a rectangular surface."""
        samples: List[np.ndarray] = []
        for i in range(samples_per_dim):
            for j in range(samples_per_dim):
                u = (i + 0.5) / samples_per_dim
                v = (j + 0.5) / samples_per_dim
                pt = pmin + (pmax - pmin) * np.array([u, v, 0])
                samples.append(pt)
        return samples

    def _coverage_hint(self, surf: Dict, status: str) -> Optional[str]:
        sname = surf.get("name", surf["id"])
        if status == "not_observed":
            return f"Take 3+ photos of {sname} from different angles."
        elif status == "weak":
            return f"Add 1-2 more photos of {sname} for better coverage."
        return None

    def _ray_intersects_box(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        box_min: np.ndarray,
        box_max: np.ndarray,
    ) -> bool:
        """Simple ray-AABB intersection test."""
        tmin = 0.0
        tmax = float('inf')
        for i in range(3):
            if abs(direction[i]) < 1e-9:
                if origin[i] < box_min[i] or origin[i] > box_max[i]:
                    return False
            else:
                t1 = (box_min[i] - origin[i]) / direction[i]
                t2 = (box_max[i] - origin[i]) / direction[i]
                t1, t2 = min(t1, t2), max(t1, t2)
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmax < tmin:
                    return False
        return tmax >= 0 and tmin <= tmax


__all__ = ["CoverageMapper", "SurfaceCoverage", "CoverageMap"]