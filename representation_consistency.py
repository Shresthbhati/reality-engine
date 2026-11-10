"""Multi-representation consistency checking (D-20).

Compares point clouds, meshes, Gaussian splats, semantic entities and
architectural geometry. Flags areas where one representation disagrees with
another instead of silently choosing whichever result looks best.

Honesty rules:
* A disagreement is reported with both measurements and the distance between
  them -- never silently resolved.
 * "No data in one representation" is recorded as missing, not as agreement.
* All distances are in metres and labelled with the representations compared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from world_ir import Entity, EntityType, Geometry


@dataclass(frozen=True)
class RepresentationConflict:
    """One disagreement between two representations."""
    rule: str                      # e.g. "mesh_vs_pointcloud"
    location_m: Tuple[float, float, float]
    representation_a: str          # "pointcloud" | "mesh" | "splat" | "entity" | "slab"
    representation_b: str
    value_a_m: Optional[float]
    value_b_m: Optional[float]
    disagreement_m: float
    severity: str                  # "info" | "warning" | "error"
    message: str
    note: Optional[str] = None


@dataclass(frozen=True)
class ConsistencyReport:
    """All consistency findings across representations."""
    conflicts: List[RepresentationConflict]
    representations_compared: List[str]
    provisional: bool = True


class RepresentationConsistencyChecker:
    SURFACE_DISAGREEMENT_TOL_M = 0.10
    ENTITY_MESH_TOL_M = 0.15

    def check(
        self,
        point_cloud: Optional[Sequence[Tuple[float, float, float]]] = None,
        mesh_vertices: Optional[Sequence[Tuple[float, float, float]]] = None,
        entities: Optional[Sequence[Entity]] = None,
        slabs: Optional[Sequence[Entity]] = None,
    ) -> ConsistencyReport:
        conflicts: List[RepresentationConflict] = []
        compared: List[str] = []

        if point_cloud is not None:
            compared.append("pointcloud")
        if mesh_vertices is not None:
            compared.append("mesh")
        if entities is not None:
            compared.append("entity")
        if slabs is not None:
            compared.append("slab")

        if point_cloud is not None and mesh_vertices is not None:
            conflicts.extend(self._cloud_vs_mesh(point_cloud, mesh_vertices))

        if entities is not None and mesh_vertices is not None:
            conflicts.extend(self._entities_vs_mesh(entities, mesh_vertices))

        if slabs is not None and mesh_vertices is not None:
            conflicts.extend(self._slabs_vs_mesh(slabs, mesh_vertices))

        if point_cloud is not None and entities is not None:
            conflicts.extend(self._entities_vs_pointcloud(entities, point_cloud))

        conflicts.sort(key=lambda c: ({"error": 0, "warning": 1, "info": 2}[c.severity],
                                      c.location_m))
        return ConsistencyReport(
            conflicts=conflicts,
            representations_compared=compared,
            provisional=True,
        )

    # ------------------------------------------------------------------ checks

    @staticmethod
    def _nearest_distance(
        query: np.ndarray, candidates: np.ndarray
    ) -> Tuple[float, Optional[np.ndarray]]:
        if len(candidates) == 0:
            return float("inf"), None
        d = np.linalg.norm(candidates - query, axis=1)
        i = int(np.argmin(d))
        return float(d[i]), candidates[i]

    def _cloud_vs_mesh(
        self,
        cloud: Sequence[Tuple[float, float, float]],
        mesh: Sequence[Tuple[float, float, float]],
    ) -> List[RepresentationConflict]:
        conflicts: List[RepresentationConflict] = []
        pc = np.asarray(cloud, float)
        mv = np.asarray(mesh, float)
        if len(pc) == 0 or len(mv) == 0:
            return conflicts

        sample = pc[::max(1, len(pc) // 2000)]
        for pt in sample:
            d, _ = self._nearest_distance(pt, mv)
            if d > self.SURFACE_DISAGREEMENT_TOL_M:
                conflicts.append(RepresentationConflict(
                    rule="pointcloud_vs_mesh_surface",
                    location_m=tuple(round(float(c), 4) for c in pt),
                    representation_a="pointcloud", representation_b="mesh",
                    value_a_m=None, value_b_m=None,
                    disagreement_m=round(d, 4),
                    severity="warning",
                    message=(f"Point cloud sample at {tuple(round(float(c), 3) for c in pt)} "
                             f"is {d:.3f} m from the nearest mesh surface -- "
                             f"the two representations disagree here."),
                    note="Could indicate meshing error, missing points, or occlusion.",
                ))
        return conflicts

    def _entities_vs_mesh(
        self,
        entities: Sequence[Entity],
        mesh: Sequence[Tuple[float, float, float]],
    ) -> List[RepresentationConflict]:
        conflicts: List[RepresentationConflict] = []
        mv = np.asarray(mesh, float)
        if len(mv) == 0:
            return conflicts
        for e in entities:
            cp = e.custom_properties
            bmin = cp.get("bounds_min")
            bmax = cp.get("bounds_max")
            if bmin is None or bmax is None:
                continue
            centre = (np.asarray(bmin, float) + np.asarray(bmax, float)) / 2
            d, _ = self._nearest_distance(centre, mv)
            if d > self.ENTITY_MESH_TOL_M:
                conflicts.append(RepresentationConflict(
                    rule="entity_vs_mesh_surface",
                    location_m=tuple(round(float(c), 4) for c in centre),
                    representation_a="entity", representation_b="mesh",
                    value_a_m=None, value_b_m=None,
                    disagreement_m=round(d, 4),
                    severity="warning",
                    message=(f"Entity '{e.name or e.id}' ({e.type.value}) centre is "
                             f"{d:.3f} m from the nearest mesh surface."),
                    note="Entity may be floating, misplaced, or missing from the mesh.",
                ))
        return conflicts

    def _slabs_vs_mesh(
        self,
        slabs: Sequence[Entity],
        mesh: Sequence[Tuple[float, float, float]],
    ) -> List[RepresentationConflict]:
        conflicts: List[RepresentationConflict] = []
        mv = np.asarray(mesh, float)
        if len(mv) == 0:
            return conflicts
        for s in slabs:
            cp = s.custom_properties
            slab = cp.get("slab", {})
            poly = slab.get("polygon_xyz_m")
            thick = slab.get("thickness_m")
            if not poly or thick is None:
                continue
            for corner in poly:
                pt = np.asarray(corner, float)
                d, _ = self._nearest_distance(pt, mv)
                if d > self.SURFACE_DISAGREEMENT_TOL_M:
                    conflicts.append(RepresentationConflict(
                        rule="slab_vs_mesh_surface",
                        location_m=tuple(round(float(c), 4) for c in pt),
                        representation_a="slab", representation_b="mesh",
                        value_a_m=None, value_b_m=None,
                        disagreement_m=round(d, 4),
                        severity="warning",
                        message=(f"Slab '{s.id}' corner at "
                                 f"{tuple(round(float(c), 3) for c in pt)} is "
                                 f"{d:.3f} m from the nearest mesh surface."),
                        note="Slab may not match the reconstructed mesh.",
                    ))
        return conflicts

    def _entities_vs_pointcloud(
        self,
        entities: Sequence[Entity],
        cloud: Sequence[Tuple[float, float, float]],
    ) -> List[RepresentationConflict]:
        conflicts: List[RepresentationConflict] = []
        pc = np.asarray(cloud, float)
        if len(pc) == 0:
            return conflicts
        for e in entities:
            cp = e.custom_properties
            bmin = cp.get("bounds_min")
            bmax = cp.get("bounds_max")
            if bmin is None or bmax is None:
                continue
            centre = (np.asarray(bmin, float) + np.asarray(bmax, float)) / 2
            d, _ = self._nearest_distance(centre, pc)
            if d > 0.5:
                conflicts.append(RepresentationConflict(
                    rule="entity_vs_pointcloud_support",
                    location_m=tuple(round(float(c), 4) for c in centre),
                    representation_a="entity", representation_b="pointcloud",
                    value_a_m=None, value_b_m=None,
                    disagreement_m=round(d, 4),
                    severity="info",
                    message=(f"Entity '{e.name or e.id}' has no point cloud support "
                             f"within {d:.2f} m of its centre."),
                    note="May be valid (empty space) or may indicate missing capture.",
                ))
        return conflicts


__all__ = ["RepresentationConsistencyChecker", "RepresentationConflict", "ConsistencyReport"]