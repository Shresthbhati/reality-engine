"""Ray-cast surface coverage and evidence state evaluation (Blueprint Task 5.3, Pipeline Stage S13).

Calculates true ray-cast surface coverage for architectural surfaces (walls, floors, ceilings)
from registered camera poses and intermediate scene occluders (furniture, partition walls, columns):
- Discretizes each surface into 3D grid sample points.
- Checks camera frustum inclusion (field of view, facing angle, near/far range).
- Casts rays from camera optical centers to surface points to detect line-of-sight occlusion by obstacles.
- Assigns each surface an honest coverage fraction (0..1) and evidence state:
  OBSERVED (fully or mostly visible), PARTIALLY_OBSERVED / OCCLUDED (partially hidden by obstacles),
  UNOBSERVED (outside all camera viewing frustums).
- Updates world.hierarchy nodes, world.surfaces, and surface entities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from world_ir.schema_v1 import Entity, EntityType, Provenance, Surface as WorldIRSurface


@dataclass(frozen=True)
class Ray:
    """A 3D ray with origin and unit direction."""
    origin: np.ndarray      # (3,) float
    direction: np.ndarray   # (3,) float unit vector


@dataclass(frozen=True)
class AABB:
    """Axis-aligned bounding box."""
    min_pt: np.ndarray      # (3,) float
    max_pt: np.ndarray      # (3,) float

    def intersect(self, ray: Ray) -> Optional[float]:
        """Slab intersection test. Returns distance t >= 0 to first hit, or None."""
        t_min = 0.0
        t_max = float("inf")
        for i in range(3):
            orig = float(ray.origin[i])
            d = float(ray.direction[i])
            b_min = float(self.min_pt[i])
            b_max = float(self.max_pt[i])

            if abs(d) < 1e-12:
                if orig < b_min or orig > b_max:
                    return None
            else:
                t1 = (b_min - orig) / d
                t2 = (b_max - orig) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                t_min = max(t_min, t1)
                t_max = min(t_max, t2)
                if t_min > t_max:
                    return None
        return t_min


@dataclass
class CameraView:
    """Registered camera viewpoint for ray-casting coverage."""
    camera_id: str
    position: np.ndarray        # (3,) float
    direction: np.ndarray       # (3,) float forward viewing vector
    fov_deg: float = 60.0       # Half-angle or cone angle
    near_m: float = 0.1
    far_m: float = 30.0

    @classmethod
    def from_pose(
        cls,
        camera_id: str,
        position: Sequence[float],
        forward_vector: Sequence[float],
        fov_deg: float = 60.0,
    ) -> CameraView:
        pos = np.array(position, dtype=np.float64)
        fwd = np.array(forward_vector, dtype=np.float64)
        norm = np.linalg.norm(fwd)
        fwd_unit = fwd / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        return cls(camera_id=camera_id, position=pos, direction=fwd_unit, fov_deg=fov_deg)


@dataclass
class SurfaceCoverageResult:
    """Detailed coverage calculation for one architectural surface."""
    surface_id: str
    entity_id: str
    total_samples: int
    observed_samples: int
    occluded_samples: int
    unobserved_samples: int
    coverage_fraction: float
    evidence_state: str  # OBSERVED | PARTIALLY_OBSERVED | OCCLUDED | UNOBSERVED
    occluding_entity_ids: List[str] = field(default_factory=list)


@dataclass
class SceneCoverageReport:
    """Full scene coverage report across all architectural surfaces."""
    surfaces: Dict[str, SurfaceCoverageResult] = field(default_factory=dict)
    mean_coverage: float = 0.0
    fully_observed_count: int = 0
    partially_observed_count: int = 0
    occluded_count: int = 0
    unobserved_count: int = 0


def _sample_surface_points(
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    normal: Sequence[float],
    grid_size: int = 5,
) -> List[np.ndarray]:
    """Sample a regular grid of points on a planar surface patch."""
    bmin = np.array(bounds_min, dtype=np.float64)
    bmax = np.array(bounds_max, dtype=np.float64)
    n = np.array(normal, dtype=np.float64)
    n_unit = n / np.linalg.norm(n) if np.linalg.norm(n) > 1e-12 else np.array([0.0, 0.0, 1.0])

    # Find the two principal axes spanning the plane (not aligned with normal)
    abs_n = np.abs(n_unit)
    if abs_n[0] >= abs_n[1] and abs_n[0] >= abs_n[2]:
        # X is normal axis; span Y and Z
        axes = (1, 2)
    elif abs_n[1] >= abs_n[0] and abs_n[1] >= abs_n[2]:
        # Y is normal axis; span X and Z
        axes = (0, 2)
    else:
        # Z is normal axis; span X and Y
        axes = (0, 1)

    pts: List[np.ndarray] = []
    ax1, ax2 = axes
    span1 = np.linspace(bmin[ax1], bmax[ax1], max(2, grid_size))
    span2 = np.linspace(bmin[ax2], bmax[ax2], max(2, grid_size))

    center = (bmin + bmax) * 0.5

    for v1 in span1:
        for v2 in span2:
            p = center.copy()
            p[ax1] = v1
            p[ax2] = v2
            pts.append(p)

    return pts


def compute_surface_coverage(
    surface_id: str,
    entity_id: str,
    bounds_min: Sequence[float],
    bounds_max: Sequence[float],
    normal: Sequence[float],
    cameras: Sequence[CameraView],
    occluders: Optional[Sequence[Tuple[str, AABB]]] = None,
    grid_size: int = 5,
) -> SurfaceCoverageResult:
    """Ray-cast coverage computation for a single surface.

    Tests each sample point against camera frustums and occluding AABBs.
    """
    occluders = occluders or []
    sample_points = _sample_surface_points(bounds_min, bounds_max, normal, grid_size=grid_size)
    total_samples = len(sample_points)

    if total_samples == 0:
        return SurfaceCoverageResult(
            surface_id=surface_id,
            entity_id=entity_id,
            total_samples=0,
            observed_samples=0,
            occluded_samples=0,
            unobserved_samples=0,
            coverage_fraction=0.0,
            evidence_state="UNOBSERVED",
        )

    surf_normal = np.array(normal, dtype=np.float64)
    norm_len = np.linalg.norm(surf_normal)
    surf_normal_unit = surf_normal / norm_len if norm_len > 1e-12 else np.array([0.0, 0.0, 1.0])

    observed_count = 0
    occluded_count = 0
    unobserved_count = 0
    hit_occluders: Set[str] = set()

    fov_cos_thresholds = [math.cos(math.radians(c.fov_deg * 0.5)) for c in cameras]

    for pt in sample_points:
        pt_visible = False
        pt_occluded = False

        for cam, fov_cos in zip(cameras, fov_cos_thresholds):
            vec_to_pt = pt - cam.position
            dist = float(np.linalg.norm(vec_to_pt))

            if dist < cam.near_m or dist > cam.far_m:
                continue

            dir_to_pt = vec_to_pt / dist

            # Check frustum orientation (camera looking toward point)
            cos_angle = float(np.dot(cam.direction, dir_to_pt))
            if cos_angle < fov_cos:
                continue

            # Check surface facing angle (camera must see front of surface)
            facing_dot = float(np.dot(surf_normal_unit, -dir_to_pt))
            if facing_dot <= 0.05:  # Glancing angle or backface
                continue

            # Point is in camera frustum! Now test occlusion
            ray = Ray(origin=cam.position, direction=dir_to_pt)
            blocked = False
            for occ_id, aabb in occluders:
                t_hit = aabb.intersect(ray)
                if t_hit is not None and 1e-3 < t_hit < (dist - 0.05):
                    blocked = True
                    hit_occluders.add(occ_id)
                    break

            if not blocked:
                pt_visible = True
                break
            else:
                pt_occluded = True

        if pt_visible:
            observed_count += 1
        elif pt_occluded:
            occluded_count += 1
        else:
            unobserved_count += 1

    coverage_fraction = float(observed_count) / float(total_samples)

    if coverage_fraction >= 0.80:
        ev_state = "OBSERVED"
    elif coverage_fraction > 0.05:
        ev_state = "PARTIALLY_OBSERVED"
    elif occluded_count > 0:
        ev_state = "OCCLUDED"
    else:
        ev_state = "UNOBSERVED"

    return SurfaceCoverageResult(
        surface_id=surface_id,
        entity_id=entity_id,
        total_samples=total_samples,
        observed_samples=observed_count,
        occluded_samples=occluded_count,
        unobserved_samples=unobserved_count,
        coverage_fraction=coverage_fraction,
        evidence_state=ev_state,
        occluding_entity_ids=sorted(hit_occluders),
    )


def compute_world_surface_coverage(
    world,
    camera_poses: Sequence[Tuple[str, Sequence[float], Sequence[float]]],
    fov_deg: float = 60.0,
    grid_size: int = 5,
) -> SceneCoverageReport:
    """Compute ray-cast coverage across all architectural surfaces in WorldIR.

    Parameters:
        world: WorldIR instance
        camera_poses: List of (evidence_id, position [x,y,z], forward_dir [dx,dy,dz])
        fov_deg: Camera field of view angle in degrees
        grid_size: Number of sampling points along each dimension of surface patch

    Updates in-place:
        - world.hierarchy.nodes surface nodes with `coverage` and `evidence_state`.
        - world.surfaces with `coverage` and `evidence_state` custom attributes.
        - Surface entities with `coverage` and `evidence_state` in custom_properties.
    """
    cameras = [
        CameraView.from_pose(cid, pos, fwd, fov_deg=fov_deg)
        for cid, pos, fwd in camera_poses
    ]

    # Gather occluding entities (e.g. furniture, structures)
    surface_types = {EntityType.WALL, EntityType.FLOOR, EntityType.CEILING}
    occluders: List[Tuple[str, AABB]] = []

    for eid, ent in world.entities.items():
        if ent.type not in surface_types and ent.type not in {EntityType.ROOM, EntityType.CORRIDOR, EntityType.LEVEL, EntityType.BUILDING}:
            cp = ent.custom_properties or {}
            bmin = cp.get("bounds_min")
            bmax = cp.get("bounds_max")
            if bmin and bmax:
                occluders.append((eid, AABB(min_pt=np.array(bmin, float), max_pt=np.array(bmax, float))))

    report = SceneCoverageReport()
    surface_ents = [e for e in world.entities.values() if e.type in surface_types]

    for sf_ent in surface_ents:
        cp = sf_ent.custom_properties or {}
        bmin = cp.get("bounds_min")
        bmax = cp.get("bounds_max")
        
        # If no explicit bounds in custom properties, check associated geometries
        if not bmin or not bmax:
            if sf_ent.geometry_ids:
                geom = world.geometries.get(sf_ent.geometry_ids[0])
                if geom and geom.bounds_min and geom.bounds_max:
                    bmin = (geom.bounds_min.x, geom.bounds_min.y, geom.bounds_min.z)
                    bmax = (geom.bounds_max.x, geom.bounds_max.y, geom.bounds_max.z)

        if not bmin or not bmax:
            # Synthetic default bounds around transform position
            pos = (ent.transform or {}).get("position") if isinstance(ent.transform, dict) else None
            px = pos.get("x", 0.0) if pos else 0.0
            py = pos.get("y", 0.0) if pos else 0.0
            pz = pos.get("z", 0.0) if pos else 0.0
            bmin = (px - 1.0, py - 1.0, pz - 0.05)
            bmax = (px + 1.0, py + 1.0, pz + 0.05)

        normal = cp.get("normal", (0.0, 0.0, 1.0))
        if sf_ent.type == EntityType.FLOOR:
            normal = (0.0, 0.0, 1.0)
        elif sf_ent.type == EntityType.CEILING:
            normal = (0.0, 0.0, -1.0)

        # Exclude self from occluders
        active_occluders = [(oid, box) for oid, box in occluders if oid != sf_ent.id]

        res = compute_surface_coverage(
            surface_id=f"surf-{sf_ent.id}",
            entity_id=sf_ent.id,
            bounds_min=bmin,
            bounds_max=bmax,
            normal=normal,
            cameras=cameras,
            occluders=active_occluders,
            grid_size=grid_size,
        )

        report.surfaces[sf_ent.id] = res

        # Update entity custom_properties
        cp["coverage"] = res.coverage_fraction
        cp["evidence_state"] = res.evidence_state
        sf_ent.custom_properties = cp

        # Update world.surfaces
        surf_key = f"surf-{sf_ent.id}"
        if hasattr(world, "surfaces") and surf_key in world.surfaces:
            world.surfaces[surf_key].custom_attributes["coverage"] = res.coverage_fraction
            world.surfaces[surf_key].custom_attributes["evidence_state"] = res.evidence_state

        # Update world.hierarchy
        if hasattr(world, "hierarchy") and world.hierarchy:
            node_id = f"node-{sf_ent.id}"
            if node_id in world.hierarchy.nodes:
                old_node = world.hierarchy.nodes[node_id]
                from dataclasses import replace
                world.hierarchy.nodes[node_id] = replace(
                    old_node,
                    coverage=res.coverage_fraction,
                    evidence_state=res.evidence_state,
                )

        if res.evidence_state == "OBSERVED":
            report.fully_observed_count += 1
        elif res.evidence_state == "PARTIALLY_OBSERVED":
            report.partially_observed_count += 1
        elif res.evidence_state == "OCCLUDED":
            report.occluded_count += 1
        else:
            report.unobserved_count += 1

    if report.surfaces:
        report.mean_coverage = float(sum(s.coverage_fraction for s in report.surfaces.values())) / float(len(report.surfaces))

    return report
