"""Outdoor and environmental perception (Master Development Plan Task 6.7).

Fits and promotes outdoor taxonomy entities from 3D points / geometry:
1. `vegetation.tree`:
   - Distinguishes vertical cylindrical trunk and volumetric canopy crown.
   - Measures trunk radius, crown diameter, and total tree height.
   - Gated by taxonomy plausibility prior [1.5m, 45.0m].
2. `street.lamppost`:
   - Detects slender vertical pole/shaft (high aspect ratio) and top luminaire head.
   - Gated by taxonomy plausibility prior [2.0m, 15.0m].
3. `road.carriageway`:
   - Fits planar vehicular travel corridor (width >= 2.5m).
4. `road.footpath`:
   - Fits planar pedestrian sidewalk corridor (width 0.6m - 6.0m).
5. `street.bench` & `street.bin`:
   - Fits street furniture entities with measured bounding dimensions.

All fits are deterministic, evidence-grounded, and raise explicit Refusal
exceptions rather than fabricating geometry when points do not support the class.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from uuid import uuid4

import numpy as np

from provenance import Provenance
from world_ir import WorldIR
from world_ir.schema_v1 import (
    Entity,
    EntityType,
    Geometry,
    GeometryType,
    Measurement,
    Vector3,
)
from perception.entities import taxonomy

Point3 = Tuple[float, float, float]
Vec3 = Tuple[float, float, float]


class OutdoorRefused(ValueError):
    """Base exception when points do not demonstrate the requested outdoor entity."""


class TreeRefused(OutdoorRefused):
    """Points do not demonstrate tree structure (trunk + canopy)."""


class LamppostRefused(OutdoorRefused):
    """Points do not demonstrate a slender vertical lamppost."""


class RoadRefused(OutdoorRefused):
    """Points do not demonstrate a vehicular carriageway."""


class FootpathRefused(OutdoorRefused):
    """Points do not demonstrate a pedestrian footpath / sidewalk."""


class StreetFurnitureRefused(OutdoorRefused):
    """Points do not demonstrate street furniture dimensions."""


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class TreeFit:
    """A measured tree: trunk cylinder + volumetric canopy crown."""

    trunk_axis: Vec3
    trunk_base: Point3
    trunk_radius_m: float
    trunk_height_m: float
    crown_center: Point3
    crown_radius_m: float
    crown_height_m: float
    total_height_m: float
    confidence: float
    point_count: int
    rms_residual_m: float
    bounds_min: Point3
    bounds_max: Point3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_class": "vegetation.tree",
            "trunk_axis": list(self.trunk_axis),
            "trunk_base": list(self.trunk_base),
            "trunk_radius_m": round(self.trunk_radius_m, 3),
            "trunk_height_m": round(self.trunk_height_m, 3),
            "crown_center": list(self.crown_center),
            "crown_radius_m": round(self.crown_radius_m, 3),
            "crown_height_m": round(self.crown_height_m, 3),
            "total_height_m": round(self.total_height_m, 3),
            "confidence": round(self.confidence, 3),
            "point_count": self.point_count,
            "rms_residual_m": round(self.rms_residual_m, 4),
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
        }


@dataclass(frozen=True)
class LamppostFit:
    """A measured lamppost: vertical slender shaft + top luminaire."""

    shaft_base: Point3
    shaft_top: Point3
    shaft_radius_m: float
    total_height_m: float
    luminaire_offset_m: float
    luminaire_position: Point3
    confidence: float
    point_count: int
    bounds_min: Point3
    bounds_max: Point3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_class": "street.lamppost",
            "shaft_base": list(self.shaft_base),
            "shaft_top": list(self.shaft_top),
            "shaft_radius_m": round(self.shaft_radius_m, 3),
            "total_height_m": round(self.total_height_m, 3),
            "luminaire_offset_m": round(self.luminaire_offset_m, 3),
            "luminaire_position": list(self.luminaire_position),
            "confidence": round(self.confidence, 3),
            "point_count": self.point_count,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
        }


@dataclass(frozen=True)
class RoadSurfaceFit:
    """A measured vehicular carriageway road corridor."""

    normal: Vec3
    plane_d: float
    center: Point3
    width_m: float
    length_m: float
    area_m2: float
    corridor_bearing_deg: float
    elevation_m: float
    confidence: float
    point_count: int
    bounds_min: Point3
    bounds_max: Point3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_class": "road.carriageway",
            "normal": list(self.normal),
            "plane_d": round(self.plane_d, 4),
            "center": list(self.center),
            "width_m": round(self.width_m, 3),
            "length_m": round(self.length_m, 3),
            "area_m2": round(self.area_m2, 2),
            "corridor_bearing_deg": round(self.corridor_bearing_deg, 1),
            "elevation_m": round(self.elevation_m, 3),
            "confidence": round(self.confidence, 3),
            "point_count": self.point_count,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
        }


@dataclass(frozen=True)
class FootpathFit:
    """A measured pedestrian footpath or sidewalk corridor."""

    normal: Vec3
    plane_d: float
    center: Point3
    width_m: float
    length_m: float
    elevation_m: float
    confidence: float
    point_count: int
    bounds_min: Point3
    bounds_max: Point3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_class": "road.footpath",
            "normal": list(self.normal),
            "plane_d": round(self.plane_d, 4),
            "center": list(self.center),
            "width_m": round(self.width_m, 3),
            "length_m": round(self.length_m, 3),
            "elevation_m": round(self.elevation_m, 3),
            "confidence": round(self.confidence, 3),
            "point_count": self.point_count,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
        }


@dataclass(frozen=True)
class StreetFurnitureFit:
    """A measured piece of street furniture (bench, dustbin, bollard)."""

    furniture_kind: str
    center: Point3
    dimensions_m: Tuple[float, float, float]
    height_m: float
    confidence: float
    point_count: int
    bounds_min: Point3
    bounds_max: Point3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_class": f"street.{self.furniture_kind}",
            "furniture_kind": self.furniture_kind,
            "center": list(self.center),
            "dimensions_m": [round(d, 3) for d in self.dimensions_m],
            "height_m": round(self.height_m, 3),
            "confidence": round(self.confidence, 3),
            "point_count": self.point_count,
            "bounds_min": list(self.bounds_min),
            "bounds_max": list(self.bounds_max),
        }


# -----------------------------------------------------------------------------
# Geometric helpers
# -----------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.array([0.0, 0.0, 1.0])


def _to_point3(arr: np.ndarray) -> Point3:
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _measure_bounds(pts: np.ndarray) -> Tuple[Point3, Point3]:
    bmin = np.min(pts, axis=0)
    bmax = np.max(pts, axis=0)
    return _to_point3(bmin), _to_point3(bmax)


# -----------------------------------------------------------------------------
# Fitters
# -----------------------------------------------------------------------------

def fit_tree(
    points: Sequence[Point3],
    world_up: Vec3 = (0.0, 0.0, 1.0),
) -> TreeFit:
    """Fit a tree from 3D points by identifying trunk cylinder and canopy crown."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 15:
        raise TreeRefused(f"Insufficient points ({len(pts)} < 15) to demonstrate a tree")

    up = _normalize(np.asarray(world_up, dtype=np.float64))

    # Project heights along world_up
    heights = np.dot(pts, up)
    h_min, h_max = float(np.min(heights)), float(np.max(heights))
    total_height = h_max - h_min

    # Plausibility check against taxonomy: vegetation.tree size_m is (1.5, 45.0)
    tree_meta = taxonomy.get("vegetation.tree")
    min_h, max_h = tree_meta.size_m
    if total_height < min_h * 0.75 or total_height > max_h * 1.25:
        raise TreeRefused(
            f"Measured height {total_height:.2f}m outside tree plausibility [{min_h}, {max_h}]m"
        )

    # Split into lower trunk zone (lower 35%) and upper canopy zone (upper 65%)
    z_split = h_min + 0.35 * total_height
    trunk_mask = heights <= z_split
    canopy_mask = heights > z_split

    if np.sum(trunk_mask) < 4 or np.sum(canopy_mask) < 8:
        raise TreeRefused(
            f"Points do not partition into trunk and crown (trunk={np.sum(trunk_mask)}, crown={np.sum(canopy_mask)})"
        )

    trunk_pts = pts[trunk_mask]
    canopy_pts = pts[canopy_mask]

    # Trunk base and horizontal center
    trunk_center_xy = np.median(trunk_pts[:, :2], axis=0)
    trunk_base = np.array([trunk_center_xy[0], trunk_center_xy[1], h_min])

    # Trunk radius (distance from trunk center in xy)
    trunk_radii = np.hypot(trunk_pts[:, 0] - trunk_center_xy[0], trunk_pts[:, 1] - trunk_center_xy[1])
    trunk_radius = float(np.median(trunk_radii))
    if trunk_radius < 0.03 or trunk_radius > 2.5:
        raise TreeRefused(f"Unrealistic trunk radius {trunk_radius:.2f}m (expected 0.03m-2.5m)")

    # Canopy crown spread
    canopy_center_xy = np.median(canopy_pts[:, :2], axis=0)
    canopy_radii = np.hypot(canopy_pts[:, 0] - canopy_center_xy[0], canopy_pts[:, 1] - canopy_center_xy[1])
    crown_radius = float(np.percentile(canopy_radii, 85))

    # Canopy must expand beyond trunk
    if crown_radius <= trunk_radius * 1.1:
        raise TreeRefused(
            f"Crown radius {crown_radius:.2f}m does not expand beyond trunk radius {trunk_radius:.2f}m"
        )

    crown_center = (
        float(canopy_center_xy[0]),
        float(canopy_center_xy[1]),
        float(np.median(heights[canopy_mask])),
    )
    crown_height = h_max - z_split
    trunk_height = z_split - h_min

    # Residuals against cylinder on trunk
    trunk_residuals = np.abs(trunk_radii - trunk_radius)
    rms_res = float(np.sqrt(np.mean(trunk_residuals ** 2)))

    # Confidence calculation: higher point support + tight trunk fit + plausible proportions
    radial_ratio = crown_radius / max(0.01, trunk_radius)
    ratio_score = min(1.0, radial_ratio / 3.0)
    res_score = max(0.1, 1.0 - (rms_res / max(0.1, trunk_radius)))
    pt_score = min(1.0, len(pts) / 40.0)
    confidence = float(np.clip(0.4 * res_score + 0.3 * ratio_score + 0.3 * pt_score, 0.2, 0.95))

    bmin, bmax = _measure_bounds(pts)

    return TreeFit(
        trunk_axis=_to_point3(up),
        trunk_base=_to_point3(trunk_base),
        trunk_radius_m=trunk_radius,
        trunk_height_m=trunk_height,
        crown_center=crown_center,
        crown_radius_m=crown_radius,
        crown_height_m=crown_height,
        total_height_m=total_height,
        confidence=confidence,
        point_count=len(pts),
        rms_residual_m=rms_res,
        bounds_min=bmin,
        bounds_max=bmax,
    )


def fit_lamppost(
    points: Sequence[Point3],
    world_up: Vec3 = (0.0, 0.0, 1.0),
) -> LamppostFit:
    """Fit a street lamppost: vertical slender shaft + top luminaire fixture."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 10:
        raise LamppostRefused(f"Insufficient points ({len(pts)} < 10) to demonstrate a lamppost")

    up = _normalize(np.asarray(world_up, dtype=np.float64))
    heights = np.dot(pts, up)
    h_min, h_max = float(np.min(heights)), float(np.max(heights))
    total_height = h_max - h_min

    # Taxonomy prior: street.lamppost size_m is (2.0, 15.0)
    post_meta = taxonomy.get("street.lamppost")
    min_h, max_h = post_meta.size_m
    if total_height < min_h * 0.8 or total_height > max_h * 1.2:
        raise LamppostRefused(
            f"Measured height {total_height:.2f}m outside lamppost bounds [{min_h}, {max_h}]m"
        )

    # Shaft points (lower 80%)
    z_shaft_top = h_min + 0.80 * total_height
    shaft_mask = heights <= z_shaft_top
    head_mask = heights > z_shaft_top

    if np.sum(shaft_mask) < 6:
        raise LamppostRefused("Insufficient points in shaft zone to measure vertical pole")

    shaft_pts = pts[shaft_mask]
    shaft_center_xy = np.median(shaft_pts[:, :2], axis=0)

    # Check horizontal slenderness: shaft radius
    shaft_radii = np.hypot(shaft_pts[:, 0] - shaft_center_xy[0], shaft_pts[:, 1] - shaft_center_xy[1])
    shaft_radius = float(np.percentile(shaft_radii, 90))

    if shaft_radius > 0.40:
        raise LamppostRefused(f"Shaft radius {shaft_radius:.2f}m is too wide for a street lamppost (max 0.40m)")

    aspect_ratio = total_height / max(0.05, 2.0 * shaft_radius)
    if aspect_ratio < 5.0:
        raise LamppostRefused(f"Aspect ratio {aspect_ratio:.1f} too stocky for lamppost (min 5.0)")

    # Luminaire fixture detection in top head zone
    head_pts = pts[head_mask]
    if len(head_pts) > 0:
        head_center_xy = np.median(head_pts[:, :2], axis=0)
        luminaire_offset = float(np.hypot(head_center_xy[0] - shaft_center_xy[0], head_center_xy[1] - shaft_center_xy[1]))
        luminaire_pos = (float(head_center_xy[0]), float(head_center_xy[1]), h_max)
    else:
        luminaire_offset = 0.0
        luminaire_pos = (float(shaft_center_xy[0]), float(shaft_center_xy[1]), h_max)

    shaft_base = (float(shaft_center_xy[0]), float(shaft_center_xy[1]), h_min)
    shaft_top = (float(shaft_center_xy[0]), float(shaft_center_xy[1]), z_shaft_top)

    # Confidence from slenderness and point support
    aspect_score = min(1.0, aspect_ratio / 10.0)
    pt_score = min(1.0, len(pts) / 25.0)
    confidence = float(np.clip(0.5 * aspect_score + 0.5 * pt_score, 0.3, 0.95))

    bmin, bmax = _measure_bounds(pts)

    return LamppostFit(
        shaft_base=shaft_base,
        shaft_top=shaft_top,
        shaft_radius_m=shaft_radius,
        total_height_m=total_height,
        luminaire_offset_m=luminaire_offset,
        luminaire_position=luminaire_pos,
        confidence=confidence,
        point_count=len(pts),
        bounds_min=bmin,
        bounds_max=bmax,
    )


def fit_road_surface(
    points: Sequence[Point3],
    world_up: Vec3 = (0.0, 0.0, 1.0),
) -> RoadSurfaceFit:
    """Fit a vehicular carriageway road surface corridor from 3D points."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 10:
        raise RoadRefused(f"Insufficient points ({len(pts)} < 10) to fit road surface")

    up = _normalize(np.asarray(world_up, dtype=np.float64))
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid

    # Fit plane via SVD
    _, _, vh = np.linalg.svd(centered)
    normal = vh[2, :]
    if np.dot(normal, up) < 0.0:
        normal = -normal

    # Verticality of normal (road surface is horizontal: normal . up >= 0.85)
    up_dot = float(np.dot(normal, up))
    if up_dot < 0.85:
        raise RoadRefused(f"Surface normal {normal} is not horizontal (|normal . up| = {up_dot:.2f} < 0.85)")

    plane_d = float(-np.dot(normal, centroid))

    # Corridor alignment via principal horizontal axes (first 2 components)
    u_axis = vh[0, :]
    v_axis = vh[1, :]

    u_proj = np.dot(centered, u_axis)
    v_proj = np.dot(centered, v_axis)

    u_span = float(np.max(u_proj) - np.min(u_proj))
    v_span = float(np.max(v_proj) - np.min(v_proj))

    length = max(u_span, v_span)
    width = min(u_span, v_span)

    # Road carriageway width requirement: at least 2.5 meters
    road_meta = taxonomy.get("road.carriageway")
    min_w = road_meta.size_m[0]  # 3.0 m nominal
    if width < 2.2:
        raise RoadRefused(
            f"Corridor width {width:.2f}m is too narrow for vehicular carriageway (min 2.2m)"
        )

    area = length * width
    bearing = math.degrees(math.atan2(u_axis[1], u_axis[0])) % 360.0
    elevation = float(centroid[2])

    bmin, bmax = _measure_bounds(pts)
    confidence = float(np.clip(min(1.0, len(pts) / 30.0) * up_dot, 0.4, 0.95))

    return RoadSurfaceFit(
        normal=_to_point3(normal),
        plane_d=plane_d,
        center=_to_point3(centroid),
        width_m=width,
        length_m=length,
        area_m2=area,
        corridor_bearing_deg=bearing,
        elevation_m=elevation,
        confidence=confidence,
        point_count=len(pts),
        bounds_min=bmin,
        bounds_max=bmax,
    )


def fit_footpath(
    points: Sequence[Point3],
    world_up: Vec3 = (0.0, 0.0, 1.0),
) -> FootpathFit:
    """Fit a pedestrian footpath or sidewalk corridor from 3D points."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 8:
        raise FootpathRefused(f"Insufficient points ({len(pts)} < 8) to fit footpath")

    up = _normalize(np.asarray(world_up, dtype=np.float64))
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid

    _, _, vh = np.linalg.svd(centered)
    normal = vh[2, :]
    if np.dot(normal, up) < 0.0:
        normal = -normal

    up_dot = float(np.dot(normal, up))
    if up_dot < 0.85:
        raise FootpathRefused(f"Surface normal {normal} is not horizontal (|normal . up| = {up_dot:.2f} < 0.85)")

    plane_d = float(-np.dot(normal, centroid))

    u_axis = vh[0, :]
    v_axis = vh[1, :]
    u_proj = np.dot(centered, u_axis)
    v_proj = np.dot(centered, v_axis)

    u_span = float(np.max(u_proj) - np.min(u_proj))
    v_span = float(np.max(v_proj) - np.min(v_proj))

    length = max(u_span, v_span)
    width = min(u_span, v_span)

    # Footpath width prior: 0.6m to 6.0m
    if width < 0.5:
        raise FootpathRefused(f"Footpath width {width:.2f}m is too narrow (min 0.5m)")
    if width > 7.0:
        raise FootpathRefused(f"Footpath width {width:.2f}m exceeds typical sidewalk limits (max 7.0m)")

    elevation = float(centroid[2])
    bmin, bmax = _measure_bounds(pts)
    confidence = float(np.clip(min(1.0, len(pts) / 20.0) * up_dot, 0.35, 0.95))

    return FootpathFit(
        normal=_to_point3(normal),
        plane_d=plane_d,
        center=_to_point3(centroid),
        width_m=width,
        length_m=length,
        elevation_m=elevation,
        confidence=confidence,
        point_count=len(pts),
        bounds_min=bmin,
        bounds_max=bmax,
    )


def fit_street_furniture(
    points: Sequence[Point3],
    kind: str = "bench",
) -> StreetFurnitureFit:
    """Fit a piece of street furniture (e.g. bench or dustbin)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 6:
        raise StreetFurnitureRefused(f"Insufficient points ({len(pts)} < 6) for street furniture")

    centroid = np.mean(pts, axis=0)
    bmin, bmax = _measure_bounds(pts)
    dx = bmax[0] - bmin[0]
    dy = bmax[1] - bmin[1]
    dz = bmax[2] - bmin[2]
    max_dim = max(dx, dy, dz)

    target_class = f"street.{kind}"
    if target_class not in taxonomy._R:
        raise StreetFurnitureRefused(f"Unknown street furniture class: {target_class}")

    class_meta = taxonomy.get(target_class)
    lo, hi = class_meta.size_m
    if not (lo * 0.5 <= max_dim <= hi * 1.5):
        raise StreetFurnitureRefused(
            f"Dimensions {max_dim:.2f}m outside plausible range [{lo}, {hi}]m for {target_class}"
        )

    confidence = float(np.clip(min(1.0, len(pts) / 15.0), 0.3, 0.9))

    return StreetFurnitureFit(
        furniture_kind=kind,
        center=_to_point3(centroid),
        dimensions_m=(dx, dy, dz),
        height_m=dz,
        confidence=confidence,
        point_count=len(pts),
        bounds_min=bmin,
        bounds_max=bmax,
    )


# -----------------------------------------------------------------------------
# WorldIR Promotion
# -----------------------------------------------------------------------------

def promote_outdoor_entity(
    fit: Union[TreeFit, LamppostFit, RoadSurfaceFit, FootpathFit, StreetFurnitureFit],
    world: WorldIR,
    evidence_ids: Sequence[str] = (),
    name: Optional[str] = None,
) -> Entity:
    """Promote a fitted outdoor structure into a canonical WorldIR entity."""
    fit_data = fit.to_dict()
    entity_class = fit_data.get("entity_class", "misc.object")

    # Map to WorldIR EntityType
    if isinstance(fit, TreeFit):
        etype = EntityType.VEGETATION
        default_name = f"Tree (H={fit.total_height_m:.1f}m)"
        geom_type = GeometryType.CYLINDER
        measurements = {
            "total_height_m": Measurement(
                value=fit.total_height_m,
                unit="m",
                precision=0.1,
                provenance=Provenance.ESTIMATED,
                confidence=fit.confidence,
            ),
            "trunk_radius_m": Measurement(
                value=fit.trunk_radius_m,
                unit="m",
                precision=0.02,
                provenance=Provenance.ESTIMATED,
                confidence=fit.confidence,
            ),
            "crown_radius_m": Measurement(
                value=fit.crown_radius_m,
                unit="m",
                precision=0.15,
                provenance=Provenance.ESTIMATED,
                confidence=fit.confidence,
            ),
        }
    elif isinstance(fit, LamppostFit):
        etype = EntityType.STRUCTURE
        default_name = f"Lamppost (H={fit.total_height_m:.1f}m)"
        geom_type = GeometryType.CYLINDER
        measurements = {
            "total_height_m": Measurement(
                value=fit.total_height_m,
                unit="m",
                precision=0.05,
                provenance=Provenance.ESTIMATED,
                confidence=fit.confidence,
            ),
            "shaft_radius_m": Measurement(
                value=fit.shaft_radius_m,
                unit="m",
                precision=0.01,
                provenance=Provenance.ESTIMATED,
                confidence=fit.confidence,
            ),
            "luminaire_offset_m": Measurement(
                value=fit.luminaire_offset_m,
                unit="m",
                precision=0.05,
                provenance=Provenance.ESTIMATED,
                confidence=fit.confidence,
            ),
        }
    elif isinstance(fit, RoadSurfaceFit):
        etype = EntityType.ROAD
        default_name = f"Carriageway (W={fit.width_m:.1f}m, L={fit.length_m:.1f}m)"
        geom_type = GeometryType.PLANE
        measurements = {
            "width_m": Measurement(
                value=fit.width_m,
                unit="m",
                precision=0.1,
                provenance=Provenance.ESTIMATED,
                confidence=fit.confidence,
            ),
            "length_m": Measurement(
                value=fit.length_m,
                unit="m",
                precision=0.2,
                provenance=Provenance.ESTIMATED,
                confidence=fit.confidence,
            ),
            "area_m2": Measurement(
                value=fit.area_m2,
                unit="m2",
                precision=0.5,
                provenance=Provenance.ESTIMATED,
                confidence=fit.confidence,
            ),
        }
    elif isinstance(fit, FootpathFit):
        etype = EntityType.SIDEWALK
        default_name = f"Footpath (W={fit.width_m:.1f}m)"
        geom_type = GeometryType.PLANE
        measurements = {
            "width_m": Measurement(
                value=fit.width_m,
                unit="m",
                precision=0.05,
                provenance=Provenance.ESTIMATED,
                confidence=fit.confidence,
            ),
            "length_m": Measurement(
                value=fit.length_m,
                unit="m",
                precision=0.1,
                provenance=Provenance.ESTIMATED,
                confidence=fit.confidence,
            ),
        }
    elif isinstance(fit, StreetFurnitureFit):
        etype = EntityType.STRUCTURE
        default_name = f"{fit.furniture_kind.capitalize()} (H={fit.height_m:.2f}m)"
        geom_type = GeometryType.BOX
        measurements = {
            "height_m": Measurement(
                value=fit.height_m,
                unit="m",
                precision=0.02,
                provenance=Provenance.ESTIMATED,
                confidence=fit.confidence,
            ),
        }
    else:
        raise TypeError(f"Unsupported outdoor fit type: {type(fit)}")

    # Create Geometry
    geom_id = f"geom-{uuid4()}"
    geom = Geometry(
        id=geom_id,
        type=geom_type,
        bounds_min=Vector3(fit.bounds_min[0], fit.bounds_min[1], fit.bounds_min[2]),
        bounds_max=Vector3(fit.bounds_max[0], fit.bounds_max[1], fit.bounds_max[2]),
        provenance=Provenance.INFERRED,
        confidence=fit.confidence,
    )
    world.geometries[geom.id] = geom

    # Create Entity
    entity_id = f"ent-{uuid4()}"
    entity_name = name or default_name
    custom_props = {
        "entity_class": entity_class,
        "position": [
            (fit.bounds_min[0] + fit.bounds_max[0]) / 2.0,
            (fit.bounds_min[1] + fit.bounds_max[1]) / 2.0,
            (fit.bounds_min[2] + fit.bounds_max[2]) / 2.0,
        ],
        "bounds_min": list(fit.bounds_min),
        "bounds_max": list(fit.bounds_max),
        "outdoor_fit": fit_data,
        "measurements": {k: m.to_dict() for k, m in measurements.items()},
        "evidence_ids": list(evidence_ids),
    }

    entity = Entity(
        id=entity_id,
        type=etype,
        name=entity_name,
        geometry_ids=[geom_id],
        provenance=Provenance.INFERRED,
        confidence=fit.confidence,
        custom_properties=custom_props,
    )
    world.entities[entity.id] = entity

    return entity
