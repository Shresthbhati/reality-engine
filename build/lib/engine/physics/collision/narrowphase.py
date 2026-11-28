"""Narrowphase contact generation for sphere/box/plane pairs.

Each function returns a ContactGeom (normal pointing from A to B,
penetration depth >= 0, world contact point) or None if the shapes
don't overlap. Boxes are treated as axis-aligned (see shapes.Box) so
box-vs-box here is genuine AABB-vs-AABB, not full OBB/SAT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from engine.physics.math3 import Vec3
from .shapes import Box, Plane, Sphere


@dataclass(frozen=True)
class ContactGeom:
    normal: Vec3  # points from A to B
    penetration: float
    point: Vec3


def sphere_vs_sphere(pos_a: Vec3, a: Sphere, pos_b: Vec3, b: Sphere) -> Optional[ContactGeom]:
    delta = pos_b - pos_a
    dist = delta.length()
    radius_sum = a.radius + b.radius
    if dist >= radius_sum:
        return None
    normal = delta.normalized() if dist > 1e-9 else Vec3(0.0, 1.0, 0.0)
    penetration = radius_sum - dist
    point = pos_a + normal * a.radius
    return ContactGeom(normal=normal, penetration=penetration, point=point)


def sphere_vs_plane(pos: Vec3, sphere: Sphere, plane: Plane) -> Optional[ContactGeom]:
    dist = plane.signed_distance(pos)
    if dist >= sphere.radius:
        return None
    penetration = sphere.radius - dist
    point = pos - plane.normal * dist
    # Normal points from A (plane, treated as body A by convention below)
    # to B (sphere) -- i.e. along the plane normal.
    return ContactGeom(normal=plane.normal, penetration=penetration, point=point)


def box_vs_plane(pos: Vec3, box: Box, plane: Plane) -> Optional[ContactGeom]:
    he = box.half_extents
    corners = [
        pos + Vec3(sx * he.x, sy * he.y, sz * he.z)
        for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
    ]
    min_dist = min(plane.signed_distance(c) for c in corners)
    if min_dist >= 0:
        return None
    penetration = -min_dist
    deepest = min(corners, key=lambda c: plane.signed_distance(c))
    point = deepest - plane.normal * plane.signed_distance(deepest)
    return ContactGeom(normal=plane.normal, penetration=penetration, point=point)


def box_vs_box(pos_a: Vec3, a: Box, pos_b: Vec3, b: Box) -> Optional[ContactGeom]:
    delta = pos_b - pos_a
    overlap_x = (a.half_extents.x + b.half_extents.x) - abs(delta.x)
    overlap_y = (a.half_extents.y + b.half_extents.y) - abs(delta.y)
    overlap_z = (a.half_extents.z + b.half_extents.z) - abs(delta.z)
    if overlap_x <= 0 or overlap_y <= 0 or overlap_z <= 0:
        return None

    # Separate along the axis of least penetration (minimum translation vector).
    penetration = min(overlap_x, overlap_y, overlap_z)
    if penetration == overlap_x:
        normal = Vec3(1.0 if delta.x >= 0 else -1.0, 0.0, 0.0)
    elif penetration == overlap_y:
        normal = Vec3(0.0, 1.0 if delta.y >= 0 else -1.0, 0.0)
    else:
        normal = Vec3(0.0, 0.0, 1.0 if delta.z >= 0 else -1.0)

    point = pos_a + delta * 0.5
    return ContactGeom(normal=normal, penetration=penetration, point=point)
