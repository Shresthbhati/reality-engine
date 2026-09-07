"""Ray intersection tests against the primitive shapes, used by
IPhysicsBackend.query_raycast.
"""

from __future__ import annotations

import math
from typing import Optional

from engine.physics.math3 import Vec3
from .shapes import Box, Plane, Sphere


def ray_vs_sphere(origin: Vec3, direction: Vec3, center: Vec3, sphere: Sphere) -> Optional[float]:
    """Returns hit distance t (smallest positive) or None."""
    m = origin - center
    b = m.dot(direction)
    c = m.length_sq() - sphere.radius**2
    if c > 0.0 and b > 0.0:
        return None
    discriminant = b * b - c
    if discriminant < 0.0:
        return None
    t = -b - math.sqrt(discriminant)
    if t < 0.0:
        t = 0.0
    return t


def ray_vs_box(origin: Vec3, direction: Vec3, center: Vec3, box: Box) -> Optional[float]:
    """Slab method against an axis-aligned box centered at `center`."""
    he = box.half_extents
    box_min = (center.x - he.x, center.y - he.y, center.z - he.z)
    box_max = (center.x + he.x, center.y + he.y, center.z + he.z)
    o = (origin.x, origin.y, origin.z)
    d = (direction.x, direction.y, direction.z)

    t_min, t_max = 0.0, math.inf
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if o[i] < box_min[i] or o[i] > box_max[i]:
                return None
            continue
        t1 = (box_min[i] - o[i]) / d[i]
        t2 = (box_max[i] - o[i]) / d[i]
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None
    return t_min


def ray_vs_plane(origin: Vec3, direction: Vec3, plane: Plane) -> Optional[float]:
    denom = plane.normal.dot(direction)
    if abs(denom) < 1e-12:
        return None
    t = (plane.distance - plane.normal.dot(origin)) / denom
    if t < 0.0:
        return None
    return t
