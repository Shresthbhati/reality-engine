"""Collision primitives (spec sec 44: COLLISION_LOD_0 simple bounds,
COLLISION_LOD_1 primitive decomposition). Sphere, axis-aligned box, and
infinite plane -- enough for the falling-cube / stacked-box golden
scenes (sec 91). Convex decomposition (LOD_2) and mesh collision
(LOD_3/4) are later work.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.physics.math3 import Vec3


@dataclass(frozen=True)
class Sphere:
    radius: float

    def aabb_half_extents(self) -> Vec3:
        return Vec3(self.radius, self.radius, self.radius)

    def to_dict(self) -> dict:
        return {"kind": "sphere", "radius": self.radius}


@dataclass(frozen=True)
class Box:
    """Axis-aligned box. Does not rotate with the body's orientation --
    a known simplification for this foundation pass (true OBB collision
    needs orientation-aware SAT, deferred to when it's actually load-
    bearing for a golden scene).
    """

    half_extents: Vec3

    def aabb_half_extents(self) -> Vec3:
        return self.half_extents

    def to_dict(self) -> dict:
        return {"kind": "box", "half_extents": self.half_extents.to_dict()}


@dataclass(frozen=True)
class Plane:
    """Infinite plane: all points p with normal.dot(p) == distance."""

    normal: Vec3
    distance: float

    def __post_init__(self):
        n = self.normal.length()
        if abs(n - 1.0) > 1e-6:
            object.__setattr__(self, "normal", self.normal.normalized())

    def signed_distance(self, point: Vec3) -> float:
        return self.normal.dot(point) - self.distance

    def to_dict(self) -> dict:
        return {"kind": "plane", "normal": self.normal.to_dict(), "distance": self.distance}


Shape = Sphere | Box | Plane


def shape_from_dict(data: dict) -> Shape:
    kind = data["kind"]
    if kind == "sphere":
        return Sphere(radius=data["radius"])
    if kind == "box":
        return Box(half_extents=Vec3.from_dict(data["half_extents"]))
    if kind == "plane":
        return Plane(normal=Vec3.from_dict(data["normal"]), distance=data["distance"])
    raise ValueError(f"unknown shape kind '{kind}'")


def aabb_of(shape: Sphere | Box, center: Vec3) -> tuple[Vec3, Vec3]:
    """World-space (min, max) AABB for a sphere/box centered at `center`."""
    half = shape.aabb_half_extents()
    return (center - half, center + half)
