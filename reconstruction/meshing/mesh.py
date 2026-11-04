"""MeshData: the real triangle-mesh artifact representation (P0.12).

MeshData now lives in ``world_ir.geometry_data`` (the canonical contract
layer). This module re-exports it for backward compatibility and keeps
the ``mesh_summary`` helper here.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from engine.math import Quat, Vec3
from world_ir.geometry_data import MeshData, Point, Triangle

Vert = Point
Tri = Triangle
UV = Tuple[float, float]


def mesh_summary(mesh: MeshData) -> dict:
    """Observed facts for provenance/reporting. watertight is reported,
    never enforced -- trimmed Poisson output is typically open."""
    (mnx, mny, mnz), (mxx, mxy, mxz) = mesh.bounds()
    extent = (mxx - mnx, mxy - mny, mxz - mnz)
    return {
        "vertices": len(mesh.vertices),
        "faces": len(mesh.triangles),
        "bounds_min": [mnx, mny, mnz],
        "bounds_max": [mxx, mxy, mxz],
        "extent_m": [round(e, 4) for e in extent],
        "watertight": mesh.is_watertight_ish(),
        "has_normals": mesh.normals is not None,
        "has_uvs": mesh.uvs is not None,
        "units": mesh.units,
        "frame": mesh.frame,
    }